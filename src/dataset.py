"""
dataset.py — Clases Dataset PyTorch para Landslide4Sense.

Implementa:
  - Landslide4SenseDataset : Dataset base que lee parches .h5
  - FoldDataset            : Dataset con máscara de índices para K-Fold CV
  - get_dataloaders()      : Función helper para crear DataLoaders
  - get_transforms()       : Augmentaciones reproducibles por partición
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import (
    CHANNEL_MEAN,
    CHANNEL_STD,
    N_CHANNELS,
    PATCH_SIZE,
    TrainingConfig,
)


# ────────────────────────────────────────────────────────────
# Utilidades de transformación
# ────────────────────────────────────────────────────────────

def normalize_patch(
    patch: np.ndarray,
    mean: List[float] = CHANNEL_MEAN,
    std:  List[float] = CHANNEL_STD,
) -> np.ndarray:
    """Normalización Z-score por canal. patch: (H, W, C) → float32."""
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    std_arr  = np.array(std,  dtype=np.float32).reshape(1, 1, -1)
    return (patch - mean_arr) / (std_arr + 1e-8)


def minmax_patch(patch: np.ndarray) -> np.ndarray:
    """Normalización min-max por canal a [0, 1]."""
    mins = patch.min(axis=(0, 1), keepdims=True)
    maxs = patch.max(axis=(0, 1), keepdims=True)
    return (patch - mins) / (maxs - mins + 1e-8)


class Augmenter:
    """Augmentaciones geométricas y de intensidad para parches multicanal."""

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg

    def __call__(
        self, patch: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        patch: (H, W, C) float32
        mask:  (H, W)    uint8 o None
        """
        # Flip horizontal
        if random.random() < self.cfg.aug_hflip_prob:
            patch = np.flip(patch, axis=1).copy()
            if mask is not None:
                mask = np.flip(mask, axis=1).copy()

        # Flip vertical
        if random.random() < self.cfg.aug_vflip_prob:
            patch = np.flip(patch, axis=0).copy()
            if mask is not None:
                mask = np.flip(mask, axis=0).copy()

        # Rotación 90° (0, 90, 180, 270)
        if random.random() < self.cfg.aug_rotate90_prob:
            k = random.randint(1, 3)
            patch = np.rot90(patch, k=k, axes=(0, 1)).copy()
            if mask is not None:
                mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

        # Perturbación de brillo por canal (solo canales ópticos 0-6)
        if random.random() < self.cfg.aug_brightness_prob:
            factors = np.random.uniform(0.8, 1.2, size=(1, 1, patch.shape[2]))
            factors[:, :, 7:] = 1.0   # No perturbar SAR/DEM
            patch = (patch * factors).astype(np.float32)

        # Ruido gaussiano
        if random.random() < self.cfg.aug_noise_prob:
            noise = np.random.normal(0, self.cfg.aug_noise_std, patch.shape).astype(np.float32)
            patch = patch + noise

        return patch, mask


# ────────────────────────────────────────────────────────────
# Dataset base
# ────────────────────────────────────────────────────────────

class Landslide4SenseDataset(Dataset):
    """
    Dataset para Landslide4Sense.

    Lee parches .h5 con forma (128, 128, 14) y máscaras binarias (128, 128).
    Retorna tensores (C, H, W) para el parche y (1, H, W) para la máscara
    (tarea de segmentación) o escalar para clasificación de parche.

    Args:
        img_dir:        Directorio con archivos imagen .h5
        mask_dir:       Directorio con archivos máscara .h5 (None para test/val sin etiquetas)
        indices:        Lista de índices a usar (para K-Fold). None = todos.
        transform:      Función de augmentation (Augmenter o None)
        normalize:      Aplicar normalización Z-score
        task:           'classification' | 'segmentation'
        cfg:            TrainingConfig (opcional, para parámetros de normalización)
    """

    def __init__(
        self,
        img_dir: str,
        mask_dir: Optional[str] = None,
        indices: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        task: str = "classification",
        cfg: Optional[TrainingConfig] = None,
    ):
        self.img_dir   = Path(img_dir)
        self.mask_dir  = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.normalize = normalize
        self.task      = task
        self.cfg       = cfg

        # Listar archivos imagen ordenados
        all_img_files = sorted(self.img_dir.glob("*.h5"))
        if not all_img_files:
            raise FileNotFoundError(f"No se encontraron archivos .h5 en {self.img_dir}")

        if indices is not None:
            self.img_files = [all_img_files[i] for i in indices]
        else:
            self.img_files = all_img_files

        # Verificar que existan las máscaras correspondientes
        if self.mask_dir is not None:
            self.mask_files = []
            for img_f in self.img_files:
                mask_f = self.mask_dir / img_f.name
                if not mask_f.exists():
                    raise FileNotFoundError(f"Máscara no encontrada: {mask_f}")
                self.mask_files.append(mask_f)
        else:
            self.mask_files = None

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Dict:
        # ── Cargar parche ──────────────────────────────────
        with h5py.File(self.img_files[idx], "r") as f:
            # Buscar la key de imagen (puede variar entre versiones del dataset)
            key = "img" if "img" in f else list(f.keys())[0]
            patch = f[key][()].astype(np.float32)   # (H, W, C)

        # ── Cargar máscara ─────────────────────────────────
        mask = None
        label = -1
        if self.mask_files is not None:
            with h5py.File(self.mask_files[idx], "r") as f:
                mkey = "mask" if "mask" in f else list(f.keys())[0]
                mask = f[mkey][()].astype(np.float32)  # (H, W)

            # Etiqueta de parche: 1 si cualquier píxel es deslizamiento
            label = int(mask.max() > 0)

        # ── Normalización ─────────────────────────────────
        if self.normalize:
            if self.cfg and hasattr(self.cfg, "norm_strategy"):
                if self.cfg.norm_strategy == "minmax":
                    patch = minmax_patch(patch)
                else:
                    patch = normalize_patch(patch)
            else:
                patch = normalize_patch(patch)

        # ── Augmentation ──────────────────────────────────
        if self.transform is not None:
            patch, mask = self.transform(patch, mask)

        # ── Convertir a tensores (C, H, W) ────────────────
        patch_t = torch.from_numpy(patch.transpose(2, 0, 1))   # (C, H, W)

        result = {
            "image":    patch_t,
            "label":    torch.tensor(label, dtype=torch.float32),
            "filename": self.img_files[idx].stem,
        }

        if mask is not None:
            if self.task == "segmentation":
                result["mask"] = torch.from_numpy(mask).unsqueeze(0)   # (1, H, W)
            else:
                result["mask"] = torch.from_numpy(mask)                 # (H, W)

        return result

    def get_labels(self) -> np.ndarray:
        """Devuelve array de etiquetas (0/1) para todos los parches. Útil para stratify en K-Fold."""
        if self.mask_files is None:
            raise ValueError("No hay máscaras disponibles para calcular etiquetas.")
        labels = []
        for mf in self.mask_files:
            with h5py.File(mf, "r") as f:
                mkey = "mask" if "mask" in f else list(f.keys())[0]
                mask = f[mkey][()]
            labels.append(int(mask.max() > 0))
        return np.array(labels)


# ────────────────────────────────────────────────────────────
# K-Fold helpers
# ────────────────────────────────────────────────────────────

def get_fold_indices(
    dataset: Landslide4SenseDataset,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """
    Genera índices de train/val para K-Fold estratificado.

    Returns:
        Lista de (train_indices, val_indices) por fold.
    """
    from sklearn.model_selection import StratifiedKFold

    labels = dataset.get_labels()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    indices = np.arange(len(dataset))
    for train_idx, val_idx in skf.split(indices, labels):
        folds.append((train_idx.tolist(), val_idx.tolist()))
    return folds


# ────────────────────────────────────────────────────────────
# DataLoaders
# ────────────────────────────────────────────────────────────

def get_dataloaders(
    cfg: TrainingConfig,
    train_indices: Optional[List[int]] = None,
    val_indices:   Optional[List[int]] = None,
) -> Dict[str, DataLoader]:
    """
    Crea DataLoaders de entrenamiento y validación.

    Si train_indices / val_indices son None, usa la partición completa de TrainData.
    """
    data_root = Path(cfg.data_root)
    img_dir   = data_root / "TrainData" / "img"
    mask_dir  = data_root / "TrainData" / "mask"

    augmenter = Augmenter(cfg) if cfg.augmentation else None

    train_ds = Landslide4SenseDataset(
        img_dir=str(img_dir),
        mask_dir=str(mask_dir),
        indices=train_indices,
        transform=augmenter,
        normalize=cfg.normalize,
        task="segmentation" if cfg.model_arch == "unet_resnet34" else "classification",
        cfg=cfg,
    )

    val_ds = Landslide4SenseDataset(
        img_dir=str(img_dir),
        mask_dir=str(mask_dir),
        indices=val_indices,
        transform=None,              # Sin augmentation en validación
        normalize=cfg.normalize,
        task="segmentation" if cfg.model_arch == "unet_resnet34" else "classification",
        cfg=cfg,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return {"train": train_loader, "val": val_loader}


def get_test_loader(
    cfg: TrainingConfig,
    partition: str = "ValidData",
) -> DataLoader:
    """Crea DataLoader para partición de validación o test (sin etiquetas)."""
    data_root = Path(cfg.data_root)
    img_dir   = data_root / partition / "img"

    ds = Landslide4SenseDataset(
        img_dir=str(img_dir),
        mask_dir=None,
        transform=None,
        normalize=cfg.normalize,
        cfg=cfg,
    )

    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
