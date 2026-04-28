"""
dataset.py — Clases Dataset PyTorch para Landslide4Sense.
Ajustado para manejar prefijos image_ y mask_ (EAFIT Research).
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

        # Rotación 90°
        if random.random() < self.cfg.aug_rotate90_prob:
            k = random.randint(1, 3)
            patch = np.rot90(patch, k=k, axes=(0, 1)).copy()
            if mask is not None:
                mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

        return patch, mask


# ────────────────────────────────────────────────────────────
# Dataset base
# ────────────────────────────────────────────────────────────

class Landslide4SenseDataset(Dataset):
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
        all_img_files = sorted(list(self.img_dir.glob("*.h5")))
        if not all_img_files:
            raise FileNotFoundError(f"No se encontraron archivos .h5 en {self.img_dir}")

        if indices is not None:
            self.img_files = [all_img_files[i] for i in indices]
        else:
            self.img_files = all_img_files

        # --- CORRECCIÓN DE NOMBRES DE MÁSCARA ---
        if self.mask_dir is not None:
            self.mask_files = []
            for img_f in self.img_files:
                # Reemplazamos el prefijo 'image_' por 'mask_' para que coincida con tu Drive
                mask_name = img_f.name.replace("image_", "mask_")
                mask_f = self.mask_dir / mask_name
                
                if not mask_f.exists():
                    raise FileNotFoundError(f"🚨 Error: No existe la máscara {mask_f.name} para la imagen {img_f.name}")
                self.mask_files.append(mask_f)
        else:
            self.mask_files = None

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Dict:
        # Cargar parche (14 bandas)
        with h5py.File(self.img_files[idx], "r") as f:
            key = "img" if "img" in f else list(f.keys())[0]
            patch = f[key][()].astype(np.float32)

        mask = None
        label = -1
        if self.mask_files is not None:
            with h5py.File(self.mask_files[idx], "r") as f:
                mkey = "mask" if "mask" in f else list(f.keys())[0]
                mask = f[mkey][()].astype(np.float32)
            label = int(mask.max() > 0)

        # Normalización
        if self.normalize:
            patch = normalize_patch(patch)

        # Augmentation
        if self.transform is not None:
            patch, mask = self.transform(patch, mask)

        # Transpose a (C, H, W) para PyTorch
        patch_t = torch.from_numpy(patch.transpose(2, 0, 1))

        result = {
            "image": patch_t,
            "label": torch.tensor(label, dtype=torch.float32),
            "filename": self.img_files[idx].stem,
        }

        if mask is not None:
            result["mask"] = torch.from_numpy(mask).unsqueeze(0) if self.task == "segmentation" else torch.from_numpy(mask)

        return result

    def get_labels(self) -> np.ndarray:
        labels = []
        for mf in self.mask_files:
            with h5py.File(mf, "r") as f:
                mkey = "mask" if "mask" in f else list(f.keys())[0]
                mask = f[mkey][()]
                labels.append(int(mask.max() > 0))
        return np.array(labels)

# ────────────────────────────────────────────────────────────
# Helpers (K-Fold y DataLoaders)
# ────────────────────────────────────────────────────────────

def get_fold_indices(dataset, n_folds=5, seed=42):
    from sklearn.model_selection import StratifiedKFold
    labels = dataset.get_labels()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(t.tolist(), v.tolist()) for t, v in skf.split(np.arange(len(dataset)), labels)]

def get_dataloaders(cfg, train_indices=None, val_indices=None):
    data_root = Path(cfg.data_root)
    img_dir, mask_dir = data_root/"TrainData"/"img", data_root/"TrainData"/"mask"
    
    params = {"img_dir": str(img_dir), "mask_dir": str(mask_dir), "normalize": cfg.normalize, "cfg": cfg}
    
    train_ds = Landslide4SenseDataset(**params, indices=train_indices, transform=Augmenter(cfg))
    val_ds = Landslide4SenseDataset(**params, indices=val_indices, transform=None)

    return {
        "train": DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True),
        "val": DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    }
