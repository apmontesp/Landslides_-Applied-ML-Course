"""
tests/test_dataset.py — Tests unitarios para src/dataset.py

Ejecutar con:
    pytest tests/test_dataset.py -v
    pytest tests/test_dataset.py -v --data_root ./data  # Con datos reales
"""

import sys
import os
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TrainingConfig, get_debug_config, N_CHANNELS, PATCH_SIZE
from src.dataset import (
    normalize_patch,
    minmax_patch,
    Augmenter,
    Landslide4SenseDataset,
    get_fold_indices,
)


# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_patch():
    """Parche sintético (128, 128, 14) con valores aleatorios."""
    np.random.seed(42)
    return np.random.uniform(0.0, 1.0, (PATCH_SIZE, PATCH_SIZE, N_CHANNELS)).astype(np.float32)


@pytest.fixture
def dummy_mask():
    """Máscara binaria sintética con ~5% de píxeles positivos."""
    mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    mask[60:65, 60:65] = 1
    return mask


@pytest.fixture
def debug_cfg():
    """Configuración de debug para tests."""
    cfg = get_debug_config()
    cfg.data_root = "./data"
    return cfg


# ────────────────────────────────────────────────────────────
# Tests de normalización
# ────────────────────────────────────────────────────────────

class TestNormalization:

    def test_normalize_patch_shape(self, dummy_patch):
        """La normalización Z-score no cambia la forma del parche."""
        result = normalize_patch(dummy_patch)
        assert result.shape == dummy_patch.shape

    def test_normalize_patch_dtype(self, dummy_patch):
        """La normalización devuelve float32."""
        result = normalize_patch(dummy_patch)
        assert result.dtype == np.float32

    def test_normalize_patch_zero_mean(self, dummy_patch):
        """Tras normalización Z-score, la media por canal debe ser ~0."""
        result = normalize_patch(dummy_patch)
        for c in range(N_CHANNELS):
            chan_mean = result[:, :, c].mean()
            # La media no será exactamente 0 porque usamos estadísticas globales del dataset
            assert abs(chan_mean) < 2.0, f"Canal {c}: media demasiado lejos de cero ({chan_mean:.3f})"

    def test_minmax_patch_range(self, dummy_patch):
        """Min-max debe producir valores en [0, 1] por canal."""
        result = minmax_patch(dummy_patch)
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_minmax_patch_shape(self, dummy_patch):
        """Min-max no cambia la forma."""
        result = minmax_patch(dummy_patch)
        assert result.shape == dummy_patch.shape

    def test_normalize_patch_custom_stats(self, dummy_patch):
        """Permite especificar mean y std personalizados."""
        mean = [0.5] * N_CHANNELS
        std  = [0.1] * N_CHANNELS
        result = normalize_patch(dummy_patch, mean=mean, std=std)
        assert result.shape == dummy_patch.shape


# ────────────────────────────────────────────────────────────
# Tests de Augmentation
# ────────────────────────────────────────────────────────────

class TestAugmenter:

    def test_augmenter_preserves_shape(self, dummy_patch, dummy_mask, debug_cfg):
        """El augmenter no cambia la forma del parche ni de la máscara."""
        aug = Augmenter(debug_cfg)
        for _ in range(10):
            p_aug, m_aug = aug(dummy_patch.copy(), dummy_mask.copy())
            assert p_aug.shape == dummy_patch.shape
            assert m_aug.shape == dummy_mask.shape

    def test_augmenter_mask_binary(self, dummy_patch, dummy_mask, debug_cfg):
        """Tras augmentación, la máscara sigue siendo binaria (0 o 1)."""
        aug = Augmenter(debug_cfg)
        for _ in range(10):
            _, m_aug = aug(dummy_patch.copy(), dummy_mask.copy())
            unique_vals = np.unique(m_aug)
            assert all(v in [0, 1] for v in unique_vals), f"Valores no binarios: {unique_vals}"

    def test_augmenter_no_mask(self, dummy_patch, debug_cfg):
        """El augmenter funciona sin máscara (mask=None)."""
        aug = Augmenter(debug_cfg)
        p_aug, m_aug = aug(dummy_patch.copy(), None)
        assert p_aug.shape == dummy_patch.shape
        assert m_aug is None

    def test_augmenter_preserves_label(self, dummy_patch, dummy_mask, debug_cfg):
        """La augmentación no debe cambiar si hay deslizamiento (máx de la máscara)."""
        aug = Augmenter(debug_cfg)
        original_label = int(dummy_mask.max() > 0)
        for _ in range(20):
            _, m_aug = aug(dummy_patch.copy(), dummy_mask.copy())
            aug_label = int(m_aug.max() > 0)
            assert aug_label == original_label, "La augmentación cambió la etiqueta del parche."

    def test_augmenter_flip_probability(self, dummy_patch, debug_cfg):
        """Con p=1.0, el parche siempre cambia. Con p=0.0, nunca cambia."""
        # Flip siempre
        cfg_always = get_debug_config()
        cfg_always.aug_hflip_prob  = 1.0
        cfg_always.aug_vflip_prob  = 0.0
        cfg_always.aug_rotate90_prob = 0.0
        cfg_always.aug_brightness_prob = 0.0
        cfg_always.aug_noise_prob  = 0.0
        aug = Augmenter(cfg_always)
        p_aug, _ = aug(dummy_patch.copy(), None)
        assert not np.allclose(p_aug, dummy_patch), "Con p=1.0, el parche debería cambiar."

        # Sin augmentación
        cfg_never = get_debug_config()
        cfg_never.aug_hflip_prob   = 0.0
        cfg_never.aug_vflip_prob   = 0.0
        cfg_never.aug_rotate90_prob = 0.0
        cfg_never.aug_brightness_prob = 0.0
        cfg_never.aug_noise_prob   = 0.0
        aug_off = Augmenter(cfg_never)
        p_noaug, _ = aug_off(dummy_patch.copy(), None)
        assert np.allclose(p_noaug, dummy_patch), "Sin augmentación, el parche no debería cambiar."


# ────────────────────────────────────────────────────────────
# Tests de Dataset (con datos sintéticos)
# ────────────────────────────────────────────────────────────

class TestLandslide4SenseDataset:
    """Tests usando archivos .h5 sintéticos creados en un directorio temporal."""

    @pytest.fixture(autouse=True)
    def setup_temp_dataset(self, tmp_path):
        """Crea un dataset sintético en tmp_path."""
        import h5py

        self.img_dir  = tmp_path / "img"
        self.mask_dir = tmp_path / "mask"
        self.img_dir.mkdir()
        self.mask_dir.mkdir()

        np.random.seed(42)
        self.n_samples = 20
        self.labels = []

        for i in range(self.n_samples):
            fname = f"sample_{i:04d}.h5"
            patch = np.random.uniform(0, 1, (PATCH_SIZE, PATCH_SIZE, N_CHANNELS)).astype(np.float32)
            mask  = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
            label = i % 2  # Alternamente positivo y negativo
            if label:
                mask[60:65, 60:65] = 1
            self.labels.append(label)

            with h5py.File(self.img_dir / fname, "w")  as f: f.create_dataset("img",  data=patch)
            with h5py.File(self.mask_dir / fname, "w") as f: f.create_dataset("mask", data=mask)

    def test_dataset_length(self):
        ds = Landslide4SenseDataset(
            img_dir=str(self.img_dir), mask_dir=str(self.mask_dir), normalize=False
        )
        assert len(ds) == self.n_samples

    def test_dataset_item_shape(self):
        ds = Landslide4SenseDataset(
            img_dir=str(self.img_dir), mask_dir=str(self.mask_dir), normalize=False
        )
        item = ds[0]
        assert item["image"].shape == (N_CHANNELS, PATCH_SIZE, PATCH_SIZE)
        assert item["label"].shape == ()
        assert isinstance(item["filename"], str)

    def test_dataset_label_type(self):
        ds = Landslide4SenseDataset(
            img_dir=str(self.img_dir), mask_dir=str(self.mask_dir), normalize=False
        )
        for i in range(self.n_samples):
            item = ds[i]
            label = item["label"].item()
            assert label in [0.0, 1.0], f"Etiqueta inesperada: {label}"

    def test_dataset_labels_match(self):
        ds = Landslide4SenseDataset(
            img_dir=str(self.img_dir), mask_dir=str(self.mask_dir), normalize=False
        )
        for i in range(self.n_samples):
            item = ds[i]
            assert int(item["label"].item()) == self.labels[i], \
                f"Etiqueta en índice {i}: esperada {self.labels[i]}, obtenida {item['label'].item()}"

    def test_dataset_with_indices(self):
        indices = [0, 2, 4, 6]
        ds = Landslide4SenseDataset(
            img_dir=str(self.img_dir), mask_dir=str(self.mask_dir),
            indices=indices, normalize=False
        )
        assert len(ds) == len(indices)

    def test_dataset_normalize(self):
        ds = Landslide4SenseDataset(
            img_dir=str(self.img_dir), mask_dir=str(self.mask_dir), normalize=True
        )
        item = ds[0]
        # Valores normalizados no deben estar todos en [0,1]
        assert item["image"].dtype == torch.float32

    def test_dataset_no_mask_dir(self):
        ds = Landslide4SenseDataset(
            img_dir=str(self.img_dir), mask_dir=None, normalize=False
        )
        item = ds[0]
        assert "mask" not in item
        assert item["label"].item() == -1

    def test_dataset_segmentation_task(self):
        ds = Landslide4SenseDataset(
            img_dir=str(self.img_dir), mask_dir=str(self.mask_dir),
            normalize=False, task="segmentation"
        )
        item = ds[0]
        assert "mask" in item
        assert item["mask"].shape == (1, PATCH_SIZE, PATCH_SIZE)

    def test_get_labels(self):
        ds = Landslide4SenseDataset(
            img_dir=str(self.img_dir), mask_dir=str(self.mask_dir), normalize=False
        )
        labels = ds.get_labels()
        assert len(labels) == self.n_samples
        assert set(labels).issubset({0, 1})

    def test_fold_indices(self):
        ds = Landslide4SenseDataset(
            img_dir=str(self.img_dir), mask_dir=str(self.mask_dir), normalize=False
        )
        folds = get_fold_indices(ds, n_folds=2, seed=42)
        assert len(folds) == 2
        for train_idx, val_idx in folds:
            assert len(train_idx) + len(val_idx) == self.n_samples
            assert len(set(train_idx) & set(val_idx)) == 0  # Sin solapamiento


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
