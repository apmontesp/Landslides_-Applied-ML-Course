"""
config.py — Configuración centralizada del proyecto Landslide4Sense.

Todas las rutas, hiperparámetros y constantes se definen aquí para
facilitar la reproducibilidad y el ajuste de experimentos.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

# ────────────────────────────────────────────────────────────
# Constantes del dataset
# ────────────────────────────────────────────────────────────

# Número de canales de entrada (14 bandas multiespectrales)
N_CHANNELS: int = 14

# Tamaño del parche (H × W)
PATCH_SIZE: int = 128

# Nombres de los canales para visualización y logging
CHANNEL_NAMES: List[str] = [
    "S2-B2 Azul",       # 0
    "S2-B3 Verde",      # 1
    "S2-B4 Rojo",       # 2
    "S2-B8 NIR",        # 3
    "S2-B8A NIR-A",     # 4
    "S2-B11 SWIR1",     # 5
    "S2-B12 SWIR2",     # 6
    "S1-VV SAR",        # 7
    "S1-VH SAR",        # 8
    "ALOS DEM",         # 9
    "ALOS Pendiente",   # 10
    "S2-B5 RedEdge1",   # 11
    "S2-B6 RedEdge2",   # 12
    "S2-B7 RedEdge3",   # 13
]

# Índices de grupos de canales
CHANNELS_SENTINEL2_OPT: List[int] = [0, 1, 2, 3, 4, 5, 6]
CHANNELS_SAR:           List[int] = [7, 8]
CHANNELS_DEM:           List[int] = [9, 10]
CHANNELS_RED_EDGE:      List[int] = [11, 12, 13]

# Índices RGB para visualización (Rojo, Verde, Azul)
RGB_CHANNELS: Tuple[int, int, int] = (2, 1, 0)

# Estadísticas del dataset (media y std por canal, estimadas sobre TrainData)
# Calculadas con eda_landslide4sense.py sobre muestra representativa
CHANNEL_MEAN: List[float] = [
    0.1245, 0.1438, 0.1312, 0.2891, 0.3015, 0.2134, 0.1789,  # Sentinel-2 óptico
    0.0823, 0.0641,                                             # SAR VV, VH
    0.4521, 0.2189,                                             # DEM, Pendiente
    0.3102, 0.3478, 0.3812,                                     # Red-Edge
]

CHANNEL_STD: List[float] = [
    0.0512, 0.0621, 0.0589, 0.0934, 0.0978, 0.0734, 0.0612,
    0.0341, 0.0289,
    0.2134, 0.1456,
    0.0812, 0.0867, 0.0923,
]

# Peso de la clase positiva (deslizamiento) calculado del EDA
# n_neg / n_pos = 1568 / 2231 ≈ 0.703
POS_WEIGHT: float = 0.703


# ────────────────────────────────────────────────────────────
# Dataclass de configuración
# ────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Configuración completa de un experimento de entrenamiento."""

    # ── Identificación ──────────────────────────────────────
    experiment_name: str = "resnet50_finetuned"
    model_arch: str = "resnet50"          # resnet50 | efficientnet_b4 | unet_resnet34 | random_forest

    # ── Rutas ───────────────────────────────────────────────
    data_root: str = "./data"
    output_dir: str = "./results"
    checkpoint_dir: str = "./checkpoints"

    # ── Dataset ─────────────────────────────────────────────
    n_channels: int = N_CHANNELS
    patch_size: int = PATCH_SIZE
    n_folds: int = 5
    val_fold: int = 0                     # Índice del fold de validación (0-4)
    seed: int = 42

    # ── Entrenamiento ────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # ── Optimizador ──────────────────────────────────────────
    optimizer: str = "adamw"              # adamw | sgd | adam
    lr_head: float = 1e-4                 # LR para capas nuevas (cabeza)
    lr_backbone: float = 1e-5            # LR para backbone preentrenado
    weight_decay: float = 1e-4
    freeze_epochs: int = 5               # Épocas con backbone congelado

    # ── Scheduler ────────────────────────────────────────────
    scheduler: str = "cosine"            # cosine | step | plateau
    T_max: int = 50                      # Para CosineAnnealingLR
    eta_min: float = 1e-7

    # ── Función de pérdida ───────────────────────────────────
    loss: str = "weighted_bce"           # weighted_bce | dice_bce | focal
    pos_weight: float = POS_WEIGHT
    dice_weight: float = 0.5             # Para dice_bce: peso del Dice Loss

    # ── Umbral de decisión ───────────────────────────────────
    threshold: float = 0.5
    optimize_threshold: bool = True      # Buscar umbral óptimo post-entrenamiento

    # ── Early stopping ────────────────────────────────────────
    early_stopping: bool = True
    patience: int = 15
    monitor: str = "val_f1"             # Métrica a monitorear
    mode: str = "max"                   # max para F1, min para loss

    # ── Data augmentation ────────────────────────────────────
    augmentation: bool = True
    aug_hflip_prob: float = 0.5
    aug_vflip_prob: float = 0.5
    aug_rotate90_prob: float = 0.5
    aug_brightness_prob: float = 0.3
    aug_noise_prob: float = 0.2
    aug_noise_std: float = 0.02

    # ── Normalización ────────────────────────────────────────
    normalize: bool = True
    norm_strategy: str = "per_channel"  # per_channel | global | minmax

    # ── Preentrenamiento ──────────────────────────────────────
    pretrained: bool = True
    pretrained_weights: str = "imagenet"

    # ── Hardware ─────────────────────────────────────────────
    device: str = "auto"                 # auto | cuda | cpu | mps

    # ── Logging ──────────────────────────────────────────────
    log_interval: int = 10              # Pasos entre logs
    save_best_only: bool = True
    tensorboard: bool = True
    verbose: bool = True

    # ── Opciones adicionales por arquitectura ─────────────────
    extra: dict = field(default_factory=dict)

    # ────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Carga configuración desde un archivo YAML."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Separar campos extra no definidos en el dataclass
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        extra = {k: v for k, v in data.items() if k not in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.extra = extra
        return cfg

    def to_yaml(self, path: str) -> None:
        """Serializa la configuración a YAML."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, allow_unicode=True)

    def resolve_device(self) -> str:
        """Resuelve el dispositivo de cómputo disponible."""
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def __repr__(self) -> str:
        lines = [f"TrainingConfig(experiment='{self.experiment_name}')"]
        for k, v in self.__dict__.items():
            if k != "extra":
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# Configuraciones rápidas (para tests y demos)
# ────────────────────────────────────────────────────────────

def get_debug_config() -> TrainingConfig:
    """Configuración mínima para debug rápido (2 épocas, batch pequeño)."""
    return TrainingConfig(
        experiment_name="debug",
        epochs=2,
        batch_size=8,
        n_folds=2,
        early_stopping=False,
        tensorboard=False,
        augmentation=False,
        verbose=True,
    )


def get_resnet50_config(data_root: str = "./data", output_dir: str = "./results") -> TrainingConfig:
    """Configuración por defecto para ResNet-50 fine-tuned."""
    return TrainingConfig(
        experiment_name="resnet50_finetuned",
        model_arch="resnet50",
        data_root=data_root,
        output_dir=output_dir,
        epochs=50,
        batch_size=32,
        lr_head=1e-4,
        lr_backbone=1e-5,
        freeze_epochs=5,
        loss="weighted_bce",
        augmentation=True,
    )


def get_efficientnet_config(data_root: str = "./data", output_dir: str = "./results") -> TrainingConfig:
    """Configuración por defecto para EfficientNet-B4 fine-tuned."""
    return TrainingConfig(
        experiment_name="efficientnet_b4_finetuned",
        model_arch="efficientnet_b4",
        data_root=data_root,
        output_dir=output_dir,
        epochs=50,
        batch_size=24,              # EfficientNet-B4 es más grande → menor batch
        lr_head=1e-4,
        lr_backbone=1e-5,
        freeze_epochs=5,
        loss="weighted_bce",
        augmentation=True,
    )


def get_unet_config(data_root: str = "./data", output_dir: str = "./results") -> TrainingConfig:
    """Configuración por defecto para U-Net + ResNet-34 segmentación."""
    return TrainingConfig(
        experiment_name="unet_resnet34_segmentation",
        model_arch="unet_resnet34",
        data_root=data_root,
        output_dir=output_dir,
        epochs=60,
        batch_size=16,
        lr_head=1e-4,
        lr_backbone=1e-5,
        freeze_epochs=0,            # U-Net se entrena completo desde el inicio
        loss="dice_bce",
        dice_weight=0.5,
        monitor="val_iou",
        augmentation=True,
    )
