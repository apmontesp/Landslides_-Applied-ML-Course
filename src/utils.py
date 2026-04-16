"""
utils.py — Utilidades generales para el proyecto Landslide4Sense.

Implementa:
  - set_seed()          : Semilla reproducible para Python, NumPy y PyTorch
  - AverageMeter        : Acumulador de métricas con media móvil
  - format_time()       : Formatea segundos a mm:ss o hh:mm:ss
  - save_checkpoint()   : Guarda estado de modelo y optimizador
  - load_checkpoint()   : Carga estado desde checkpoint
  - visualize_patch()   : Visualiza un parche multibanda en composiciones clave
  - visualize_prediction(): Superpone predicción sobre parche original
  - plot_channel_stats(): Estadísticas por canal (mean ± std)
  - get_device()        : Selección automática de dispositivo (CUDA / MPS / CPU)
  - count_h5_files()    : Cuenta archivos .h5 en un directorio
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ────────────────────────────────────────────────────────────
# Reproducibilidad
# ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Establece la semilla aleatoria en todos los marcos para reproducibilidad.

    Fija: Python random, NumPy, PyTorch CPU y GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Determinismo en cuDNN (ligera pérdida de rendimiento)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ────────────────────────────────────────────────────────────
# Medición de métricas
# ────────────────────────────────────────────────────────────

class AverageMeter:
    """Calcula y almacena la media móvil de un valor."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self) -> str:
        return f"AverageMeter(avg={self.avg:.4f}, count={self.count})"


# ────────────────────────────────────────────────────────────
# Tiempo
# ────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Formatea segundos a una cadena legible (hh:mm:ss o mm:ss)."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


# ────────────────────────────────────────────────────────────
# Checkpoints
# ────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: str,
    extra: Optional[Dict] = None,
) -> None:
    """Guarda el estado del modelo y el optimizador en disco."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "metrics":    metrics,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Carga un checkpoint en el modelo (y opcionalmente en el optimizador).

    Returns:
        El diccionario completo del checkpoint.
    """
    map_location = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Checkpoint cargado: {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt


# ────────────────────────────────────────────────────────────
# Dispositivo
# ────────────────────────────────────────────────────────────

def get_device(prefer: str = "auto") -> torch.device:
    """
    Selecciona el dispositivo de cómputo disponible.

    Args:
        prefer: 'auto' | 'cuda' | 'mps' | 'cpu'
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ────────────────────────────────────────────────────────────
# Utilidades del dataset
# ────────────────────────────────────────────────────────────

def count_h5_files(directory: str) -> int:
    """Cuenta archivos .h5 en un directorio (sin recursión)."""
    return len(list(Path(directory).glob("*.h5")))


def check_data_structure(data_root: str) -> Dict:
    """
    Verifica que el dataset esté correctamente estructurado.

    Returns:
        Diccionario con conteos por partición y estado de validez.
    """
    import h5py

    root  = Path(data_root)
    info  = {}
    valid = True

    for partition, has_mask in [("TrainData", True), ("ValidData", False), ("TestData", False)]:
        img_dir  = root / partition / "img"
        mask_dir = root / partition / "mask"

        n_img  = count_h5_files(str(img_dir))  if img_dir.exists()  else -1
        n_mask = count_h5_files(str(mask_dir)) if mask_dir.exists() else -1

        status = "OK" if n_img > 0 else "ERROR"
        if has_mask and n_mask != n_img:
            status = "MISMATCH"
            valid  = False

        info[partition] = {
            "n_img":  n_img,
            "n_mask": n_mask if has_mask else "N/A",
            "status": status,
        }

        print(f"[{status:7s}] {partition}/img  → {n_img:4d} archivos .h5")
        if has_mask:
            print(f"[{status:7s}] {partition}/mask → {n_mask:4d} archivos .h5")

    # Verificar forma del primer parche
    try:
        first_img  = next((root / "TrainData" / "img").glob("*.h5"))
        first_mask = next((root / "TrainData" / "mask").glob("*.h5"))
        with h5py.File(first_img, "r") as f:
            key   = list(f.keys())[0]
            shape = f[key].shape
        with h5py.File(first_mask, "r") as f:
            mkey   = list(f.keys())[0]
            mshape = f[mkey].shape
        print(f"\n[OK] Forma de parche: {shape}  dtype=float32")
        print(f"[OK] Forma de máscara: {mshape}  dtype=uint8")
        info["patch_shape"] = shape
        info["mask_shape"]  = mshape
    except Exception as e:
        print(f"[WARN] No se pudo verificar la forma: {e}")
        valid = False

    info["valid"] = valid
    return info


# ────────────────────────────────────────────────────────────
# Visualización de parches
# ────────────────────────────────────────────────────────────

def _clip_percentile(arr: np.ndarray, pct: float = 2.0) -> np.ndarray:
    """Recorta percentiles para visualización de imágenes."""
    lo, hi = np.percentile(arr, pct), np.percentile(arr, 100 - pct)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def visualize_patch(
    patch: np.ndarray,
    mask: Optional[np.ndarray] = None,
    title: str = "Parche Landslide4Sense",
    output_path: Optional[str] = None,
    rgb_idx: Tuple[int, int, int] = (2, 1, 0),
    nir_idx: Tuple[int, int, int] = (3, 2, 1),
    sar_idx: int = 7,
    dem_idx: int = 9,
) -> None:
    """
    Visualiza un parche en cuatro composiciones:
      1. Color verdadero (RGB)
      2. Falso color NIR
      3. SAR VV
      4. DEM
    Si mask no es None, agrega una quinta columna con overlay.

    Args:
        patch:   (H, W, C) float32 (puede estar normalizado o en bruto)
        mask:    (H, W) binaria, o None
    """
    if not HAS_MPL:
        print("matplotlib no disponible.")
        return

    n_cols = 5 if mask is not None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # RGB
    rgb = np.stack([patch[:, :, rgb_idx[i]] for i in range(3)], axis=-1)
    axes[0].imshow(_clip_percentile(rgb))
    axes[0].set_title("RGB")

    # Falso color NIR
    nir = np.stack([patch[:, :, nir_idx[i]] for i in range(3)], axis=-1)
    axes[1].imshow(_clip_percentile(nir))
    axes[1].set_title("Falso Color NIR")

    # SAR VV
    sar = patch[:, :, sar_idx]
    axes[2].imshow(_clip_percentile(sar), cmap="gray")
    axes[2].set_title("SAR VV")

    # DEM
    dem = patch[:, :, dem_idx]
    im  = axes[3].imshow(dem, cmap="terrain")
    plt.colorbar(im, ax=axes[3], fraction=0.046)
    axes[3].set_title("DEM")

    # Máscara overlay
    if mask is not None:
        rgb_disp = _clip_percentile(rgb)
        axes[4].imshow(rgb_disp)
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask > 0] = [1.0, 0.2, 0.2, 0.5]   # Rojo semi-transparente
        axes[4].imshow(overlay)
        red_patch = mpatches.Patch(color="red", alpha=0.5, label="Deslizamiento")
        axes[4].legend(handles=[red_patch], loc="lower right", fontsize=9)
        axes[4].set_title("Máscara Overlay")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def visualize_prediction(
    patch: np.ndarray,
    pred_mask: np.ndarray,
    true_mask: Optional[np.ndarray] = None,
    title: str = "Predicción vs. Verdad",
    output_path: Optional[str] = None,
    rgb_idx: Tuple[int, int, int] = (2, 1, 0),
) -> None:
    """
    Muestra la predicción del modelo sobre el parche original.
    Columnas: RGB | Prob. pred | Pred binarizada | True mask (si disponible)
    """
    if not HAS_MPL:
        return

    n_cols = 4 if true_mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    rgb = _clip_percentile(np.stack([patch[:, :, i] for i in rgb_idx], axis=-1))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Original")

    # Mapa de probabilidad (heatmap)
    im = axes[1].imshow(pred_mask, cmap="RdYlGn_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[1].set_title("Prob. Deslizamiento")

    # Predicción binarizada
    binary = (pred_mask >= 0.5).astype(float)
    axes[2].imshow(rgb)
    overlay = np.zeros((*binary.shape, 4))
    overlay[binary > 0] = [1.0, 0.2, 0.2, 0.5]
    axes[2].imshow(overlay)
    axes[2].set_title("Pred. Binarizada")

    if true_mask is not None:
        axes[3].imshow(true_mask, cmap="Reds", vmin=0, vmax=1)
        axes[3].set_title("Máscara Real")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_channel_stats(
    mean_pos: np.ndarray,
    mean_neg: np.ndarray,
    channel_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Gráfico de barras dobles: media por canal para clases positiva y negativa.
    Útil para visualizar la discriminabilidad de cada canal.
    """
    if not HAS_MPL:
        return

    n = len(mean_pos)
    names = channel_names or [f"Ch{i:02d}" for i in range(n)]
    x = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Media por clase
    width = 0.35
    axes[0].bar(x - width/2, mean_pos, width, label="Deslizamiento", color="#e74c3c", alpha=0.8)
    axes[0].bar(x + width/2, mean_neg, width, label="No-deslizamiento", color="#3498db", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[0].set_ylabel("Media normalizada")
    axes[0].set_title("Media por canal y clase", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Delta (positivo - negativo)
    delta = mean_pos - mean_neg
    colors = ["#e74c3c" if d > 0 else "#3498db" for d in delta]
    axes[1].bar(x, delta, color=colors, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[1].set_ylabel("Δ Media (Pos − Neg)")
    axes[1].set_title("Diferencia discriminativa por canal", fontweight="bold")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
