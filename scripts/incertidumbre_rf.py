#!/usr/bin/env python3
"""
incertidumbre_rf.py — Cuantificación de Incertidumbre en Random Forest
=======================================================================
Proyecto: Detección de deslizamientos con ML — EAFIT / Landslide4Sense
Autora  : Ana Patricia Montes Pimienta

DESCRIPCIÓN
-----------
Cuantifica la incertidumbre epistémica del modelo Random Forest usando la
varianza de predicción entre los árboles individuales del ensemble.
La idea es análoga al Monte Carlo Dropout en redes neuronales: cada árbol
del RF actúa como un modelo muestreado de la distribución posterior.

TIPOS DE INCERTIDUMBRE ANALIZADOS
-----------------------------------
  1. Varianza de predicción entre árboles (incertidumbre epistémica)
     σ² = Var[P(y=1 | xᵢ, θₖ)]  para k=1,...,K árboles
     Alta varianza → el modelo no tiene confianza → zona de transición o
     región poco representada en entrenamiento (análogo a las regiones
     de datos escasos en FWI, Montoya et al. 2024).

  2. Entropía predictiva (incertidumbre total)
     H = -p·log(p) - (1-p)·log(1-p)
     Máxima en p=0.5 (mayor ambigüedad). Combina incertidumbre epistémica
     y aleatórica.

  3. Calibración del modelo
     Curva de calibración (reliability diagram): ¿coincide la probabilidad
     predicha con la frecuencia observada? Un RF calibrado tiene curva
     cercana a la diagonal.

  4. Mapas de incertidumbre espacial
     Para parches individuales: mapa pixel-wise de varianza entre árboles
     (requiere modo segmentación, aproximado con ventanas locales).

USO
---
  python scripts/incertidumbre_rf.py

  Con opciones:
  python scripts/incertidumbre_rf.py --n_samples 500 --n_estimators 300

DEPENDENCIAS
------------
  pip install scikit-learn h5py numpy matplotlib seaborn tqdm
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMG_DIR  = ROOT / "TrainData" / "img"
TRAIN_MASK_DIR = ROOT / "TrainData" / "mask"
DEFAULT_OUT    = ROOT / "results" / "incertidumbre"

CHANNEL_NAMES = [
    "S2-B2 Azul", "S2-B3 Verde", "S2-B4 Rojo", "S2-B8 NIR", "S2-B8A NIR-A",
    "S2-B11 SWIR1", "S2-B12 SWIR2", "S1-VV SAR", "S1-VH SAR",
    "ALOS DEM", "ALOS Pendiente", "S2-B5 RedEdge1", "S2-B6 RedEdge2", "S2-B7 RedEdge3",
]
CHANNEL_MEAN = [
    0.1245, 0.1438, 0.1312, 0.2891, 0.3015, 0.2134, 0.1789,
    0.0823, 0.0641, 0.4521, 0.2189, 0.3102, 0.3478, 0.3812,
]
CHANNEL_STD = [
    0.0512, 0.0621, 0.0589, 0.0934, 0.0978, 0.0734, 0.0612,
    0.0341, 0.0289, 0.2134, 0.1456, 0.0812, 0.0867, 0.0923,
]


# ════════════════════════════════════════════════════════════════════════════
# 1. CARGA Y EXTRACCIÓN DE FEATURES
# ════════════════════════════════════════════════════════════════════════════

def load_patch(img_path: Path, mask_path: Path):
    with h5py.File(img_path, "r") as f:
        key = "img" if "img" in f else list(f.keys())[0]
        patch = f[key][()].astype(np.float32)
    with h5py.File(mask_path, "r") as f:
        key = "mask" if "mask" in f else list(f.keys())[0]
        mask = f[key][()].astype(np.uint8)
    return patch, mask


def normalize(patch: np.ndarray) -> np.ndarray:
    mean = np.array(CHANNEL_MEAN, dtype=np.float32).reshape(1, 1, -1)
    std  = np.array(CHANNEL_STD,  dtype=np.float32).reshape(1, 1, -1)
    return (patch - mean) / (std + 1e-8)


def extract_features(patch: np.ndarray) -> np.ndarray:
    """32 features: 14 medias + 14 stds + NDVI + NDWI + SAR-CR + EVI."""
    eps = 1e-8
    nir, rojo, verde, azul = patch[:,:,3], patch[:,:,2], patch[:,:,1], patch[:,:,0]
    vv,  vh               = patch[:,:,7], patch[:,:,8]

    ndvi  = (nir - rojo)  / (nir + rojo  + eps)
    ndwi  = (verde - nir) / (verde + nir + eps)
    sar_cr = vh / (vv + eps)
    evi    = 2.5*(nir - rojo) / (nir + 6*rojo - 7.5*azul + 1 + eps)

    return np.concatenate([
        patch.mean(axis=(0,1)),
        patch.std(axis=(0,1)),
        [ndvi.mean(), ndwi.mean(), sar_cr.mean(), evi.mean()],
    ]).astype(np.float32)


def load_dataset(img_dir: Path, mask_dir: Path, n_samples: int, seed: int = 42):
    img_files = sorted(img_dir.glob("*.h5"))
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(img_files), size=min(n_samples, len(img_files)), replace=False))

    X_list, y_list, patches_list, masks_list = [], [], [], []
    print(f"\n📂 Cargando {len(chosen)} parches...")
    for i in tqdm(chosen, desc="Extrayendo features", unit="parche"):
        img_f  = img_files[i]
        mask_f = mask_dir / img_f.name.replace("image_", "mask_")
        if not mask_f.exists():
            continue
        patch, mask = load_patch(img_f, mask_f)
        patch_n = normalize(patch)
        X_list.append(extract_features(patch_n))
        y_list.append(int(mask.max() > 0))
        patches_list.append(patch)
        masks_list.append(mask)

    X = np.stack(X_list)
    y = np.array(y_list)
    n_pos = y.sum()
    print(f"   Dataset: {X.shape[0]} muestras | {n_pos} positivos ({100*y.mean():.1f}%)")
    return X, y, patches_list, masks_list


# ════════════════════════════════════════════════════════════════════════════
# 2. ENTRENAMIENTO RF Y CUANTIFICACIÓN DE INCERTIDUMBRE
# ════════════════════════════════════════════════════════════════════════════

def train_rf(X_train, y_train, n_estimators: int = 200, seed: int = 42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    print(f"\n🌲 Entrenando RF ({n_estimators} árboles)...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_train, y_train)

    cv_f1 = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)
    print(f"   CV F1 (5-fold): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    return rf


def compute_tree_predictions(rf, X: np.ndarray) -> np.ndarray:
    """
    Obtiene predicciones de probabilidad de CADA árbol por separado.
    Retorna array (n_samples, n_estimators) con P(y=1) por árbol.
    """
    print(f"\n🔍 Calculando predicciones por árbol ({rf.n_estimators} árboles)...")
    tree_preds = np.zeros((len(X), rf.n_estimators), dtype=np.float32)

    for k, tree in enumerate(tqdm(rf.estimators_, desc="Árboles", unit="árbol")):
        proba = tree.predict_proba(X)
        # Manejar caso de árbol con una sola clase (raro pero posible)
        if proba.shape[1] == 2:
            tree_preds[:, k] = proba[:, 1]
        else:
            # Si el árbol solo predice clase 0 o 1
            cls = tree.classes_[0]
            tree_preds[:, k] = float(cls)

    return tree_preds


def compute_uncertainty_metrics(tree_preds: np.ndarray) -> dict:
    """
    Calcula métricas de incertidumbre a partir de las predicciones por árbol.

    Returns dict con:
      p_mean      : probabilidad media del ensemble
      p_var       : varianza epistémica (varianza entre árboles)
      p_std       : desviación estándar
      entropy     : entropía predictiva H(p)
      p_q05       : percentil 5 (límite inferior IC 90%)
      p_q95       : percentil 95 (límite superior IC 90%)
      ci_width    : amplitud del intervalo de confianza 90%
    """
    p_mean = tree_preds.mean(axis=1)
    p_var  = tree_preds.var(axis=1)
    p_std  = tree_preds.std(axis=1)
    p_q05  = np.percentile(tree_preds, 5,  axis=1)
    p_q95  = np.percentile(tree_preds, 95, axis=1)

    eps = 1e-10
    entropy = -(p_mean * np.log(p_mean + eps) + (1 - p_mean) * np.log(1 - p_mean + eps))

    return {
        "p_mean":   p_mean,
        "p_var":    p_var,
        "p_std":    p_std,
        "entropy":  entropy,
        "p_q05":    p_q05,
        "p_q95":    p_q95,
        "ci_width": p_q95 - p_q05,
    }


def compute_patch_uncertainty_map(patch: np.ndarray, rf, window: int = 16) -> dict:
    """
    Genera mapa espacial de incertidumbre para un parche 128×128×14 usando
    ventanas deslizantes. Cada ventana produce un vector de features,
    se clasifica con cada árbol y se computa la varianza local.

    Parámetro window: tamaño de ventana en píxeles (debe dividir 128).
    Retorna mapas de baja resolución escalados a 128×128.
    """
    H, W, C = patch.shape
    step    = window
    n_rows  = H // step
    n_cols  = W // step

    patch_n = normalize(patch)
    feat_map = []
    for r in range(n_rows):
        row_feats = []
        for c in range(n_cols):
            win = patch_n[r*step:(r+1)*step, c*step:(c+1)*step, :]
            row_feats.append(extract_features(win))
        feat_map.append(row_feats)

    feat_grid = np.array(feat_map).reshape(n_rows * n_cols, -1)  # (64, 32)

    # Predicciones por árbol
    tree_p = np.zeros((len(feat_grid), rf.n_estimators), dtype=np.float32)
    for k, tree in enumerate(rf.estimators_):
        proba = tree.predict_proba(feat_grid)
        tree_p[:, k] = proba[:, 1] if proba.shape[1] == 2 else float(tree.classes_[0])

    p_mean = tree_p.mean(axis=1).reshape(n_rows, n_cols)
    p_var  = tree_p.var(axis=1).reshape(n_rows, n_cols)

    # Escalar a 128×128 con interpolación nearest
    scale = step
    p_mean_full = np.kron(p_mean, np.ones((scale, scale), dtype=np.float32))
    p_var_full  = np.kron(p_var,  np.ones((scale, scale), dtype=np.float32))

    return {"p_mean": p_mean_full, "p_var": p_var_full}


# ════════════════════════════════════════════════════════════════════════════
# 3. ANÁLISIS DE CALIBRACIÓN
# ════════════════════════════════════════════════════════════════════════════

def calibration_analysis(p_mean: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Reliability diagram: fracción de positivos observados vs. probabilidad predicha.
    Retorna (bin_centers, fraction_pos, mean_confidence, ece).
    ECE = Expected Calibration Error.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    fraction_pos, mean_conf, bin_counts = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_mean >= lo) & (p_mean < hi)
        n = mask.sum()
        bin_counts.append(n)
        if n > 0:
            fraction_pos.append(y_true[mask].mean())
            mean_conf.append(p_mean[mask].mean())
        else:
            fraction_pos.append(np.nan)
            mean_conf.append(bin_centers[len(mean_conf)])

    fraction_pos = np.array(fraction_pos)
    mean_conf    = np.array(mean_conf)
    bin_counts   = np.array(bin_counts)
    total        = bin_counts.sum()

    # ECE = suma ponderada del error de calibración por bin
    ece = np.nansum(bin_counts / total * np.abs(fraction_pos - mean_conf))

    return bin_centers, fraction_pos, mean_conf, ece


# ════════════════════════════════════════════════════════════════════════════
# 4. FIGURAS
# ════════════════════════════════════════════════════════════════════════════

def fig_uncertainty_distributions(metrics: dict, y_true: np.ndarray, out_path: Path):
    """
    Distribuciones de varianza y entropía predictiva separadas por clase.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pos_mask = y_true == 1
    neg_mask = y_true == 0

    for ax, (key, label, xlabel) in zip(axes, [
        ("p_var",   "Varianza epistémica",   "Varianza entre árboles σ²"),
        ("entropy", "Entropía predictiva H", "Entropía H(p)"),
        ("ci_width","Amplitud IC 90%",        "P95 − P05"),
    ]):
        vals = metrics[key]
        bins = np.linspace(vals.min(), vals.max(), 40)
        ax.hist(vals[neg_mask], bins=bins, alpha=0.7, color="#2E5FA3",
                label="Sin deslizamiento", density=True)
        ax.hist(vals[pos_mask], bins=bins, alpha=0.7, color="#D97706",
                label="Deslizamiento", density=True)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Densidad", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

        # Estadísticas
        m_pos = vals[pos_mask].mean()
        m_neg = vals[neg_mask].mean()
        ax.axvline(m_pos, color="#D97706", lw=1.5, ls="--",
                   label=f"μ desliz.={m_pos:.4f}")
        ax.axvline(m_neg, color="#2E5FA3", lw=1.5, ls="--",
                   label=f"μ no-desliz.={m_neg:.4f}")
        ax.legend(fontsize=7.5)

    fig.suptitle("Distribución de Incertidumbre del RF por Clase\n"
                 "(varianza entre árboles = incertidumbre epistémica)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✔ {out_path.name}")


def fig_reliability_diagram(p_mean: np.ndarray, y_true: np.ndarray, out_path: Path):
    """Reliability diagram (curva de calibración) + histograma de confianza."""
    bin_centers, frac_pos, mean_conf, ece = calibration_analysis(p_mean, y_true)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    # ── Reliability diagram ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Calibración perfecta")
    valid = ~np.isnan(frac_pos)
    ax.plot(mean_conf[valid], frac_pos[valid], "o-",
            color="#D97706", lw=2, ms=7, label=f"Random Forest (ECE={ece:.4f})")

    # Zona de sobre/sub confianza
    ax.fill_between([0,1],[0,1],[1,1], alpha=0.07, color="red",
                    label="Sobre-confianza")
    ax.fill_between([0,1],[0,0],[0,1], alpha=0.07, color="blue",
                    label="Infra-confianza")

    ax.set_xlabel("Confianza media predicha P(y=1)", fontsize=11)
    ax.set_ylabel("Fracción de positivos observados", fontsize=11)
    ax.set_title(f"Curva de Calibración (Reliability Diagram)\nECE = {ece:.4f}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines[["top","right"]].set_visible(False)

    # ── Histograma de confianza ──────────────────────────────────────────
    ax = axes[1]
    ax.hist(p_mean[y_true == 0], bins=20, alpha=0.7, color="#2E5FA3",
            label="Sin deslizamiento", density=True)
    ax.hist(p_mean[y_true == 1], bins=20, alpha=0.7, color="#D97706",
            label="Deslizamiento", density=True)
    ax.axvline(0.5, color="red", lw=1.2, ls="--", label="Umbral p=0.5")
    ax.set_xlabel("Probabilidad predicha P(y=1)", fontsize=11)
    ax.set_ylabel("Densidad", fontsize=11)
    ax.set_title("Distribución de confianza\npor clase real", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✔ {out_path.name}")


def fig_uncertainty_vs_error(metrics: dict, y_true: np.ndarray, out_path: Path):
    """
    Scatter: incertidumbre (varianza) vs. error de predicción.
    Verifica si alta varianza → mayor tasa de error (ideal: sí).
    """
    p_mean = metrics["p_mean"]
    p_var  = metrics["p_var"]
    y_pred = (p_mean >= 0.5).astype(int)
    errors = (y_pred != y_true).astype(int)

    # Discretizar varianza en bins
    n_bins = 10
    var_bins = np.linspace(0, p_var.max(), n_bins + 1)
    bin_centers = (var_bins[:-1] + var_bins[1:]) / 2
    error_rates, counts = [], []

    for lo, hi in zip(var_bins[:-1], var_bins[1:]):
        mask = (p_var >= lo) & (p_var < hi)
        n = mask.sum()
        counts.append(n)
        error_rates.append(errors[mask].mean() if n > 0 else np.nan)

    error_rates = np.array(error_rates)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Scatter varianza vs. predicción ─────────────────────────────────
    ax = axes[0]
    colors_sc = ["#2E5FA3" if e == 0 else "#EF4444" for e in errors]
    ax.scatter(p_var, p_mean, c=colors_sc, s=8, alpha=0.4, linewidths=0)
    ax.axhline(0.5, color="gray", lw=1, ls="--")
    ax.set_xlabel("Varianza epistémica σ² (entre árboles)", fontsize=11)
    ax.set_ylabel("Probabilidad media del ensemble P(y=1)", fontsize=11)
    ax.set_title("Incertidumbre vs. Predicción\n(rojo = error de clasificación)",
                 fontsize=11, fontweight="bold")

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#2E5FA3", label="Correcto"),
        mpatches.Patch(color="#EF4444", label="Error"),
    ], fontsize=9)
    ax.spines[["top","right"]].set_visible(False)

    # ── Tasa de error por bin de varianza ────────────────────────────────
    ax = axes[1]
    valid = ~np.isnan(error_rates)
    bar_colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, valid.sum()))

    ax.bar(bin_centers[valid], error_rates[valid],
           width=(var_bins[1]-var_bins[0]) * 0.85,
           color=bar_colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Varianza epistémica σ² (bin)", fontsize=11)
    ax.set_ylabel("Tasa de error de clasificación", fontsize=11)
    ax.set_title("Tasa de Error vs. Incertidumbre\n"
                 "(alta varianza → mayor error → modelo lo sabe)",
                 fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Correlación
    corr = np.corrcoef(p_var, errors)[0,1]
    fig.text(0.5, -0.02,
             f"Correlación Pearson (σ², error): r = {corr:.4f}   —   "
             "Interpretación: r > 0 indica que la varianza predice el error (buen indicador de incertidumbre)",
             ha="center", fontsize=9, style="italic", color="#4B5563")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✔ {out_path.name}")


def fig_spatial_uncertainty(patch: np.ndarray, mask: np.ndarray,
                            rf, out_path: Path):
    """
    Mapa espacial de incertidumbre para un parche ejemplo.
    Muestra: RGB | Probabilidad media | Varianza epistémica | GT máscara
    """
    print("   → Calculando mapa espacial de incertidumbre (ventanas 16×16)...")
    unc_map = compute_patch_uncertainty_map(patch, rf, window=16)

    rgb = patch[:, :, [2, 1, 0]]
    rgb = np.clip((rgb - rgb.min()) / (rgb.ptp() + 1e-8), 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    panels = [
        (rgb,                   "RGB Falso Color (S2)",            "gray",   False),
        (unc_map["p_mean"],     "P(deslizamiento) — Ensemble RF",  "RdYlGn_r", True),
        (unc_map["p_var"],      "Varianza σ² (incertidumbre epist.)","YlOrRd", True),
        (mask.astype(float),    "Máscara real (Ground Truth)",      "Reds",   True),
    ]

    for ax, (data, title, cmap, cbar) in zip(axes, panels):
        if title.startswith("RGB"):
            ax.imshow(np.clip(rgb, 0, 1))
        else:
            im = ax.imshow(data, cmap=cmap, vmin=0,
                           vmax=1 if "P(" in title or "Máscara" in title else None)
            if cbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Mapa Espacial de Incertidumbre Epistémica — RF (ventana 16×16 px)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✔ {out_path.name}")


def fig_convergence_with_trees(X: np.ndarray, y: np.ndarray,
                               n_estimators_max: int, seed: int,
                               out_path: Path):
    """
    Convergencia de F1 y varianza media en función del número de árboles.
    Muestra a partir de cuántos árboles la incertidumbre se estabiliza.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                               stratify=y, random_state=seed)

    checkpoints = list(range(10, n_estimators_max + 1, max(1, n_estimators_max // 20)))
    f1_list, var_list = [], []

    print(f"\n📈 Curva de convergencia ({len(checkpoints)} puntos)...")
    rf_full = RandomForestClassifier(
        n_estimators=n_estimators_max, class_weight="balanced",
        min_samples_leaf=2, n_jobs=-1, random_state=seed,
    )
    rf_full.fit(X_tr, y_tr)

    for k in tqdm(checkpoints, desc="Árboles", unit="checkpoint"):
        sub_trees = rf_full.estimators_[:k]
        tree_p = np.zeros((len(X_te), k), dtype=np.float32)
        for i, tree in enumerate(sub_trees):
            proba = tree.predict_proba(X_te)
            tree_p[:, i] = proba[:, 1] if proba.shape[1] == 2 else float(tree.classes_[0])

        p_mean = tree_p.mean(axis=1)
        y_pred = (p_mean >= 0.5).astype(int)
        f1_list.append(f1_score(y_te, y_pred, zero_division=0))
        var_list.append(tree_p.var(axis=1).mean())

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(checkpoints, f1_list, "o-", color="#2E5FA3", ms=5, lw=2, label="F1-Score")
    ax2.plot(checkpoints, var_list, "s--", color="#D97706", ms=4, lw=1.8,
             alpha=0.9, label="Varianza media σ²")

    ax1.set_xlabel("Número de árboles K", fontsize=11)
    ax1.set_ylabel("F1-Score", fontsize=11, color="#2E5FA3")
    ax2.set_ylabel("Varianza epistémica media σ²", fontsize=11, color="#D97706")
    ax1.tick_params(axis="y", labelcolor="#2E5FA3")
    ax2.tick_params(axis="y", labelcolor="#D97706")

    # Convergencia aproximada: cuando la varianza cae < 10% de su rango
    var_arr = np.array(var_list)
    thresh = var_arr.max() - 0.1 * (var_arr.max() - var_arr.min())
    converge_idx = np.argmax(var_arr <= thresh)
    if converge_idx > 0:
        ax1.axvline(checkpoints[converge_idx], color="gray", lw=1.2,
                    ls=":", label=f"Convergencia ≈ {checkpoints[converge_idx]} árboles")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")

    ax1.set_title("Convergencia del RF: F1 y Varianza Epistémica vs. Número de Árboles",
                  fontsize=12, fontweight="bold")
    ax1.spines[["top"]].set_visible(False)
    ax1.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✔ {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# REPORTE EN CONSOLA
# ════════════════════════════════════════════════════════════════════════════

def print_uncertainty_report(metrics: dict, y_true: np.ndarray):
    p_mean = metrics["p_mean"]
    p_var  = metrics["p_var"]
    entropy = metrics["entropy"]

    y_pred = (p_mean >= 0.5).astype(int)
    from sklearn.metrics import f1_score, roc_auc_score, classification_report
    f1  = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, p_mean)

    _, frac_pos, _, ece = calibration_analysis(p_mean, y_true)

    # Separar incertidumbre por clase
    pos_var = p_var[y_true == 1].mean()
    neg_var = p_var[y_true == 0].mean()
    pos_ent = entropy[y_true == 1].mean()
    neg_ent = entropy[y_true == 0].mean()

    # Errores con alta vs. baja incertidumbre
    threshold_var = np.percentile(p_var, 75)
    high_unc = p_var >= threshold_var
    err_high = (y_pred[high_unc]  != y_true[high_unc]).mean()
    err_low  = (y_pred[~high_unc] != y_true[~high_unc]).mean()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║     REPORTE DE INCERTIDUMBRE — RANDOM FOREST            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Rendimiento del ensemble:")
    print(f"    F1-Score : {f1:.4f}")
    print(f"    AUC-ROC  : {auc:.4f}")
    print(f"    ECE      : {ece:.4f}  (0 = calibración perfecta)")
    print()
    print(f"  Incertidumbre epistémica (varianza entre árboles):")
    print(f"    Media global          : {p_var.mean():.6f}")
    print(f"    En deslizamientos     : {pos_var:.6f}")
    print(f"    En no-deslizamientos  : {neg_var:.6f}")
    print(f"    Ratio pos/neg         : {pos_var/neg_var:.2f}x")
    print()
    print(f"  Entropía predictiva:")
    print(f"    En deslizamientos     : {pos_ent:.4f}")
    print(f"    En no-deslizamientos  : {neg_ent:.4f}")
    print()
    print(f"  Tasa de error por nivel de incertidumbre:")
    print(f"    Alta varianza (Q75+)  : {err_high:.3f}  ({100*err_high:.1f}%)")
    print(f"    Baja varianza (Q0-75) : {err_low:.3f}  ({100*err_low:.1f}%)")
    print(f"    → Ratio alta/baja     : {err_high/err_low:.2f}x")
    print()
    if err_high > err_low:
        print("  ✅ La varianza es un buen indicador de incertidumbre:")
        print("     mayor varianza → mayor tasa de error (comportamiento deseable).")
    else:
        print("  ⚠️  La varianza no discrimina bien los errores.")
    print()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Cuantificación de incertidumbre RF — Landslide4Sense",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--img_dir",      type=Path, default=TRAIN_IMG_DIR)
    p.add_argument("--mask_dir",     type=Path, default=TRAIN_MASK_DIR)
    p.add_argument("--n_samples",    type=int,  default=500)
    p.add_argument("--n_estimators", type=int,  default=200, help="Árboles del RF")
    p.add_argument("--output_dir",   type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--skip_spatial", action="store_true",
                   help="Omitir mapa espacial (más rápido)")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  CUANTIFICACIÓN DE INCERTIDUMBRE — RF / LANDSLIDE4SENSE")
    print("  EAFIT — Ana Patricia Montes Pimienta")
    print("=" * 65)

    # 1. Cargar datos
    X, y, patches, masks = load_dataset(
        args.img_dir, args.mask_dir, args.n_samples, args.seed
    )

    # 2. Entrenar RF
    rf = train_rf(X, y, args.n_estimators, args.seed)

    # 3. Predicciones por árbol
    tree_preds = compute_tree_predictions(rf, X)
    metrics    = compute_uncertainty_metrics(tree_preds)

    # 4. Reporte consola
    print_uncertainty_report(metrics, y)

    # 5. Figuras
    print("📊 Generando figuras...")
    fig_uncertainty_distributions(
        metrics, y,
        args.output_dir / "distribucion_incertidumbre.png",
    )
    fig_reliability_diagram(
        metrics["p_mean"], y,
        args.output_dir / "curva_calibracion.png",
    )
    fig_uncertainty_vs_error(
        metrics, y,
        args.output_dir / "varianza_vs_error.png",
    )
    fig_convergence_with_trees(
        X, y, args.n_estimators, args.seed,
        args.output_dir / "convergencia_arboles.png",
    )

    # Mapa espacial: primer parche positivo
    if not args.skip_spatial:
        for patch, mask in zip(patches, masks):
            if mask.max() > 0:
                fig_spatial_uncertainty(
                    patch, mask, rf,
                    args.output_dir / "mapa_incertidumbre_espacial.png",
                )
                break

    print(f"\n✅ Figuras guardadas en: {args.output_dir}")
    print("   Archivos generados:")
    for f in sorted(args.output_dir.glob("*.png")):
        print(f"   • {f.name}")


if __name__ == "__main__":
    main()
