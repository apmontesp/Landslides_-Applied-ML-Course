#!/usr/bin/env python3
"""
features_fisicos.py — Features Informados por Física para Detección de Deslizamientos
========================================================================================
Proyecto: Detección de deslizamientos con ML — EAFIT / Landslide4Sense
Autora  : Ana Patricia Montes Pimienta

DESCRIPCIÓN
-----------
Calcula features basados en principios geomecánicos directamente a partir de
los canales DEM y Pendiente del dataset L4S, en analogía con las Physics-
Informed Neural Networks (PINNs) revisadas en Montoya et al. (2024).

FEATURES FÍSICOS IMPLEMENTADOS
--------------------------------
  1. TWI (Índice Topográfico de Humedad)
     TWI = ln(A / tan(β))
     donde A ≈ área de aportación hídrica (aproximada por kernel de flujo)
     y β = pendiente local en radianes.
     Indica acumulación de agua — alto TWI → suelo saturado → mayor riesgo.

  2. Factor de Seguridad Simplificado (FoS)
     FS = (c' + (γ·z·cos²β - u)·tan(φ')) / (γ·z·sin(β)·cos(β))
     Implementado con parámetros geotécnicos típicos de suelos andinos.
     FS < 1.0 → inestable | 1.0–1.5 → crítico | >1.5 → estable.

  3. Índice de Erodabilidad (EI)
     EI = pendiente × σ_SAR_VH
     Combina geometría topográfica con textura de retrodispersión SAR.

  4. Índice de Humedad SAR-Topográfico (SHI)
     SHI = VH × cos(β) / (TWI + ε)
     Integra señal SAR con terreno para detectar saturación del suelo.

USO
---
  python scripts/features_fisicos.py

  Con opciones:
  python scripts/features_fisicos.py --n_samples 500 --phi 30 --cohesion 15

DEPENDENCIAS
------------
  pip install scikit-learn h5py numpy matplotlib seaborn scipy tqdm
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
from scipy.ndimage import uniform_filter
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Rutas ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMG_DIR  = ROOT / "TrainData" / "img"
TRAIN_MASK_DIR = ROOT / "TrainData" / "mask"
DEFAULT_OUT    = ROOT / "results" / "features_fisicos"

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

# Parámetros geotécnicos típicos — suelos andinos tropicales
# (ajustables por argumento de línea de comandos)
GAMMA_SUELO = 18.0    # kN/m³  — peso unitario del suelo
Z_PROF      = 2.0     # m      — profundidad del plano de falla
U_PRESION   = 5.0     # kPa    — presión de poro (condición saturada parcial)


# ════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ════════════════════════════════════════════════════════════════════════════

def load_patch_raw(img_path: Path, mask_path: Path):
    """Carga parche (128×128×14) y máscara sin normalizar."""
    with h5py.File(img_path, "r") as f:
        key = "img" if "img" in f else list(f.keys())[0]
        patch = f[key][()].astype(np.float32)
    with h5py.File(mask_path, "r") as f:
        key = "mask" if "mask" in f else list(f.keys())[0]
        mask = f[key][()].astype(np.uint8)
    return patch, mask


def load_samples(img_dir: Path, mask_dir: Path, n_samples: int, seed: int = 42):
    """Carga n_samples parches. Retorna lista de (patch, mask)."""
    img_files = sorted(img_dir.glob("*.h5"))
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(img_files), size=min(n_samples, len(img_files)), replace=False))

    samples = []
    print(f"\n📂 Cargando {len(chosen)} parches...")
    for i in tqdm(chosen, desc="Cargando", unit="parche"):
        img_f  = img_files[i]
        mask_f = mask_dir / img_f.name.replace("image_", "mask_")
        if not mask_f.exists():
            continue
        patch, mask = load_patch_raw(img_f, mask_f)
        samples.append((patch, mask, img_f.stem))
    return samples


# ════════════════════════════════════════════════════════════════════════════
# 2. CÁLCULO DE FEATURES FÍSICOS (nivel de parche)
# ════════════════════════════════════════════════════════════════════════════

def denormalize_channel(values: np.ndarray, ch_idx: int) -> np.ndarray:
    """Desnormaliza un canal usando estadísticas del dataset."""
    return values * CHANNEL_STD[ch_idx] + CHANNEL_MEAN[ch_idx]


def compute_twi(dem_patch: np.ndarray, slope_patch: np.ndarray,
                kernel_size: int = 9) -> np.ndarray:
    """
    Topographic Wetness Index (TWI) — Beven & Kirkby (1979).

    TWI = ln(A / tan(β))
      A: área de aportación (aproximada con filtro de área acumulada)
      β: pendiente en radianes

    El DEM en L4S está normalizado — lo trabajamos en escala relativa.
    """
    # Aproximar área de aportación con media local (proxy de cuenca)
    # (sin DEM real en coordenadas métricas, usamos escala relativa)
    area_approx = uniform_filter(dem_patch.astype(np.float64), size=kernel_size) + 1e-6

    # Pendiente: convertir de escala normalizada a radianes
    # El canal ALOS Pendiente suele estar en [0, ~1.5] representando ~0-60°
    slope_deg = slope_patch * 60.0         # escala empírica L4S
    slope_rad = np.deg2rad(np.clip(slope_deg, 0.1, 89.9))

    twi = np.log(area_approx / np.tan(slope_rad))
    return twi.astype(np.float32)


def compute_factor_of_safety(slope_patch: np.ndarray,
                              phi_deg: float = 28.0,
                              cohesion: float = 12.0) -> np.ndarray:
    """
    Factor de Seguridad infinito en pendiente (Skempton & DeLory, 1957).

    FS = [c' + (γ·z·cos²β - u)·tan(φ')] / [γ·z·sin(β)·cos(β)]

    Parámetros típicos suelos andinos tropicales:
      φ' = 28° (ángulo de fricción interna)
      c' = 12 kPa (cohesión efectiva)
      γ  = 18 kN/m³
      z  = 2 m  (profundidad de falla)
      u  = 5 kPa (presión de poro, suelo húmedo)

    FS < 1.0  → Falla inminente
    1.0–1.5   → Zona crítica
    > 1.5     → Estable
    """
    slope_deg = slope_patch * 60.0
    slope_rad = np.deg2rad(np.clip(slope_deg, 0.1, 89.9))

    cos_b = np.cos(slope_rad)
    sin_b = np.sin(slope_rad)
    tan_phi = np.tan(np.deg2rad(phi_deg))

    sigma_n = GAMMA_SUELO * Z_PROF * cos_b**2   # tensión normal efectiva [kPa]
    tau_d   = GAMMA_SUELO * Z_PROF * sin_b * cos_b  # tensión cortante [kPa]

    fs_num = cohesion + (sigma_n - U_PRESION) * tan_phi
    fs_den = tau_d + 1e-6

    fs = fs_num / fs_den
    fs = np.clip(fs, 0.0, 5.0)   # limitar para visualización
    return fs.astype(np.float32)


def compute_sar_topo_humidity(vh_patch: np.ndarray, twi: np.ndarray) -> np.ndarray:
    """
    Índice de Humedad SAR-Topográfico (SHI) — combinación de VH y TWI.
    SHI = VH × cos(pendiente_implicita) / (TWI normalizado + ε)
    """
    twi_norm = (twi - twi.min()) / (twi.ptp() + 1e-8)
    vh_norm  = (vh_patch - vh_patch.min()) / (vh_patch.ptp() + 1e-8)
    shi = vh_norm / (twi_norm + 0.1)
    return shi.astype(np.float32)


def extract_physics_features(patch: np.ndarray, phi_deg: float = 28.0,
                              cohesion: float = 12.0) -> dict:
    """
    Extrae todos los features físicos de un parche 128×128×14.
    Retorna dict con arrays 2D (128×128) y escalares resumen.
    """
    dem_ch   = patch[:, :, 9]    # ALOS DEM
    slope_ch = patch[:, :, 10]   # ALOS Pendiente
    vh_ch    = patch[:, :, 8]    # S1-VH SAR

    twi = compute_twi(dem_ch, slope_ch)
    fos = compute_factor_of_safety(slope_ch, phi_deg, cohesion)
    shi = compute_sar_topo_humidity(vh_ch, twi)

    # Erodabilidad: pendiente × rugosidad SAR-VH
    slope_deg = slope_ch * 60.0
    erodabilidad = slope_deg * np.std(vh_ch) * (vh_ch / (vh_ch.mean() + 1e-8))

    return {
        "twi":          twi,
        "fos":          fos,
        "shi":          shi,
        "erodabilidad": erodabilidad.astype(np.float32),
        # Estadísticas resumen (escalar por parche)
        "twi_mean":   float(twi.mean()),
        "twi_std":    float(twi.std()),
        "fos_min":    float(fos.min()),
        "fos_mean":   float(fos.mean()),
        "fos_p25":    float(np.percentile(fos, 25)),
        "area_inest": float((fos < 1.0).mean()),   # fracción pixels inestables
        "shi_mean":   float(shi.mean()),
        "ero_mean":   float(erodabilidad.mean()),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. COMPARACIÓN RF: CON Y SIN FEATURES FÍSICOS
# ════════════════════════════════════════════════════════════════════════════

def build_feature_vector(patch: np.ndarray, phi_deg: float, cohesion: float) -> dict:
    """Construye vector de features espectrales + físicos para un parche."""
    mean_arr = np.array(CHANNEL_MEAN, dtype=np.float32).reshape(1, 1, -1)
    std_arr  = np.array(CHANNEL_STD,  dtype=np.float32).reshape(1, 1, -1)
    patch_n  = (patch - mean_arr) / (std_arr + 1e-8)

    eps = 1e-8
    nir, rojo, verde, azul = patch_n[:,:,3], patch_n[:,:,2], patch_n[:,:,1], patch_n[:,:,0]
    vv,  vh               = patch_n[:,:,7], patch_n[:,:,8]
    ndvi  = (nir - rojo)   / (nir + rojo   + eps)
    ndwi  = (verde - nir)  / (verde + nir  + eps)
    sar_cr = vh / (vv + eps)

    spectral = np.concatenate([
        patch_n.mean(axis=(0,1)),
        patch_n.std(axis=(0,1)),
        [ndvi.mean(), ndwi.mean(), sar_cr.mean()],
    ])

    phys  = extract_physics_features(patch, phi_deg, cohesion)
    phys_vec = np.array([
        phys["twi_mean"], phys["twi_std"],
        phys["fos_min"],  phys["fos_mean"], phys["fos_p25"],
        phys["area_inest"], phys["shi_mean"], phys["ero_mean"],
    ])

    return {"spectral": spectral, "physical": phys_vec, "both": np.concatenate([spectral, phys_vec])}


def run_comparison(samples: list, phi_deg: float, cohesion: float, seed: int = 42):
    """
    Entrena 3 RF:
      A) solo features espectrales
      B) solo features físicos
      C) espectrales + físicos (fusión)
    y compara métricas de clasificación.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import f1_score, roc_auc_score

    X_spec, X_phys, X_both, y_all = [], [], [], []

    print("\n⚙️  Extrayendo features espectrales + físicos...")
    for patch, mask, _ in tqdm(samples, desc="Features", unit="parche"):
        label = int(mask.max() > 0)
        fv = build_feature_vector(patch, phi_deg, cohesion)
        X_spec.append(fv["spectral"])
        X_phys.append(fv["physical"])
        X_both.append(fv["both"])
        y_all.append(label)

    X_spec = np.stack(X_spec)
    X_phys = np.stack(X_phys)
    X_both = np.stack(X_both)
    y_all  = np.array(y_all)

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for name, X in [("Espectral", X_spec), ("Físico", X_phys), ("Espectral + Físico", X_both)]:
        rf = RandomForestClassifier(
            n_estimators=150, class_weight="balanced",
            min_samples_leaf=2, n_jobs=-1, random_state=seed,
        )
        f1_scores = cross_val_score(rf, X, y_all, cv=skf, scoring="f1", n_jobs=-1)
        auc_scores = cross_val_score(rf, X, y_all, cv=skf, scoring="roc_auc", n_jobs=-1)
        results[name] = {
            "f1_mean": f1_scores.mean(), "f1_std": f1_scores.std(),
            "auc_mean": auc_scores.mean(), "auc_std": auc_scores.std(),
        }
        print(f"   {name:<22}: F1={f1_scores.mean():.4f}±{f1_scores.std():.4f}  "
              f"AUC={auc_scores.mean():.4f}±{auc_scores.std():.4f}")

    return results, X_spec, X_phys, X_both, y_all


# ════════════════════════════════════════════════════════════════════════════
# 4. FIGURAS
# ════════════════════════════════════════════════════════════════════════════

def fig_physics_maps(patch: np.ndarray, mask: np.ndarray, phi_deg: float,
                     cohesion: float, out_path: Path):
    """
    Visualización de mapas de features físicos para un parche ejemplo.
    Muestra: RGB | DEM | Pendiente | TWI | FS | Máscara
    """
    phys = extract_physics_features(patch, phi_deg, cohesion)

    # RGB falso color (Rojo=2, Verde=1, Azul=0)
    rgb = patch[:, :, [2, 1, 0]]
    rgb = (rgb - rgb.min()) / (rgb.ptp() + 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    panels = [
        (rgb,                   "RGB Falso Color (S2)",      "gray",    False),
        (patch[:,:,9],          "ALOS DEM (norm.)",           "terrain", True),
        (patch[:,:,10] * 60,    "Pendiente (°aprox.)",        "YlOrRd",  True),
        (phys["twi"],           "TWI — Índice Topogr. Humedad","Blues_r", True),
        (phys["fos"],           "Factor de Seguridad (FS)",   "RdYlGn",  True),
        (mask.astype(np.float32),"Máscara GT (deslizamiento)","Reds",    True),
    ]

    for ax, (data, title, cmap, show_cbar) in zip(axes.flatten(), panels):
        if title.startswith("RGB"):
            ax.imshow(np.clip(rgb, 0, 1))
        else:
            im = ax.imshow(data, cmap=cmap)
            if show_cbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    # Anotación FS
    axes[1][1].set_title(
        f"Factor de Seguridad (FS)\n"
        f"FS_min={phys['fos_min']:.2f}  FS_medio={phys['fos_mean']:.2f}  "
        f"Área inestable={100*phys['area_inest']:.1f}%",
        fontsize=9, fontweight="bold",
    )

    # Parámetros geotécnicos como pie de figura
    fig.text(0.5, 0.01,
             f"Parámetros geotécnicos: φ'={phi_deg}°  c'={cohesion} kPa  "
             f"γ={GAMMA_SUELO} kN/m³  z={Z_PROF} m  u={U_PRESION} kPa",
             ha="center", fontsize=9, style="italic", color="#4B5563")

    fig.suptitle("Features Físicos (Geomecánicos) — Parche Ejemplo L4S",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✔ {out_path.name}")


def fig_fos_distribution(samples: list, phi_deg: float, cohesion: float, out_path: Path):
    """
    Distribución del FS mínimo por parche, separado por clase (desliz. / no-desliz.).
    """
    fos_pos, fos_neg = [], []
    for patch, mask, _ in samples:
        phys = extract_physics_features(patch, phi_deg, cohesion)
        if mask.max() > 0:
            fos_pos.append(phys["fos_mean"])
        else:
            fos_neg.append(phys["fos_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histograma
    ax = axes[0]
    bins = np.linspace(0, 3, 40)
    ax.hist(fos_neg, bins=bins, alpha=0.7, color="#2E5FA3", label="Sin deslizamiento", density=True)
    ax.hist(fos_pos, bins=bins, alpha=0.7, color="#D97706", label="Deslizamiento", density=True)
    ax.axvline(1.0, color="red",    lw=1.5, ls="--", label="FS = 1.0 (falla)")
    ax.axvline(1.5, color="orange", lw=1.5, ls="--", label="FS = 1.5 (límite)")
    ax.set_xlabel("Factor de Seguridad medio por parche", fontsize=11)
    ax.set_ylabel("Densidad", fontsize=11)
    ax.set_title("Distribución del Factor de Seguridad por clase", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)

    # Boxplot
    ax = axes[1]
    data_bp = [fos_neg, fos_pos]
    bp = ax.boxplot(data_bp, patch_artist=True, notch=True,
                    medianprops={"color": "white", "linewidth": 2})
    colors_bp = ["#2E5FA3", "#D97706"]
    for patch_box, color in zip(bp["boxes"], colors_bp):
        patch_box.set_facecolor(color)
        patch_box.set_alpha(0.8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Sin deslizamiento", "Deslizamiento"], fontsize=10)
    ax.set_ylabel("FS medio", fontsize=11)
    ax.set_title("Boxplot FS por clase\n(muescas = IC 95% de la mediana)",
                 fontsize=11, fontweight="bold")
    ax.axhline(1.0, color="red",    lw=1.2, ls="--")
    ax.axhline(1.5, color="orange", lw=1.2, ls="--")
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✔ {out_path.name}")


def fig_comparison_bars(results: dict, out_path: Path):
    """Barras comparativas F1 y AUC: espectral vs. físico vs. fusión."""
    models = list(results.keys())
    f1_means  = [results[m]["f1_mean"]  for m in models]
    f1_stds   = [results[m]["f1_std"]   for m in models]
    auc_means = [results[m]["auc_mean"] for m in models]
    auc_stds  = [results[m]["auc_std"]  for m in models]

    x      = np.arange(len(models))
    width  = 0.35
    colors = ["#2E5FA3", "#16A34A", "#D97706"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, (vals, errs, metric_name) in zip(
        axes,
        [(f1_means, f1_stds, "F1-Score"), (auc_means, auc_stds, "AUC-ROC")],
    ):
        bars = ax.bar(x, vals, width*2, yerr=errs, color=colors, edgecolor="white",
                      linewidth=0.8, capsize=6, alpha=0.9,
                      error_kw={"linewidth": 1.5, "ecolor": "#374151"})
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_ylim(0.5, 1.02)
        ax.set_title(f"Comparación {metric_name} — RF 5-fold CV", fontsize=11, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Impacto de los Features Físicos (Geomecánicos) en el Rendimiento del RF",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✔ {out_path.name}")


def fig_twi_vs_fos(samples: list, phi_deg: float, cohesion: float, out_path: Path):
    """
    Scatter: TWI medio vs. FS medio, coloreado por clase.
    Muestra la separabilidad en el espacio de features físicos.
    """
    twi_vals, fos_vals, labels = [], [], []
    for patch, mask, _ in samples:
        phys = extract_physics_features(patch, phi_deg, cohesion)
        twi_vals.append(phys["twi_mean"])
        fos_vals.append(phys["fos_mean"])
        labels.append(int(mask.max() > 0))

    twi_arr = np.array(twi_vals)
    fos_arr = np.array(fos_vals)
    lab_arr = np.array(labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(twi_arr[lab_arr==0], fos_arr[lab_arr==0],
               c="#2E5FA3", alpha=0.4, s=15, label="Sin deslizamiento", linewidths=0)
    ax.scatter(twi_arr[lab_arr==1], fos_arr[lab_arr==1],
               c="#D97706", alpha=0.5, s=15, label="Deslizamiento", linewidths=0)

    ax.axhline(1.0, color="red",    lw=1.2, ls="--", alpha=0.6, label="FS = 1.0")
    ax.axhline(1.5, color="orange", lw=1.2, ls="--", alpha=0.6, label="FS = 1.5")

    ax.set_xlabel("TWI medio por parche", fontsize=11)
    ax.set_ylabel("Factor de Seguridad medio por parche", fontsize=11)
    ax.set_title("Espacio de Features Físicos: TWI vs. FS por clase\n"
                 "(Separabilidad geomecánica del riesgo de deslizamiento)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✔ {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Features físicos geomecánicos — Landslide4Sense",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--img_dir",    type=Path,  default=TRAIN_IMG_DIR)
    p.add_argument("--mask_dir",   type=Path,  default=TRAIN_MASK_DIR)
    p.add_argument("--n_samples",  type=int,   default=400, help="Número de parches")
    p.add_argument("--output_dir", type=Path,  default=DEFAULT_OUT)
    p.add_argument("--phi",        type=float, default=28.0, help="Ángulo de fricción interna φ' (grados)")
    p.add_argument("--cohesion",   type=float, default=12.0, help="Cohesión efectiva c' (kPa)")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  FEATURES FÍSICOS GEOMECÁNICOS — LANDSLIDE4SENSE / EAFIT")
    print(f"  Parámetros: φ'={args.phi}°  c'={args.cohesion} kPa")
    print("=" * 65)

    # 1. Cargar datos
    samples = load_samples(args.img_dir, args.mask_dir, args.n_samples, args.seed)
    n_pos = sum(1 for _, m, _ in samples if m.max() > 0)
    print(f"   Positivos: {n_pos}/{len(samples)} ({100*n_pos/len(samples):.1f}%)")

    # 2. Visualizar parche ejemplo (primer parche positivo)
    print("\n📊 Generando figuras...")
    for patch, mask, name in samples:
        if mask.max() > 0:
            fig_physics_maps(
                patch, mask, args.phi, args.cohesion,
                args.output_dir / "mapa_features_fisicos_ejemplo.png",
            )
            break

    # 3. Distribución FS por clase
    fig_fos_distribution(
        samples, args.phi, args.cohesion,
        args.output_dir / "distribucion_factor_seguridad.png",
    )

    # 4. Scatter TWI vs FS
    fig_twi_vs_fos(
        samples, args.phi, args.cohesion,
        args.output_dir / "scatter_twi_vs_fos.png",
    )

    # 5. Comparación RF con y sin features físicos
    print("\n🌲 Comparando RF: espectral vs. físico vs. fusión (5-fold CV)...")
    results, _, _, _, _ = run_comparison(samples, args.phi, args.cohesion, args.seed)
    fig_comparison_bars(results, args.output_dir / "comparacion_rf_features.png")

    # 6. Resumen numérico
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   RESUMEN: IMPACTO DE FEATURES FÍSICOS                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    for model, r in results.items():
        print(f"  {model:<22}: F1={r['f1_mean']:.4f}±{r['f1_std']:.4f}  "
              f"AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f}")
    best = max(results, key=lambda m: results[m]["f1_mean"])
    print(f"\n  → Mejor configuración: {best}  (F1={results[best]['f1_mean']:.4f})")

    mejora = results["Espectral + Físico"]["f1_mean"] - results["Espectral"]["f1_mean"]
    print(f"  → Mejora por features físicos: {mejora:+.4f} F1")
    print(f"\n✅ Figuras guardadas en: {args.output_dir}")


if __name__ == "__main__":
    main()
