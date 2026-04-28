#!/usr/bin/env python3
"""
analisis_shap.py — Análisis de importancia de canales con valores SHAP
=======================================================================
Proyecto: Detección de deslizamientos con ML — EAFIT / Landslide4Sense
Autora  : Ana Patricia Montes Pimienta

DESCRIPCIÓN
-----------
Entrena un Random Forest sobre los parches L4S y calcula los valores SHAP
(Shapley Additive Explanations) para cuantificar la contribución de cada
canal espectral a la predicción de deslizamientos.

ANÁLISIS GENERADOS
------------------
  1. Importancia SHAP media por canal (beeswarm + bar chart)
  2. Importancia SHAP por grupo de sensores (S2 óptico, SAR, DEM, RedEdge)
  3. Correlación entre canales (cross-talk / interferencia espectral)
  4. Dependencia SHAP vs. valor de feature (scatter plots top-5 canales)

USO
---
  # Desde el directorio raíz del proyecto:
  python scripts/analisis_shap.py

  # Con opciones:
  python scripts/analisis_shap.py --n_samples 800 --output_dir results/shap

DEPENDENCIAS
------------
  pip install shap scikit-learn h5py numpy matplotlib seaborn tqdm
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")          # backend sin pantalla
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Rutas por defecto (relativas al directorio raíz del proyecto) ────────────
ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMG_DIR  = ROOT / "TrainData" / "img"
TRAIN_MASK_DIR = ROOT / "TrainData" / "mask"
DEFAULT_OUT    = ROOT / "results" / "shap"

# ── Metadatos de canales (igual que config.py) ───────────────────────────────
CHANNEL_NAMES = [
    "S2-B2 Azul",     "S2-B3 Verde",    "S2-B4 Rojo",
    "S2-B8 NIR",      "S2-B8A NIR-A",   "S2-B11 SWIR1",
    "S2-B12 SWIR2",   "S1-VV SAR",      "S1-VH SAR",
    "ALOS DEM",       "ALOS Pendiente", "S2-B5 RedEdge1",
    "S2-B6 RedEdge2", "S2-B7 RedEdge3",
]

CHANNEL_GROUPS = {
    "S2 Óptico": [0, 1, 2, 3, 4, 5, 6],
    "SAR":       [7, 8],
    "DEM":       [9, 10],
    "Red-Edge":  [11, 12, 13],
}

# Colores por grupo de sensor
GROUP_COLORS = {
    "S2 Óptico": "#2E5FA3",
    "SAR":       "#D97706",
    "DEM":       "#16A34A",
    "Red-Edge":  "#9333EA",
}

CHANNEL_MEAN = [
    0.1245, 0.1438, 0.1312, 0.2891, 0.3015, 0.2134, 0.1789,
    0.0823, 0.0641,
    0.4521, 0.2189,
    0.3102, 0.3478, 0.3812,
]
CHANNEL_STD = [
    0.0512, 0.0621, 0.0589, 0.0934, 0.0978, 0.0734, 0.0612,
    0.0341, 0.0289,
    0.2134, 0.1456,
    0.0812, 0.0867, 0.0923,
]


# ════════════════════════════════════════════════════════════════════════════
# 1. CARGA Y EXTRACCIÓN DE FEATURES
# ════════════════════════════════════════════════════════════════════════════

def load_patch(img_path: Path, mask_path: Path):
    """Carga un parche H5 y su máscara. Retorna (patch 128×128×14, label 0/1)."""
    with h5py.File(img_path, "r") as f:
        key = "img" if "img" in f else list(f.keys())[0]
        patch = f[key][()].astype(np.float32)   # (128, 128, 14)
    with h5py.File(mask_path, "r") as f:
        key = "mask" if "mask" in f else list(f.keys())[0]
        mask = f[key][()]
    label = int(mask.max() > 0)
    return patch, label


def normalize(patch: np.ndarray) -> np.ndarray:
    """Z-score por canal."""
    mean = np.array(CHANNEL_MEAN, dtype=np.float32).reshape(1, 1, -1)
    std  = np.array(CHANNEL_STD,  dtype=np.float32).reshape(1, 1, -1)
    return (patch - mean) / (std + 1e-8)


def extract_features(patch: np.ndarray) -> np.ndarray:
    """
    Extrae un vector de features por parche (128×128×14).

    Features (28 base + 4 índices espectrales = 32 features):
      - Media  por canal  (×14)
      - Std    por canal  (×14)
      - NDVI   = (NIR - Rojo) / (NIR + Rojo + ε)            [S2-B8, S2-B4]
      - NDWI   = (Verde - NIR) / (Verde + NIR + ε)           [S2-B3, S2-B8]
      - SAR-CR = VH / (VV + ε)    (cross-ratio, sensible a estructura)
      - EVI    = 2.5*(NIR-Rojo)/(NIR+6*Rojo-7.5*Azul+1)
    """
    # Canales individuales (media espacial)
    nir, rojo, verde, azul = patch[:,:,3], patch[:,:,2], patch[:,:,1], patch[:,:,0]
    vv,  vh               = patch[:,:,7], patch[:,:,8]
    eps = 1e-8

    ndvi = (nir - rojo)   / (nir + rojo   + eps)
    ndwi = (verde - nir)  / (verde + nir  + eps)
    sar_cr = vh / (vv + eps)
    evi    = 2.5 * (nir - rojo) / (nir + 6*rojo - 7.5*azul + 1 + eps)

    feat = np.concatenate([
        patch.mean(axis=(0, 1)),      # 14 medias espaciales
        patch.std(axis=(0, 1)),       # 14 desviaciones estándar
        [ndvi.mean(), ndwi.mean(), sar_cr.mean(), evi.mean()],  # 4 índices
    ])
    return feat.astype(np.float32)


def build_feature_names() -> list[str]:
    """Nombres de los 32 features extraídos."""
    names  = [f"μ_{n}" for n in CHANNEL_NAMES]
    names += [f"σ_{n}" for n in CHANNEL_NAMES]
    names += ["NDVI", "NDWI", "SAR-CR", "EVI"]
    return names


def load_dataset(img_dir: Path, mask_dir: Path, n_samples: int, seed: int = 42):
    """Carga n_samples parches aleatorios y extrae features."""
    img_files = sorted(img_dir.glob("*.h5"))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(img_files), size=min(n_samples, len(img_files)), replace=False)
    chosen = sorted(chosen)

    X_list, y_list = [], []
    print(f"\n📂 Cargando {len(chosen)} parches desde {img_dir}...")
    for i in tqdm(chosen, desc="Extrayendo features", unit="parche"):
        img_f  = img_files[i]
        mask_f = mask_dir / img_f.name.replace("image_", "mask_")
        if not mask_f.exists():
            continue
        patch, label = load_patch(img_f, mask_f)
        patch = normalize(patch)
        X_list.append(extract_features(patch))
        y_list.append(label)

    X = np.stack(X_list)
    y = np.array(y_list)
    print(f"   Dataset: {X.shape[0]} muestras | {y.sum()} positivos ({100*y.mean():.1f}%)")
    return X, y


# ════════════════════════════════════════════════════════════════════════════
# 2. ENTRENAMIENTO RF
# ════════════════════════════════════════════════════════════════════════════

def train_rf(X_train, y_train, seed: int = 42):
    """Entrena Random Forest con hiperparámetros del proyecto."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    print("\n🌲 Entrenando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_train, y_train)

    cv_f1 = cross_val_score(rf, X_train, y_train, cv=3, scoring="f1", n_jobs=-1)
    print(f"   CV F1 (3-fold): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    return rf


# ════════════════════════════════════════════════════════════════════════════
# 3. CÁLCULO DE VALORES SHAP
# ════════════════════════════════════════════════════════════════════════════

def compute_shap_values(rf, X, feature_names: list[str]):
    """Calcula valores SHAP con TreeExplainer."""
    try:
        import shap
    except ImportError:
        print("❌  Instala shap: pip install shap")
        sys.exit(1)

    print("\n🔍 Calculando valores SHAP (TreeExplainer)...")
    explainer = shap.TreeExplainer(rf)

    # Para datasets grandes, limitar a 500 muestras para velocidad
    n = min(500, len(X))
    X_sub = X[:n]
    shap_values = explainer.shap_values(X_sub)

    # shap_values puede ser list[2] (clase 0 y 1) o array único
    if isinstance(shap_values, list):
        sv = shap_values[1]   # clase positiva (deslizamiento)
    else:
        sv = shap_values

    print(f"   SHAP calculado para {n} muestras × {sv.shape[1]} features")
    return sv, X_sub, explainer


# ════════════════════════════════════════════════════════════════════════════
# 4. FIGURAS
# ════════════════════════════════════════════════════════════════════════════

def _channel_color(feat_name: str) -> str:
    """Asigna color de grupo al nombre de feature."""
    for group, idxs in CHANNEL_GROUPS.items():
        for idx in idxs:
            short = CHANNEL_NAMES[idx].replace(" ", "")
            if short in feat_name.replace(" ", "") or feat_name.endswith(CHANNEL_NAMES[idx]):
                return GROUP_COLORS[group]
    # Índices espectrales derivados
    if feat_name in ("NDVI", "EVI"):
        return GROUP_COLORS["S2 Óptico"]
    if feat_name == "NDWI":
        return GROUP_COLORS["S2 Óptico"]
    if feat_name == "SAR-CR":
        return GROUP_COLORS["SAR"]
    return "#6B7280"


def fig_shap_bar(shap_values: np.ndarray, feature_names: list[str], out_path: Path):
    """Bar chart de importancia SHAP media por feature (top-20)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = [_channel_color(feature_names[i]) for i in order]
    y_pos   = np.arange(len(order))

    ax.barh(y_pos, mean_abs[order], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia SHAP media |φᵢ|", fontsize=11)
    ax.set_title("Importancia de features — Random Forest (SHAP)", fontsize=13, fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Leyenda de grupos
    patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9, title="Sensor")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✔ {out_path.name}")


def fig_shap_by_group(shap_values: np.ndarray, feature_names: list[str], out_path: Path):
    """Bar chart agrupado por tipo de sensor (media SHAP dentro de cada grupo)."""
    mean_abs = np.abs(shap_values).mean(axis=0)

    # Mapear cada feature a su grupo
    GROUP_GROUPS = list(CHANNEL_GROUPS.keys())
    group_sums: dict = {g: [] for g in GROUP_GROUPS}
    group_sums["Índices espectrales"] = []

    indices_espectrales = {"NDVI", "NDWI", "SAR-CR", "EVI"}

    for j, name in enumerate(feature_names):
        assigned = False
        # Buscar en μ_ y σ_ prefijos
        base = name[2:] if name.startswith(("μ_", "σ_")) else name
        for group, idxs in CHANNEL_GROUPS.items():
            if any(base == CHANNEL_NAMES[i] for i in idxs):
                group_sums[group].append(mean_abs[j])
                assigned = True
                break
        if not assigned and base in indices_espectrales:
            group_sums["Índices espectrales"].append(mean_abs[j])

    # Calcular estadísticas por grupo
    groups, means, stds = [], [], []
    all_groups = list(CHANNEL_GROUPS.keys()) + ["Índices espectrales"]
    all_colors = list(GROUP_COLORS.values()) + ["#EF4444"]

    for g in all_groups:
        vals = group_sums[g]
        if vals:
            groups.append(g)
            means.append(np.mean(vals))
            stds.append(np.std(vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(groups))
    cols = [all_colors[all_groups.index(g)] for g in groups]

    bars = ax.bar(x, means, yerr=stds, color=cols, edgecolor="white",
                  linewidth=0.8, capsize=5, error_kw={"linewidth": 1.5})

    # Anotar valores
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{m:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("Importancia SHAP media por canal del grupo", fontsize=10)
    ax.set_title("Contribución SHAP por tipo de sensor", fontsize=13, fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✔ {out_path.name}")


def fig_shap_beeswarm(shap_values: np.ndarray, X_sub: np.ndarray,
                      feature_names: list[str], out_path: Path):
    """
    Beeswarm-style: scatter de SHAP vs. índice de muestra, coloreado por valor
    de feature, para los top-10 features por importancia SHAP.
    (Alternativa ligera a shap.plots.beeswarm que no requiere JS)
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    top10    = np.argsort(mean_abs)[::-1][:10]

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()

    for ax, j in zip(axes, top10):
        sv_j   = shap_values[:, j]
        feat_j = X_sub[:, j]
        order  = np.argsort(feat_j)

        sc = ax.scatter(
            np.arange(len(sv_j)),
            sv_j[order],
            c=feat_j[order],
            cmap="RdYlGn",
            s=4, alpha=0.6, linewidths=0,
        )
        ax.axhline(0, color="gray", lw=0.6, ls="--")
        ax.set_title(feature_names[j], fontsize=8, pad=4)
        ax.set_xlabel("Muestra (ord. por valor)", fontsize=6)
        ax.set_ylabel("SHAP", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        plt.colorbar(sc, ax=ax, pad=0.02).ax.tick_params(labelsize=5)

    fig.suptitle("Valores SHAP — Top 10 features (color = valor del feature)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✔ {out_path.name}")


def fig_channel_correlation(X: np.ndarray, feature_names: list[str], out_path: Path):
    """
    Mapa de calor de correlación entre los 14 canales base (medias μ_*).
    Análogo al análisis de cross-talk en FWI (Montoya et al., 2024).
    """
    # Extraer solo las 14 medias (features 0-13)
    X_means = X[:, :14]

    corr = np.corrcoef(X_means.T)

    fig, ax = plt.subplots(figsize=(9, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # solo triángulo inferior

    im = ax.imshow(np.where(mask, np.nan, corr), cmap="RdBu_r",
                   vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(14))
    ax.set_yticks(range(14))
    short = [n.split(" ")[0] for n in CHANNEL_NAMES]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)

    # Anotar valores
    for i in range(14):
        for j in range(14):
            if not mask[i, j]:
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black" if abs(corr[i,j]) < 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Correlación de Pearson", shrink=0.8)
    ax.set_title("Correlación entre canales espectrales (media por parche)\n"
                 "Análogo al análisis de cross-talk en FWI (Montoya et al., 2024)",
                 fontsize=11, fontweight="bold", pad=12)

    # Líneas divisorias entre grupos
    boundaries = [7, 9, 11]   # después de S2-opt, SAR, DEM
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", lw=1.5)
        ax.axvline(b - 0.5, color="white", lw=1.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✔ {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# 5. REPORTE EN CONSOLA
# ════════════════════════════════════════════════════════════════════════════

def print_shap_summary(shap_values: np.ndarray, feature_names: list[str]):
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║     IMPORTANCIA SHAP — TOP 20 FEATURES              ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  {'Rango':<6} {'Feature':<28} {'SHAP medio |φ|':>16}")
    print("  " + "─"*52)
    for rank, i in enumerate(order[:20], 1):
        print(f"  {rank:<6} {feature_names[i]:<28} {mean_abs[i]:>16.6f}")

    # Por grupo de sensor
    print("\n  RESUMEN POR SENSOR:")
    print("  " + "─"*40)
    group_importance: dict[str, float] = {}
    for group, idxs in CHANNEL_GROUPS.items():
        # Features μ_ y σ_ de esos canales
        group_feat_idx = []
        for j, name in enumerate(feature_names):
            base = name[2:] if name.startswith(("μ_", "σ_")) else name
            if any(base == CHANNEL_NAMES[i] for i in idxs):
                group_feat_idx.append(j)
        if group_feat_idx:
            group_importance[group] = mean_abs[group_feat_idx].mean()
    for g, imp in sorted(group_importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 5000)
        print(f"  {g:<18} {imp:.6f}  {bar}")
    print()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Análisis SHAP de importancia de canales — Landslide4Sense",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--img_dir",    type=Path, default=TRAIN_IMG_DIR,  help="Directorio con archivos image_*.h5")
    p.add_argument("--mask_dir",   type=Path, default=TRAIN_MASK_DIR, help="Directorio con archivos mask_*.h5")
    p.add_argument("--n_samples",  type=int,  default=600,            help="Número de parches a cargar")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUT,    help="Carpeta de salida para figuras")
    p.add_argument("--seed",       type=int,  default=42,             help="Semilla aleatoria")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ANÁLISIS SHAP — IMPORTANCIA DE CANALES ESPECTRALES")
    print("  Proyecto Landslide Detection ML — EAFIT")
    print("=" * 60)

    # 1. Cargar datos
    X, y = load_dataset(args.img_dir, args.mask_dir, args.n_samples, args.seed)
    feature_names = build_feature_names()

    # 2. Entrenar RF
    rf = train_rf(X, y, args.seed)

    # 3. Calcular SHAP
    shap_values, X_sub, _ = compute_shap_values(rf, X, feature_names)

    # 4. Imprimir resumen
    print_shap_summary(shap_values, feature_names)

    # 5. Generar figuras
    print("\n📊 Generando figuras...")
    fig_shap_bar(
        shap_values, feature_names,
        args.output_dir / "shap_importancia_features.png",
    )
    fig_shap_by_group(
        shap_values, feature_names,
        args.output_dir / "shap_por_sensor.png",
    )
    fig_shap_beeswarm(
        shap_values, X_sub, feature_names,
        args.output_dir / "shap_beeswarm_top10.png",
    )
    fig_channel_correlation(
        X, feature_names,
        args.output_dir / "correlacion_canales.png",
    )

    print(f"\n✅ Figuras guardadas en: {args.output_dir}")
    print("   Archivos generados:")
    for f in sorted(args.output_dir.glob("*.png")):
        print(f"   • {f.name}")


if __name__ == "__main__":
    main()
