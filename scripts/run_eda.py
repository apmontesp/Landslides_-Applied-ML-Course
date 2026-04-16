#!/usr/bin/env python3
"""
eda_landslide4sense.py
======================
Script EDA standalone para Landslide4Sense Dataset.
Genera estadísticas, figuras y reporte de calidad de datos.

Uso:
    python eda_landslide4sense.py --data_root /ruta/al/dataset --output_dir ./eda_outputs

Requiere:
    pip install h5py numpy pandas matplotlib seaborn scipy scikit-learn tqdm
"""

import os
import glob
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from scipy import stats as sp_stats
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
CHANNEL_NAMES = [
    'S2-B2 Azul',    'S2-B3 Verde',   'S2-B4 Rojo',    'S2-B8 NIR',
    'S2-B8A NIR-A',  'S2-B11 SWIR1',  'S2-B12 SWIR2',
    'S1-VV SAR',     'S1-VH SAR',
    'ALOS DEM',      'DEM Slope',
    'S2-B5 RedEdge1','S2-B6 RedEdge2','S2-B7 RedEdge3'
]
CHANNEL_GROUPS = {
    'Sentinel-2 Óptico': list(range(7)),
    'Sentinel-1 SAR':    [7, 8],
    'ALOS DEM':          [9, 10],
    'S2 Red-Edge':       [11, 12, 13]
}
GROUP_COLORS = {
    'Sentinel-2 Óptico': '#3498db',
    'Sentinel-1 SAR':    '#e74c3c',
    'ALOS DEM':          '#2ecc71',
    'S2 Red-Edge':       '#9b59b6'
}
SEED = 42
np.random.seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────────────────────
def get_channel_group(ch_idx):
    for group, channels in CHANNEL_GROUPS.items():
        if ch_idx in channels:
            return group
    return 'Otro'


def normalize_display(arr_2d, percentile=99):
    """Normaliza un canal 2D a [0,1] para visualización."""
    a = arr_2d.copy().astype(np.float32)
    mask = a > 0.001
    if mask.any():
        p_high = np.percentile(a[mask], percentile)
        a = np.clip(a / max(float(p_high), 1e-6), 0, 1)
    return a


def load_image(filepath):
    with h5py.File(filepath, 'r') as hf:
        return hf['img'][:].astype(np.float32)


def load_mask(filepath):
    with h5py.File(filepath, 'r') as hf:
        key = list(hf.keys())[0]
        return hf[key][:].astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 1. CARGA Y VALIDACIÓN
# ──────────────────────────────────────────────────────────────────────────────
def discover_dataset(data_root):
    """Descubre qué particiones están disponibles."""
    partitions = {}
    for split in ['TrainData', 'ValData', 'TestData']:
        img_dir  = os.path.join(data_root, split, 'img')
        mask_dir = os.path.join(data_root, split, 'mask')
        img_files  = sorted(glob.glob(os.path.join(img_dir, '*.h5'))) if os.path.exists(img_dir) else []
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.h5'))) if os.path.exists(mask_dir) else []
        partitions[split] = {'img': img_files, 'mask': mask_files}
    return partitions


def print_dataset_summary(partitions):
    print("\n" + "="*60)
    print(" DATASET SUMMARY — Landslide4Sense")
    print("="*60)
    for split, data in partitions.items():
        n_img  = len(data['img'])
        n_mask = len(data['mask'])
        status = '✅' if n_img > 0 else '❌'
        mask_status = f'+ {n_mask} máscaras' if n_mask > 0 else '(sin máscaras)'
        print(f"  {status} {split:<12}: {n_img:4d} imágenes {mask_status}")

    # Inspeccionar un archivo
    sample = None
    for split in ['TrainData', 'ValData', 'TestData']:
        if partitions[split]['img']:
            sample = partitions[split]['img'][0]
            break
    if sample:
        arr = load_image(sample)
        print(f"\n  Shape por parche: {arr.shape}  |  dtype: {arr.dtype}")
        print(f"  Rango global de valores: [{arr.min():.4f}, {arr.max():.4f}]")
    print("="*60)


# ──────────────────────────────────────────────────────────────────────────────
# 2. ESTADÍSTICAS POR CANAL
# ──────────────────────────────────────────────────────────────────────────────
def compute_channel_stats(img_files, n_sample=100, subsample_step=8):
    """Calcula estadísticas por canal de forma incremental (sin cargar todo en RAM)."""
    files_sample = img_files[::max(1, len(img_files) // n_sample)][:n_sample]

    sum1  = np.zeros(14, dtype=np.float64)
    sum2  = np.zeros(14, dtype=np.float64)
    n_pix = 0
    c_min = np.full(14,  np.inf)
    c_max = np.full(14, -np.inf)
    hist_samples = {c: [] for c in range(14)}

    for fp in tqdm(files_sample, desc='  Estadísticas por canal'):
        arr  = load_image(fp)                  # (128, 128, 14)
        flat = arr.reshape(-1, 14)             # (16384, 14)
        n_pix += flat.shape[0]
        sum1  += flat.sum(0)
        sum2  += (flat ** 2).sum(0)
        c_min  = np.minimum(c_min, flat.min(0))
        c_max  = np.maximum(c_max, flat.max(0))
        for c in range(14):
            hist_samples[c].extend(flat[::subsample_step, c].tolist())

    mean = sum1 / n_pix
    std  = np.sqrt(np.maximum(sum2 / n_pix - mean ** 2, 0))

    rows = []
    for c in range(14):
        vals = np.array(hist_samples[c])
        p = np.percentile(vals, [5, 25, 50, 75, 95])
        rows.append({
            'Canal': c,
            'Nombre': CHANNEL_NAMES[c],
            'Grupo': get_channel_group(c),
            'Min':   round(float(c_min[c]), 4),
            'Max':   round(float(c_max[c]), 4),
            'Media': round(float(mean[c]),  4),
            'Std':   round(float(std[c]),   4),
            'P5':    round(float(p[0]),     4),
            'P25':   round(float(p[1]),     4),
            'P50':   round(float(p[2]),     4),
            'P75':   round(float(p[3]),     4),
            'P95':   round(float(p[4]),     4),
            'CV%':   round(float(std[c] / max(mean[c], 1e-8) * 100), 1),
        })

    stats_df = pd.DataFrame(rows)
    return stats_df, hist_samples, len(files_sample)


def compute_correlation_matrix(img_files, n_imgs=30):
    """Calcula la matriz de correlación de Pearson entre los 14 canales."""
    flat_list = []
    for fp in tqdm(img_files[:n_imgs], desc='  Correlación'):
        arr = load_image(fp)
        flat_list.append(arr.reshape(-1, 14))
    flat = np.vstack(flat_list)
    return np.corrcoef(flat.T)


# ──────────────────────────────────────────────────────────────────────────────
# 3. ANÁLISIS DE ETIQUETAS (si hay máscaras)
# ──────────────────────────────────────────────────────────────────────────────
def analyze_labels(mask_files):
    """Analiza balance de clases y distribución de áreas de deslizamiento."""
    labels = []
    areas  = []
    for mf in tqdm(mask_files, desc='  Analizando etiquetas'):
        mask = load_mask(mf)
        has_landslide = int(mask.sum() > 0)
        labels.append(has_landslide)
        if has_landslide:
            areas.append(float(mask.sum()) / (128 * 128) * 100)

    labels = np.array(labels)
    n_pos  = labels.sum()
    n_neg  = len(labels) - n_pos
    ratio  = n_neg / max(n_pos, 1)

    print(f"\n  📊 Balance de clases:")
    print(f"     Positivos (deslizamiento):    {n_pos:4d}  ({100*n_pos/len(labels):.1f}%)")
    print(f"     Negativos (no-deslizamiento): {n_neg:4d}  ({100*n_neg/len(labels):.1f}%)")
    print(f"     Ratio de desbalance: 1 : {ratio:.2f}")
    if areas:
        print(f"     Área media del deslizamiento (parches +): {np.mean(areas):.1f}% del parche")
        print(f"     Área mediana:                             {np.median(areas):.1f}%")
    return labels, np.array(areas)


# ──────────────────────────────────────────────────────────────────────────────
# 4. VERIFICACIÓN DE LEAKAGE
# ──────────────────────────────────────────────────────────────────────────────
def check_leakage(img_files_a, img_files_b, label_a, label_b, n_check=80):
    """
    Verifica ausencia de duplicados entre dos particiones.
    Usa fingerprint de media por canal + similitud coseno.
    """
    def compute_fingerprints(files, n):
        fps = []
        sample = files[::max(1, len(files)//n)][:n]
        for fp in tqdm(sample, desc=f'  Fingerprints {label_a}↔{label_b}', leave=False):
            arr = load_image(fp)
            fps.append(arr.mean(axis=(0, 1)))
        return np.array(fps)

    fa = compute_fingerprints(img_files_a, n_check)
    fb = compute_fingerprints(img_files_b, n_check)

    na = np.linalg.norm(fa, axis=1, keepdims=True)
    nb = np.linalg.norm(fb, axis=1, keepdims=True)
    sim = (fa / (na + 1e-8)) @ (fb / (nb + 1e-8)).T
    max_sim = sim.max()

    duplicates = int((sim > 0.995).sum())
    print(f"\n  🔍 Leakage {label_a} ↔ {label_b}:")
    print(f"     Similitud coseno máxima: {max_sim:.4f}")
    status = '✅ Sin leakage' if duplicates == 0 else f'⚠️  {duplicates} posibles duplicados'
    print(f"     Resultado: {status}")
    return duplicates, max_sim


def check_nan_quality(img_files, n_check=50):
    """Verifica valores NaN, porcentaje de ceros y calidad general."""
    nan_total  = 0
    zero_ratios = []
    for fp in tqdm(img_files[:n_check], desc='  Calidad de datos', leave=False):
        arr = load_image(fp)
        nan_total += int(np.isnan(arr).sum())
        zero_ratios.append(float((arr == 0).mean()))

    print(f"\n  🧹 Verificación de calidad:")
    print(f"     NaN en {n_check} imágenes:  {nan_total}  {'✅' if nan_total == 0 else '⚠️'}")
    print(f"     Ratio de ceros promedio:   {np.mean(zero_ratios)*100:.1f}%")
    n_border = sum(1 for r in zero_ratios if r > 0.30)
    print(f"     Parches borde (>30% ceros): {n_border}  (candidatos a filtrar)")
    return nan_total, zero_ratios


# ──────────────────────────────────────────────────────────────────────────────
# 5. GENERACIÓN DE FIGURAS
# ──────────────────────────────────────────────────────────────────────────────
def fig_sample_images(img_files, output_dir, n_samples=5):
    """Figura 1: Grid de imágenes de muestra en 4 representaciones."""
    sample = [img_files[i * (len(img_files) // n_samples)] for i in range(n_samples)]
    fig, axes = plt.subplots(4, n_samples, figsize=(4*n_samples, 16))
    fig.suptitle('Landslide4Sense — Parches de Muestra (128×128 px)\n'
                 'Fila 1: RGB  |  Fila 2: Falso Color NIR  |  Fila 3: SAR VV  |  Fila 4: DEM',
                 fontsize=13, fontweight='bold')
    row_labels = ['RGB\n(B4-B3-B2)', 'Falso Color NIR\n(B8-B3-B2)', 'SAR VV\n(Sentinel-1)', 'DEM\nElevación']

    for col, fp in enumerate(sample):
        arr = load_image(fp)
        rgb  = np.stack([normalize_display(arr[:,:,2]),
                         normalize_display(arr[:,:,1]),
                         normalize_display(arr[:,:,0])], axis=-1)
        nirf = np.stack([normalize_display(arr[:,:,3]),
                         normalize_display(arr[:,:,1]),
                         normalize_display(arr[:,:,0])], axis=-1)
        axes[0, col].imshow(rgb);    axes[0, col].axis('off')
        axes[1, col].imshow(nirf);   axes[1, col].axis('off')
        axes[2, col].imshow(normalize_display(arr[:,:,7]), cmap='gray'); axes[2, col].axis('off')
        axes[3, col].imshow(arr[:,:,9], cmap='terrain');                  axes[3, col].axis('off')
        axes[0, col].set_title(f'Parche {col+1}', fontsize=9)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9, fontweight='bold', rotation=90, labelpad=5)

    plt.tight_layout()
    out = os.path.join(output_dir, 'fig1_sample_images.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✅ fig1_sample_images.png")
    return out


def fig_channel_histograms(hist_samples, stats_df, output_dir):
    """Figura 2: Histogramas de distribución por canal."""
    fig, axes = plt.subplots(4, 4, figsize=(17, 13))
    fig.suptitle('Distribución de Valores por Canal\n'
                 '(densidad, excluyendo ceros — n≈100 imágenes de muestra)',
                 fontsize=13, fontweight='bold')

    for c in range(14):
        ax   = axes[c // 4, c % 4]
        vals = np.array(hist_samples[c])
        vals = vals[vals > 0.001]
        group = get_channel_group(c)
        color = GROUP_COLORS[group]

        ax.hist(vals, bins=55, color=color, alpha=0.75, density=True, edgecolor='none')
        mu = stats_df.loc[c, 'Media']
        me = stats_df.loc[c, 'P50']
        sd = stats_df.loc[c, 'Std']
        ax.axvline(mu, color='red',  lw=1.8, ls='--', alpha=0.9, label='Media')
        ax.axvline(me, color='navy', lw=1.8, ls=':',  alpha=0.9, label='Mediana')
        ax.set_title(f'Ch{c}: {CHANNEL_NAMES[c]}', fontsize=8.5, fontweight='bold', color=color)
        ax.set_xlabel('Valor', fontsize=7);  ax.set_ylabel('Densidad', fontsize=7)
        ax.tick_params(labelsize=7);          ax.legend(fontsize=6, framealpha=0.7)
        ax.text(0.97, 0.96, f'μ={mu:.3f}\nσ={sd:.3f}', transform=ax.transAxes,
                fontsize=7.5, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        ax.grid(axis='y', alpha=0.3, lw=0.5)

    for ax in [axes[3, 2], axes[3, 3]]:
        ax.set_visible(False)

    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=legend_patches, loc='lower right', ncol=2, fontsize=9,
               bbox_to_anchor=(0.98, 0.01))
    plt.tight_layout()
    out = os.path.join(output_dir, 'fig2_channel_histograms.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✅ fig2_channel_histograms.png")
    return out


def fig_correlation_heatmap(corr_matrix, output_dir):
    """Figura 3: Heatmap de correlación entre canales."""
    short_names = [
        'S2-B2\nAzul',  'S2-B3\nVerde', 'S2-B4\nRojo',   'S2-B8\nNIR',
        'S2-B8A\nNIR-A','S2-B11\nSWIR1','S2-B12\nSWIR2',
        'S1-VV\nSAR',   'S1-VH\nSAR',
        'ALOS\nDEM',    'DEM\nSlope',
        'S2-B5\nRE1',   'S2-B6\nRE2',   'S2-B7\nRE3'
    ]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1,
                ax=ax, linewidths=0.5, xticklabels=short_names, yticklabels=short_names,
                annot_kws={'size': 7.5}, cbar_kws={'label': 'Correlación de Pearson (r)'})
    ax.set_title('Matriz de Correlación entre los 14 Canales\n'
                 'Landslide4Sense (n=30 imágenes)', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(fontsize=8);  plt.yticks(fontsize=8, rotation=0)
    plt.tight_layout()
    out = os.path.join(output_dir, 'fig3_correlation_heatmap.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✅ fig3_correlation_heatmap.png")

    # Imprimir pares de alta correlación
    pairs = [(corr_matrix[i,j], i, j)
             for i in range(14) for j in range(i+1, 14)]
    pairs.sort(key=lambda x: abs(x[0]), reverse=True)
    print("\n  🔗 Top 5 pares de mayor correlación:")
    for r, i, j in pairs[:5]:
        print(f"     Ch{i:02d} {CHANNEL_NAMES[i]:16s} ↔ Ch{j:02d} {CHANNEL_NAMES[j]:16s}  r={r:+.4f}")
    return out


def fig_channel_stats_bars(stats_df, output_dir):
    """Figura 4: Barras de media ± std y ratio CV por canal."""
    means = stats_df['Media'].values
    stds  = stats_df['Std'].values
    cvs   = stats_df['CV%'].values
    colors = [GROUP_COLORS[get_channel_group(c)] for c in range(14)]
    ch_labels = [f'Ch{c}\n{CHANNEL_NAMES[c].split()[0]}' for c in range(14)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.bar(range(14), means, yerr=stds, color=colors, alpha=0.8,
            capsize=4, edgecolor='gray', linewidth=0.5)
    ax1.set_xticks(range(14));  ax1.set_xticklabels(ch_labels, fontsize=8)
    ax1.set_ylabel('Valor medio (μ)', fontsize=10)
    ax1.set_title('Media ± Desviación Estándar por Canal', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.4)
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(facecolor=c, label=g) for g,c in GROUP_COLORS.items()],
               fontsize=8, loc='upper left')

    ax2.bar(range(14), cvs, color=colors, alpha=0.8, edgecolor='gray', linewidth=0.5)
    ax2.set_xticks(range(14));  ax2.set_xticklabels(ch_labels, fontsize=8)
    ax2.set_ylabel('Coef. de Variación (CV%)', fontsize=10)
    ax2.set_title('Coeficiente de Variación por Canal\n(σ/μ × 100)', fontsize=11, fontweight='bold')
    ax2.axhline(100, color='red', ls='--', lw=1.2, alpha=0.7, label='CV=100%')
    ax2.legend(fontsize=8);  ax2.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    out = os.path.join(output_dir, 'fig4_channel_stats.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✅ fig4_channel_stats.png")
    return out


def fig_class_balance(labels, areas, output_dir):
    """Figura 5: Balance de clases y distribución de áreas."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.pie([n_pos, n_neg],
            labels=[f'Deslizamiento\n({n_pos})', f'No-deslizamiento\n({n_neg})'],
            colors=['#e74c3c', '#3498db'], autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 10})
    ax1.set_title('Balance de Clases — TrainData', fontsize=11, fontweight='bold')

    if len(areas) > 0:
        ax2.hist(areas, bins=30, color='#e74c3c', alpha=0.8, edgecolor='white')
        ax2.axvline(np.mean(areas), color='navy', ls='--', lw=2,
                    label=f'Media: {np.mean(areas):.1f}%')
        ax2.set_xlabel('% del parche con deslizamiento', fontsize=10)
        ax2.set_ylabel('Frecuencia', fontsize=10)
        ax2.set_title('Área del Deslizamiento\n(parches positivos)', fontsize=11, fontweight='bold')
        ax2.legend();  ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, 'fig5_class_balance.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✅ fig5_class_balance.png")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 6. REPORTE JSON
# ──────────────────────────────────────────────────────────────────────────────
def save_eda_report(partitions, stats_df, corr_matrix, output_dir,
                    labels=None, areas=None, leakage_results=None):
    report = {
        'dataset': 'Landslide4Sense',
        'partitions': {k: {'n_images': len(v['img']), 'n_masks': len(v['mask'])}
                       for k, v in partitions.items()},
        'channel_stats': stats_df.to_dict(orient='records'),
        'correlation_matrix': corr_matrix.tolist(),
        'top_correlations': [],
        'class_balance': None,
        'leakage': leakage_results or {},
    }

    pairs = [(corr_matrix[i,j], i, j) for i in range(14) for j in range(i+1,14)]
    pairs.sort(key=lambda x: abs(x[0]), reverse=True)
    report['top_correlations'] = [
        {'ch_a': i, 'ch_b': j, 'name_a': CHANNEL_NAMES[i],
         'name_b': CHANNEL_NAMES[j], 'r': round(float(r), 4)}
        for r, i, j in pairs[:10]
    ]

    if labels is not None:
        n_pos = int(labels.sum())
        n_neg = int(len(labels) - n_pos)
        report['class_balance'] = {
            'n_positive': n_pos, 'n_negative': n_neg,
            'ratio_neg_pos': round(n_neg / max(n_pos, 1), 2),
            'pct_positive': round(100 * n_pos / len(labels), 1),
            'area_mean_pct': round(float(np.mean(areas)), 2) if len(areas)>0 else None,
            'area_median_pct': round(float(np.median(areas)), 2) if len(areas)>0 else None,
        }

    out = os.path.join(output_dir, 'eda_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  ✅ eda_report.json")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='EDA completo para el dataset Landslide4Sense')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Ruta raíz del dataset (debe contener TrainData/, ValData/, TestData/)')
    parser.add_argument('--output_dir', type=str, default='./eda_outputs',
                        help='Directorio de salida para figuras y reportes')
    parser.add_argument('--n_sample', type=int, default=100,
                        help='Número de imágenes para cálculo de estadísticas (default: 100)')
    parser.add_argument('--n_corr', type=int, default=30,
                        help='Número de imágenes para correlación (default: 30)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n🌍 Landslide4Sense EDA Script")
    print(f"   data_root:  {args.data_root}")
    print(f"   output_dir: {args.output_dir}")

    # ─── 1. Descubrir dataset ──────────────────────────────────────────────
    partitions = discover_dataset(args.data_root)
    print_dataset_summary(partitions)

    # Seleccionar partición para EDA de canales
    eda_files = (partitions['TrainData']['img']
                 or partitions['ValData']['img']
                 or partitions['TestData']['img'])
    if not eda_files:
        print("❌ No se encontraron archivos .h5. Verifica --data_root")
        return

    split_name = ('TrainData' if partitions['TrainData']['img'] else
                  'ValData'   if partitions['ValData']['img'] else 'TestData')
    print(f"\n📊 Ejecutando EDA sobre: {split_name} ({len(eda_files)} imágenes)")

    # ─── 2. Estadísticas por canal ────────────────────────────────────────
    print("\n[1/6] Calculando estadísticas por canal...")
    stats_df, hist_samples, n_files = compute_channel_stats(eda_files, n_sample=args.n_sample)
    print(f"\n  Estadísticas calculadas sobre {n_files} imágenes:")
    print(stats_df[['Canal','Nombre','Min','Max','Media','Std','CV%']].to_string(index=False))

    # ─── 3. Correlación ──────────────────────────────────────────────────
    print("\n[2/6] Calculando matriz de correlación...")
    corr_matrix = compute_correlation_matrix(eda_files, n_imgs=args.n_corr)

    # ─── 4. Análisis de etiquetas (si disponible) ─────────────────────────
    labels, areas = None, np.array([])
    if partitions['TrainData']['mask']:
        print("\n[3/6] Analizando balance de clases...")
        labels, areas = analyze_labels(partitions['TrainData']['mask'])
    else:
        print("\n[3/6] Sin máscaras disponibles — omitiendo análisis de clases")

    # ─── 5. Verificación de leakage ──────────────────────────────────────
    leakage_results = {}
    print("\n[4/6] Verificando calidad y leakage...")
    _, _ = check_nan_quality(eda_files)
    if (partitions['TrainData']['img'] and partitions['ValData']['img']):
        dups, max_sim = check_leakage(
            partitions['TrainData']['img'],
            partitions['ValData']['img'],
            'TrainData', 'ValData'
        )
        leakage_results['train_val'] = {'duplicates': dups, 'max_sim': round(max_sim, 4)}
    if (partitions['TrainData']['img'] and partitions['TestData']['img']):
        dups, max_sim = check_leakage(
            partitions['TrainData']['img'],
            partitions['TestData']['img'],
            'TrainData', 'TestData'
        )
        leakage_results['train_test'] = {'duplicates': dups, 'max_sim': round(max_sim, 4)}

    # ─── 6. Generar figuras ──────────────────────────────────────────────
    print("\n[5/6] Generando figuras...")
    fig_sample_images(eda_files, args.output_dir)
    fig_channel_histograms(hist_samples, stats_df, args.output_dir)
    fig_correlation_heatmap(corr_matrix, args.output_dir)
    fig_channel_stats_bars(stats_df, args.output_dir)
    if labels is not None:
        fig_class_balance(labels, areas, args.output_dir)

    # ─── 7. Reporte JSON ─────────────────────────────────────────────────
    print("\n[6/6] Guardando reporte...")
    save_eda_report(partitions, stats_df, corr_matrix, args.output_dir,
                    labels, areas, leakage_results)

    print(f"\n{'='*60}")
    print(f"✅ EDA completado. Resultados en: {args.output_dir}/")
    print(f"   fig1_sample_images.png")
    print(f"   fig2_channel_histograms.png")
    print(f"   fig3_correlation_heatmap.png")
    print(f"   fig4_channel_stats.png")
    if labels is not None:
        print(f"   fig5_class_balance.png")
    print(f"   eda_report.json")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
