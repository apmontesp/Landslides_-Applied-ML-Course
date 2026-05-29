import json, copy

# ── Paleta y constantes ──────────────────────────────────────────
COLORES = {
    'RF':      '#DC2626',
    'SVM':     '#2563EB',
    'LR':      '#64748B',
    'ResNet':  '#7C3AED',
    'EfficientNet': '#0891B2',
    'UNet':    '#9CA3AF',
    'RedEdge': '#7C3AED',
    'Topo':    '#D97706',
    'SAR':     '#0369A1',
    'Optico':  '#059669',
}
FIG_DIR = '../../../data/figures'

# ── Celda: setup ────────────────────────────────────────────────
SETUP = """\
import pandas as pd, numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

FIG_DIR = '../../../data/figures'
DATA_DIR = '../../../data'

COLORES = {
    'RF':           '#DC2626',
    'SVM':          '#2563EB',
    'LR':           '#64748B',
    'ResNet':       '#7C3AED',
    'EfficientNet': '#0891B2',
    'UNet':         '#9CA3AF',
    'RedEdge':      '#7C3AED',
    'Topo':         '#D97706',
    'SAR':          '#0369A1',
    'Optico':       '#059669',
}

df = pd.read_csv(f'{DATA_DIR}/comparison_table.csv')
ch = pd.read_csv(f'{DATA_DIR}/channel_stats_by_class.csv')
with open(f'{DATA_DIR}/final_summary.json') as f:
    summary = json.load(f)
with open(f'{DATA_DIR}/folds/random_forest_folds.json') as f:
    rf_folds = json.load(f)
with open(f'{DATA_DIR}/folds/svm_folds.json') as f:
    svm_folds = json.load(f)
with open(f'{DATA_DIR}/folds/logistic_regression_folds.json') as f:
    lr_folds = json.load(f)
with open(f'{DATA_DIR}/folds/unet_folds.json') as f:
    unet_folds = json.load(f)
with open(f'{DATA_DIR}/folds/resnet50_folds.json') as f:
    resnet_folds = json.load(f)

print("Datos cargados:", df.shape, "modelos")
print(df[['Modelo','F1 medio','Std']].to_string(index=False))
"""

# ── H1: F1 por modelo ───────────────────────────────────────────
H1_MD = """\
---
## Hallazgo 1 — Rendimiento global por modelo
**Pregunta:** ¿Qué tan bien detecta cada modelo un deslizamiento de tierra?

Métrica: **F1-Score** — equilibrio entre no fallar en detectar deslizamientos reales (recall) y no generar falsas alarmas (precisión). Rango 0–1, mayor es mejor.

> *Datos: Landslide4Sense — 5-fold cross-validation con separación geoespacial. Sin interpretación ni hipótesis sobre Colombia.*
"""

H1_CODE = """\
modelos = df['Modelo'].tolist()
f1s     = df['F1 medio'].tolist()
stds    = df['Std'].tolist()
col_map = {'Random Forest':'RF','SVM (RBF)':'SVM','Logistic Regression':'LR',
           'ResNet-50':'ResNet','EfficientNet-B4':'EfficientNet','U-Net ResNet-34':'UNet'}
cols = [COLORES[col_map[m]] for m in modelos]

fig, ax = plt.subplots(figsize=(9, 4.2))
bars = ax.barh(modelos, f1s, color=cols, height=0.55, zorder=3, xerr=stds,
               error_kw=dict(elinewidth=1.2, ecolor='#374151', capsize=3))

for bar, val, std in zip(bars, f1s, stds):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f} ±{std:.3f}', va='center', fontsize=9.5, color='#374151')

ax.axvline(0.80, color='#374151', lw=1.2, ls='--', zorder=2, alpha=0.6)
ax.text(0.801, 5.65, 'F1 = 0.80', fontsize=8, color='#374151')
ax.set_xlim(0, 1.0)
ax.set_xlabel('F1-Score (media ± desviación estándar entre folds)')
ax.set_title('Hallazgo 1 — F1-Score por modelo\\n(5-fold, protocolo geoespacial)', fontsize=11, pad=10)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/nb01_h1_f1_modelos.png', dpi=150, bbox_inches='tight')
plt.show()
"""

# ── H2: Precisión vs Recall ─────────────────────────────────────
H2_MD = """\
---
## Hallazgo 2 — Precisión vs. Cobertura (Recall)
**Pregunta:** ¿Los modelos detectan muchos deslizamientos o son muy precisos al hacerlo?

Cada modelo enfrenta un trade-off: subir el recall (no perder eventos reales) baja la precisión (más falsas alarmas), y viceversa.

> *Sólo los modelos clásicos tienen valores de precisión y recall reportados en el dataset.*
"""

H2_CODE = """\
pr_data = [
    ('Random Forest',       0.7439, 0.9569, 'RF'),
    ('SVM (RBF)',           0.8193, 0.7777, 'SVM'),
    ('Logistic Regression', 0.7971, 0.7806, 'LR'),
]

fig, ax = plt.subplots(figsize=(7, 5))
for nm, prec, rec, ckey in pr_data:
    col = COLORES[ckey]
    ax.scatter(rec, prec, s=160, color=col, zorder=4)
    ax.annotate(nm, (rec, prec), textcoords='offset points',
                xytext=(8, 4), fontsize=9, color=col)

ax.set_xlabel('Recall (cobertura — qué fracción de deslizamientos reales detecta)')
ax.set_ylabel('Precisión (qué fracción de alertas son correctas)')
ax.set_title('Hallazgo 2 — Precisión vs. Cobertura\\nTrade-off entre falsos negativos y falsas alarmas',
             fontsize=10, pad=10)
ax.set_xlim(0.72, 1.0)
ax.set_ylim(0.70, 0.86)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/nb01_h2_prec_recall.png', dpi=150, bbox_inches='tight')
plt.show()
"""

# ── H3+H4 fusionados ────────────────────────────────────────────
H3_MD = """\
---
## Hallazgo 3 — Poder discriminativo de los canales satelitales
**Pregunta:** ¿Cuáles de las 14 bandas espectrales separan mejor los píxeles con deslizamiento de los que no?

Dos vistas complementarias: **izquierda** — valores reales de reflectancia por clase (magnitud de separación); **derecha** — brecha Δ total ordenada de mayor a menor.
"""

H3_CODE = """\
dot_data = [
    ('S2-B7 RedEdge3', 2.0209, 1.2136, 'RedEdge'),
    ('S2-B6 RedEdge2', 1.4782, 0.9157, 'RedEdge'),
    ('ALOS DEM',       1.2739, 1.0786, 'Topo'),
    ('S1-VH SAR',      1.2488, 1.0606, 'SAR'),
    ('S2-B8A NIR-A',   1.0397, 1.0176, 'Optico'),
    ('DEM Slope',      1.0703, 1.0274, 'Topo'),
]
deltas = [d[1]-d[2] for d in dot_data]

fig, (ax_dot, ax_bar) = plt.subplots(1, 2, figsize=(13, 4.8),
                                      gridspec_kw={'width_ratios': [1.15, 0.85]})
fig.subplots_adjust(wspace=0.35)

# Panel izquierdo — dot plot
for i, (nm, pos, neg, grp) in enumerate(dot_data):
    col = COLORES[grp]
    ax_dot.plot([neg, pos], [i, i], color='#D1D5DB', lw=1.8, zorder=2)
    ax_dot.scatter(pos, i, s=90, color=col, zorder=4)
    ax_dot.scatter(neg, i, s=90, color='white', zorder=4,
                   edgecolors=col, linewidths=1.8)
    ax_dot.text(max(pos,neg)+0.04, i, f'Δ={pos-neg:.2f}',
                va='center', fontsize=9, color='#6B7280')

ax_dot.set_yticks(range(len(dot_data)))
ax_dot.set_yticklabels([d[0] for d in dot_data], fontsize=9.5)
ax_dot.set_xlim(0.8, 2.55)
ax_dot.set_xlabel('Reflectancia media normalizada', fontsize=9)
ax_dot.set_title('Valores por clase\\n● con deslizamiento  ○ sin deslizamiento', fontsize=9.5, pad=8)
ax_dot.spines['left'].set_visible(False)
ax_dot.tick_params(axis='y', length=0)

# Panel derecho — barras Δ
cols_bar = [COLORES[d[3]] for d in dot_data]
bars = ax_bar.barh([d[0] for d in dot_data], deltas, color=cols_bar, height=0.55, zorder=3)
for bar, val in zip(bars, deltas):
    ax_bar.text(val+0.01, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9.5, color='#374151')
ax_bar.set_xlim(0, 1.0)
ax_bar.set_xlabel('Brecha Δ (con − sin deslizamiento)', fontsize=9)
ax_bar.set_title('Brecha Δ total\\n(mayor = más discriminativo)', fontsize=9.5, pad=8)
ax_bar.set_yticklabels([])
ax_bar.spines['left'].set_visible(False)
ax_bar.tick_params(axis='y', length=0)

leyenda = [
    mpatches.Patch(color=COLORES['RedEdge'], label='RedEdge — Sentinel-2'),
    mpatches.Patch(color=COLORES['Topo'],    label='Topografía (DEM)'),
    mpatches.Patch(color=COLORES['SAR'],     label='Radar SAR — Sentinel-1'),
    mpatches.Patch(color=COLORES['Optico'],  label='Óptico NIR — Sentinel-2'),
]
fig.legend(handles=leyenda, loc='lower center', ncol=4, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.03))
plt.suptitle('Hallazgo 3 — Poder discriminativo de canales satelitales',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/nb01_h3_canales.png', dpi=150, bbox_inches='tight')
plt.show()
"""

# ── H4: Variabilidad entre folds ────────────────────────────────
H4_MD = """\
---
## Hallazgo 4 — Variabilidad entre experimentos (5 folds)
**Pregunta:** ¿El rendimiento de cada modelo es estable o depende del subconjunto de datos?

Cada punto es el F1 en uno de los 5 folds. La caja muestra la dispersión. Un modelo con caja pequeña es consistente.
"""

H4_CODE = """\
folds_data = {
    'Random Forest':  [f['f1'] for f in rf_folds],
    'SVM (RBF)':      [f['f1'] for f in svm_folds],
    'Log. Regression':[f['f1'] for f in lr_folds],
    'ResNet-50':      [f['f1'] for f in resnet_folds],
    'U-Net':          [f['f1'] for f in unet_folds],
}
col_box = [COLORES['RF'], COLORES['SVM'], COLORES['LR'],
           COLORES['ResNet'], COLORES['UNet']]
nombres = list(folds_data.keys())
values  = list(folds_data.values())

fig, ax = plt.subplots(figsize=(9, 4.5))
bp = ax.boxplot(values, vert=False, patch_artist=True,
                medianprops=dict(color='white', linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2))

for patch, col in zip(bp['boxes'], col_box):
    patch.set_facecolor(col)
    patch.set_alpha(0.75)

for i, (vals, col) in enumerate(zip(values, col_box)):
    ax.scatter(vals, [i+1]*len(vals), color=col, s=40, zorder=4, alpha=0.9)

ax.set_yticks(range(1, len(nombres)+1))
ax.set_yticklabels(nombres)
ax.set_xlabel('F1-Score por fold')
ax.set_title('Hallazgo 4 — Variabilidad entre experimentos\\nCada punto = un fold; caja = dispersión', fontsize=11, pad=10)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/nb01_h4_variabilidad.png', dpi=150, bbox_inches='tight')
plt.show()
"""

# ── Síntesis ─────────────────────────────────────────────────────
SINTESIS_MD = """\
---
## Síntesis — 4 hallazgos observacionales

| # | Pregunta | Observación |
|---|----------|-------------|
| 1 | **F1 por modelo** | Random Forest lidera (0.837); U-Net es el peor (0.444) |
| 2 | **Precisión vs Recall** | RF prioriza cobertura (Recall=0.96); SVM prioriza precisión |
| 3 | **Canales satelitales** | RedEdge3 tiene Δ=0.807 — brecha 4× mayor que SAR-VH |
| 4 | **Variabilidad** | RF es el más consistente entre folds (Std=0.008) |

> **Estos hallazgos son descriptivos.** No asumen causalidad ni proponen recomendaciones.  
> Las implicaciones para Colombia y la selección de modelo se desarrollan en el **Notebook 02**.
"""

# ── Ensamblar notebook ───────────────────────────────────────────
INTRO_MD = """\
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apmontesp/Landslides_-Applied-ML-Course/blob/main/visualizacion_datos/entregables/corrección/01_eda_hallazgos.ipynb)

# Notebook 01 — Análisis Exploratorio: Datos y Hallazgos

**Curso:** Visualización de Datos  
**Dataset:** Landslide4Sense — 14 bandas espectrales · Nepal, Perú e Italia  
**Protocolo:** 5-fold cross-validation con separación geoespacial  
**Soporte ML:** `fase1_proyecto_ML.pptx`

---

## Propósito

Este notebook **describe** los datos y los resultados de los modelos. No genera hipótesis, no propone soluciones, no hace referencia a Colombia.

- H1 — Rendimiento global por modelo (F1)
- H2 — Trade-off Precisión vs. Recall
- H3 — Poder discriminativo de canales satelitales (ranking + magnitud)
- H4 — Variabilidad entre experimentos (estabilidad)

> Las preguntas de negocio y las implicaciones para Colombia están en el **Notebook 02**.
"""

def mkmd(src): return {"cell_type":"markdown","metadata":{},"source":[src]}
def mkcode(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[src]}

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"}
    },
    "cells": [
        mkmd(INTRO_MD),
        mkcode(SETUP),
        mkmd(H1_MD), mkcode(H1_CODE),
        mkmd(H2_MD), mkcode(H2_CODE),
        mkmd(H3_MD), mkcode(H3_CODE),
        mkmd(H4_MD), mkcode(H4_CODE),
        mkmd(SINTESIS_MD),
    ]
}

out = '/sessions/fervent-charming-galileo/mnt/Landslide_ML/visualizacion_datos/entregables/corrección/01_eda_hallazgos.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"NB01 escrito: {len(nb['cells'])} celdas")
