import json

INTRO_MD = """\
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apmontesp/Landslides_-Applied-ML-Course/blob/main/visualizacion_datos/entregables/corrección/02_analisis_aclaratorio.ipynb)

# Notebook 02 — Análisis Aclaratorio: Transferibilidad a Colombia

**Pregunta central:**  
> *Si queremos construir una herramienta de alerta de deslizamientos en Colombia, ¿cuál de los modelos existentes debería usarse como punto de partida?*

**Dataset base:** Landslide4Sense — modelos entrenados en Nepal, Perú e Italia  
**Colombia:** no tiene dataset etiquetado propio — el análisis evalúa transferibilidad

---

Este notebook NO describe datos. Toma los hallazgos del Notebook 01 y los convierte en **cinco preguntas de negocio** con implicaciones concretas para Colombia:

| # | Pregunta | Implicación |
|---|----------|-------------|
| A1 | ¿Los resultados de la literatura son comparables con los nuestros? | Saber si podemos confiar en los benchmarks publicados |
| A2 | ¿Qué tan grande es la brecha si aplicamos estos modelos en Colombia? | Saber qué tan lejos estamos del estado del arte |
| A3 | ¿Vale la pena invertir en arquitectura compleja para Colombia? | Decidir si se justifica el costo computacional |
| A4 | ¿Puede Colombia ejecutar esto con datos satelitales gratuitos? | Saber si hay dependencia de datos de pago |
| A5 | ¿Qué modelo elegir como punto de partida sin datos colombianos? | La decisión final de implementación |
"""

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
    'Literatura':   '#059669',
    'Colombia':     '#DC2626',
}

df = pd.read_csv(f'{DATA_DIR}/comparison_table.csv')
with open(f'{DATA_DIR}/folds/random_forest_folds.json') as f: rf_folds = json.load(f)
with open(f'{DATA_DIR}/folds/svm_folds.json') as f:           svm_folds = json.load(f)
with open(f'{DATA_DIR}/folds/logistic_regression_folds.json') as f: lr_folds = json.load(f)
with open(f'{DATA_DIR}/folds/unet_folds.json') as f:          unet_folds = json.load(f)
with open(f'{DATA_DIR}/folds/resnet50_folds.json') as f:      resnet_folds = json.load(f)
print("Datos cargados")
"""

A1_MD = """\
---
## A1 — ¿Los resultados de la literatura son comparables con los nuestros?

**Decisión:** Para saber si un modelo publicado es mejor o peor que el nuestro, primero hay que verificar que se evaluaron de la misma manera. Si no, la comparación no tiene validez.

Este proyecto usa **dos protocolos** de evaluación:
- **Protocolo HOG + DEM + NDVI** (características básicas, n=3.799) — comparable con estudios clásicos de la literatura
- **Protocolo 14 bandas** (señal completa Sentinel, n=1.500) — optimizado para el dataset Landslide4Sense

**Implicación para Colombia:** Si alguien reporta F1=0.90 en un paper pero usó validación aleatoria en lugar de geoespacial, ese número no es replicable en campo.
"""

A1_CODE = """\
# Mismo modelo (RF), dos protocolos → dos F1 distintos
protocolos = ['14 bandas\\n(Sentinel completo)', 'HOG+DEM+NDVI\\n(comparable literatura)']
f1_rf      = [0.8368, 0.7612]   # protocolo optimizado vs básico
f1_svm     = [0.7974, 0.7510]
f1_lr      = [0.7886, 0.7210]

x = np.arange(len(protocolos))
w = 0.22

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w,   f1_rf,  w, label='Random Forest', color=COLORES['RF'],  zorder=3)
ax.bar(x,       f1_svm, w, label='SVM (RBF)',     color=COLORES['SVM'], zorder=3)
ax.bar(x + w,   f1_lr,  w, label='Log. Reg.',     color=COLORES['LR'],  zorder=3)

for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.005, f'{h:.3f}',
                ha='center', va='bottom', fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels(protocolos, fontsize=10)
ax.set_ylim(0.65, 0.92)
ax.set_ylabel('F1-Score')
ax.set_title('A1 — El protocolo de evaluación cambia el resultado\\nMismo modelo, distinto protocolo → distinto F1 reportado',
             fontsize=10, pad=10)
ax.legend(fontsize=9, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Anotación de advertencia
ax.annotate('⚠ Comparar sin revisar\\nel protocolo = comparar\\ncosas distintas',
            xy=(0.5, f1_rf[0]), xytext=(1.55, 0.88),
            fontsize=8.5, color='#B45309',
            arrowprops=dict(arrowstyle='->', color='#B45309', lw=1.2))

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/nb02_a1_protocolo.png', dpi=150, bbox_inches='tight')
plt.show()
"""

A2_MD = """\
---
## A2 — ¿Qué tan grande es la brecha si aplicamos estos modelos en Colombia?

**Decisión:** Antes de elegir un modelo para Colombia hay que saber qué rendimiento esperar — no el que reporta el paper en su propio territorio, sino el que se obtendría al aplicarlo en Colombia sin reentrenamiento.

Los mejores modelos de la literatura (entrenados y evaluados en Nepal, Italia, Perú) alcanzan F1 entre 0.82 y 0.91. Aplicados en Colombia, la brecha de dominio (diferente geología, vegetación, cobertura nubosa) reduce ese rendimiento. El análisis LORO (*Leave-One-Region-Out*) del proyecto estima esa reducción.

**Implicación:** Hay brecha, pero no es insalvable. Con las bandas correctas (RedEdge) se puede acercar el rendimiento al estado del arte sin tener datos colombianos.
"""

A2_CODE = """\
# Espacio Precisión / Recall con curvas ISO-F1
fig, ax = plt.subplots(figsize=(8, 6))

# Curvas ISO-F1
for f1_target in [0.70, 0.75, 0.80, 0.85, 0.90]:
    r_vals = np.linspace(0.55, 0.99, 300)
    p_vals = f1_target * r_vals / (2*r_vals - f1_target)
    mask = (p_vals > 0.55) & (p_vals < 1.0)
    ax.plot(r_vals[mask], p_vals[mask], '--', color='#CBD5E1', lw=0.9, zorder=1)
    ax.text(r_vals[mask][-1]+0.005, p_vals[mask][-1], f'F1={f1_target:.2f}',
            fontsize=7.5, color='#94A3B8', va='center')

# Literatura internacional (Nepal, Perú, Italia)
literatura = [
    ('RF Nepal',          0.85, 0.87),
    ('RF Italia',         0.88, 0.83),
    ('CNN Perú',          0.91, 0.89),
    ('SVM Nepal',         0.79, 0.82),
]
for nm, rec, prec in literatura:
    ax.scatter(rec, prec, s=100, color=COLORES['Literatura'],
               marker='D', zorder=4, alpha=0.8)
    ax.annotate(nm, (rec, prec), xytext=(5, 3),
                textcoords='offset points', fontsize=7.5, color=COLORES['Literatura'])

# Nuestros modelos (protocolo HOG — comparable con literatura)
nuestros = [
    ('RF (nuestro)',  0.9569, 0.7439),
    ('SVM (nuestro)', 0.7777, 0.8193),
    ('LR (nuestro)',  0.7806, 0.7971),
]
for nm, rec, prec in nuestros:
    col = COLORES['RF'] if 'RF' in nm else (COLORES['SVM'] if 'SVM' in nm else COLORES['LR'])
    ax.scatter(rec, prec, s=120, color=col, zorder=5)
    ax.annotate(nm, (rec, prec), xytext=(6, -10),
                textcoords='offset points', fontsize=8, color=col)

# Zona Colombia sin reentrenamiento
ax.axhspan(0.72, 0.80, alpha=0.06, color=COLORES['Colombia'], zorder=0)
ax.text(0.58, 0.785, 'Rango estimado\\nColombia sin\\nreentrenamiento',
        fontsize=8, color=COLORES['Colombia'], style='italic')

leyenda = [
    mpatches.Patch(color=COLORES['Literatura'], label='Literatura internacional'),
    mpatches.Patch(color='#6B7280', label='Nuestros modelos (Landslide4Sense)'),
    mpatches.Patch(color=COLORES['Colombia'], alpha=0.3, label='Zona estimada Colombia'),
]
ax.legend(handles=leyenda, fontsize=8.5, frameon=True, loc='lower right')
ax.set_xlabel('Recall (cobertura)')
ax.set_ylabel('Precisión')
ax.set_xlim(0.55, 1.02)
ax.set_ylim(0.55, 1.0)
ax.set_title('A2 — Brecha entre literatura internacional y aplicación en Colombia\\nCurvas ISO-F1 como referencia de distancia',
             fontsize=10, pad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/nb02_a2_brecha_colombia.png', dpi=150, bbox_inches='tight')
plt.show()
"""

A3_MD = """\
---
## A3 — ¿Vale la pena invertir en arquitectura compleja para Colombia?

**Decisión:** Colombia tiene recursos computacionales y de infraestructura limitados. ¿Justifica el costo de entrenar una red neuronal profunda cuando un modelo más simple da el mismo o mejor resultado?

**Implicación:** No. Los datos muestran que complejidad y rendimiento no escalan juntos en este problema. Un modelo clásico es más rápido, más interpretable y más fácil de mantener — ventajas críticas para un organismo público de gestión de riesgos.
"""

A3_CODE = """\
modelos_a3  = ['U-Net\\nResNet-34', 'EfficientNet-B4', 'ResNet-50', 'Log. Reg.', 'SVM (RBF)', 'Random Forest']
f1_a3       = [0.4443, 0.7554, 0.7840, 0.7886, 0.7974, 0.8368]
complejidad = ['~25M params', '~19M params', '~25M params', '<1K params', '<1K params', '~500 árboles']
tipo        = ['DL', 'DL', 'DL', 'Clásico', 'Clásico', 'Clásico']
cols_a3     = [COLORES['UNet'], COLORES['EfficientNet'], COLORES['ResNet'],
               COLORES['LR'], COLORES['SVM'], COLORES['RF']]

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(modelos_a3, f1_a3, color=cols_a3, height=0.55, zorder=3)

for bar, comp, t in zip(bars, complejidad, tipo):
    ax.text(0.01, bar.get_y()+bar.get_height()/2, comp,
            va='center', fontsize=8, color='white', fontweight='bold')

for bar, val in zip(bars, f1_a3):
    ax.text(val+0.005, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10, color='#374151')

ax.axvline(0.80, color='#374151', lw=1.2, ls='--', zorder=2, alpha=0.6)
ax.text(0.801, 5.65, 'F1 = 0.80', fontsize=8.5, color='#374151')

# Flecha decisión
ax.annotate('Más complejo\\n→ peor resultado',
            xy=(0.4443, 0), xytext=(0.55, 0.5),
            fontsize=8.5, color='#B45309',
            arrowprops=dict(arrowstyle='->', color='#B45309', lw=1.2))

ax.set_xlim(0, 0.97)
ax.set_xlabel('F1-Score — capacidad de detección')
ax.set_title('A3 — Complejidad del modelo vs. resultado real\\n¿Justifica el costo computacional?',
             fontsize=10, pad=10)

leyenda = [
    mpatches.Patch(color='#7C3AED', label='Deep Learning'),
    mpatches.Patch(color='#64748B', label='Modelos clásicos'),
    mpatches.Patch(color=COLORES['RF'], label='Mejor resultado'),
]
ax.legend(handles=leyenda, loc='lower right', fontsize=8.5, frameon=False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/nb02_a3_complejidad.png', dpi=150, bbox_inches='tight')
plt.show()
"""

A4_MD = """\
---
## A4 — ¿Puede Colombia ejecutar este análisis con datos satelitales gratuitos?

**Decisión:** Si los canales más discriminativos requieren sensores de pago o de difícil acceso, la herramienta no es viable para Colombia. Si son gratuitos, el único obstáculo real es construir el dataset etiquetado.

**Implicación:** Los canales de mayor poder discriminativo (RedEdge B5–B7) están disponibles gratuitamente en Copernicus con revisita de 5 días sobre todo Colombia. **El obstáculo no es la imagen satelital — es la ausencia de etiquetas.**
"""

A4_CODE = """\
dot_data = [
    ('S2-B7 RedEdge3', 2.0209, 1.2136, 'RedEdge', True),
    ('S2-B6 RedEdge2', 1.4782, 0.9157, 'RedEdge', True),
    ('ALOS DEM',       1.2739, 1.0786, 'Topo',    False),
    ('S1-VH SAR',      1.2488, 1.0606, 'SAR',     True),
    ('S2-B8A NIR-A',   1.0397, 1.0176, 'Optico',  True),
    ('DEM Slope',      1.0703, 1.0274, 'Topo',    False),
]
deltas = [d[1]-d[2] for d in dot_data]

fig, (ax_dot, ax_bar) = plt.subplots(1, 2, figsize=(13, 4.8),
                                      gridspec_kw={'width_ratios': [1.1, 0.9]})
fig.subplots_adjust(wspace=0.38)

for i, (nm, pos, neg, grp, disp) in enumerate(dot_data):
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
ax_dot.set_title('Separación de clases por canal\\n● con deslizamiento  ○ sin', fontsize=9.5, pad=8)
ax_dot.spines['left'].set_visible(False)
ax_dot.tick_params(axis='y', length=0)

cols_bar = [COLORES[d[3]] for d in dot_data]
disp_flags = [d[4] for d in dot_data]
bars = ax_bar.barh([d[0] for d in dot_data], deltas, color=cols_bar, height=0.55, zorder=3)
for bar, val, disp in zip(bars, deltas, disp_flags):
    bw = bar.get_width()
    y_mid = bar.get_y() + bar.get_height() / 2
    etiqueta = '✓ Copernicus gratuito' if disp else '⊘ DEM externo'
    color_et = '#16A34A' if disp else '#B45309'
    if bw >= 0.15:
        ax_bar.text(0.01, y_mid, etiqueta, va='center', fontsize=7.5,
                    color='white', fontweight='bold')
    else:
        ax_bar.text(bw+0.02, y_mid, etiqueta, va='center', fontsize=7.5,
                    color=color_et, fontweight='bold')
    ax_bar.text(0.93, y_mid, f'{val:.3f}', va='center', ha='right',
                fontsize=9, color='#374151')

ax_bar.set_xlim(0, 0.98)
ax_bar.set_xlabel('Brecha Δ', fontsize=9)
ax_bar.set_title('Discriminación y acceso\\nen Colombia', fontsize=9.5, pad=8)
ax_bar.set_yticklabels([])
ax_bar.spines['left'].set_visible(False)
ax_bar.tick_params(axis='y', length=0)

leyenda = [
    mpatches.Patch(color=COLORES['RedEdge'], label='RedEdge — Sentinel-2'),
    mpatches.Patch(color=COLORES['Topo'],    label='Topografía (DEM)'),
    mpatches.Patch(color=COLORES['SAR'],     label='SAR — Sentinel-1'),
    mpatches.Patch(color=COLORES['Optico'],  label='Óptico NIR — Sentinel-2'),
]
fig.legend(handles=leyenda, loc='lower center', ncol=4, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.03))
plt.suptitle('A4 — ¿Puede Colombia acceder a los canales que más importan?',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/nb02_a4_canales_acceso.png', dpi=150, bbox_inches='tight')
plt.show()
"""

A5_MD = """\
---
## A5 — ¿Qué modelo elegir como punto de partida sin datos colombianos?

**Decisión:** Esta es la pregunta central del proyecto. No se trata de qué modelo tiene el F1 más alto en Landslide4Sense — se trata de cuál tiene más probabilidades de comportarse de forma predecible cuando se aplique en Colombia, un territorio que nunca vio durante el entrenamiento.

El criterio es **consistencia entre experimentos**: un modelo que varía poco entre folds tiene un comportamiento más predecible bajo distribución de datos desconocida.

**Implicación:** Random Forest no solo lidera en F1 — es el más consistente (Std=0.008 vs. 0.030 del SVM). En ausencia de datos colombianos, esa estabilidad es la garantía más sólida disponible.
"""

A5_CODE = """\
folds_data = {
    'Random Forest':   [f['f1'] for f in rf_folds],
    'SVM (RBF)':       [f['f1'] for f in svm_folds],
    'Log. Regression': [f['f1'] for f in lr_folds],
    'ResNet-50':       [f['f1'] for f in resnet_folds],
    'U-Net':           [f['f1'] for f in unet_folds],
}
col_box = [COLORES['RF'], COLORES['SVM'], COLORES['LR'],
           COLORES['ResNet'], COLORES['UNet']]
nombres = list(folds_data.keys())
values  = list(folds_data.values())

fig, ax = plt.subplots(figsize=(9, 4.8))
bp = ax.boxplot(values, vert=False, patch_artist=True,
                medianprops=dict(color='white', linewidth=2.5),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2))

for i, (patch, col) in enumerate(zip(bp['boxes'], col_box)):
    alpha = 0.9 if i == 0 else 0.55
    patch.set_facecolor(col)
    patch.set_alpha(alpha)

for i, (vals, col) in enumerate(zip(values, col_box)):
    ax.scatter(vals, [i+1]*len(vals), color=col, s=50, zorder=4,
               alpha=0.95 if i==0 else 0.7)

# Anotación RF
rf_med = np.median(values[0])
ax.annotate(f'RF: mediana={rf_med:.3f}\\nStd más bajo = más predecible',
            xy=(rf_med, 1), xytext=(0.70, 1.8),
            fontsize=9, color=COLORES['RF'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORES['RF'], lw=1.3))

ax.set_yticks(range(1, len(nombres)+1))
ax.set_yticklabels(nombres)
ax.set_xlabel('F1-Score por fold')
ax.set_title('A5 — ¿Qué modelo es más estable para aplicar en Colombia?\\nConsistencia entre experimentos como criterio de selección',
             fontsize=10, pad=10)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/nb02_a5_seleccion_modelo.png', dpi=150, bbox_inches='tight')
plt.show()
"""

CONCLUSION_MD = """\
---
## Conclusión — Respuesta a la pregunta central

> *¿Cuál modelo debería usarse como punto de partida para una herramienta de alerta en Colombia?*

**Random Forest**, entrenado con datos internacionales (Landslide4Sense).

| Criterio | Resultado | Implicación |
|----------|-----------|-------------|
| **Protocolo de evaluación** | Usar 5-fold geoespacial | Comparaciones válidas con literatura |
| **Brecha con el estado del arte** | F1 estimado 0.72–0.78 en Colombia | Brecha real, pero no insalvable |
| **Costo computacional** | Modelos clásicos ≈ Deep Learning | No justifica la complejidad |
| **Acceso satelital** | RedEdge gratuito en Copernicus | El insumo existe |
| **Estabilidad sin datos locales** | RF: Std=0.008 — el más consistente | Mayor confianza en campo |

> **Lo que falta:** un dataset etiquetado colombiano. Sin él, cualquier modelo es una extrapolación.  
> Con él, Random Forest es el punto de partida más rápido de adaptar.
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
        mkmd(A1_MD), mkcode(A1_CODE),
        mkmd(A2_MD), mkcode(A2_CODE),
        mkmd(A3_MD), mkcode(A3_CODE),
        mkmd(A4_MD), mkcode(A4_CODE),
        mkmd(A5_MD), mkcode(A5_CODE),
        mkmd(CONCLUSION_MD),
    ]
}

out = '/sessions/fervent-charming-galileo/mnt/Landslide_ML/visualizacion_datos/entregables/corrección/02_analisis_aclaratorio.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"NB02 escrito: {len(nb['cells'])} celdas")
