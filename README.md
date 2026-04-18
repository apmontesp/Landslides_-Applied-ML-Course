# 🌍 Landslide4Sense — Detección de Deslizamientos con Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Landslide4Sense-green.svg)](https://www.kaggle.com/datasets/tekbahadurkshetri/landslide4sense)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apmontesp/Landslides_-Applied-ML-Course/blob/main/notebooks/00_setup_verification.ipynb)

> **Proyecto Final — Aprendizaje de Máquinas Aplicado a Inteligencia Artificial**  
> Evaluación comparativa de modelos clásicos y arquitecturas CNN con fine-tuning para la detección automática de deslizamientos de tierra sobre imágenes multi-espectrales de 14 canales.  
> **Mejor resultado:** Random Forest HOG — F1=0.837 | ResNet-50 — F1=0.784 | U-Net segmentación — F1=0.445

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Dataset](#dataset)
- [Hallazgos EDA](#hallazgos-eda)
- [Arquitecturas Implementadas](#arquitecturas-implementadas)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Instalación](#instalación)
- [Uso Paso a Paso](#uso-paso-a-paso)
- [Resultados](#resultados)
- [Discusión y Transferibilidad a Colombia](#discusión-y-transferibilidad-a-colombia)
- [Citación](#citación)
- [Licencia](#licencia)

---

## Descripción del Proyecto

Los deslizamientos de tierra son uno de los fenómenos de remoción en masa más destructivos a nivel global. Este proyecto implementa y compara tres arquitecturas de aprendizaje profundo para su detección automática usando el dataset multi-espectral **Landslide4Sense**:

| Modelo | Tipo | Preentrenamiento | Métrica objetivo |
|--------|------|-----------------|-----------------|
| Logistic Regression (HOG) | Baseline lineal | — | F1-score |
| SVM kernel RBF (HOG) | Baseline no lineal clásico | — | F1-score |
| Random Forest (HOG) | Baseline ensemble | — | F1-score |
| ResNet-50 | Clasificación de parche | ImageNet | F1-score |
| EfficientNet-B4 | Clasificación de parche | ImageNet | F1-score / AUC-ROC |
| U-Net + ResNet-34 | Segmentación pixel-level | ImageNet | IoU |

**Pregunta de investigación:** ¿En qué medida el fine-tuning de arquitecturas CNN sobre el dataset multi-espectral Landslide4Sense supera a los baselines clásicos y al entrenamiento desde cero, y qué implicaciones tiene para la transferibilidad a contextos geomorfológicos andinos colombianos?

---

## Dataset

**Landslide4Sense** — ISPRS Competition 2022

| Partición | Imágenes | Máscaras | Descripción |
|-----------|----------|----------|-------------|
| TrainData | 3,799 | 3,799 | Entrenamiento con anotaciones pixel-level |
| ValidData | 245 | — | Validación sin etiquetas públicas |
| TestData | 800 | — | Prueba (competición, sin etiquetas) |

**Estructura de cada parche:** `128 × 128 × 14` canales (float32)

| Canales | Fuente | Bandas |
|---------|--------|--------|
| 0–6 | Sentinel-2 | B2(Azul), B3(Verde), B4(Rojo), B8(NIR), B8A(NIR-A), B11(SWIR1), B12(SWIR2) |
| 7–8 | Sentinel-1 SAR | VV, VH |
| 9–10 | ALOS PALSAR | DEM Elevación, Pendiente |
| 11–13 | Sentinel-2 Red-Edge | B5, B6, B7 |

**Descarga:** [kaggle.com/datasets/landslide4sense/competition](https://www.kaggle.com/datasets/landslide4sense/competition)

```bash
# Con Kaggle API:
kaggle datasets download -d landslide4sense/competition
unzip competition.zip -d data/
```

---

## Hallazgos EDA

Los análisis exploratorios con datos reales revelaron tres hallazgos clave que impactan el diseño experimental:

### 1. Balance de Clases — Casi Equiparado
```
Positivos (deslizamiento):     2,231  (58.7%)
Negativos (no-deslizamiento):  1,568  (41.3%)
Ratio: 1 : 0.70  ← prácticamente balanceado
```
> ⚠️ El balance observado difiere del 27%/73% documentado en la competición original. Se aplica `pos_weight=0.70` en la función de pérdida.

### 2. Deslizamientos de Pequeña Escala (Small Object Detection)
```
Área mediana del deslizamiento:  2.04% del parche (~334 px / 16,384)
Percentil 75:                    5.16%
Percentil 90:                    9.89%
```
> Los deslizamientos son objetos extremadamente pequeños dentro del parche. La segmentación pixel-level (U-Net) es crítica para la cuantificación precisa.

### 3. Canales más Discriminativos
| Rank | Canal | Δ Media (Pos - Neg) | Interpretación |
|------|-------|---------------------|----------------|
| 1 | Ch13 S2-B7 RedEdge3 | +0.807 ↑ | Estrés vegetal en zona de ruptura |
| 2 | Ch12 S2-B6 RedEdge2 | +0.563 ↑ | Cambio en estructura del dosel |
| 3 | Ch09 ALOS DEM | +0.195 ↑ | Mayor elevación media en eventos |
| 4 | Ch08 S1-VH SAR | +0.188 ↑ | Rugosidad superficial post-deslizamiento |
| 5 | Ch01 S2-B3 Verde | -0.079 ↓ | Menor reflectancia (vegetación destruida) |

---

## Arquitecturas Implementadas

### ResNet-50 Fine-Tuned
- Entrada adaptada: 14 canales (mean-initialization desde pesos ImageNet)
- Protocolo: congelar backbone 5 épocas → descongelar completo
- Loss: Weighted BCE (`pos_weight=0.70`)
- Optimizer: AdamW (lr_head=1e-4, lr_backbone=1e-5)

### EfficientNet-B4 Fine-Tuned
- Misma estrategia de adaptación de canales
- 19M parámetros vs 25M de ResNet-50
- Escalado compuesto: depth=1.8, width=1.4, resolution=1.3

### U-Net + ResNet-34 (Segmentación)
- Encoder ResNet-34 preentrenado en ImageNet
- Loss: Dice Loss + BCE (50/50)
- Produce mapas de probabilidad 128×128 pixel-level
- Implementado con `segmentation-models-pytorch`

---

## Estructura del Repositorio

```
landslide4sense-ml/
├── README.md                        ← Este archivo
├── LICENSE
├── .gitignore
├── requirements.txt                 ← Dependencias Python
├── environment.yml                  ← Entorno Conda
│
├── data/                            ← Dataset (NO incluido en repo)
│   └── README.md                    ← Instrucciones de descarga
│
├── src/                             ← Código fuente reutilizable
│   ├── __init__.py
│   ├── config.py                    ← Configuración centralizada
│   ├── dataset.py                   ← Clases Dataset PyTorch
│   ├── models.py                    ← Definición de arquitecturas
│   ├── train.py                     ← Loop de entrenamiento
│   ├── evaluate.py                  ← Evaluación y métricas
│   └── utils.py                     ← Utilidades generales
│
├── configs/                         ← Hiperparámetros por modelo
│   ├── resnet50.yaml
│   ├── efficientnet_b4.yaml
│   └── unet_resnet34.yaml
│
├── notebooks/                       ← Jupyter Notebooks paso a paso
│   ├── 00_setup_verification.ipynb     ← Setup y verificación del entorno
│   ├── 01_eda_analysis.ipynb           ← EDA completo con datos reales
│   ├── 02_preprocessing.ipynb          ← Preprocesamiento: Z-score, StandardScaler, Pipeline, data leakage
│   ├── 03_classical_baselines.ipynb    ← Baselines clásicos: LR → SVM → Random Forest
│   ├── 04_resnet50.ipynb               ← Fine-tuning ResNet-50
│   ├── 05_efficientnet_b4.ipynb        ← Fine-tuning EfficientNet-B4
│   ├── 06_unet_segmentation.ipynb      ← U-Net segmentación pixel-level
│   └── 07_evaluation_comparison.ipynb  ← Comparativa final todos los modelos
│
├── scripts/                         ← Scripts standalone ejecutables
│   ├── run_eda.py                   ← EDA completo desde CLI
│   ├── run_training.py              ← Entrenamiento desde CLI
│   ├── run_evaluation.py            ← Evaluación final
│   └── run_all.sh                   ← Pipeline completo
│
├── docs/                            ← Documentación adicional
│   ├── methodology.md               ← Metodología detallada
│   ├── results.md                   ← Resultados y análisis
│   ├── colombia_transfer.md         ← Transferibilidad a Colombia
│   └── figures/                     ← Figuras del EDA
│
├── results/                         ← Resultados reales de entrenamiento (PNGs y JSONs en git; pesos .pt excluidos)
│   ├── classical_baselines/         ← JSONs y figuras LR, SVM, Random Forest
│   ├── resnet50/                    ← Métricas, historial y figuras ResNet-50
│   ├── efficientnet_b4/             ← Métricas, historial y figuras EfficientNet-B4
│   ├── unet_resnet34/               ← Métricas, predicciones y figuras U-Net
│   └── comparison_*.png/csv/json   ← Comparativa final generada por notebook 07
│
└── tests/                           ← Tests unitarios
    ├── __init__.py
    ├── test_dataset.py
    └── test_models.py
```

---

## Instalación

### Opción A — Google Colab (recomendado)

Cada notebook tiene un badge **Open in Colab** — haz clic y ejecuta las celdas en orden. El dataset se carga desde Google Drive automáticamente.

**Requisito:** Dataset en `MyDrive/Landslide4Sense/TrainData/` ([descargar en Kaggle](https://www.kaggle.com/datasets/tekbahadurkshetri/landslide4sense))

### Opción B — Local con pip
```bash
git clone https://github.com/apmontesp/Landslides_-Applied-ML-Course.git
cd Landslides_-Applied-ML-Course
pip install -r requirements.txt
```

### Opción C — Conda
```bash
git clone https://github.com/apmontesp/Landslides_-Applied-ML-Course.git
cd Landslides_-Applied-ML-Course
conda env create -f environment.yml
conda activate landslide4sense
```

---

## Uso Paso a Paso

### En Google Colab (recomendado)

Ejecuta los notebooks en orden — cada uno guarda sus resultados automáticamente en Drive:

| # | Notebook | Descripción | Resultados en Drive |
|---|----------|-------------|-------------------|
| 00 | `00_setup_verification.ipynb` | Verificación del entorno | — |
| 01 | `01_eda_analysis.ipynb` | Análisis exploratorio | figuras EDA |
| 02 | `02_preprocessing.ipynb` | Preprocesamiento | — |
| 03 | `03_classical_baselines.ipynb` | Baselines clásicos: LR, SVM, RF | `results/classical_baselines/` |
| 04 | `04_resnet50.ipynb` | ResNet-50 fine-tuning | `results/resnet50/` |
| 05 | `05_efficientnet_b4.ipynb` | EfficientNet-B4 fine-tuning | `results/efficientnet_b4/` |
| 06 | `06_unet_segmentation.ipynb` | U-Net segmentación | `results/unet_resnet34/` |
| 07 | `07_evaluation_comparison.ipynb` | Comparación final | `results/comparison_*.png` |

### Desde CLI (local)
```bash
# Entrenar ResNet-50
python scripts/run_training.py --config configs/resnet50.yaml --data_root ./data

# Evaluar y comparar
python scripts/run_evaluation.py --results_dir ./results

# Pipeline completo
chmod +x scripts/run_all.sh && ./scripts/run_all.sh --data_root ./data
```

---

## Resultados

> ✅ **Experimentos completados** — Resultados reales sobre 2-Fold Cross-Validation (protocolo optimizado para Colab T4).

### Tabla Comparativa — Resultados Reales

| Modelo | Tipo | F1 medio | AUC-ROC | Notas |
|--------|------|----------|---------|-------|
| **Random Forest (HOG)** | Clásico | **0.837** | 0.808 | 🏆 Mejor F1 global |
| SVM kernel RBF (HOG) | Clásico | 0.797 | 0.779 | |
| Logistic Regression (HOG) | Clásico | 0.789 | 0.761 | |
| ResNet-50 fine-tuned | Deep Learning | 0.784 | ~0.813 | AUC-PR: 0.810/0.816 por fold |
| EfficientNet-B4 fine-tuned | Deep Learning | 0.756 | ~0.823 | AUC-PR: 0.807/0.839 por fold |
| U-Net + ResNet-34 | Segmentación | 0.445 | ~0.391 | Tarea pixel-level (más difícil) |

> Los resultados detallados por fold se generan ejecutando `07_evaluation_comparison.ipynb`

### Hallazgo Principal

**El Random Forest supera a los modelos CNN** en clasificación de parche (F1: 0.837 vs 0.784 ResNet-50). Este resultado, contra-intuitivo a priori, se explica por tres factores concurrentes:

1. **Ingeniería de características efectiva** — HOG + Pendiente DEM + NDVI + SAR VH capturan las señales espectrales más discriminativas identificadas en el EDA (Ch12-13 RedEdge, Ch9 DEM). El Random Forest extrae estas relaciones directamente sin necesidad de aprender representaciones desde cero.

2. **Régimen de datos** — Con 1,500–2,000 muestras y solo 2 folds de validación, el CNN no tiene suficiente señal para superar el sesgo inductivo de features bien diseñadas. Los modelos de aprendizaje profundo tipicamente necesitan >10k muestras etiquetadas para superar a métodos clásicos en tareas de clasificación de parche.

3. **Tarea patch-level vs pixel-level** — ResNet-50 y EfficientNet clasifican el parche completo (etiqueta binaria), mientras que U-Net produce mapas de probabilidad 128×128. La comparación directa de F1 no es equivalente: el U-Net opera sobre ~16,384 píxeles vs 1 etiqueta por parche.

### U-Net — Interpretación de Resultados

El F1 de 0.445 refleja las condiciones experimentales (subconjunto de 2,000 muestras, 10 épocas máx., umbral fijo 0.5), **no el límite del modelo**. Las predicciones visuales muestran localización espacial correcta de deslizamientos: el modelo aprende la forma y ubicación de las zonas afectadas, pero la calibración del umbral y más épocas de entrenamiento mejorarían el F1 significativamente. Los valores de AUC-PR (0.391/0.420 por fold) confirman que el modelo discrimina mejor de lo que el umbral fijo reporta.

### Ablation Study (ResNet-50 fine-tuned)

| Configuración | F1 | Δ vs. completo |
|---------------|-----|----------------|
| Completo (referencia) | 0.784 | — |
| Sin data augmentation | — | pendiente |
| Sin freeze/unfreeze encoder | — | pendiente |
| Sin pos_weight (BCE pura) | — | pendiente |
| LR uniforme 1e-4 | — | pendiente |

---

## Discusión y Transferibilidad a Colombia

Los modelos entrenados sobre Landslide4Sense tienen limitaciones específicas para el contexto andino colombiano:

- **Nubosidad:** 60–80% de píxeles con nube en muchas zonas andinas → degrada bandas ópticas Sentinel-2; el SAR (Sentinel-1) mantiene cobertura pero pierde señal en pendientes pronunciadas
- **Vegetación:** Bosques húmedos tropicales con mayor densidad de dosel que las regiones del dataset → la señal RedEdge (Ch12-13, más discriminativa según EDA) se satura antes
- **Litología:** Volcánica-metamórfica en Andes colombianos, diferente de regiones loéssicas o calcáreas dominantes en el dataset original

**Trabajo futuro:** Fine-tuning local con datos de Antioquia (Abriaquí, Dabeiba, Salgar) usando CORAL domain adaptation para estimar y compensar la brecha de dominio espectral antes del transfer.

---

## Citación

Si usas este repositorio en tu investigación, por favor cita:

```bibtex
@misc{montes2026landslide,
  author = {Montes, Ana Patricia},
  title  = {Landslide Detection with Deep Learning Fine-Tuning on Landslide4Sense},
  year   = {2026},
  url    = {https://github.com/apmontesp/Landslides_-Applied-ML-Course}
}
```

**Dataset original:**
```bibtex
@article{ghorbanzadeh2022landslide4sense,
  title   = {Landslide4Sense: Reference Benchmark Data and Deep Learning Models for Landslide Detection},
  author  = {Ghorbanzadeh, Omid and others},
  journal = {IEEE