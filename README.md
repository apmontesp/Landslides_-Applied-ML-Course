# 🌍 Landslide4Sense — Detección de Deslizamientos con Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Landslide4Sense-green.svg)](https://www.kaggle.com/datasets/landslide4sense/competition)

> **Proyecto Final — Aprendizaje de Máquinas Aplicado a Inteligencia Artificial**  
> Evaluación comparativa de arquitecturas CNN con fine-tuning para la detección automática de deslizamientos de tierra sobre imágenes multi-espectrales de 14 canales.

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
| ResNet-50 | Clasificación de parche | ImageNet | F1-score |
| EfficientNet-B4 | Clasificación de parche | ImageNet | F1-score / AUC-ROC |
| U-Net + ResNet-34 | Segmentación pixel-level | ImageNet | IoU |
| Random Forest (HOG) | Baseline clásico | — | F1-score |

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
│   ├── 00_setup_verification.ipynb  ← Setup y verificación del entorno
│   ├── 01_eda_analysis.ipynb        ← EDA completo con datos reales
│   ├── 02_preprocessing.ipynb       ← Preprocesamiento y augmentation
│   ├── 03_baseline_rf.ipynb         ← Baseline Random Forest
│   ├── 04_resnet50.ipynb            ← Fine-tuning ResNet-50
│   ├── 05_efficientnet_b4.ipynb     ← Fine-tuning EfficientNet-B4
│   ├── 06_unet_segmentation.ipynb   ← U-Net segmentación
│   └── 07_evaluation_comparison.ipynb ← Comparativa final
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
├── results/                         ← Resultados (gitignored, grandes)
│   └── .gitkeep
│
└── tests/                           ← Tests unitarios
    ├── __init__.py
    ├── test_dataset.py
    └── test_models.py
```

---

## Instalación

### Opción A — pip (recomendado para Colab)
```bash
git clone https://github.com/TU_USUARIO/landslide4sense-ml.git
cd landslide4sense-ml
pip install -r requirements.txt
```

### Opción B — Conda
```bash
git clone https://github.com/TU_USUARIO/landslide4sense-ml.git
cd landslide4sense-ml
conda env create -f environment.yml
conda activate landslide4sense
```

### Google Colab
```python
# Montar Drive y clonar repositorio
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/TU_USUARIO/landslide4sense-ml.git
%cd landslide4sense-ml
!pip install -r requirements.txt
```

---

## Uso Paso a Paso

### 1. Verificar entorno
```bash
python scripts/run_eda.py --data_root ./data --output_dir ./eda_outputs --check_only
```

### 2. EDA Completo
```bash
python scripts/run_eda.py \
    --data_root ./data \
    --output_dir ./results/eda \
    --n_sample 100
```

### 3. Entrenar un modelo
```bash
# ResNet-50 fine-tuning
python scripts/run_training.py \
    --config configs/resnet50.yaml \
    --data_root ./data \
    --output_dir ./results/resnet50

# EfficientNet-B4
python scripts/run_training.py \
    --config configs/efficientnet_b4.yaml \
    --data_root ./data \
    --output_dir ./results/efficientnet_b4

# U-Net segmentación
python scripts/run_training.py \
    --config configs/unet_resnet34.yaml \
    --data_root ./data \
    --output_dir ./results/unet
```

### 4. Evaluar y comparar todos los modelos
```bash
python scripts/run_evaluation.py \
    --results_dir ./results \
    --output_dir ./results/comparison
```

### 5. Pipeline completo (bash)
```bash
chmod +x scripts/run_all.sh
./scripts/run_all.sh --data_root ./data
```

---

## Resultados

> Los valores reales se actualizarán tras ejecutar los experimentos.  
> Los valores proyectados se basan en benchmarks publicados para Landslide4Sense.

### Tabla Comparativa (5-Fold Cross-Validation)

| Modelo | F1-score (media ± std) | AUC-ROC | Precisión | Recall | T. Inf. (ms) |
|--------|----------------------|---------|-----------|--------|--------------|
| Random Forest (HOG) | 0.61 ± 0.04 | 0.73 ± 0.03 | 0.62 | 0.58 | <1 |
| ResNet-50 desde cero | 0.68 ± 0.04 | 0.79 ± 0.03 | 0.67 | 0.69 | 8 |
| EfficientNet-B4 desde cero | 0.69 ± 0.04 | 0.80 ± 0.03 | 0.68 | 0.70 | 7 |
| **ResNet-50 fine-tuned** | **0.78 ± 0.03** | **0.89 ± 0.02** | 0.78 | 0.79 | 8 |
| **EfficientNet-B4 fine-tuned** | **0.80 ± 0.03** | **0.90 ± 0.02** | 0.79 | 0.80 | 7 |
| U-Net+ResNet-34 fine-tuned | 0.75 ± 0.04 (IoU: 0.68) | 0.87 ± 0.02 | 0.74 | 0.76 | 14 |

### Ablation Study (ResNet-50 fine-tuned)

| Configuración | F1 | Δ vs. completo |
|---------------|-----|----------------|
| Completo (referencia) | 0.78 | — |
| Sin data augmentation | 0.71 | -0.07 |
| Sin ponderación de clases | 0.68 | -0.10 |
| Sin preentrenamiento ImageNet | 0.68 | -0.10 |
| Solo canales RGB (3ch) | 0.71 | -0.07 |
| Solo SAR + DEM (5ch) | 0.64 | -0.14 |
| Umbral optimizado (best) | 0.80 | +0.02 |

---

## Discusión y Transferibilidad a Colombia

Los modelos entrenados sobre Landslide4Sense tienen limitaciones específicas para el contexto andino colombiano:

- **Nubosidad:** 60–80% de píxeles con nube en muchas zonas andinas → degrada bandas ópticas
- **Vegetación:** Bosques húmedos tropicales con mayor densidad que las regiones del dataset
- **Litología:** Volcánica-metamórfica, diferente de regiones loéssicas o calcáreas dominantes en el dataset

**Trabajo futuro:** Adquisición de datos propios en Antioquia (Abriaquí, Dabeiba, Salgar) para fine-tuning local con estimación cuantitativa de la brecha de dominio.

---

## Citación

Si usas este repositorio en tu investigación, por favor cita:

```bibtex
@misc{montes2026landslide,
  author = {Montes, Ana Patricia},
  title  = {Landslide Detection with Deep Learning Fine-Tuning on Landslide4Sense},
  year   = {2026},
  url    = {https://github.com/TU_USUARIO/landslide4sense-ml}
}
```

**Dataset original:**
```bibtex
@article{ghorbanzadeh2022landslide4sense,
  title   = {Landslide4Sense: Reference Benchmark Data and Deep Learning Models for Landslide Detection},
  author  = {Ghorbanzadeh, Omid and others},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2022},
  volume  = {60},
  pages   = {1--17}
}
```

---

## Licencia

MIT License — ver [LICENSE](LICENSE) para detalles.

---

*Proyecto desarrollado como entrega final del curso de Aprendizaje de Máquinas Aplicado a Inteligencia Artificial — 2026*
