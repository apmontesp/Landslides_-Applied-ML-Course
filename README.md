#  Landslide4Sense — Detección de Deslizamientos con Aprendizaje Automático

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Landslide4Sense-green.svg)](https://www.kaggle.com/datasets/tekbahadurkshetri/landslide4sense)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apmontesp/Landslides_-Applied-ML-Course/blob/main/notebooks/00_setup_verification.ipynb)

> **Proyecto Final — Aprendizaje de Máquinas Aplicado a Inteligencia Artificial**
> Universidad EAFIT · APMP · 2026
>
> Evaluación comparativa de modelos clásicos y arquitecturas CNN con fine-tuning para la detección automática de movimientos en masa sobre imágenes multi-espectrales de 14 canales.
>
> **Mejor resultado:** Random Forest HOG — F1=0.837 | ResNet-50 — F1=0.784 | U-Net segmentación — F1=0.445

---

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Dataset](#dataset)
- [Hallazgos EDA](#hallazgos-eda)
- [Arquitecturas Implementadas](#arquitecturas-implementadas)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Instalación](#instalación)
- [Uso Paso a Paso](#uso-paso-a-paso)
- [Resultados](#resultados)
- [Discusión y Transferibilidad a Colombia](#discusión-y-transferibilidad-a-colombia)
- [Referencias](#referencias)


## Descripción del Proyecto

Los deslizamientos de laderas son uno de los fenómenos de remoción en masa más destructivos a nivel global, con estimaciones de más de 4,000 fatalidades anuales y daños económicos superiores a 100 millones de dólares por evento [1]. Su ocurrencia está condicionada por factores intrínsecos —topografía, geología, hidrología, cobertura vegetal— y se ve agravada por detonantes externos como precipitaciones intensas y sismos [2][3].

En Colombia, la complejidad de la geomorfología andina —altas pendientes, meteorización avanzada, cobertura vegetal densa y alta sismicidad— configura un escenario de elevada susceptibilidad [4]. Más del 70% de la población colombiana reside en la región Andina, donde los movimientos en masa afectan regularmente infraestructura vial, comunidades rurales y zonas periurbanas, con tendencias crecientes asociadas al Fenómeno de La Niña [5]. A pesar de los avances en metodologías de detección, su efectividad es limitada en el contexto montañoso y tropical del país, donde la alta nubosidad, la cobertura vegetal densa y la complejidad geomorfológica reducen el desempeño de los modelos entrenados en otros contextos [15].

Este proyecto implementa y compara seis modelos de aprendizaje automático para la detección automática de deslizamientos sobre el dataset multi-espectral **Landslide4Sense** [17], con análisis explícito de las implicaciones de transferibilidad al contexto andino colombiano.

### Pregunta de Investigación

> **¿En qué medida el fine-tuning de arquitecturas de Redes Neuronales Convolucionales sobre el dataset Landslide4Sense supera a los modelos clásicos con ingeniería de características HOG en términos de F1-score y AUC-ROC, bajo validación cruzada estratificada?**

### Objetivos

**Objetivo General:** Determinar en qué medida el fine-tuning de Redes Neuronales Convolucionales supera a los modelos clásicos con ingeniería de características HOG en la detección de deslizamientos sobre Landslide4Sense, evaluado mediante F1-score y AUC-ROC bajo validación cruzada estratificada de 2 pliegues.

**Objetivos Específicos:**

- **OE1** — Establecer el desempeño de referencia de tres clasificadores clásicos (Regresión Logística, SVM y Random Forest con HOG) sobre los 3,799 parches de Landslide4Sense, reportando F1-score y AUC-ROC mediante 2-Fold Stratified CV.
- **OE2** — Determinar la ganancia de desempeño del fine-tuning de ResNet-50 y EfficientNet-B4 con entrada de 14 canales, respecto a los baselines clásicos del OE1, en términos de F1-score y AUC-ROC bajo el mismo protocolo de validación.
- **OE3** — Cuantificar la capacidad de segmentación pixel-level de U-Net+ResNet-34 sobre parches 128×128 del dataset, reportando IoU y Dice bajo el mismo protocolo de validación cruzada.
- **OE4** — Identificar las brechas de dominio espectral, cobertura vegetal y contexto geológico que limitan la transferibilidad directa de los modelos evaluados al contexto andino colombiano.

### Modelos Evaluados

| Modelo | Tipo | Preentrenamiento | Métrica objetivo |
|--------|------|-----------------|-----------------|
| Logistic Regression (HOG) | Baseline lineal | — | F1-score |
| SVM kernel RBF (HOG) | Baseline no lineal clásico | — | F1-score |
| Random Forest (HOG) | Baseline ensemble | — | F1-score |
| ResNet-50 | Clasificación de parche | ImageNet | F1-score |
| EfficientNet-B4 | Clasificación de parche | ImageNet | F1-score / AUC-ROC |
| U-Net + ResNet-34 | Segmentación pixel-level | ImageNet | IoU / Dice |

---

## Dataset

**Landslide4Sense** — ISPRS Competition 2022 [17]

| Partición | Imágenes | Máscaras | Descripción |
|-----------|----------|----------|-------------|
| TrainData | 3,799 | 3,799 | Entrenamiento con anotaciones pixel-level |
| ValidData | 245 | — | Validación sin etiquetas públicas |
| TestData | 800 | — | Prueba (competición, sin etiquetas) |

**Estructura de cada parche:** `128 × 128 × 14` canales (float32)

| Canales | Fuente | Bandas |
|---------|--------|--------|
| 0–6 | Sentinel-2 | B2 (Azul), B3 (Verde), B4 (Rojo), B8 (NIR), B8A (NIR-A), B11 (SWIR1), B12 (SWIR2) |
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
Ratio: 1 : 0.70
```
> El balance observado difiere del 27%/73% documentado en la competición original [17]. Se aplica `pos_weight=0.70` en la función de pérdida de los modelos CNN.

### 2. Deslizamientos de Pequeña Escala (Small Object Detection)
```
Área mediana del deslizamiento:  2.04% del parche (~334 px / 16,384)
Percentil 75:                    5.16%
Percentil 90:                    9.89%
```
> Los deslizamientos son objetos extremadamente pequeños dentro del parche. La segmentación pixel-level (U-Net) es crítica para la cuantificación espacial precisa.

### 3. Canales más Discriminativos
| Rank | Canal | Δ Media (Pos − Neg) | Interpretación |
|------|-------|---------------------|----------------|
| 1 | Ch13 S2-B7 RedEdge3 | +0.807 ↑ | Estrés vegetal en zona de ruptura |
| 2 | Ch12 S2-B6 RedEdge2 | +0.563 ↑ | Cambio en estructura del dosel |
| 3 | Ch09 ALOS DEM | +0.195 ↑ | Mayor elevación media en eventos |
| 4 | Ch08 S1-VH SAR | +0.188 ↑ | Rugosidad superficial post-deslizamiento |
| 5 | Ch01 S2-B3 Verde | −0.079 ↓ | Menor reflectancia (vegetación destruida) |

---

## Arquitecturas Implementadas

El proyecto sigue una escalera de complejidad deliberada: primero se establecen baselines clásicos con ingeniería de características manual, luego se evalúan arquitecturas CNN con fine-tuning, y finalmente segmentación pixel-level. Esta progresión permite aislar la contribución real de cada nivel de complejidad.

### Baselines Clásicos (notebook 03)

Los tres modelos clásicos usan un vector de características HOG sobre falso color RGB (Sentinel-2 B4/B3/B2) + pendiente DEM + media NDVI + media SAR VH. Se aplica un Pipeline `SimpleImputer(median) → StandardScaler` antes de cada modelo sensible a escala. Su uso en tareas de susceptibilidad está ampliamente documentado en la literatura [8][9].

**Logistic Regression** — Baseline lineal de referencia. `C=1.0`, `class_weight='balanced'`, `max_iter=1000`.

**SVM kernel RBF** — Captura fronteras de decisión no lineales. `C=1.0`, `gamma='scale'`, `class_weight='balanced'`. StandardScaler obligatorio.

**Random Forest (ensemble)** — No requiere escalado. `n_estimators=100`, `class_weight='balanced'`. Proporciona importancia de features interpretable.

### ResNet-50 Fine-Tuned (notebook 04)
- Entrada adaptada: 14 canales (mean-initialization desde pesos ImageNet)
- Protocolo: congelar backbone → descongelar completo con LR diferencial (lr_head=1e-4, lr_backbone=1e-5)
- Loss: Weighted BCE (`pos_weight=0.70`) · Optimizer: AdamW + OneCycleLR · hasta 20 épocas por pliegue
- Trabajos recientes validan la efectividad de esta estrategia en detección multiespectral [11][12]

### EfficientNet-B4 Fine-Tuned (notebook 05)
- Misma estrategia de adaptación de canales que ResNet-50
- Escalado compuesto: depth=1.8, width=1.4, resolution=1.3 · 19M parámetros vs 25M de ResNet-50

### U-Net + ResNet-34 — Segmentación pixel-level (notebook 06)
- Encoder ResNet-34 preentrenado en ImageNet via `segmentation-models-pytorch`
- Loss: DiceBCELoss (0.5 × Dice + 0.5 × BCE) · AMP + early stopping (patience=3) · batch_size=16
- Produce mapas de probabilidad 128×128 — tarea cualitativamente distinta a los clasificadores de parche
- Arquitectura validada en múltiples trabajos de detección con imágenes de teledetección [13][14]

---

## Estructura del Repositorio

```
Landslide_ML/
├── README.md                           <- Este archivo
├── LICENSE
├── requirements.txt                    <- Dependencias Python
├── environment.yml                     <- Entorno Conda
│
├── data/                               <- Dataset (NO incluido en repo)
│   └── README.md                       <- Instrucciones de descarga
│
├── src/                                <- Código fuente reutilizable
│   ├── config.py                       <- Configuración centralizada
│   ├── dataset.py                      <- Clases Dataset PyTorch
│   ├── models.py                       <- Definición de arquitecturas
│   ├── train.py                        <- Loop de entrenamiento
│   ├── evaluate.py                     <- Evaluación y métricas
│   └── utils.py                        <- Utilidades generales
│
├── configs/                            <- Hiperparámetros por modelo
│   ├── resnet50.yaml
│   ├── efficientnet_b4.yaml
│   └── unet_resnet34.yaml
│
├── notebooks/                          <- Jupyter Notebooks paso a paso
│   ├── 00_setup_verification.ipynb     <- Setup y verificación del entorno
│   ├── 01_eda_analysis.ipynb           <- EDA completo con datos reales
│   ├── 02_preprocessing.ipynb          <- Preprocesamiento y protocolo CV
│   ├── 03_classical_baselines.ipynb    <- LR, SVM, Random Forest + HOG
│   ├── 04_resnet50.ipynb               <- Fine-tuning ResNet-50
│   ├── 05_efficientnet_b4.ipynb        <- Fine-tuning EfficientNet-B4
│   ├── 06_unet_segmentation.ipynb      <- U-Net segmentación pixel-level
│   └── 07_evaluation_comparison.ipynb  <- Comparativa final
│
├── scripts/                            <- Scripts standalone ejecutables
│   ├── run_eda.py
│   ├── run_training.py
│   ├── run_evaluation.py
│   └── run_all.sh
│
├── results/                            <- Resultados (PNGs y JSONs; pesos .pt excluidos)
│   ├── classical_baselines/
│   ├── resnet50/
│   ├── efficientnet_b4/
│   ├── unet_resnet34/
│   └── comparison_*.png / *.csv / *.json
│
├── docs/                               <- Documentación adicional
│   ├── methodology.md
│   ├── results.md
│   └── colombia_transfer.md
│
└── tests/
    ├── test_dataset.py
    └── test_models.py
```

---

## Instalación

### Opción A — Google Colab (recomendado)

Cada notebook tiene un badge **Open in Colab**. El dataset se carga desde Google Drive automáticamente.

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

| # | Notebook | Descripción | Salida |
|---|----------|-------------|--------|
| 00 | `00_setup_verification.ipynb` | Verificación del entorno | — |
| 01 | `01_eda_analysis.ipynb` | Análisis exploratorio de datos | figuras EDA |
| 02 | `02_preprocessing.ipynb` | Preprocesamiento y splits | — |
| 03 | `03_classical_baselines.ipynb` | LR, SVM, RF con HOG | `results/classical_baselines/` |
| 04 | `04_resnet50.ipynb` | ResNet-50 fine-tuning | `results/resnet50/` |
| 05 | `05_efficientnet_b4.ipynb` | EfficientNet-B4 fine-tuning | `results/efficientnet_b4/` |
| 06 | `06_unet_segmentation.ipynb` | U-Net segmentación pixel-level | `results/unet_resnet34/` |
| 07 | `07_evaluation_comparison.ipynb` | Comparación final | `results/comparison_*.png` |

### Desde CLI (local)
```bash
python scripts/run_training.py --config configs/resnet50.yaml --data_root ./data
python scripts/run_evaluation.py --results_dir ./results
bash scripts/run_all.sh --data_root ./data
```

---

## Resultados

> Experimentos completados sobre **2-Fold Stratified Cross-Validation** (protocolo optimizado para Colab T4).

### Tabla Comparativa

| Modelo | Tipo | F1 medio | AUC-ROC | Notas |
|--------|------|----------|---------|-------|
| **Random Forest (HOG)** | Clásico | **0.837** ★ | 0.808 | Mejor F1 global |
| SVM kernel RBF (HOG) | Clásico | 0.797 | 0.779 | |
| Logistic Regression (HOG) | Clásico | 0.789 | 0.761 | |
| ResNet-50 fine-tuned | Deep Learning | 0.784 | ~0.813 | AUC-PR: 0.810/0.816 por fold |
| EfficientNet-B4 fine-tuned | Deep Learning | 0.756 | ~0.823 | AUC-PR: 0.807/0.839 por fold |
| U-Net + ResNet-34 | Segmentación | 0.445 | ~0.391 | Tarea pixel-level — no comparable directamente |

> Los resultados detallados por fold se generan ejecutando `07_evaluation_comparison.ipynb`.

### Hallazgo Principal

**El Random Forest supera a los modelos CNN** en clasificación de parche (F1: 0.837 vs 0.784 ResNet-50, +5.3 pp). Este resultado, contraintuitivo respecto a la narrativa predominante en literatura reciente [10][18], se explica por tres factores concurrentes:

1. **Ingeniería de características efectiva** — HOG + Pendiente DEM + NDVI + SAR VH capturan las señales más discriminativas del EDA (Ch12–13 RedEdge, Ch9 DEM, Ch8 SAR-VH) de forma directa, sin aprender representaciones desde datos.

2. **Régimen de datos limitado** — Con 1,500–2,000 muestras por pliegue, las arquitecturas CNN (25M parámetros ResNet-50) no tienen señal suficiente para superar el sesgo inductivo de features bien diseñadas. La literatura sugiere que el aprendizaje profundo tiende a superar métodos clásicos a partir de ~10k muestras [8].

3. **Diferencia de tarea** — ResNet-50 y EfficientNet clasifican el parche completo (etiqueta binaria), mientras que U-Net produce mapas de probabilidad 128×128. La comparación directa de F1 no es equivalente.

### U-Net — Interpretación de Resultados

El F1=0.445 refleja las condiciones experimentales restrictivas (N=2,000, 10 épocas máx., umbral fijo 0.5), **no el límite arquitectónico del modelo**. Las predicciones visuales muestran localización espacial coherente de zonas de deslizamiento. Los valores de AUC-PR (0.391/0.420 por fold) confirman que el modelo discrimina mejor de lo que el umbral estándar reporta. Variantes optimizadas con módulos de atención y doble canal demuestran mejoras significativas en este tipo de tarea [13][14].

---

## Discusión y Transferibilidad a Colombia

La mayoría de los modelos publicados han sido entrenados en contextos geomorfológicos de China, Italia y Grecia —con características topográficas y climáticas distintas al territorio colombiano [8][15]—. Implementar estos modelos sin adaptación puede implicar una reducción de hasta el 25% en las métricas de desempeño [15]. Enfoques recientes de mapeo cross-domain [19][20] sugieren que la armonización de datos multi-sensor puede reducir estas brechas. Este proyecto identifica tres brechas de dominio concretas:

**Nubosidad** — 60–80% de cobertura de nubes en zonas andinas colombianas degrada sistemáticamente las bandas ópticas de Sentinel-2. El SAR (Sentinel-1 VV, VH) mantiene penetración atmosférica pero pierde sensibilidad en pendientes pronunciadas por efecto de layover [2].

**Vegetación** — Los bosques húmedos tropicales presentan mayor densidad de dosel que las regiones del dataset. La señal RedEdge (Ch12–13), la más discriminativa según el EDA, se satura en coberturas densas [3].

**Cuantificación de la brecha:** Wang y Brenning [15] demuestran que la transferencia de modelos entre contextos geomorfológicos sin adaptación de dominio puede producir degradaciones de F1 de hasta 25 puntos porcentuales. Aplicado al mejor resultado obtenido (RF F1=0.837), esto proyecta un desempeño de F1≈0.63 en datos colombianos sin reentrenamiento local, lo que subraya la necesidad de adaptación antes de cualquier despliegue operativo. 

**Litología** — El ambiente volcánico-metamórfico de los Andes colombianos difiere de las regiones loéssicas o calcáreas predominantes en el dataset original, afectando las firmas espectrales en bandas SWIR y SAR [4].

**Cuantificación de la brecha:** Aplicando la estimación de Wang y Brenning [15], el RF con F1=0.837 proyectaría F1≈0.63 en datos colombianos sin reentrenamiento local.

**Trabajo futuro:** (i) colecta de datos de campo en municipios de alta susceptibilidad de Antioquia (Abriaquí, Dabeiba, Salgar) con DJI Mini 4 Pro [6]; (ii) construcción de inventario georreferenciado con el Sistema de Información de Movimientos en Masa (SIMMA/SGC) [4]; (iii) fine-tuning sobre datos mixtos (Landslide4Sense + datos locales) con técnicas de domain adaptation (CORAL).

---

## Referencias

> Fuentes ordenadas por primera aparición en el texto. Las referencias de literatura especializada provienen de la revisión sistemática realizada en el marco del proyecto.

**Contexto y monitoreo**

[1] N. K. Biswas, T. A. Stanley, D. B. Kirschbaum, P. M. Amatya, C. Meechaiya, A. Poortinga, y P. Towashiraporn, "A dynamic landslide hazard monitoring framework for the Lower Mekong Region," *Front. Earth Sci.*, vol. 10, nov. 2022, doi: 10.3389/feart.2022.1057796.

[2] X. Ge, Q. Zhao, B. Wang, y M. Chen, "Lightweight Landslide Detection Network for Emergency Scenarios," *Remote Sens.*, vol. 15, núm. 4, p. 1085, feb. 2023, doi: 10.3390/rs15041085.

[3] H. Thirugnanam, S. Uhlemann, R. Reghunadh, M. V. Ramesh, y V. P. Rangan, "Review of Landslide Monitoring Techniques With IoT Integration Opportunities," *IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.*, vol. 15, pp. 5317–5338, 2022, doi: 10.1109/JSTARS.2022.3183684.

**Contexto colombiano**

[4] Servicio Geológico Colombiano, *Las amenazas por movimientos en masa de Colombia, una visión a escala 1:100.000*. Bogotá: SGC, 2017, doi: 10.32685/9789589952887.

[5] J. Ayala-García y K. Ospino-Ramos, "Desastres naturales en Colombia: un análisis regional," *Documentos de Trabajo sobre Economía Regional y Urbana*, Banco de la República, 2019. [Online]. Available: https://www.banrep.gov.co

**Teledetección y UAV**

[6] J. Sun, G. Yuan, L. Song, y H. Zhang, "Unmanned Aerial Vehicles (UAVs) in Landslide Investigation and Monitoring: A Review," *Drones*, vol. 8, núm. 1, p. 30, ene. 2024, doi: 10.3390/drones8010030.

**Revisión sistemática y métodos estadísticos**

[7] P. Reichenbach, M. Rossi, B. D. Malamud, M. Mihir, y F. Guzzetti, "A review of statistically-based landslide susceptibility models," *Earth Sci. Rev.*, vol. 180, pp. 60–91, may. 2018, doi: 10.1016/j.earscirev.2018.03.001.

**Aprendizaje automático clásico**

[8] A. M. Youssef y H. R. Pourghasemi, "Landslide susceptibility mapping using machine learning algorithms and comparison of their performance at Abha Basin, Asir Region, Saudi Arabia," *Geoscience Frontiers*, vol. 12, núm. 2, pp. 639–655, mar. 2021, doi: 10.1016/j.gsf.2020.05.010.

[9] Z. Zhou, "Integrated Ensemble Learning, Feature Selection, and Hyperparameter Optimization for Landslide Susceptibility Mapping," *Remote Sens.*, 2024.

**Aprendizaje profundo — detección**

[10] T. Chen, X. Gao, G. Liu, C. Wang, Z. Zhao, J. Dou, R. Niu, y A. J. Plaza, "BisDeNet: A New Lightweight Deep Learning-Based Framework for Efficient Landslide Detection," *IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.*, vol. 17, pp. 3648–3663, 2024, doi: 10.1109/JSTARS.2024.3351873.

[11] R. Uribe-Ventura, "GeoNeXt: Efficient Landslide Mapping Using a Pre-Trained ConvNeXt V2 Encoder with Transformer-Based Decoder," *Remote Sens.*, 2025.

[12] Y. Song, "Landslide Detection Using Deep Learning on Remotely Sensed Images," *Remote Sens.*, 2025.

**Segmentación pixel-level (U-Net)**

[13] Y. Song, Y. Zou, Y. Li, Y. He, W. Wu, R. Niu, y S. Xu, "Enhancing Landslide Detection with SBConv-Optimized U-Net Architecture Based on Remote Sensing Images," *IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.*, 2025.

[14] J. Wang, Q. Zhang, H. Xie, Y. Chen, y R. Sun, "Enhanced Dual-Channel Model-Based with Improved Unet++ Network for Landslide Monitoring," *Remote Sens.*, 2024.

**Transferibilidad y domain adaptation**

[15] Z. Wang y A. Brenning, "Unsupervised active–transfer learning for automated landslide mapping," *Comput. Geosci.*, vol. 181, p. 105457, dic. 2023, doi: 10.1016/j.cageo.2023.105457.

[16] L. Nava, E. Carraro, C. Reyes-Carmona, S. Puliero, K. Bhuyan, A. Rosi, O. Monserrat, M. Floris, S. R. Meena, J. P. Galve, y F. Catani, "Landslide displacement forecasting using deep learning and monitoring data across selected sites," *Landslides*, vol. 20, núm. 10, pp. 2111–2129, oct. 2023, doi: 10.1007/s10346-023-02104-9.

**Dataset**

[17] O. Ghorbanzadeh, H. Shahabi, A. Crivellari, S. Homayouni, T. Blaschke, y P. Ghamisi, "Landslide4Sense: Reference Benchmark Data and Deep Learning Models for Landslide Detection," *IEEE Trans. Geosci. Remote Sens.*, vol. 60, pp. 1–17, 2022, doi: 10.1109/TGRS.2022.3215209.

**Vision Transformers y comparación**

[18] P. Lv, L. Ma, Q. Li, y F. Du, "ShapeFormer: A Shape-Enhanced Vision Transformer Model for Optical Remote Sensing Image Landslide Detection," *IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.*, vol. 16, pp. 2681–2689, 2023, doi: 10.1109/JSTARS.2023.3253769.

**Mapeo cross-domain**

[19] B. Yu, F. Chen, W. Chen, G. Shi, C. Xu, N. Wang, y L. Wang, "Cross-Domain Landslide Mapping by Harmonizing Heterogeneous Remote Sensing Datasets," *Remote Sens.*, 2025.

[20] J. Chen, J. Liu, X. Zeng, S. Zhou, G. Sun, S. Rao, Y. Guo, y J. Zhu, "A Cross-Domain Landslide Extraction Method Utilizing Image Masking and Morphological Operations," *Remote Sens.*, 2025.

---

*Proyecto desarrollado en el marco del programa de Maestría en Inteligencia Artificial de la Universidad EAFIT · Medellín, Colombia · 2025.*
