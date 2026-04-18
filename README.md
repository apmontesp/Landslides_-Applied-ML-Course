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
- [Referencias](#referencias)
  - Dataset · Arquitecturas CNN · Métodos clásicos · Teledetección · Herramientas · Contexto colombiano

---

## Descripción del Proyecto

Los deslizamientos de tierra son uno de los fenómenos de remoción en masa más destructivos a nivel global, con estimaciones de más de 4,000 fatalidades anuales y daños económicos superiores a 100 millones de dólares por evento [8][13]. Su ocurrencia está condicionada por factores intrínsecos —topografía, geología, hidrología, cobertura vegetal— y se ve agravada por detonantes externos como precipitaciones intensas, eventos sísmicos e intervención antrópica [13].

En Colombia, la complejidad geológica, sumada a lluvias intensas, pendientes fuertes y el estado de meteorización de los materiales, configura un escenario de elevada susceptibilidad a movimientos en masa [13]. Estos fenómenos se generan principalmente en la región Andina, donde se asienta más del 70% de la población colombiana y se concentra la mayor parte de la infraestructura vial y productiva del país. Las implicaciones humanas y económicas presentan tendencias crecientes, fuertemente asociadas a eventos de lluvia y al Fenómeno de La Niña [14]. A pesar de los avances en metodologías de detección, su efectividad es limitada en el contexto montañoso y tropical del país, donde la alta nubosidad, la cobertura vegetal densa y la complejidad geomorfológica reducen el desempeño de los modelos entrenados en otros contextos [13][15].

Este proyecto implementa y compara modelos de aprendizaje automático clásico y arquitecturas de aprendizaje profundo para la detección automática de deslizamientos sobre el dataset multi-espectral **Landslide4Sense** [1], con análisis explícito de las implicaciones de transferibilidad al contexto colombiano:

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

**Landslide4Sense** — ISPRS Competition 2022 [1]

| Partición | Imágenes | Máscaras | Descripción |
|-----------|----------|----------|-------------|
| TrainData | 3,799 | 3,799 | Entrenamiento con anotaciones pixel-level |
| ValidData | 245 | — | Validación sin etiquetas públicas |
| TestData | 800 | — | Prueba (competición, sin etiquetas) |

**Estructura de cada parche:** `128 × 128 × 14` canales (float32)

| Canales | Fuente | Bandas |
|---------|--------|--------|
| 0–6 | Sentinel-2 [9] | B2(Azul), B3(Verde), B4(Rojo), B8(NIR), B8A(NIR-A), B11(SWIR1), B12(SWIR2) |
| 7–8 | Sentinel-1 SAR | VV, VH |
| 9–10 | ALOS PALSAR | DEM Elevación, Pendiente |
| 11–13 | Sentinel-2 Red-Edge [9] | B5, B6, B7 |

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

El proyecto sigue una escalera de complejidad deliberada: primero se establecen baselines clásicos con ingeniería de características manual, luego se evalúan arquitecturas CNN con fine-tuning, y finalmente segmentación pixel-level. Esta progresión permite aislar la contribución real de cada nivel de complejidad.

### Baselines Clásicos (notebook 03)

Los tres modelos clásicos usan un vector de características extraído de cada parche: **HOG** [5] sobre falso color RGB (Sentinel-2 B4/B3/B2) + pendiente DEM + media NDVI + media SAR VH. Se aplica un Pipeline `SimpleImputer(median) → StandardScaler` antes de cada modelo sensible a escala. Las implementaciones se realizan con scikit-learn [11].

**Logistic Regression**
- Pipeline con StandardScaler obligatorio
- `C=1.0`, `class_weight='balanced'`, `max_iter=1000`
- Sirve como baseline lineal de referencia

**SVM kernel RBF** [7]
- Pipeline con StandardScaler obligatorio
- `C=1.0`, `gamma='scale'`, `class_weight='balanced'`
- Captura fronteras de decisión no lineales en el espacio de features

**Random Forest (ensemble)** [6]
- No requiere escalado (invariante a monotonías)
- `n_estimators=100`, `class_weight='balanced'`
- Proporciona importancia de features interpretable

### ResNet-50 Fine-Tuned (notebook 04) [2]
- Entrada adaptada: 14 canales (mean-initialization desde pesos ImageNet)
- Protocolo: congelar backbone 5 épocas → descongelar completo
- Loss: Weighted BCE (`pos_weight=0.70`)
- Optimizer: AdamW (lr_head=1e-4, lr_backbone=1e-5)
- Implementado en PyTorch [10]

### EfficientNet-B4 Fine-Tuned (notebook 05) [3]
- Misma estrategia de adaptación de canales que ResNet-50
- 19M parámetros vs 25M de ResNet-50
- Escalado compuesto: depth=1.8, width=1.4, resolution=1.3
- Implementado en PyTorch [10]

### U-Net + ResNet-34 — Segmentación pixel-level (notebook 06) [4]
- Encoder ResNet-34 preentrenado en ImageNet
- Loss: Dice Loss + BCE (50/50)
- Produce mapas de probabilidad 128×128 — tarea fundamentalmente distinta a los modelos anteriores (clasificación de parche)
- Implementado con `segmentation-models-pytorch` [12]

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

1. **Ingeniería de características efectiva** — HOG [5] + Pendiente DEM + NDVI + SAR VH capturan las señales espectrales más discriminativas identificadas en el EDA (Ch12-13 RedEdge, Ch9 DEM). El Random Forest [6] extrae estas relaciones directamente sin necesidad de aprender representaciones desde cero.

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

La mayoría de los modelos publicados en la literatura han sido entrenados y validados en contextos geomorfológicos específicos —principalmente China, Italia y Grecia— con características topográficas, climáticas e hidrológicas completamente distintas a las del territorio colombiano. Implementar estos modelos sin reentrenamiento ni adaptación puede implicar una reducción de hasta el 25% en las métricas de desempeño [15]. Este proyecto, entrenado sobre Landslide4Sense, no es la excepción: sus resultados proveen una línea base robusta, pero presentan tres brechas de dominio que condicionan su transferibilidad directa al contexto andino:

- **Nubosidad:** 60–80% de cobertura de nubes en zonas montañosas colombianas degrada sistemáticamente las bandas ópticas de Sentinel-2; el SAR (Sentinel-1) mantiene penetración atmosférica pero pierde sensibilidad en pendientes pronunciadas por efecto de layover [13]
- **Vegetación:** Los bosques húmedos tropicales presentan mayor densidad y complejidad de dosel que las regiones del dataset; la señal RedEdge (Ch12-13), la más discriminativa según el EDA, se satura en coberturas densas, reduciendo su poder de separación entre clases [13]
- **Litología:** El ambiente volcánico-metamórfico de los Andes colombianos difiere de las regiones loéssicas o calcáreas predominantes en el dataset original, afectando las firmas espectrales en bandas SWIR y SAR [13]
- **Disponibilidad de inventarios:** La limitada cobertura del inventario nacional (SGC/SIMMA) en zonas de difícil acceso restringe la posibilidad de validación local y reentrenamiento con datos propios [13]

**Cuantificación de la brecha:** Wang y Brenning [15] demuestran que la transferencia de modelos entre contextos geomorfológicos sin adaptación de dominio puede producir degradaciones de F1 de hasta 25 puntos porcentuales. Aplicado al mejor resultado obtenido (RF F1=0.837), esto proyecta un desempeño de F1≈0.63 en datos colombianos sin reentrenamiento local, lo que subraya la necesidad de adaptación antes de cualquier despliegue operativo.

**Trabajo futuro — integración UAV e IA local:** Los vehículos aéreos no tripulados (UAV) constituyen una herramienta versátil para la adquisición de información de alta resolución en entornos de difícil acceso, con capacidad de obtener ortomosaicos y modelos digitales de elevación a escala centimétrica [16]. La integración de datos UAV con algoritmos de inteligencia artificial calibrados para las condiciones locales representa la ruta más directa para superar las brechas identificadas. El trabajo futuro contempla: colecta de datos de campo en municipios de alta susceptibilidad de Antioquia (Abriaquí, Dabeiba, Salgar) mediante inspecciones UAV; construcción de un inventario georreferenciado con el Sistema de Información de Movimientos en Masa (SIMMA/SGC) [13]; y fine-tuning de los modelos sobre datos mixtos (Landslide4Sense + datos locales) mediante técnicas de domain adaptation (CORAL) para estimar y compensar la brecha de dominio espectral.

---

## Referencias

**Dataset y competición**

[1] O. Ghorbanzadeh, H. Shahabi, A. Crivellari, S. Homayouni, T. Blaschke, and P. Ghamisi, "Landslide4Sense: Reference Benchmark Data and Deep Learning Models for Landslide Detection," *IEEE Trans. Geosci. Remote Sens.*, vol. 60, pp. 1–17, 2022, doi: 10.1109/TGRS.2022.3215209.

**Arquitecturas CNN**

[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Las Vegas, NV, USA, 2016, pp. 770–778.

[3] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. 36th Int. Conf. Mach. Learn. (ICML)*, Long Beach, CA, USA, 2019, pp. 6105–6114.

[4] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in *Proc. Med. Image Comput. Comput.-Assisted Intervention (MICCAI)*, Munich, Germany, 2015, pp. 234–241.

**Métodos clásicos y extracción de características**

[5] N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, San Diego, CA, USA, 2005, pp. 886–893.

[6] L. Breiman, "Random Forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.

[7] C. Cortes and V. Vapnik, "Support-Vector Networks," *Mach. Learn.*, vol. 20, no. 3, pp. 273–297, 1995.

**Teledetección y deslizamientos**

[8] H. Yamagishi and F. Yamazaki, "Landslides by the 2018 Hokkaido Iburi-Tobu Earthquake on September 6," *Landslides*, vol. 15, no. 12, pp. 2521–2524, 2018.

[9] European Space Agency, "Sentinel-2 Level-2A Product Guide," Copernicus Data Space Ecosystem, 2023. [Online]. Available: https://dataspace.copernicus.eu

**Herramientas y librerías**

[10] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, Vancouver, Canada, 2019, vol. 32.

[11] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[12] P. Iakubovskii, "Segmentation Models PyTorch," GitHub, 2019. [Online]. Available: https://github.com/qubvel/segmentation_models.pytorch

**Contexto colombiano y transferibilidad**

[13] Servicio Geológico Colombiano, *Las amenazas por movimientos en masa de Colombia, una visión a escala 1:100.000*. Bogotá: SGC, 2017, doi: 10.32685/9789589952887.

[14] J. Ayala-García and K. Ospino-Ramos, "Desastres naturales en Colombia: un análisis regional," *Documentos de Trabajo sobre Economía Regional y Urbana*, Banco de la República, 2019. [Online]. Available: https://www.banrep.gov.co

[15] Z. Wang and A. Brenning, "Unsupervised active–transfer learning for automated landslide mapping," *Comput. Geosci.*, vol. 181, p. 105457, Dec. 2023, doi: 10.1016/j.cageo.2023.105457.

[16] J. Sun, G. Yuan, L. Song, and H. Zhang, "Unmanned Aerial Vehicles (UAVs) in Landslide Investigation and Monitoring: A Review," *Drones*, vol. 8, no. 1, p. 30, Jan. 2024, doi: 10.3390/drones8010030.