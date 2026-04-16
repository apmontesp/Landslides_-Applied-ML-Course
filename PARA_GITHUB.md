# Guía de Subida a GitHub — Landslide4Sense

> Este archivo es solo para referencia local. **No sube a GitHub** (agrégalo al .gitignore si quieres).

---

## ✅ LO QUE SÍ SUBE A GITHUB

```
landslide4sense-ml/
│
├── README.md                          ✅ Portada del repo con badges y resultados
├── LICENSE                            ✅ MIT License
├── .gitignore                         ✅ Reglas de exclusión
├── requirements.txt                   ✅ Dependencias pip (con versiones)
├── environment.yml                    ✅ Entorno Conda reproducible
│
├── src/                               ✅ MÓDULO PYTHON COMPLETO
│   ├── __init__.py
│   ├── config.py                      ✅ TrainingConfig, constantes del dataset
│   ├── dataset.py                     ✅ Dataset PyTorch + Augmenter + DataLoaders
│   ├── models.py                      ✅ ResNet-50, EfficientNet-B4, U-Net + adapt_first_conv
│   ├── train.py                       ✅ Loop entrenamiento, K-Fold, EarlyStopping
│   ├── evaluate.py                    ✅ Métricas, curvas ROC/PR, comparativa
│   └── utils.py                       ✅ set_seed, checkpoints, visualizaciones
│
├── configs/                           ✅ HIPERPARÁMETROS POR MODELO
│   ├── resnet50.yaml
│   ├── efficientnet_b4.yaml
│   └── unet_resnet34.yaml
│
├── notebooks/                         ✅ PASO A PASO (8 notebooks)
│   ├── 00_setup_verification.ipynb    ✅ Verificar entorno y dataset
│   ├── 01_eda_analysis.ipynb          ✅ EDA con datos reales
│   ├── 02_preprocessing.ipynb         ✅ Normalización y augmentation
│   ├── 03_baseline_rf.ipynb           ✅ Baseline Random Forest + HOG
│   ├── 04_resnet50.ipynb              ✅ Fine-tuning ResNet-50
│   ├── 05_efficientnet_b4.ipynb       ✅ Fine-tuning EfficientNet-B4
│   ├── 06_unet_segmentation.ipynb     ✅ U-Net segmentación pixel-level
│   └── 07_evaluation_comparison.ipynb ✅ Comparativa final + ablation study
│
├── scripts/                           ✅ SCRIPTS CLI
│   ├── run_eda.py                     ✅ EDA desde terminal
│   ├── run_training.py                ✅ Entrenamiento K-Fold desde terminal
│   ├── run_evaluation.py              ✅ Evaluación y comparativa
│   └── run_all.sh                     ✅ Pipeline completo automatizado
│
├── docs/                              ✅ DOCUMENTACIÓN TÉCNICA
│   ├── methodology.md                 ✅ Diseño experimental detallado
│   ├── results.md                     ✅ Tabla de resultados y ablation study
│   ├── colombia_transfer.md           ✅ Transferibilidad a Colombia (Andes)
│   └── figures/                       ✅ Figuras EDA generadas con datos reales
│       ├── fig1_samples_pos_neg.png
│       ├── fig2_class_balance_areas.png
│       ├── fig3_channel_class_comparison.png
│       ├── fig4_histograms_by_class.png
│       ├── fig5_correlation_matrix.png
│       ├── fig6_leakage_check.png
│       └── fig7_mask_details.png
│
├── tests/                             ✅ TESTS UNITARIOS
│   ├── __init__.py
│   ├── test_dataset.py                ✅ Tests de normalización, augmentation, Dataset
│   └── test_models.py                 ✅ Tests de arquitecturas, forward pass, factory
│
├── data/                              ✅ Solo las instrucciones (sin datos .h5)
│   └── README.md                      ✅ Instrucciones de descarga del dataset
│
└── results/
    └── .gitkeep                       ✅ Mantiene la carpeta vacía en el repo
```

---

## ❌ LO QUE NO SUBE A GITHUB

| Archivo / Carpeta | Razón |
|---|---|
| `TrainData/`, `ValidData/`, `TestData/` | Dataset ~3 GB — descarga separada desde Kaggle |
| `*.h5` | Archivos de imagen del dataset |
| `eda_outputs/` | Salidas generadas localmente (JSON, PNG de análisis) |
| `results/` (excepto `.gitkeep`) | Checkpoints `.pth` y métricas del entrenamiento |
| `checkpoints/` | Pesos del modelo entrenado (~100 MB+ por fold) |
| `Articulo_Final_LandslideDetection_ML.docx` | Artículo académico (entrega separada) |
| `CLAUDE.md` | Archivo interno de la herramienta de desarrollo |
| `eda_landslide4sense.py` (raíz) | Reemplazado por `scripts/run_eda.py` |
| `fig*.png` (en raíz) | Duplicados de `docs/figures/` y `eda_outputs/` |
| `notebooks/Landslide4Sense_EDA_Pipeline_legacy.ipynb` | Notebook antiguo, reemplazado por `notebooks/01-07` |
| `*.pyc`, `__pycache__/` | Bytecode de Python |
| `.ipynb_checkpoints/` | Checkpoints de Jupyter |
| `kaggle.json` | Credenciales API de Kaggle |

---

## 🚀 Pasos para Subir a GitHub

```bash
# 1. Crear el repositorio en GitHub (vacío, sin README)
#    → ir a github.com → New repository → "landslide4sense-ml"

# 2. Desde la carpeta del proyecto:
cd /ruta/a/Landslide_ML

# 3. Inicializar git
git init
git branch -M main

# 4. Conectar con GitHub
git remote add origin https://github.com/TU_USUARIO/landslide4sense-ml.git

# 5. Agregar solo los archivos correctos
git add README.md LICENSE .gitignore requirements.txt environment.yml
git add src/ configs/ notebooks/ scripts/ docs/ tests/ data/README.md results/.gitkeep

# 6. Primer commit
git commit -m "Initial commit: Landslide4Sense deep learning pipeline

Includes ResNet-50, EfficientNet-B4, U-Net+ResNet-34 fine-tuning
on Landslide4Sense 14-channel multispectral dataset.
5-Fold CV, EDA real, ablation study, Colombia transferability docs.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

# 7. Subir
git push -u origin main
```

### Verificar antes de subir

```bash
# Ver qué archivos irían al commit (sin subir)
git status
git diff --cached --name-only

# Verificar que .h5 y resultados están excluidos
git check-ignore -v TrainData/img/image_1.h5   # Debe decir: ignorado
git check-ignore -v eda_outputs/               # Debe decir: ignorado
```

---

## 📁 Tamaño estimado del repositorio

| Sección | Tamaño aprox. |
|---------|--------------|
| `src/` (código Python) | ~150 KB |
| `configs/` (YAML) | ~5 KB |
| `notebooks/` (8 notebooks) | ~200 KB |
| `scripts/` (3 scripts) | ~40 KB |
| `docs/figures/` (7 PNG del EDA) | ~3–5 MB |
| `docs/` (Markdown) | ~40 KB |
| `tests/` | ~20 KB |
| **Total repositorio** | **~5–6 MB** |

Tamaño muy manejable para GitHub (límite gratuito: 1 GB por repo).
