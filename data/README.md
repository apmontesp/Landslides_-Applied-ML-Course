# Dataset — Instrucciones de Descarga

Este directorio contiene los datos del proyecto. **Los archivos de datos NO se incluyen en el repositorio** debido a su tamaño (~3 GB).

## Landslide4Sense — ISPRS Competition 2022

### Estructura esperada

```
data/
├── TrainData/
│   ├── img/         ← 3,799 parches .h5  (128×128×14 float32)
│   └── mask/        ← 3,799 máscaras .h5 (128×128 uint8)
├── ValidData/
│   └── img/         ← 245 parches .h5 (sin máscaras públicas)
└── TestData/
    └── img/         ← 800 parches .h5  (sin máscaras, competición)
```

### Opción 1 — Kaggle API (recomendado)

```bash
# 1. Instalar Kaggle CLI
pip install kaggle --break-system-packages

# 2. Configurar credenciales
#    Ir a https://www.kaggle.com/settings → API → Create New Token
#    Guardar kaggle.json en ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 3. Descargar dataset
kaggle datasets download -d landslide4sense/competition -p data/
unzip data/competition.zip -d data/
```

### Opción 2 — Descarga manual

1. Ingresar a [kaggle.com/datasets/landslide4sense/competition](https://www.kaggle.com/datasets/landslide4sense/competition)
2. Click en **Download** (requiere cuenta Kaggle gratuita)
3. Descomprimir `competition.zip` en este directorio `data/`

### Opción 3 — Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

# Si el dataset ya está en Drive:
import shutil
shutil.copytree('/content/drive/MyDrive/landslide4sense', '/content/data')

# O con Kaggle API en Colab:
!pip install kaggle
from google.colab import files
files.upload()  # subir kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d landslide4sense/competition -p /content/data/
!unzip /content/data/competition.zip -d /content/data/
```

### Verificación

Una vez descargado, verificar la integridad:

```bash
python scripts/run_eda.py --data_root ./data --check_only
```

Salida esperada:
```
[OK] TrainData/img  → 3,799 archivos .h5
[OK] TrainData/mask → 3,799 archivos .h5
[OK] ValidData/img  → 245 archivos .h5
[OK] TestData/img   → 800 archivos .h5
[OK] Forma de parche: (128, 128, 14)  dtype=float32
[OK] Forma de máscara: (128, 128)     dtype=uint8
```

### Referencia del Dataset

```bibtex
@article{ghorbanzadeh2022landslide4sense,
  title   = {Landslide4Sense: Reference Benchmark Data and Deep Learning Models for Landslide Detection},
  author  = {Ghorbanzadeh, Omid and others},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2022},
  volume  = {60},
  pages   = {1--17},
  doi     = {10.1109/TGRS.2022.3215209}
}
```
