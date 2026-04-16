#!/usr/bin/env bash
# =============================================================
# run_all.sh — Pipeline completo de Landslide4Sense
#
# Ejecuta en orden: EDA → Baseline RF → ResNet-50 → EfficientNet-B4
#                   → U-Net → Evaluación comparativa final
#
# Uso:
#   chmod +x scripts/run_all.sh
#   ./scripts/run_all.sh --data_root ./data
#   ./scripts/run_all.sh --data_root ./data --debug   # Pipeline de prueba
# =============================================================

set -euo pipefail

# ── Colores ───────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'  # Sin color

# ── Valores por defecto ───────────────────────────────────────
DATA_ROOT="./data"
OUTPUT_ROOT="./results"
N_SAMPLE=100
DEBUG=false
SKIP_EDA=false
SKIP_RF=false
SKIP_TRAINING=false

# ── Parser de argumentos ──────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)    DATA_ROOT="$2"; shift 2 ;;
        --output_dir)   OUTPUT_ROOT="$2"; shift 2 ;;
        --n_sample)     N_SAMPLE="$2"; shift 2 ;;
        --debug)        DEBUG=true; shift ;;
        --skip_eda)     SKIP_EDA=true; shift ;;
        --skip_rf)      SKIP_RF=true; shift ;;
        --skip_training) SKIP_TRAINING=true; shift ;;
        -h|--help)
            echo "Uso: $0 [--data_root DIR] [--debug] [--skip_eda] [--skip_rf] [--skip_training]"
            exit 0 ;;
        *) echo "Argumento desconocido: $1"; exit 1 ;;
    esac
done

DEBUG_FLAG=""
if [ "$DEBUG" = true ]; then
    DEBUG_FLAG="--debug"
    N_SAMPLE=20
    echo -e "${YELLOW}[DEBUG] Pipeline de prueba — parámetros reducidos.${NC}"
fi

# ── Banner ────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Landslide4Sense — Pipeline Completo                ║${NC}"
echo -e "${BLUE}║   Proyecto Final — Aprendizaje de Máquinas 2026      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Datos     : $DATA_ROOT"
echo "  Resultados: $OUTPUT_ROOT"
echo "  Debug     : $DEBUG"
echo ""

# ── Función de log ────────────────────────────────────────────
log_step() {
    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  PASO $1: $2${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
}

log_ok() {
    echo -e "${GREEN}[✓] $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[!] $1${NC}"
}

log_error() {
    echo -e "${RED}[✗] $1${NC}"
    exit 1
}

# ── 0. Verificar entorno ──────────────────────────────────────
log_step 0 "Verificación del entorno Python"

python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')" || \
    log_error "PyTorch no está instalado. Ejecuta: pip install -r requirements.txt"

python -c "import h5py; print(f'h5py {h5py.__version__}')" || \
    log_error "h5py no instalado."

python -c "import segmentation_models_pytorch; print('segmentation_models_pytorch OK')" || \
    log_warn "segmentation_models_pytorch no instalado. Instalando..."
    pip install segmentation-models-pytorch --break-system-packages -q

log_ok "Entorno verificado."

# ── 0b. Verificar datos ───────────────────────────────────────
log_step "0b" "Verificación de la estructura del dataset"
python scripts/run_training.py --config configs/resnet50.yaml \
    --data_root "$DATA_ROOT" --check_data || \
    log_error "Dataset no válido. Consulta data/README.md para descargarlo."
log_ok "Dataset verificado."

# ── 1. EDA ────────────────────────────────────────────────────
if [ "$SKIP_EDA" = false ]; then
    log_step 1 "Análisis Exploratorio de Datos (EDA)"
    mkdir -p "$OUTPUT_ROOT/eda"
    python eda_landslide4sense.py \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_ROOT/eda" \
        --n_sample "$N_SAMPLE" && log_ok "EDA completado." || log_warn "EDA falló — continuando."
else
    log_warn "EDA omitido (--skip_eda)."
fi

# ── 2. Baseline Random Forest ─────────────────────────────────
if [ "$SKIP_RF" = false ]; then
    log_step 2 "Baseline Random Forest (HOG features)"
    mkdir -p "$OUTPUT_ROOT/random_forest"
    python - <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
import numpy as np
import h5py
import json
from skimage.feature import hog

DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else './data'
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else './results/random_forest'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("Cargando datos para Random Forest (HOG features)...")
img_dir  = Path(DATA_ROOT) / "TrainData" / "img"
mask_dir = Path(DATA_ROOT) / "TrainData" / "mask"

img_files  = sorted(img_dir.glob("*.h5"))[:500]  # Muestra de 500
X, y = [], []

for img_f in img_files:
    mask_f = mask_dir / img_f.name
    with h5py.File(img_f,  "r") as f: patch = f[list(f.keys())[0]][()]
    with h5py.File(mask_f, "r") as f: mask  = f[list(f.keys())[0]][()]
    # HOG sobre canal RGB (canales 2,1,0)
    rgb = patch[:,:,[2,1,0]]
    rgb = ((rgb - rgb.min()) / (rgb.ptp() + 1e-8) * 255).astype("uint8")
    feats = hog(rgb, pixels_per_cell=(16,16), cells_per_block=(2,2), channel_axis=-1)
    X.append(feats)
    y.append(int(mask.max() > 0))

X, y = np.array(X), np.array(y)
print(f"  Features: {X.shape}, Positivos: {y.sum()}/{len(y)}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1s, aucs = [], []
for fold, (tr, va) in enumerate(skf.split(X, y)):
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X[tr], y[tr])
    probs = rf.predict_proba(X[va])[:,1]
    preds = (probs >= 0.5).astype(int)
    f1  = f1_score(y[va], preds, zero_division=0)
    auc = roc_auc_score(y[va], probs)
    f1s.append(f1); aucs.append(auc)
    print(f"  Fold {fold}: F1={f1:.4f}  AUC={auc:.4f}")

summary = {"model": "RandomForest_HOG", "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
           "auc_mean": np.mean(aucs), "auc_std": np.std(aucs)}
with open(f"{OUTPUT_DIR}/rf_summary.json","w") as f: json.dump(summary, f, indent=2)
print(f"\nRandom Forest — F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
PYEOF
    log_ok "Baseline Random Forest completado."
else
    log_warn "Random Forest omitido (--skip_rf)."
fi

# ── 3-5. Entrenamiento de modelos CNN ─────────────────────────
if [ "$SKIP_TRAINING" = false ]; then

    log_step 3 "Fine-tuning ResNet-50"
    python scripts/run_training.py \
        --config configs/resnet50.yaml \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_ROOT/resnet50" \
        $DEBUG_FLAG && log_ok "ResNet-50 completado." || log_warn "ResNet-50 falló."

    log_step 4 "Fine-tuning EfficientNet-B4"
    python scripts/run_training.py \
        --config configs/efficientnet_b4.yaml \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_ROOT/efficientnet_b4" \
        $DEBUG_FLAG && log_ok "EfficientNet-B4 completado." || log_warn "EfficientNet-B4 falló."

    log_step 5 "U-Net + ResNet-34 Segmentación"
    python scripts/run_training.py \
        --config configs/unet_resnet34.yaml \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_ROOT/unet_resnet34" \
        $DEBUG_FLAG && log_ok "U-Net completado." || log_warn "U-Net falló."

else
    log_warn "Entrenamiento omitido (--skip_training)."
fi

# ── 6. Evaluación comparativa ─────────────────────────────────
log_step 6 "Evaluación y Comparativa Final"
python scripts/run_evaluation.py \
    --compare \
    --results_dirs \
        "$OUTPUT_ROOT/resnet50" \
        "$OUTPUT_ROOT/efficientnet_b4" \
        "$OUTPUT_ROOT/unet_resnet34" \
    --output_dir "$OUTPUT_ROOT/comparison" && log_ok "Comparativa completada." || log_warn "Comparativa falló."

# ── Resumen ───────────────────────────────────────────────────
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Pipeline completado                                ║${NC}"
echo -e "${BLUE}╠══════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║   Resultados en: $OUTPUT_ROOT${NC}"
echo -e "${BLUE}║   Comparativa  : $OUTPUT_ROOT/comparison${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
