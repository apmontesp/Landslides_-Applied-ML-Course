#!/usr/bin/env python3
"""
run_training.py — Entrenamiento desde CLI para Landslide4Sense.

Uso:
    # ResNet-50 con config YAML
    python scripts/run_training.py --config configs/resnet50.yaml

    # EfficientNet-B4 sobreescribiendo parámetros
    python scripts/run_training.py --config configs/efficientnet_b4.yaml \
        --epochs 30 --batch_size 16

    # Debug rápido (2 épocas, sin GPU)
    python scripts/run_training.py --config configs/resnet50.yaml --debug

    # Un solo fold específico
    python scripts/run_training.py --config configs/resnet50.yaml --fold 0
"""

import argparse
import sys
from pathlib import Path

# Agregar el raíz del proyecto al PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TrainingConfig
from src.train import run_kfold, run_fold
from src.dataset import get_dataloaders, get_fold_indices, Landslide4SenseDataset
from src.utils import set_seed, check_data_structure, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelos CNN para detección de deslizamientos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Configuración base ────────────────────────────────────
    parser.add_argument(
        "--config", type=str, required=True,
        help="Ruta al archivo YAML de configuración (configs/resnet50.yaml, etc.)"
    )

    # ── Sobreescrituras de hiperparámetros ─────────────────────
    parser.add_argument("--data_root",   type=str,   default=None, help="Directorio raíz del dataset")
    parser.add_argument("--output_dir",  type=str,   default=None, help="Directorio de salida de resultados")
    parser.add_argument("--epochs",      type=int,   default=None, help="Número máximo de épocas")
    parser.add_argument("--batch_size",  type=int,   default=None, help="Tamaño del batch")
    parser.add_argument("--lr_head",     type=float, default=None, help="Learning rate para la cabeza")
    parser.add_argument("--lr_backbone", type=float, default=None, help="Learning rate para el backbone")
    parser.add_argument("--n_folds",     type=int,   default=None, help="Número de folds en K-Fold CV")
    parser.add_argument("--seed",        type=int,   default=None, help="Semilla aleatoria")

    # ── Modo de ejecución ─────────────────────────────────────
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Entrenar solo este fold (0-indexed). None = todos los folds."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Modo debug: 2 épocas, batch=8, sin TensorBoard, 2 folds."
    )
    parser.add_argument(
        "--no_pretrained", action="store_true",
        help="Entrenar desde cero (sin pesos ImageNet)."
    )
    parser.add_argument(
        "--check_data", action="store_true",
        help="Solo verificar la estructura del dataset y salir."
    )

    return parser.parse_args()


def apply_overrides(cfg: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    """Aplica las sobreescrituras de argumentos CLI sobre la configuración YAML."""
    overrides = {
        "data_root":    args.data_root,
        "output_dir":   args.output_dir,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr_head":      args.lr_head,
        "lr_backbone":  args.lr_backbone,
        "n_folds":      args.n_folds,
        "seed":         args.seed,
    }
    for key, val in overrides.items():
        if val is not None:
            setattr(cfg, key, val)

    if args.debug:
        cfg.epochs        = 2
        cfg.batch_size    = 8
        cfg.n_folds       = 2
        cfg.patience      = 2
        cfg.early_stopping = False
        cfg.tensorboard   = False
        cfg.augmentation  = False
        cfg.verbose       = True
        print("[DEBUG] Modo debug activado.")

    if args.no_pretrained:
        cfg.pretrained = False
        cfg.experiment_name += "_scratch"

    return cfg


def main() -> None:
    args = parse_args()

    # ── Cargar configuración ──────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Archivo de configuración no encontrado: {cfg_path}")
        sys.exit(1)

    cfg = TrainingConfig.from_yaml(str(cfg_path))
    cfg = apply_overrides(cfg, args)

    print(f"\n{'='*60}")
    print(f"  Landslide4Sense — Entrenamiento")
    print(f"  Experimento : {cfg.experiment_name}")
    print(f"  Arquitectura: {cfg.model_arch}")
    print(f"  Config YAML : {cfg_path}")
    print(f"  Device      : {cfg.resolve_device()}")
    print(f"{'='*60}\n")

    # ── Verificar datos ───────────────────────────────────────
    info = check_data_structure(cfg.data_root)
    if not info.get("valid", False):
        print("\n[ERROR] Estructura del dataset inválida. Verifica la descarga.")
        sys.exit(1)
    print()

    if args.check_data:
        print("[OK] Verificación completada. Saliendo (--check_data).")
        sys.exit(0)

    # ── Ejecutar entrenamiento ────────────────────────────────
    set_seed(cfg.seed)

    if args.fold is not None:
        # Entrenar un fold específico
        full_ds = Landslide4SenseDataset(
            img_dir=str(Path(cfg.data_root) / "TrainData" / "img"),
            mask_dir=str(Path(cfg.data_root) / "TrainData" / "mask"),
            normalize=False,
        )
        folds = get_fold_indices(full_ds, n_folds=cfg.n_folds, seed=cfg.seed)
        fold_idx = args.fold
        if fold_idx >= len(folds):
            print(f"[ERROR] Fold {fold_idx} fuera de rango (n_folds={cfg.n_folds}).")
            sys.exit(1)
        train_idx, val_idx = folds[fold_idx]
        print(f"Entrenando solo Fold {fold_idx} (train={len(train_idx)}, val={len(val_idx)})")
        result = run_fold(cfg, fold_idx, train_idx, val_idx)
        print(f"\nMejores métricas (Fold {fold_idx}):")
        for k, v in result["best_metrics"].items():
            print(f"  {k}: {v:.4f}")
    else:
        # K-Fold completo
        summary = run_kfold(cfg)
        print(f"\n[OK] Entrenamiento completado. Resultados en: {cfg.output_dir}")
        print(f"     Resumen: {cfg.output_dir}/kfold_summary.json")


if __name__ == "__main__":
    main()
