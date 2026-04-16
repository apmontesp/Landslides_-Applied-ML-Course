#!/usr/bin/env python3
"""
run_evaluation.py — Evaluación y comparación de modelos desde CLI.

Uso:
    # Evaluar un modelo específico (requiere checkpoint)
    python scripts/run_evaluation.py \
        --config configs/resnet50.yaml \
        --ckpt results/resnet50/fold_0/best_model.pth \
        --output_dir results/resnet50/evaluation

    # Comparar todos los modelos entrenados
    python scripts/run_evaluation.py \
        --compare \
        --results_dirs results/resnet50 results/efficientnet_b4 results/unet_resnet34 \
        --output_dir results/comparison
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import TrainingConfig
from src.dataset import Landslide4SenseDataset, get_fold_indices, get_dataloaders
from src.evaluate import evaluate_model, compare_models
from src.models import build_model
from src.utils import load_checkpoint, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación y comparativa de modelos Landslide4Sense.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Modo 1: evaluar un modelo ─────────────────────────────
    parser.add_argument("--config",     type=str, default=None, help="Config YAML del modelo a evaluar")
    parser.add_argument("--ckpt",       type=str, default=None, help="Ruta al checkpoint .pth")
    parser.add_argument("--fold",       type=int, default=0,    help="Fold de validación a usar")
    parser.add_argument("--partition",  type=str, default="val",
                        choices=["train", "val", "valid", "test"],
                        help="Partición del dataset a evaluar")
    parser.add_argument("--output_dir", type=str, default="./results/evaluation",
                        help="Directorio para guardar figuras y reporte")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Umbral fijo (si None, se optimiza automáticamente)")

    # ── Modo 2: comparar múltiples experimentos ────────────────
    parser.add_argument("--compare",     action="store_true",
                        help="Comparar todos los experimentos en --results_dirs")
    parser.add_argument("--results_dirs", type=str, nargs="+", default=None,
                        help="Directorios de resultados de cada experimento")

    return parser.parse_args()


def evaluate_single(args: argparse.Namespace) -> None:
    """Evalúa un modelo sobre la partición especificada."""
    if not args.config:
        print("[ERROR] Especifica --config para evaluar un modelo.")
        sys.exit(1)

    cfg = TrainingConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.resolve_device())

    # ── Preparar DataLoader ───────────────────────────────────
    data_root = Path(cfg.data_root)

    if args.partition in ("val", "valid"):
        full_ds = Landslide4SenseDataset(
            img_dir=str(data_root / "TrainData" / "img"),
            mask_dir=str(data_root / "TrainData" / "mask"),
            normalize=False,
        )
        folds = get_fold_indices(full_ds, n_folds=cfg.n_folds, seed=cfg.seed)
        _, val_idx = folds[args.fold]
        loaders = get_dataloaders(cfg, train_indices=None, val_indices=val_idx)
        loader  = loaders["val"]
        model_name = f"{cfg.experiment_name} (fold {args.fold})"
    elif args.partition == "train":
        full_ds = Landslide4SenseDataset(
            img_dir=str(data_root / "TrainData" / "img"),
            mask_dir=str(data_root / "TrainData" / "mask"),
            normalize=False,
        )
        folds = get_fold_indices(full_ds, n_folds=cfg.n_folds, seed=cfg.seed)
        train_idx, _ = folds[args.fold]
        loaders = get_dataloaders(cfg, train_indices=train_idx, val_indices=None)
        loader  = loaders["train"]
        model_name = f"{cfg.experiment_name} [train]"
    else:
        from src.dataset import get_test_loader
        loader     = get_test_loader(cfg, partition="TestData")
        model_name = f"{cfg.experiment_name} [test]"

    # ── Cargar modelo ─────────────────────────────────────────
    model = build_model(cfg.model_arch, n_channels=cfg.n_channels, pretrained=False)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            print(f"[ERROR] Checkpoint no encontrado: {ckpt_path}")
            sys.exit(1)
        load_checkpoint(model, str(ckpt_path), device=device)
    else:
        # Buscar automáticamente el mejor checkpoint del fold
        auto_ckpt = Path(cfg.output_dir) / f"fold_{args.fold}" / "best_model.pth"
        if auto_ckpt.exists():
            load_checkpoint(model, str(auto_ckpt), device=device)
            print(f"[OK] Checkpoint automático: {auto_ckpt}")
        else:
            print(f"[WARN] No se encontró checkpoint. Evaluando con pesos iniciales.")

    model = model.to(device)

    # ── Evaluar ───────────────────────────────────────────────
    threshold = args.threshold if args.threshold else cfg.threshold
    optimize  = args.threshold is None and cfg.optimize_threshold

    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        threshold=threshold,
        optimize_threshold=optimize,
        model_name=model_name,
        output_dir=args.output_dir,
    )

    print(f"\n[OK] Evaluación completada. Resultados en: {args.output_dir}")


def main() -> None:
    args = parse_args()

    if args.compare:
        if not args.results_dirs:
            print("[ERROR] Especifica --results_dirs para la comparativa.")
            sys.exit(1)
        print("\n" + "="*60)
        print("  Comparativa de Modelos — Landslide4Sense")
        print("="*60)
        compare_models(args.results_dirs)
    else:
        evaluate_single(args)


if __name__ == "__main__":
    main()
