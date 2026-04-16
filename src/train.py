"""
train.py — Loop de entrenamiento para Landslide4Sense.

Implementa:
  - DiceBCELoss       : Pérdida combinada Dice + BCE para segmentación
  - train_epoch()     : Una época de entrenamiento
  - eval_epoch()      : Una época de evaluación
  - build_optimizer() : Configura AdamW con LR diferencial backbone/cabeza
  - build_scheduler() : Configura el scheduler de LR
  - EarlyStopping     : Clase para early stopping con paciencia configurable
  - run_fold()        : Entrena un fold completo con todos los callbacks
  - run_kfold()       : Orquesta el experimento K-Fold completo
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import TrainingConfig
from .dataset import get_dataloaders, get_fold_indices, Landslide4SenseDataset
from .evaluate import compute_metrics
from .models import build_model, model_summary
from .utils import set_seed, AverageMeter, format_time


# ────────────────────────────────────────────────────────────
# Funciones de pérdida
# ────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Dice Loss para segmentación binaria."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits)
        flat_p  = probs.view(-1)
        flat_t  = targets.view(-1)
        inter   = (flat_p * flat_t).sum()
        dice    = (2.0 * inter + self.smooth) / (flat_p.sum() + flat_t.sum() + self.smooth)
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """Combinación ponderada de Dice Loss y BCE para segmentación."""

    def __init__(self, dice_weight: float = 0.5, pos_weight: Optional[float] = None):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = 1.0 - dice_weight
        self.dice = DiceLoss()

        pw = torch.tensor([pos_weight]) if pos_weight else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight  * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )


def build_criterion(cfg: TrainingConfig) -> nn.Module:
    """Construye la función de pérdida según la configuración."""
    if cfg.loss == "weighted_bce":
        pw = torch.tensor([cfg.pos_weight])
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    elif cfg.loss == "dice_bce":
        return DiceBCELoss(dice_weight=cfg.dice_weight, pos_weight=cfg.pos_weight)
    elif cfg.loss == "bce":
        return nn.BCEWithLogitsLoss()
    elif cfg.loss == "focal":
        # Focal Loss básica (gamma=2, alpha=cfg.pos_weight)
        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0, alpha=None):
                super().__init__()
                self.gamma = gamma
                self.alpha = alpha
            def forward(self, logits, targets):
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                )
                probs = torch.sigmoid(logits)
                pt = torch.where(targets == 1, probs, 1 - probs)
                loss = bce * ((1 - pt) ** self.gamma)
                if self.alpha is not None:
                    alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
                    loss = alpha_t * loss
                return loss.mean()
        alpha = cfg.pos_weight / (1.0 + cfg.pos_weight)
        return FocalLoss(gamma=2.0, alpha=alpha)
    else:
        raise ValueError(f"Función de pérdida desconocida: {cfg.loss}")


# ────────────────────────────────────────────────────────────
# Optimizador y Scheduler
# ────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> optim.Optimizer:
    """
    Configura AdamW con learning rate diferencial:
    - Backbone preentrenado: lr_backbone (1e-5)
    - Cabeza nueva:          lr_head     (1e-4)
    """
    backbone_params = []
    head_params     = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Parámetros que pertenecen a la cabeza de clasificación
        if any(k in name for k in ["classifier", "head", "fc", "decoder"]):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": cfg.lr_backbone},
        {"params": head_params,     "lr": cfg.lr_head},
    ]

    if cfg.optimizer == "adamw":
        return optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam":
        return optim.Adam(param_groups, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        return optim.SGD(param_groups, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Optimizador desconocido: {cfg.optimizer}")


def build_scheduler(
    optimizer: optim.Optimizer,
    cfg: TrainingConfig,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Configura el scheduler de learning rate."""
    if cfg.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.T_max, eta_min=cfg.eta_min
        )
    elif cfg.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif cfg.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
    return None


# ────────────────────────────────────────────────────────────
# Early Stopping
# ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Detiene el entrenamiento si la métrica monitoreada no mejora."""

    def __init__(self, patience: int = 15, mode: str = "max", delta: float = 1e-5):
        self.patience   = patience
        self.mode       = mode
        self.delta      = delta
        self.best       = -np.inf if mode == "max" else np.inf
        self.counter    = 0
        self.best_epoch = 0
        self.stop       = False

    def __call__(self, metric: float, epoch: int) -> bool:
        improved = (
            metric > self.best + self.delta
            if self.mode == "max"
            else metric < self.best - self.delta
        )
        if improved:
            self.best       = metric
            self.counter    = 0
            self.best_epoch = epoch
            return True   # Mejoró → guardar checkpoint
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
            return False


# ────────────────────────────────────────────────────────────
# Épocas de entrenamiento y evaluación
# ────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cfg: TrainingConfig,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """Ejecuta una época de entrenamiento y retorna métricas promedio."""
    model.train()
    loss_meter = AverageMeter()
    all_logits, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]", leave=False) if cfg.verbose else loader

    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        # Para segmentación, usar la máscara completa como target
        if cfg.model_arch == "unet_resnet34" and "mask" in batch:
            targets = batch["mask"].to(device, non_blocking=True)
        else:
            targets = labels.unsqueeze(1).float()

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, targets)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_meter.update(loss.item(), n=images.size(0))

        # Guardar logits y etiquetas para métricas al final de la época
        if cfg.model_arch == "unet_resnet34":
            # Para segmentación: máximo del mapa de probabilidades → clasificación de parche
            patch_logits = logits.sigmoid().amax(dim=(1, 2, 3))
            all_logits.append(patch_logits.detach().cpu())
        else:
            all_logits.append(logits.squeeze(1).detach().cpu())
        all_labels.append(labels.detach().cpu())

        if cfg.verbose and step % cfg.log_interval == 0:
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    # Métricas de la época
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels, threshold=cfg.threshold)
    metrics["loss"] = loss_meter.avg

    if writer:
        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)

    return metrics


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: TrainingConfig,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    split: str = "val",
) -> Dict[str, float]:
    """Ejecuta una época de evaluación y retorna métricas."""
    model.eval()
    loss_meter = AverageMeter()
    all_logits, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [val]  ", leave=False) if cfg.verbose else loader

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if cfg.model_arch == "unet_resnet34" and "mask" in batch:
            targets = batch["mask"].to(device, non_blocking=True)
        else:
            targets = labels.unsqueeze(1).float()

        logits = model(images)
        loss   = criterion(logits, targets)
        loss_meter.update(loss.item(), n=images.size(0))

        if cfg.model_arch == "unet_resnet34":
            patch_logits = logits.sigmoid().amax(dim=(1, 2, 3))
            all_logits.append(patch_logits.cpu())
        else:
            all_logits.append(logits.squeeze(1).cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels, threshold=cfg.threshold)
    metrics["loss"] = loss_meter.avg

    if writer:
        for k, v in metrics.items():
            writer.add_scalar(f"{split}/{k}", v, epoch)

    return metrics


# ────────────────────────────────────────────────────────────
# Entrenamiento de un fold
# ────────────────────────────────────────────────────────────

def run_fold(
    cfg: TrainingConfig,
    fold_idx: int,
    train_indices: List[int],
    val_indices:   List[int],
) -> Dict:
    """
    Entrena y evalúa un fold completo.

    Returns:
        Diccionario con el historial de métricas y las métricas del mejor epoch.
    """
    set_seed(cfg.seed + fold_idx)
    device = torch.device(cfg.resolve_device())
    fold_output_dir = Path(cfg.output_dir) / f"fold_{fold_idx}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset y DataLoaders ─────────────────────────────
    loaders = get_dataloaders(cfg, train_indices, val_indices)

    # ── Modelo ────────────────────────────────────────────
    model = build_model(cfg.model_arch, n_channels=cfg.n_channels, pretrained=cfg.pretrained)
    model = model.to(device)

    if cfg.verbose:
        print(model_summary(model, n_channels=cfg.n_channels))

    # ── Congelar backbone para las primeras épocas ────────
    if cfg.freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
        if cfg.verbose:
            print(f"[Fold {fold_idx}] Backbone congelado por {cfg.freeze_epochs} épocas.")

    # ── Criterio, optimizador, scheduler ─────────────────
    criterion = build_criterion(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    es        = EarlyStopping(patience=cfg.patience, mode=cfg.mode)

    # ── TensorBoard ───────────────────────────────────────
    writer = None
    if cfg.tensorboard:
        tb_dir = Path(cfg.output_dir) / "tensorboard" / f"fold_{fold_idx}"
        writer = SummaryWriter(str(tb_dir))

    # ── Loop de entrenamiento ─────────────────────────────
    history = {"train": [], "val": []}
    best_metrics = None
    best_ckpt_path = fold_output_dir / "best_model.pth"

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # Descongelar backbone tras las primeras épocas
        if epoch == cfg.freeze_epochs + 1 and hasattr(model, "unfreeze_backbone"):
            model.unfreeze_backbone()
            optimizer = build_optimizer(model, cfg)   # Reiniciar con LR diferencial completo
            scheduler = build_scheduler(optimizer, cfg)
            if cfg.verbose:
                print(f"[Fold {fold_idx}] Backbone descongelado en época {epoch}.")

        train_m = train_epoch(model, loaders["train"], optimizer, criterion, device, cfg, epoch, writer)
        val_m   = eval_epoch(model,  loaders["val"],   criterion,             device, cfg, epoch, writer)

        history["train"].append(train_m)
        history["val"].append(val_m)

        # Actualizar scheduler
        monitor_val = val_m.get(cfg.monitor.replace("val_", ""), val_m.get("f1", 0.0))
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_val)
            else:
                scheduler.step()

        # Early stopping y guardado de checkpoint
        improved = es(monitor_val, epoch)
        if improved and cfg.save_best_only:
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_m,
                "cfg":         cfg.__dict__,
            }, best_ckpt_path)
            best_metrics = val_m.copy()

        elapsed = format_time(time.time() - t0)
        if cfg.verbose:
            print(
                f"[Fold {fold_idx}] Epoch {epoch:3d}/{cfg.epochs} | "
                f"loss={train_m['loss']:.4f} | "
                f"val_loss={val_m['loss']:.4f} | "
                f"val_f1={val_m.get('f1', 0):.4f} | "
                f"val_auc={val_m.get('auc_roc', 0):.4f} | "
                f"{'⭐ best' if improved else '      '} | "
                f"{elapsed}"
            )

        if es.stop:
            if cfg.verbose:
                print(f"[Fold {fold_idx}] Early stopping en época {epoch} (mejor: {es.best_epoch})")
            break

    if writer:
        writer.close()

    return {
        "fold":         fold_idx,
        "history":      history,
        "best_metrics": best_metrics or val_m,
        "best_epoch":   es.best_epoch,
        "ckpt_path":    str(best_ckpt_path),
    }


# ────────────────────────────────────────────────────────────
# Experimento K-Fold completo
# ────────────────────────────────────────────────────────────

def run_kfold(cfg: TrainingConfig) -> Dict:
    """
    Orquesta el experimento completo de K-Fold Cross-Validation.

    Returns:
        Diccionario con resultados por fold y estadísticas agregadas.
    """
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.verbose:
        print(f"\n{'='*60}")
        print(f"Experimento: {cfg.experiment_name}")
        print(f"Arquitectura: {cfg.model_arch} | Folds: {cfg.n_folds} | Épocas: {cfg.epochs}")
        print(f"Device: {cfg.resolve_device()}")
        print(f"{'='*60}\n")

    # ── Cargar dataset completo para obtener índices de folds ──
    from pathlib import Path as _Path
    data_root   = _Path(cfg.data_root)
    full_ds = Landslide4SenseDataset(
        img_dir=str(data_root / "TrainData" / "img"),
        mask_dir=str(data_root / "TrainData" / "mask"),
        normalize=False,             # Solo necesitamos las etiquetas
    )
    folds = get_fold_indices(full_ds, n_folds=cfg.n_folds, seed=cfg.seed)

    # ── Ejecutar cada fold ─────────────────────────────────────
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if cfg.verbose:
            print(f"\n─── Fold {fold_idx + 1}/{cfg.n_folds} "
                  f"(train={len(train_idx)}, val={len(val_idx)}) ───")
        result = run_fold(cfg, fold_idx, train_idx, val_idx)
        fold_results.append(result)

    # ── Estadísticas agregadas ────────────────────────────────
    metrics_keys = list(fold_results[0]["best_metrics"].keys())
    agg = {}
    for k in metrics_keys:
        vals = [r["best_metrics"].get(k, 0) for r in fold_results]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    if cfg.verbose:
        print(f"\n{'='*60}")
        print(f"Resultados {cfg.n_folds}-Fold CV — {cfg.experiment_name}")
        print(f"{'='*60}")
        for k, v in agg.items():
            print(f"  {k:20s}: {v['mean']:.4f} ± {v['std']:.4f}")
        print(f"{'='*60}\n")

    # ── Guardar resumen ────────────────────────────────────────
    summary = {
        "experiment": cfg.experiment_name,
        "model_arch": cfg.model_arch,
        "n_folds":    cfg.n_folds,
        "fold_results": [
            {
                "fold": r["fold"],
                "best_epoch": r["best_epoch"],
                "metrics": r["best_metrics"],
            }
            for r in fold_results
        ],
        "aggregated": agg,
    }
    summary_path = output_dir / "kfold_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if cfg.verbose:
        print(f"Resumen guardado en: {summary_path}")

    return summary
