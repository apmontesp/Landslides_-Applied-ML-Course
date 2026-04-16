"""
train.py — Loop de entrenamiento optimizado para Landslide4Sense (EAFIT Research)
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

# Importaciones del proyecto
from .config import TrainingConfig
from .dataset import get_dataloaders, get_fold_indices, Landslide4SenseDataset
from .evaluate import compute_metrics
from .models import build_model  # <-- Se eliminó model_summary de aquí para evitar el error
from .utils import set_seed, AverageMeter, format_time

# ────────────────────────────────────────────────────────────
# Funciones de pérdida (Dice + BCE)
# ────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        flat_p = probs.view(-1)
        flat_t = targets.view(-1)
        inter = (flat_p * flat_t).sum()
        dice = (2.0 * inter + self.smooth) / (flat_p.sum() + flat_t.sum() + self.smooth)
        return 1.0 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, pos_weight: Optional[float] = None):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = 1.0 - dice_weight
        self.dice = DiceLoss()
        pw = torch.tensor([pos_weight]) if pos_weight else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.bce_weight * self.bce(logits, targets) + 
                self.dice_weight * self.dice(logits, targets))

def build_criterion(cfg: TrainingConfig) -> nn.Module:
    if cfg.loss == "dice_bce":
        return DiceBCELoss(dice_weight=cfg.dice_weight, pos_weight=cfg.pos_weight)
    return nn.BCEWithLogitsLoss()

# ────────────────────────────────────────────────────────────
# Optimizador y Early Stopping
# ────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> optim.Optimizer:
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if any(k in name for k in ["classifier", "head", "fc", "decoder"]):
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    param_groups = [
        {"params": backbone_params, "lr": cfg.lr_backbone},
        {"params": head_params, "lr": cfg.lr_head},
    ]
    return optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

class EarlyStopping:
    def __init__(self, patience: int = 15, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.best_epoch = 0
        self.stop = False

    def __call__(self, metric: float, epoch: int) -> bool:
        improved = metric > self.best if self.mode == "max" else metric < self.best
        if improved:
            self.best = metric
            self.counter = 0
            self.best_epoch = epoch
            return True
        self.counter += 1
        if self.counter >= self.patience: self.stop = True
        return False

# ────────────────────────────────────────────────────────────
# Loops Principales
# ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, cfg, epoch, writer=None):
    model.train()
    loss_meter = AverageMeter()
    all_logits, all_labels = [], []
    
    for batch in tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        targets = labels.unsqueeze(1).float()

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=images.size(0))
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels), threshold=cfg.threshold)
    metrics["loss"] = loss_meter.avg
    return metrics

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, cfg, epoch, writer=None):
    model.eval()
    loss_meter = AverageMeter()
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        loss = criterion(logits, labels.unsqueeze(1).float())
        
        loss_meter.update(loss.item(), n=images.size(0))
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels), threshold=cfg.threshold)
    metrics["loss"] = loss_meter.avg
    return metrics

def run_fold(cfg, fold_idx, train_indices, val_indices):
    set_seed(cfg.seed + fold_idx)
    device = torch.device(cfg.resolve_device())
    fold_dir = Path(cfg.output_dir) / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    loaders = get_dataloaders(cfg, train_indices, val_indices)
    model = build_model(cfg.model_arch, n_channels=cfg.n_channels, pretrained=cfg.pretrained).to(device)

    # --- MANEJO SEGURO DEL RESUMEN ---
    if cfg.verbose:
        try:
            from .models import model_summary
            print(model_summary(model))
        except (ImportError, NameError):
            print(f"ℹ️ Resumen de modelo no disponible. Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")

    criterion = build_criterion(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    es = EarlyStopping(patience=cfg.patience)

    best_metrics = None
    for epoch in range(1, cfg.epochs + 1):
        train_m = train_epoch(model, loaders["train"], optimizer, criterion, device, cfg, epoch)
        val_m = eval_epoch(model, loaders["val"], criterion, device, cfg, epoch)

        improved = es(val_m.get("f1", 0), epoch)
        if improved:
            torch.save(model.state_dict(), fold_dir / "best_model.pth")
            best_metrics = val_m.copy()

        print(f"[Fold {fold_idx}] Ep {epoch}: Loss={train_m['loss']:.4f} | Val F1={val_m['f1']:.4f} {'⭐' if improved else ''}")
        if es.stop: break

    return {"best_metrics": best_metrics or val_m, "best_epoch": es.best_epoch}

def run_kfold(cfg: TrainingConfig):
    set_seed(cfg.seed)
    data_root = Path(cfg.data_root)
    full_ds = Landslide4SenseDataset(str(data_root/"TrainData"/"img"), str(data_root/"TrainData"/"mask"))
    folds = get_fold_indices(full_ds, n_folds=cfg.n_folds)

    results = []
    for i, (t_idx, v_idx) in enumerate(folds):
        print(f"\n--- Iniciando Fold {i} ---")
        results.append(run_fold(cfg, i, t_idx, v_idx))
    
    return results
