"""
evaluate.py — Evaluación y métricas para Landslide4Sense.

Implementa:
  - compute_metrics()         : F1, AUC-ROC, Precisión, Recall, IoU, Accuracy
  - find_optimal_threshold()  : Búsqueda del umbral óptimo en curva PR
  - compute_iou()             : Intersection over Union para segmentación
  - plot_roc_curve()          : Curva ROC con AUC
  - plot_confusion_matrix()   : Matriz de confusión normalizada
  - plot_pr_curve()           : Curva Precisión-Recall
  - evaluate_model()          : Evaluación completa de un modelo sobre un DataLoader
  - compare_models()          : Tabla comparativa de varios experimentos
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)


# ────────────────────────────────────────────────────────────
# Métricas de clasificación de parche
# ────────────────────────────────────────────────────────────

def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calcula métricas de clasificación binaria a partir de logits y etiquetas.

    Args:
        logits:    Tensor de logits (sin sigmoid) shape (N,) o (N,1)
        labels:    Tensor de etiquetas binarias (0/1)  shape (N,)
        threshold: Umbral de decisión

    Returns:
        Diccionario con f1, auc_roc, precision, recall, accuracy, iou_binary
    """
    logits = logits.view(-1).float()
    labels = labels.view(-1).float()

    probs = torch.sigmoid(logits).numpy()
    labs  = labels.numpy().astype(int)
    preds = (probs >= threshold).astype(int)

    metrics: Dict[str, float] = {}

    # F1-score
    metrics["f1"] = float(f1_score(labs, preds, zero_division=0))

    # AUC-ROC (requiere al menos 2 clases en labels)
    try:
        metrics["auc_roc"] = float(roc_auc_score(labs, probs))
    except ValueError:
        metrics["auc_roc"] = 0.0

    # Precisión y Recall
    metrics["precision"] = float(precision_score(labs, preds, zero_division=0))
    metrics["recall"]    = float(recall_score(labs, preds, zero_division=0))

    # Accuracy
    metrics["accuracy"] = float(accuracy_score(labs, preds))

    # IoU binario (Jaccard) a nivel de parche
    tp = int(((preds == 1) & (labs == 1)).sum())
    fp = int(((preds == 1) & (labs == 0)).sum())
    fn = int(((preds == 0) & (labs == 1)).sum())
    metrics["iou"] = tp / (tp + fp + fn + 1e-8)

    return metrics


def find_optimal_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: str = "f1",
) -> Tuple[float, float]:
    """
    Encuentra el umbral óptimo maximizando F1 o J-stat en la curva PR/ROC.

    Returns:
        (threshold_optimo, metrica_en_threshold_optimo)
    """
    probs = torch.sigmoid(logits.view(-1)).numpy()
    labs  = labels.view(-1).numpy().astype(int)

    if criterion == "f1":
        precision, recall, thresholds = precision_recall_curve(labs, probs)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx  = np.argmax(f1_scores[:-1])
        return float(thresholds[best_idx]), float(f1_scores[best_idx])
    elif criterion == "youden":
        fpr, tpr, thresholds = roc_curve(labs, probs)
        j_stat = tpr - fpr
        best_idx = np.argmax(j_stat)
        return float(thresholds[best_idx]), float(j_stat[best_idx])
    else:
        raise ValueError(f"Criterio desconocido: {criterion}. Use 'f1' o 'youden'.")


# ────────────────────────────────────────────────────────────
# IoU para segmentación pixel-level
# ────────────────────────────────────────────────────────────

def compute_iou(
    pred_masks: torch.Tensor,
    true_masks: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Calcula el IoU promedio a nivel de pixel para segmentación binaria.

    Args:
        pred_masks: Logits o probabilidades (B, 1, H, W) o (B, H, W)
        true_masks: Máscaras binarias          (B, 1, H, W) o (B, H, W)
    """
    pred_bin = (torch.sigmoid(pred_masks) >= threshold).float().view(-1)
    true_bin = true_masks.float().view(-1)

    tp = (pred_bin * true_bin).sum().item()
    fp = (pred_bin * (1 - true_bin)).sum().item()
    fn = ((1 - pred_bin) * true_bin).sum().item()

    return tp / (tp + fp + fn + 1e-8)


# ────────────────────────────────────────────────────────────
# Visualizaciones
# ────────────────────────────────────────────────────────────

def plot_roc_curve(
    logits: torch.Tensor,
    labels: torch.Tensor,
    model_name: str = "Modelo",
    output_path: Optional[str] = None,
) -> None:
    """Dibuja la curva ROC con AUC."""
    if not HAS_MATPLOTLIB:
        print("matplotlib no disponible.")
        return

    probs = torch.sigmoid(logits.view(-1)).numpy()
    labs  = labels.view(-1).numpy().astype(int)

    fpr, tpr, _ = roc_curve(labs, probs)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})", color="#e74c3c")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
    ax.set_title(f"Curva ROC — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Curva ROC guardada: {output_path}")
    plt.show()
    plt.close()


def plot_confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    model_name: str = "Modelo",
    output_path: Optional[str] = None,
) -> None:
    """Dibuja la matriz de confusión normalizada."""
    if not HAS_MATPLOTLIB:
        print("matplotlib no disponible.")
        return

    probs = torch.sigmoid(logits.view(-1)).numpy()
    labs  = labels.view(-1).numpy().astype(int)
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(labs, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title_sfx, fmt in zip(
        axes,
        [cm, cm_norm],
        ["(Valores absolutos)", "(Normalizada por fila)"],
        ["d", ".2%"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=["No-deslizamiento", "Deslizamiento"],
            yticklabels=["No-deslizamiento", "Deslizamiento"],
            ax=ax, linewidths=0.5,
        )
        ax.set_xlabel("Predicción", fontsize=11)
        ax.set_ylabel("Verdadero", fontsize=11)
        ax.set_title(f"Matriz de Confusión {title_sfx}\n{model_name}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Matriz de confusión guardada: {output_path}")
    plt.show()
    plt.close()


def plot_pr_curve(
    logits: torch.Tensor,
    labels: torch.Tensor,
    model_name: str = "Modelo",
    output_path: Optional[str] = None,
) -> None:
    """Dibuja la curva Precisión-Recall con AUC-PR."""
    if not HAS_MATPLOTLIB:
        print("matplotlib no disponible.")
        return

    probs = torch.sigmoid(logits.view(-1)).numpy()
    labs  = labels.view(-1).numpy().astype(int)

    prec, rec, thresholds = precision_recall_curve(labs, probs)
    ap = auc(rec, prec)

    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx  = np.argmax(f1_scores[:-1])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, lw=2, label=f"{model_name} (AP = {ap:.4f})", color="#3498db")
    ax.fill_between(rec, prec, alpha=0.1, color="#3498db")
    ax.scatter(rec[best_idx], prec[best_idx], s=120, color="red", zorder=5,
               label=f"Umbral óptimo F1={f1_scores[best_idx]:.3f} @ {thresholds[best_idx]:.2f}")
    ax.axhline(labs.mean(), linestyle="--", color="gray", lw=1,
               label=f"Baseline (prevalencia = {labs.mean():.2f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precisión", fontsize=12)
    ax.set_title(f"Curva Precisión-Recall — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_training_history(
    history: Dict,
    fold_idx: int = 0,
    output_path: Optional[str] = None,
) -> None:
    """Dibuja la historia de entrenamiento (loss y métricas por época)."""
    if not HAS_MATPLOTLIB:
        return

    train_h = history.get("train", [])
    val_h   = history.get("val", [])

    if not train_h:
        return

    epochs  = list(range(1, len(train_h) + 1))
    metrics = [k for k in train_h[0].keys() if k != "loss"]

    n_plots = 1 + len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Loss
    axes[0].plot(epochs, [m["loss"] for m in train_h], label="Train", color="#e74c3c")
    axes[0].plot(epochs, [m["loss"] for m in val_h],   label="Val",   color="#3498db")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Otras métricas
    colors = ["#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
    for ax, metric, color in zip(axes[1:], metrics, colors):
        ax.plot(epochs, [m.get(metric, 0) for m in train_h], label="Train", color=color, alpha=0.7)
        ax.plot(epochs, [m.get(metric, 0) for m in val_h],   label="Val",   color=color)
        ax.set_title(metric.upper())
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.suptitle(f"Historia de Entrenamiento — Fold {fold_idx}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ────────────────────────────────────────────────────────────
# Evaluación completa de un modelo
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    optimize_threshold: bool = True,
    model_name: str = "Modelo",
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Evaluación completa: métricas, umbral óptimo, y figuras opcionales.

    Returns:
        Diccionario con métricas finales y umbral usado.
    """
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]

        logits = model(images)

        if logits.dim() == 4:
            # Segmentación: tomar el máximo del mapa → clasificación de parche
            logits = logits.sigmoid().amax(dim=(1, 2, 3))
            logits = torch.logit(logits.clamp(1e-6, 1 - 1e-6))
        else:
            logits = logits.squeeze(1)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Umbral óptimo
    if optimize_threshold:
        threshold, best_f1 = find_optimal_threshold(all_logits, all_labels, criterion="f1")
        print(f"Umbral óptimo encontrado: {threshold:.3f} → F1 = {best_f1:.4f}")

    metrics = compute_metrics(all_logits, all_labels, threshold=threshold)
    metrics["threshold"] = threshold

    print(f"\n{'─'*40}")
    print(f"Resultados — {model_name}")
    print(f"{'─'*40}")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    if output_dir and HAS_MATPLOTLIB:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        plot_roc_curve(all_logits, all_labels, model_name=model_name,
                       output_path=str(out / "roc_curve.png"))
        plot_pr_curve(all_logits, all_labels, model_name=model_name,
                      output_path=str(out / "pr_curve.png"))
        plot_confusion_matrix(all_logits, all_labels, threshold=threshold, model_name=model_name,
                              output_path=str(out / "confusion_matrix.png"))

    return metrics


# ────────────────────────────────────────────────────────────
# Tabla comparativa de modelos
# ────────────────────────────────────────────────────────────

def compare_models(results_dirs: List[str]) -> None:
    """
    Lee los resúmenes kfold_summary.json de varios experimentos y
    muestra una tabla comparativa de métricas.
    """
    rows = []
    for d in results_dirs:
        summary_path = Path(d) / "kfold_summary.json"
        if not summary_path.exists():
            print(f"[WARN] No encontrado: {summary_path}")
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        row = {"Modelo": s["experiment"], "Folds": s["n_folds"]}
        for k, v in s.get("aggregated", {}).items():
            row[k] = f"{v['mean']:.4f} ± {v['std']:.4f}"
        rows.append(row)

    if not rows:
        print("No se encontraron resultados para comparar.")
        return

    # Imprimir tabla
    keys = list(rows[0].keys())
    widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    header = " | ".join(k.ljust(widths[k]) for k in keys)
    sep    = "-+-".join("-" * widths[k] for k in keys)
    print("\n" + header)
    print(sep)
    for row in rows:
        print(" | ".join(str(row.get(k, "—")).ljust(widths[k]) for k in keys))
