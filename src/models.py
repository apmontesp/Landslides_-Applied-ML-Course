"""
models.py — Definición de arquitecturas CNN para Landslide4Sense.
Actualizado: Soporte robusto para EfficientNet-B4 y estabilidad en Fine-tuning.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

# ────────────────────────────────────────────────────────────
# Adaptación de la primera capa convolucional (14 canales)
# ────────────────────────────────────────────────────────────

def adapt_first_conv(
    model: nn.Module,
    n_channels: int = 14,
    layer_name: str = "conv1",
) -> nn.Module:
    """
    Adapta la primera capa para aceptar n_channels usando mean-initialization.
    Escala los pesos por (3/n_channels) para preservar la norma de activación.
    """
    old_conv: nn.Conv2d = getattr(model, layer_name)

    new_conv = nn.Conv2d(
        in_channels=n_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        old_w = old_conv.weight.data
        repeats = (n_channels // 3) + 1
        new_w = old_w.repeat(1, repeats, 1, 1)[:, :n_channels, :, :]
        new_w = new_w * (3.0 / n_channels)
        new_conv.weight.data = new_w

        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()

    setattr(model, layer_name, new_conv)
    return model


# ────────────────────────────────────────────────────────────
# EfficientNet-B4 (Optimizada para Colab)
# ────────────────────────────────────────────────────────────

class EfficientNetB4Classifier(nn.Module):
    def __init__(
        self,
        n_channels: int = 14,
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        import timm

        # timm maneja internamente la adaptación de in_chans
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            in_chans=n_channels,
            num_classes=0,
            global_pool="avg",
        )

        feat_dim = self.backbone.num_features # 1792 para B4

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 256),
            nn.SiLU(inplace=True), # Activación nativa de EfficientNet
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1),
        )

    def freeze_backbone(self) -> None:
        """Congela parámetros y pone BN en modo evaluación para estabilidad."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval() 

    def unfreeze_backbone(self) -> None:
        """Descongela para fine-tuning completo."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.classifier(feat)


# ────────────────────────────────────────────────────────────
# ResNet-50 (Tu implementación original)
# ────────────────────────────────────────────────────────────

class ResNet50Classifier(nn.Module):
    def __init__(
        self,
        n_channels: int = 14,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        import torchvision.models as tvm

        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)
        backbone = adapt_first_conv(backbone, n_channels=n_channels, layer_name="conv1")

        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool,
        )
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1),
        )

    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    def unfreeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = True
        self.features.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.flatten(self.features(x))
        return self.classifier(feat)


# ────────────────────────────────────────────────────────────
# Factory y Utilidades
# ────────────────────────────────────────────────────────────

def build_model(
    arch: str,
    n_channels: int = 14,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    arch = arch.lower().strip()

    if arch == "resnet50":
        return ResNet50Classifier(n_channels=n_channels, pretrained=pretrained, **kwargs)
    elif arch in ("efficientnet_b4", "efficientnet"):
        return EfficientNetB4Classifier(n_channels=n_channels, pretrained=pretrained, **kwargs)
    elif arch in ("unet_resnet34", "unet"):
        import segmentation_models_pytorch as smp
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=n_channels,
            classes=1,
            activation=None,
        )
    else:
        raise ValueError(f"Arquitectura '{arch}' no soportada.")

def model_summary(model: nn.Module, n_channels: int = 14) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (
        f"Modelo: {model.__class__.__name__}\n"
        f"Canales: {n_channels} | Total Params: {total:,} | Entrenables: {trainable:,}"
    )
