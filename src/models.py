"""
models.py — Definición de arquitecturas CNN para Landslide4Sense.

Implementa:
  - adapt_first_conv()     : Adapta la primera capa conv a N canales con mean-init
  - build_resnet50()       : ResNet-50 fine-tuned para clasificación binaria de parche
  - build_efficientnet_b4(): EfficientNet-B4 fine-tuned para clasificación binaria
  - build_unet_resnet34()  : U-Net + ResNet-34 encoder para segmentación pixel-level
  - build_model()          : Factory function que despacha por nombre de arquitectura
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
    Adapta la primera capa convolucional de un modelo preentrenado en ImageNet
    (que espera 3 canales) para aceptar n_channels canales de entrada.

    Estrategia de inicialización:
        Los pesos de los 3 canales originales se replican cíclicamente y se
        escalan por 3/n_channels para preservar la norma de activación.
        Esta estrategia (mean-initialization) está documentada como superior
        a inicialización aleatoria cuando los canales extra están correlacionados
        con los canales ópticos.

    Args:
        model:      Modelo PyTorch preentrenado (ResNet, EfficientNet, etc.)
        n_channels: Número de canales de entrada deseado (14 para Landslide4Sense)
        layer_name: Nombre del atributo de la primera conv en el modelo

    Returns:
        Modelo modificado in-place con nueva primera capa conv.
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
        old_w = old_conv.weight.data        # (out_c, 3, kH, kW)
        # Repetir cíclicamente los 3 canales hasta cubrir n_channels
        repeats = (n_channels // 3) + 1
        new_w   = old_w.repeat(1, repeats, 1, 1)[:, :n_channels, :, :]
        # Escalar para preservar la norma de activación esperada
        new_w   = new_w * (3.0 / n_channels)
        new_conv.weight.data = new_w

        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()

    setattr(model, layer_name, new_conv)
    return model


# ────────────────────────────────────────────────────────────
# ResNet-50
# ────────────────────────────────────────────────────────────

class ResNet50Classifier(nn.Module):
    """
    ResNet-50 adaptado para clasificación binaria de parches de 14 canales.

    La arquitectura agrega un clasificador con dropout sobre el vector de
    características 2048-dimensional del backbone.
    """

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

        # Extractor de características (sin la cabeza de clasificación original)
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool,
        )
        self.flatten = nn.Flatten()

        # Cabeza de clasificación binaria
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1),
        )

        # Guardar referencia al backbone para congelar/descongelar fácilmente
        self.backbone_layers = [
            self.features[0],   # conv1
            self.features[1],   # bn1
            self.features[4],   # layer1
            self.features[5],   # layer2
            self.features[6],   # layer3
            self.features[7],   # layer4
        ]

    def freeze_backbone(self) -> None:
        """Congela el backbone (fase inicial de fine-tuning)."""
        for layer in self.backbone_layers:
            for p in layer.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Descongela el backbone para fine-tuning completo."""
        for layer in self.backbone_layers:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → logits: (B, 1)"""
        feat = self.flatten(self.features(x))
        return self.classifier(feat)


def build_resnet50(
    n_channels: int = 14,
    pretrained: bool = True,
    dropout: float = 0.5,
) -> ResNet50Classifier:
    """Factory para ResNet-50 con 14 canales."""
    return ResNet50Classifier(n_channels=n_channels, pretrained=pretrained, dropout=dropout)


# ────────────────────────────────────────────────────────────
# EfficientNet-B4
# ────────────────────────────────────────────────────────────

class EfficientNetB4Classifier(nn.Module):
    """
    EfficientNet-B4 adaptado para clasificación binaria de 14 canales.

    Usa la librería `timm` para cargar el backbone, que permite adaptar
    in_chans directamente en la construcción del modelo.
    """

    def __init__(
        self,
        n_channels: int = 14,
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        import timm

        # timm permite especificar in_chans directamente
        # La inicialización de la capa de entrada se hace por mean-replication internamente
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            in_chans=n_channels,
            num_classes=0,           # Remover cabeza original
            global_pool="avg",
        )

        # Dimensión de features de EfficientNet-B4
        feat_dim = self.backbone.num_features  # 1792

        # Cabeza de clasificación
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1),
        )

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → logits: (B, 1)"""
        feat = self.backbone(x)
        return self.classifier(feat)


def build_efficientnet_b4(
    n_channels: int = 14,
    pretrained: bool = True,
    dropout: float = 0.4,
) -> EfficientNetB4Classifier:
    """Factory para EfficientNet-B4 con 14 canales."""
    return EfficientNetB4Classifier(n_channels=n_channels, pretrained=pretrained, dropout=dropout)


# ────────────────────────────────────────────────────────────
# U-Net + ResNet-34 (Segmentación)
# ────────────────────────────────────────────────────────────

def build_unet_resnet34(
    n_channels: int = 14,
    pretrained: bool = True,
) -> nn.Module:
    """
    U-Net con encoder ResNet-34 preentrenado en ImageNet.

    Usa segmentation-models-pytorch (smp) para la construcción.
    La adaptación de 14 canales se delega a adapt_first_conv() sobre
    el encoder ResNet-34.

    Returns:
        Modelo smp.Unet con encoder adaptado a 14 canales.
    """
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=n_channels,       # smp maneja la adaptación internamente
        classes=1,
        activation=None,              # Sin sigmoid (se aplica en la loss)
    )
    return model


# ────────────────────────────────────────────────────────────
# Factory principal
# ────────────────────────────────────────────────────────────

def build_model(
    arch: str,
    n_channels: int = 14,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Construye el modelo especificado por nombre de arquitectura.

    Args:
        arch:       Nombre de la arquitectura ('resnet50' | 'efficientnet_b4' | 'unet_resnet34')
        n_channels: Número de canales de entrada
        pretrained: Cargar pesos ImageNet
        **kwargs:   Argumentos adicionales por arquitectura

    Returns:
        Modelo PyTorch listo para entrenamiento.
    """
    arch = arch.lower().strip()

    if arch == "resnet50":
        return build_resnet50(n_channels=n_channels, pretrained=pretrained, **kwargs)
    elif arch in ("efficientnet_b4", "efficientnet"):
        return build_efficientnet_b4(n_channels=n_channels, pretrained=pretrained, **kwargs)
    elif arch in ("unet_resnet34", "unet"):
        return build_unet_resnet34(n_channels=n_channels, pretrained=pretrained)
    else:
        raise ValueError(
            f"Arquitectura desconocida: '{arch}'. "
            "Opciones: resnet50, efficientnet_b4, unet_resnet34"
        )


# ────────────────────────────────────────────────────────────
# Conteo de parámetros
# ────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    """Cuenta los parámetros del modelo."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_summary(model: nn.Module, n_channels: int = 14) -> str:
    """Resumen breve del modelo."""
    total  = count_parameters(model, only_trainable=False)
    train  = count_parameters(model, only_trainable=True)
    frozen = total - train
    return (
        f"Arquitectura: {model.__class__.__name__}\n"
        f"  Parámetros totales:     {total:,}\n"
        f"  Entrenables:            {train:,}\n"
        f"  Congelados:             {frozen:,}\n"
        f"  Canales de entrada:     {n_channels}\n"
    )
