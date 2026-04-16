import torch
import torch.nn as nn
from typing import Optional

def adapt_first_conv(model: nn.Module, n_channels: int = 14, layer_name: str = "conv1") -> nn.Module:
    """Adapta la primera capa conv a N canales usando mean-initialization."""
    old_conv = getattr(model, layer_name)
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

# --- EFFICIENTNET-B4 ---
class EfficientNetB4Classifier(nn.Module):
    def __init__(self, n_channels: int = 14, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "efficientnet_b4", 
            pretrained=pretrained, 
            in_chans=n_channels, 
            num_classes=0, 
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1),
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False
        self.backbone.eval() # Estabilidad para Batch Normalization

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True
        self.backbone.train()

    def forward(self, x):
        return self.classifier(self.backbone(x))

# --- RESNET-50 ---
class ResNet50Classifier(nn.Module):
    def __init__(self, n_channels: int = 14, pretrained: bool = True, dropout: float = 0.5):
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
            nn.Dropout(p=dropout), nn.Linear(2048, 256), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout/2), nn.Linear(256, 1)
        )

    def freeze_backbone(self):
        for p in self.features.parameters(): p.requires_grad = False
        self.features.eval()

    def unfreeze_backbone(self):
        for p in self.features.parameters(): p.requires_grad = True
        self.features.train()

    def forward(self, x):
        return self.classifier(self.flatten(self.features(x)))

# --- FACTORY ---
def build_model(arch: str, n_channels: int = 14, pretrained: bool = True, **kwargs):
    arch = arch.lower().strip()
    if arch in ("efficientnet_b4", "efficientnet"):
        return EfficientNetB4Classifier(n_channels=n_channels, pretrained=pretrained, **kwargs)
    if arch == "resnet50":
        return ResNet50Classifier(n_channels=n_channels, pretrained=pretrained, **kwargs)
    raise ValueError(f"Arquitectura {arch} no reconocida.")
