"""
AyuLeafNet Model Architecture
==============================
Hybrid CNN combining:
  • Pre-trained backbone (MobileNetV2 / EfficientNet-B0 / ResNet-50)
  • Custom local-feature CNN branch
  • Channel Attention (SE-style) mechanism
  • Fusion + Classification head with medicinal confidence scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import BACKBONE, DROPOUT_RATE, NUM_CLASSES, PRETRAINED


# ─────────────────────────────────────────────────────────────────────────────
# CHANNEL ATTENTION MODULE  (Squeeze-and-Excitation style)
# ─────────────────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """
    Recalibrates channel-wise feature responses by modelling inter-channel
    dependencies. Reduces channels by `reduction` then restores.
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        scale   = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL ATTENTION MODULE
# ─────────────────────────────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """Focuses on WHERE meaningful features are in the spatial dimensions."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv  = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(combined))
        return x * scale


# ─────────────────────────────────────────────────────────────────────────────
# CBAM BLOCK  (Combined Channel + Spatial Attention)
# ─────────────────────────────────────────────────────────────────────────────
class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL FEATURE CNN BRANCH
# ─────────────────────────────────────────────────────────────────────────────
class LocalCNNBranch(nn.Module):
    """
    Lightweight CNN branch that captures fine-grained leaf texture / edge
    details that the heavy backbone might miss.
    """
    def __init__(self, out_features: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 112 x 112

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 56 x 56

            # Block 3  — depthwise separable for efficiency
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 28 x 28

            # Block 4
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),     # 1 x 1
        )
        self.cbam = CBAM(256)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features[:-1](x)           # all but final pool
        x = self.cbam(x)
        x = self.features[-1](x)            # global avg pool
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
def _build_backbone(name: str, pretrained: bool):
    """Return (backbone_module, backbone_out_features)."""
    weights_flag = "IMAGENET1K_V1" if pretrained else None

    if name == "mobilenet_v2":
        base   = models.mobilenet_v2(weights=weights_flag)
        feat   = base.features
        out_ch = 1280
        return feat, out_ch

    elif name == "efficientnet_b0":
        base   = models.efficientnet_b0(weights=weights_flag)
        feat   = base.features
        out_ch = 1280
        return feat, out_ch

    elif name == "resnet50":
        base   = models.resnet50(weights=weights_flag)
        # Remove the final FC and avg-pool
        feat   = nn.Sequential(*list(base.children())[:-2])
        out_ch = 2048
        return feat, out_ch

    else:
        raise ValueError(f"Unknown backbone: {name}. Choose mobilenet_v2 | efficientnet_b0 | resnet50")


# ─────────────────────────────────────────────────────────────────────────────
# AYULEAFNET — MAIN HYBRID MODEL
# ─────────────────────────────────────────────────────────────────────────────
class AyuLeafNet(nn.Module):
    """
    Hybrid CNN Architecture
    -----------------------
    Dual-branch design:
      [Global Branch]  Pre-trained backbone  →  CBAM  →  GAP  →  FC(512)
      [Local Branch]   Custom LeafCNN        →  CBAM  →  GAP  →  FC(256)
    Both branches fused via concatenation then passed to the
    classification head that outputs:
      • class logits          (NUM_CLASSES)
      • confidence score       (1 — sigmoid)
    """

    def __init__(
        self,
        backbone:     str  = BACKBONE,
        num_classes:  int  = NUM_CLASSES,
        dropout_rate: float = DROPOUT_RATE,
        pretrained:   bool = PRETRAINED,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── Global (backbone) branch ──────────────────────────────────────
        self.backbone, backbone_ch = _build_backbone(backbone, pretrained)
        self.backbone_cbam = CBAM(backbone_ch)
        self.backbone_pool = nn.AdaptiveAvgPool2d(1)
        self.backbone_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_ch, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        # ── Local (custom CNN) branch ─────────────────────────────────────
        self.local_branch = LocalCNNBranch(out_features=256)

        # ── Fusion + Classification head ──────────────────────────────────
        fused_dim = 512 + 256     # 768
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
        )
        self.classifier = nn.Linear(256, num_classes)
        self.confidence  = nn.Linear(256, 1)       # medicinal confidence score

        self._init_weights()

    def _init_weights(self):
        """Xavier / He initialisation for non-pretrained layers."""
        for m in [self.backbone_proj, self.fusion,
                  self.classifier, self.confidence]:
            for layer in (m.modules() if hasattr(m, 'modules') else [m]):
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor):
        # Global branch
        g = self.backbone(x)           # (B, C, H, W)
        g = self.backbone_cbam(g)
        g = self.backbone_pool(g)
        g = self.backbone_proj(g)      # (B, 512)

        # Local branch
        l = self.local_branch(x)       # (B, 256)

        # Fusion
        fused  = torch.cat([g, l], dim=1)   # (B, 768)
        hidden = self.fusion(fused)          # (B, 256)

        # Outputs
        logits     = self.classifier(hidden)                # (B, num_classes)
        confidence = torch.sigmoid(self.confidence(hidden)) # (B, 1)

        return logits, confidence

    def predict(self, x: torch.Tensor):
        """Convenience method — returns (class_idx, probabilities, confidence)."""
        self.eval()
        with torch.no_grad():
            logits, conf = self(x)
            probs       = F.softmax(logits, dim=1)
            pred_idx    = probs.argmax(dim=1)
        return pred_idx, probs, conf


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = AyuLeafNet(pretrained=False).to(device)

    dummy  = torch.randn(4, 3, 224, 224).to(device)
    logits, conf = model(dummy)

    print("✅ AyuLeafNet architecture OK")
    print(f"   Input  : {dummy.shape}")
    print(f"   Logits : {logits.shape}")
    print(f"   Conf   : {conf.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"   Params : {total:,}")
