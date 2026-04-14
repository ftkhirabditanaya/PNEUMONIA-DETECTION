"""
densenet_attention.py  –  DenseNet121 + CBAM Attention Module.

Architecture:
    DenseNet121 backbone (pretrained, ImageNet)
        → CBAM after denseblock4
        → Global Average Pooling
        → BatchNorm → Dropout → FC(256) → ReLU → FC(2)

CBAM = Convolutional Block Attention Module (Woo et al., ECCV 2018)
       Channel attention: "what" features matter
       Spatial attention: "where" in the image matters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ── Channel Attention Module ──────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.

    Compresses spatial dimensions via AvgPool AND MaxPool,
    passes both through a shared MLP, sums, then applies sigmoid.
    This teaches the model which feature channels (e.g. edge detectors,
    texture maps) are most relevant for pneumonia detection.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced = max(in_channels // reduction_ratio, 4)  # never go below 4

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP (implemented as 1x1 convolutions for efficiency)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = torch.sigmoid(avg_out + max_out)
        return x * scale


# ── Spatial Attention Module ──────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """
    Spatial attention: which spatial locations in the feature map matter.

    Applies AvgPool and MaxPool along channel dimension,
    concatenates, then convolves to produce a spatial mask.
    For X-rays, this forces the model to focus on lung regions.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)

    def forward(self, x):
        # Pool across channels to get spatial saliency signals
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        scale = torch.sigmoid(self.conv(spatial_in))
        return x * scale


# ── CBAM = Channel + Spatial in sequence ─────────────────────────────────────
class CBAM(nn.Module):
    """
    Full CBAM: first refine channel-wise, then refine spatially.
    Sequential application is key — spatial attention works on
    channel-refined features, not raw features.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ── Main Model ────────────────────────────────────────────────────────────────
class DenseNetAttention(nn.Module):
    """
    DenseNet121 + CBAM for binary chest X-ray classification.

    Two training phases:
        Phase 1 (freeze_backbone=True):
            Only classifier trains. Fast convergence.
            LR: 1e-3 for ~10 epochs.

        Phase 2 (freeze_backbone=False):
            Last dense block + classifier fine-tune.
            LR: 5e-5 (much lower to preserve pretrained features).
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.4,
        cbam_reduction: int = 16,
        cbam_kernel: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # Load pretrained DenseNet121
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        # Extract features (everything except the classifier head)
        self.features = backbone.features
        densenet_out_channels = 1024  # DenseNet121 final feature channels

        # CBAM attention — inserted after the entire feature extractor
        self.cbam = CBAM(
            in_channels=densenet_out_channels,
            reduction_ratio=cbam_reduction,
            kernel_size=cbam_kernel,
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(densenet_out_channels),
            nn.Dropout(p=dropout_rate),
            nn.Linear(densenet_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

        # Apply Phase 1 freezing if requested
        if freeze_backbone:
            self.freeze_backbone()

        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Kaiming initialisation for the custom classifier layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        """Freeze all DenseNet feature layers. Only CBAM + classifier will train."""
        for param in self.features.parameters():
            param.requires_grad = False
        print("[Model] Backbone FROZEN — only CBAM + classifier training.")

    def unfreeze_last_block(self):
        """
        Unfreeze denseblock4 + norm5 for Phase 2 fine-tuning.
        Keep earlier blocks frozen to preserve low-level features.
        """
        for param in self.features.parameters():
            param.requires_grad = False  # freeze all first

        # Then unfreeze last block and norm
        for param in self.features.denseblock4.parameters():
            param.requires_grad = True
        for param in self.features.norm5.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Phase 2: denseblock4 + norm5 + CBAM + classifier unfrozen.")
        print(f"[Model] Trainable params: {trainable:,}")

    def unfreeze_all(self):
        """Unfreeze everything for full fine-tuning (use very low LR)."""
        for param in self.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] All layers unfrozen. Trainable params: {total:,}")

    def forward(self, x):
        # DenseNet feature extraction
        features = self.features(x)
        features = F.relu(features, inplace=True)

        # CBAM attention refinement
        features = self.cbam(features)

        # Global average pooling → flatten
        pooled = self.gap(features)
        flat = torch.flatten(pooled, 1)

        # Classification
        out = self.classifier(flat)
        return out

    def get_attention_maps(self, x):
        """
        Returns feature maps before GAP for Grad-CAM.
        Used by the explainability module.
        """
        features = self.features(x)
        features = F.relu(features, inplace=True)
        features = self.cbam(features)
        return features

