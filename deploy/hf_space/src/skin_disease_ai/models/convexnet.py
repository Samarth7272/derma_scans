from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_values, _ = torch.max(x, dim=1, keepdim=True)
        weights = torch.sigmoid(self.conv(torch.cat([avg, max_values], dim=1)))
        return x * weights


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return x.flatten(1)


class MultiScaleFusion(nn.Module):
    def __init__(self, channels: list[int], fusion_dim: int):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(ch, fusion_dim, 1, bias=False) for ch in channels])
        self.refine = nn.Sequential(
            ConvBNAct(fusion_dim, fusion_dim, 3),
            SqueezeExcitation(fusion_dim),
            SpatialAttention(),
            ConvBNAct(fusion_dim, fusion_dim, 3),
        )
        self.level_weights = nn.Parameter(torch.ones(len(channels)))

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        target_size = features[0].shape[-2:]
        weights = torch.softmax(self.level_weights, dim=0)
        projected = []
        fused = 0
        for feature, lateral, weight in zip(features, self.laterals, weights):
            x = lateral(feature)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            projected.append(x)
            fused = fused + weight * x
        return self.refine(fused), projected


class ConvexNet(nn.Module):
    """ConvNeXt/EfficientNet feature backbone with multi-scale fusion and attention."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        num_classes: int = 31,
        pretrained: bool = True,
        drop_rate: float = 0.35,
        fusion_dim: int = 512,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
        )
        channels = self.backbone.feature_info.channels()
        if len(channels) > 4:
            channels = channels[-4:]
        self.num_feature_levels = len(channels)
        self.fusion = MultiScaleFusion(channels, fusion_dim)
        self.pool = GeM()
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Dropout(drop_rate),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(drop_rate * 0.7),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if len(features) > self.num_feature_levels:
            features = features[-self.num_feature_levels :]
        fused, projected = self.fusion(features)
        fused_pool = self.pool(fused)
        level_pool = torch.stack([F.adaptive_avg_pool2d(p, 1).flatten(1) for p in projected], dim=1)
        level_pool = level_pool.mean(dim=1)
        return torch.cat([fused_pool, level_pool], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
