from __future__ import annotations

import torch
import torch.nn as nn
import timm

from skin_disease_ai.models.convexnet import ConvexNet


MODEL_ALIASES = {
    "mobilenetv2": "mobilenetv2_100",
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnetb0": "efficientnet_b0",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnetb3": "efficientnet_b3",
    "efficientnet_b3": "efficientnet_b3",
    "convnext_tiny": "convnext_tiny",
    "convnexttiny": "convnext_tiny",
    "vit": "vit_base_patch16_224",
    "vit_base": "vit_base_patch16_224",
    "swin": "swin_tiny_patch4_window7_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}

BENCHMARK_MODELS = [
    "mobilenetv2",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "efficientnet_b3",
    "convnext_tiny",
    "vit_base",
    "swin_tiny",
]


def create_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    backbone: str | None = None,
    drop_rate: float = 0.0,
    fusion_dim: int = 512,
) -> nn.Module:
    lowered = name.lower()
    if lowered == "convexnet":
        return ConvexNet(
            backbone=backbone or "convnext_tiny",
            num_classes=num_classes,
            pretrained=pretrained,
            drop_rate=drop_rate,
            fusion_dim=fusion_dim,
        )

    timm_name = MODEL_ALIASES.get(lowered, name)
    return timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if any(token in name for token in ("classifier", "head", "fc", "fusion", "pool")):
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def differential_lr_groups(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> list[dict]:
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name for token in ("classifier", "head", "fc", "fusion", "pool")):
            head_params.append(param)
        else:
            backbone_params.append(param)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    return groups


def load_model_from_checkpoint(checkpoint: dict, device: torch.device | str = "cpu") -> tuple[nn.Module, dict[str, int]]:
    model_cfg = checkpoint.get("model_config", {})
    class_to_idx = checkpoint.get("class_to_idx", {})
    num_classes = len(class_to_idx) or model_cfg.get("num_classes", 31)
    model = create_model(
        model_cfg.get("name", "convexnet"),
        num_classes=num_classes,
        pretrained=False,
        backbone=model_cfg.get("backbone", "convnext_tiny"),
        drop_rate=model_cfg.get("drop_rate", 0.0),
        fusion_dim=model_cfg.get("fusion_dim", 512),
    )
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=True)
    return model.to(device), class_to_idx
