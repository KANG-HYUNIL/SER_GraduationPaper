from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_class_counts(labels: list[int] | torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long), minlength=num_classes).float()
    return counts.clamp_min(1.0)


def build_class_weights(labels: list[int] | torch.Tensor, num_classes: int, mode: str, beta: float = 0.999) -> torch.Tensor | None:
    mode = str(mode)
    if mode == "none":
        return None

    counts = _safe_class_counts(labels, num_classes)
    if mode == "inverse_freq":
        weights = 1.0 / counts
    elif mode == "effective_num":
        beta = min(max(float(beta), 0.0), 0.9999)
        effective_num = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
        weights = (1.0 - beta) / effective_num.clamp_min(1e-8)
    else:
        raise ValueError(f"Unsupported class weight mode: {mode}")

    weights = weights / weights.sum() * num_classes
    return weights


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        if weight is None:
            self.register_buffer("weight", None)
        else:
            self.register_buffer("weight", weight.float())
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = torch.pow((1.0 - pt).clamp_min(0.0), self.gamma) * ce
        return focal.mean()


def build_criterion(cfg, train_labels: list[int], num_classes: int) -> nn.Module:
    train_cfg = cfg.train
    loss_cfg = train_cfg.get("loss", {})
    loss_name = str(loss_cfg.get("name", "cross_entropy"))
    label_smoothing = float(loss_cfg.get("label_smoothing", train_cfg.get("label_smoothing", 0.0)))
    class_weight_mode = str(loss_cfg.get("class_weight_mode", "none"))
    class_weights = build_class_weights(train_labels, num_classes, class_weight_mode)

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if loss_name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if loss_name == "focal_loss":
        gamma = float(loss_cfg.get("focal_gamma", 2.0))
        return FocalLoss(gamma=gamma, weight=class_weights, label_smoothing=label_smoothing)
    raise ValueError(f"Unsupported loss name: {loss_name}")
