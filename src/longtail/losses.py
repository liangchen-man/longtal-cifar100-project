from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_num_weights(cls_num_list: list[int], beta: float = 0.9999) -> torch.Tensor:
    counts = np.asarray(cls_num_list, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / weights.sum() * len(cls_num_list)
    return torch.tensor(weights, dtype=torch.float32)


def inverse_frequency_weights(cls_num_list: list[int]) -> torch.Tensor:
    counts = np.asarray(cls_num_list, dtype=np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.sum() * len(cls_num_list)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class LDAMLoss(nn.Module):
    def __init__(
        self,
        cls_num_list: list[int],
        max_m: float = 0.5,
        s: float = 30.0,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        margins = 1.0 / np.sqrt(np.sqrt(np.asarray(cls_num_list, dtype=np.float64)))
        margins = margins * (max_m / margins.max())
        self.register_buffer("margins", torch.tensor(margins, dtype=torch.float32))
        self.s = s
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1), True)
        batch_margins = self.margins[target].view(-1, 1)
        adjusted_logits = torch.where(index, logits - batch_margins, logits)
        return F.cross_entropy(self.s * adjusted_logits, target, weight=self.weight)


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list: list[int]):
        super().__init__()
        counts = torch.tensor(cls_num_list, dtype=torch.float32)
        self.register_buffer("log_prior", torch.log(counts / counts.sum()))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.log_prior, target)


class LogitAdjustedCE(nn.Module):
    def __init__(self, cls_num_list: list[int], tau: float = 1.0):
        super().__init__()
        counts = torch.tensor(cls_num_list, dtype=torch.float32)
        self.register_buffer("adjustment", tau * torch.log(counts / counts.sum()))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.adjustment, target)


def build_loss(cfg: dict, counts: list[int], epoch: int = 0) -> nn.Module:
    loss_cfg = cfg.get("loss", {"name": "cross_entropy"})
    name = loss_cfg.get("name", "cross_entropy").lower()

    if name in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss()
    if name in {"class_balanced_ce", "cb_ce"}:
        return nn.CrossEntropyLoss(weight=effective_num_weights(counts, loss_cfg.get("beta", 0.9999)))
    if name in {"inverse_ce", "weighted_ce"}:
        return nn.CrossEntropyLoss(weight=inverse_frequency_weights(counts))
    if name in {"focal", "focal_loss"}:
        return FocalLoss(gamma=float(loss_cfg.get("gamma", 2.0)))
    if name in {"class_balanced_focal", "cb_focal"}:
        return FocalLoss(
            gamma=float(loss_cfg.get("gamma", 2.0)),
            weight=effective_num_weights(counts, loss_cfg.get("beta", 0.9999)),
        )
    if name == "balanced_softmax":
        return BalancedSoftmaxLoss(counts)
    if name in {"logit_adjusted_ce", "la_ce"}:
        return LogitAdjustedCE(counts, tau=float(loss_cfg.get("tau", 1.0)))
    if name == "ldam":
        drw_start = int(loss_cfg.get("drw_start_epoch", 160))
        weight = None
        if epoch >= drw_start:
            weight = effective_num_weights(counts, loss_cfg.get("beta", 0.9999))
        return LDAMLoss(
            counts,
            max_m=float(loss_cfg.get("max_m", 0.5)),
            s=float(loss_cfg.get("s", 30.0)),
            weight=weight,
        )
    raise ValueError(f"Unknown loss: {name}")
