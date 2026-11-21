#!/usr/bin/env python3
"""Small additions used only by the v4 soft-vs-hard FADA experiment.

The old single-tile implementation is intentionally left untouched.  v4 keeps
its local 2K FADA discriminator and adds the binary global discriminator from
the historical Agent-Guide GAN as a second, separately reported branch.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from fada import FineGrainedDomainDiscriminator, domain_class_encoding


class GlobalDomainDiscriminator(nn.Module):
    """Global binary source/target discriminator over pooled Stage4 features."""

    def __init__(self, in_channels: int = 768) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.net = nn.Sequential(
            nn.Linear(self.in_channels, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.in_channels:
            raise ValueError(f"expected Bx{self.in_channels}xHfxWf feature, got {tuple(feature.shape)}")
        return self.net(feature.mean(dim=(2, 3))).squeeze(1)


def class_knowledge_v4(logits: torch.Tensor, *, mode: str, temperature: float,
                       confidence_cap: float, hard_threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return class knowledge and a valid-location mask.

    soft: FADA's released soft labels, temperature 1.8 and max clipping.
    hard: argmax one-hot labels only where max probability reaches a fixed
    absolute threshold; this is an ablation, not a top-ratio selector.
    """
    probs = F.softmax(logits / float(temperature), dim=1)
    max_prob = probs.max(dim=1, keepdim=True).values
    if mode == "soft":
        return torch.clamp(probs, max=float(confidence_cap)), torch.ones_like(max_prob)
    if mode != "hard":
        raise ValueError(f"unknown FADA mode: {mode}")
    valid = (max_prob >= float(hard_threshold)).to(probs.dtype)
    knowledge = F.one_hot(probs.argmax(dim=1), num_classes=probs.shape[1]).permute(0, 3, 1, 2).to(probs.dtype)
    return knowledge * valid, valid


def local_loss(discriminator: nn.Module, source_feature: torch.Tensor, target_feature: torch.Tensor,
               source_knowledge: torch.Tensor, target_knowledge: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    source_logits = discriminator(source_feature)
    target_logits = discriminator(target_feature)
    source_encoded = domain_class_encoding(source_knowledge, "source")
    target_encoded = domain_class_encoding(target_knowledge, "target")
    source_loss = -(source_encoded * F.log_softmax(source_logits, dim=1)).sum(dim=1).mean()
    target_loss = -(target_encoded * F.log_softmax(target_logits, dim=1)).sum(dim=1).mean()
    source_margin = torch.logsumexp(source_logits[:, :4], dim=1) - torch.logsumexp(source_logits[:, 4:], dim=1)
    target_margin = torch.logsumexp(target_logits[:, 4:], dim=1) - torch.logsumexp(target_logits[:, :4], dim=1)
    correct = torch.cat((source_margin >= 0, target_margin >= 0)).float().mean()
    return source_loss + target_loss, {
        "source_loss": float(source_loss.detach().item()),
        "target_loss": float(target_loss.detach().item()),
        "accuracy": float(correct.detach().item()),
    }


def local_target_adversarial_loss(discriminator: nn.Module, feature: torch.Tensor,
                                  knowledge: torch.Tensor) -> torch.Tensor:
    output = discriminator(feature)
    return -(domain_class_encoding(knowledge, "source") * F.log_softmax(output, dim=1)).sum(dim=1).mean()


def global_loss(discriminator: GlobalDomainDiscriminator, source_feature: torch.Tensor,
                target_feature: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    source = discriminator(source_feature)
    target = discriminator(target_feature)
    source_loss = F.binary_cross_entropy_with_logits(source, torch.zeros_like(source))
    target_loss = F.binary_cross_entropy_with_logits(target, torch.ones_like(target))
    prediction = torch.cat((source <= 0, target > 0))
    return 0.5 * (source_loss + target_loss), {
        "source_loss": float(source_loss.detach().item()),
        "target_loss": float(target_loss.detach().item()),
        "accuracy": float(prediction.float().mean().detach().item()),
    }


def global_target_adversarial_loss(discriminator: GlobalDomainDiscriminator,
                                   target_feature: torch.Tensor) -> torch.Tensor:
    output = discriminator(target_feature)
    return F.binary_cross_entropy_with_logits(output, torch.zeros_like(output))
