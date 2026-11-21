#!/usr/bin/env python3
"""Fine-grained adversarial alignment primitives for four-class segmentation.

The implementation follows FADA's joint domain/class discriminator: one fully
convolutional module produces 2K channels.  It is not a bank of independent
binary discriminators.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class FineGrainedDomainDiscriminator(nn.Module):
    """One low-resolution per-location discriminator with 2K output channels."""

    def __init__(self, in_channels: int = 768, num_classes: int = 4,
                 hidden_channels: tuple[int, int] = (256, 128)) -> None:
        super().__init__()
        if in_channels <= 0 or num_classes <= 1 or min(hidden_channels) <= 0:
            raise ValueError("invalid FADA discriminator dimensions")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        first, second = (int(value) for value in hidden_channels)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, first, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(first, second, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(second, 2 * self.num_classes, kernel_size=3, padding=1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.in_channels:
            raise ValueError(
                f"expected Bx{self.in_channels}xHfxWf Stage4 features, "
                f"got {tuple(feature.shape)}"
            )
        output = self.net(feature)
        expected = (feature.shape[0], 2 * self.num_classes, feature.shape[2], feature.shape[3])
        if tuple(output.shape) != expected:
            raise RuntimeError(f"FADA 2K shape invariant failed: {tuple(output.shape)} != {expected}")
        return output


def resize_segmentation_logits(logits: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4 or feature.ndim != 4:
        raise ValueError("segmentation logits and features must both be BCHW")
    if logits.shape[0] != feature.shape[0]:
        raise ValueError("segmentation-logit/feature batch mismatch")
    return F.interpolate(
        logits, size=feature.shape[-2:], mode="bilinear", align_corners=False,
    )


def apply_prior_logit_shift(logits: torch.Tensor, shift: torch.Tensor | np.ndarray | list[float]) -> torch.Tensor:
    """Apply the already-clipped tau-scaled log-prior ratio once."""
    values = torch.as_tensor(shift, dtype=logits.dtype, device=logits.device)
    if logits.ndim != 4 or values.shape != (logits.shape[1],):
        raise ValueError(f"expected K-vector shift for BCHW logits, got {tuple(values.shape)}")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("prior logit shift must be finite")
    return logits + values.view(1, -1, 1, 1)


def class_knowledge(logits: torch.Tensor, temperature: float = 1.8,
                    confidence_cap: float = 0.9,
                    prior_shift: torch.Tensor | np.ndarray | list[float] | None = None,
                    detach: bool = True) -> torch.Tensor:
    """Construct FADA soft class knowledge at the discriminator resolution.

    The >0.9 clipping mirrors the authors' released implementation.  Knowledge
    is detached so the discriminator labels cannot be gamed through the class
    encoding branch.
    """
    if logits.ndim != 4 or logits.shape[1] < 2:
        raise ValueError(f"expected BxKxHfxWf logits, got {tuple(logits.shape)}")
    if temperature <= 0 or not 0 < confidence_cap <= 1:
        raise ValueError("temperature must be positive and confidence_cap in (0,1]")
    adjusted = apply_prior_logit_shift(logits, prior_shift) if prior_shift is not None else logits
    knowledge = F.softmax(adjusted / float(temperature), dim=1)
    knowledge = torch.clamp(knowledge, max=float(confidence_cap))
    if detach:
        knowledge = knowledge.detach()
    if not bool(torch.isfinite(knowledge).all()):
        raise FloatingPointError("non-finite FADA class knowledge")
    return knowledge


def domain_class_encoding(knowledge: torch.Tensor, domain: str) -> torch.Tensor:
    """Encode [a,0] for source or [0,a] for target."""
    if knowledge.ndim != 4:
        raise ValueError("class knowledge must be BxKxHfxWf")
    zeros = torch.zeros_like(knowledge)
    if domain == "source":
        encoded = torch.cat((knowledge, zeros), dim=1)
    elif domain == "target":
        encoded = torch.cat((zeros, knowledge), dim=1)
    else:
        raise ValueError(f"domain must be source or target, got {domain!r}")
    if encoded.shape[1] != 2 * knowledge.shape[1]:
        raise RuntimeError("2K encoding invariant failed")
    return encoded


def soft_label_cross_entropy(logits_2k: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
    if logits_2k.shape != encoded.shape or logits_2k.ndim != 4:
        raise ValueError(f"2K logit/encoding mismatch: {tuple(logits_2k.shape)}/{tuple(encoded.shape)}")
    loss = -(encoded * F.log_softmax(logits_2k, dim=1)).sum(dim=1).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite FADA soft-label cross entropy")
    return loss


def discriminator_loss(discriminator: nn.Module, source_feature: torch.Tensor,
                       target_feature: torch.Tensor, source_knowledge: torch.Tensor,
                       target_knowledge: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    source_logits = discriminator(source_feature)
    target_logits = discriminator(target_feature)
    source_loss = soft_label_cross_entropy(
        source_logits, domain_class_encoding(source_knowledge, "source"),
    )
    target_loss = soft_label_cross_entropy(
        target_logits, domain_class_encoding(target_knowledge, "target"),
    )
    loss = source_loss + target_loss
    stats = discriminator_statistics(source_logits.detach(), target_logits.detach())
    stats.update({"source_loss": float(source_loss.item()), "target_loss": float(target_loss.item())})
    return loss, stats


def target_adversarial_loss(discriminator: nn.Module, target_feature: torch.Tensor,
                            target_knowledge: torch.Tensor) -> torch.Tensor:
    """Make target features look like the same class in the source half."""
    output = discriminator(target_feature)
    return soft_label_cross_entropy(output, domain_class_encoding(target_knowledge, "source"))


def marginal_domain_logit(logits_2k: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    if logits_2k.ndim != 4 or logits_2k.shape[1] != 2 * num_classes:
        raise ValueError(f"expected Bx{2 * num_classes}xHfxWf logits")
    source = torch.logsumexp(logits_2k[:, :num_classes], dim=1)
    target = torch.logsumexp(logits_2k[:, num_classes:], dim=1)
    return (target - source).mean(dim=(1, 2))


def discriminator_statistics(source_logits: torch.Tensor, target_logits: torch.Tensor,
                             num_classes: int = 4) -> dict[str, Any]:
    source_margin = marginal_domain_logit(source_logits, num_classes)
    target_margin = marginal_domain_logit(target_logits, num_classes)
    correct = torch.cat((source_margin <= 0, target_margin > 0)).float()
    return {
        "marginal_domain_accuracy": float(correct.mean().item()),
        "source_margin_mean": float(source_margin.mean().item()),
        "target_margin_mean": float(target_margin.mean().item()),
        "output_shape_source": list(source_logits.shape),
        "output_shape_target": list(target_logits.shape),
    }


class FadaDomainMarginalView(nn.Module):
    """Read-only binary view used by the legacy held-out diagnostic harness."""

    def __init__(self, discriminator: FineGrainedDomainDiscriminator) -> None:
        super().__init__()
        self.discriminator = discriminator

    def forward(self, feature: torch.Tensor, probs=None, return_hidden: bool = False):
        margin = marginal_domain_logit(self.discriminator(feature), self.discriminator.num_classes)
        hidden = feature.mean(dim=(2, 3))
        return (margin, hidden) if return_hidden else margin


def calculate_prior_shift(source_prior: np.ndarray, target_prior: np.ndarray,
                          tau: np.ndarray, epsilon: float,
                          clip: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_prior, dtype=np.float64)
    target = np.asarray(target_prior, dtype=np.float64)
    reliability = np.asarray(tau, dtype=np.float64)
    if source.shape != target.shape or source.shape != reliability.shape or source.ndim != 1:
        raise ValueError("source/target/tau priors must be aligned vectors")
    if not (np.isfinite(source).all() and np.isfinite(target).all() and np.isfinite(reliability).all()):
        raise ValueError("prior inputs must be finite")
    raw = np.log(target + epsilon) - np.log(source + epsilon)
    clipped = np.clip(raw, float(clip[0]), float(clip[1]))
    shift = reliability * clipped
    if not np.isfinite(shift).all():
        raise FloatingPointError("prior correction produced a non-finite shift")
    return raw, shift
