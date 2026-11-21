#!/usr/bin/env python3
"""Streaming metrics collected from a normal FADA review epoch.

The accumulator deliberately contains no sampling or model logic.  It only
consumes detached tensors/statistics already produced by the training step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch


NUM_CLASSES = 4
IGNORE_INDEX = 10


def gradient_probe_steps(steps_per_epoch: int, num_probes: int = 6) -> set[int]:
    """Return one-based, evenly spaced normal-training step numbers."""
    steps = int(steps_per_epoch)
    probes = int(num_probes)
    if steps < 1 or probes < 1:
        return set()
    if steps <= probes:
        return set(range(1, steps + 1))
    return {int(value) for value in np.linspace(1, steps, probes, dtype=int)}


def _f1_from_confusion(confusion: np.ndarray) -> tuple[list[float], float]:
    values = []
    for cls in range(NUM_CLASSES):
        tp = float(confusion[cls, cls])
        fp = float(confusion[:, cls].sum() - confusion[cls, cls])
        fn = float(confusion[cls, :].sum() - confusion[cls, cls])
        values.append(float(2.0 * tp / max(1.0, 2.0 * tp + fp + fn)))
    return values, float(np.mean(values))


@dataclass
class ReviewAccumulator:
    """Collect review metrics from the review epoch's existing batches."""

    num_steps: int = 0
    source_loss_sum: float = 0.0
    adv_loss_sum: float = 0.0
    d_loss_sum: float = 0.0
    source_confusion_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    )
    target_entropy_sum: float = 0.0
    target_normalized_entropy_sum: float = 0.0
    target_pixel_count: int = 0
    target_predicted_class_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_CLASSES, dtype=np.int64)
    )
    d_correct: float = 0.0
    d_total: int = 0
    cls_grad_norms: list[float] = field(default_factory=list)
    adv_grad_norms: list[float] = field(default_factory=list)
    raw_adv_to_cls_ratios: list[float] = field(default_factory=list)
    effective_gradient_ratios: list[float] = field(default_factory=list)
    safety_events: list[dict[str, Any]] = field(default_factory=list)
    target_feature_vectors: list[np.ndarray] = field(default_factory=list)
    max_feature_rows: int = 128
    source_metric_basis: str = "training_steps_with_cycle"

    def update(
        self,
        *,
        source_loss: float,
        adv_loss: float,
        d_loss: float,
        source_logits: torch.Tensor,
        source_labels: torch.Tensor,
        target_logits: torch.Tensor | None = None,
        d_accuracy: float | None = None,
        d_updates: int = 0,
        cls_grad_norm: float | None = None,
        adv_grad_norm: float | None = None,
        lambda_adv: float = 0.0,
        target_features: torch.Tensor | None = None,
        safety_event: dict[str, Any] | None = None,
    ) -> None:
        """Consume one already-computed normal training step."""
        self.num_steps += 1
        self.source_loss_sum += float(source_loss)
        self.adv_loss_sum += float(adv_loss)
        self.d_loss_sum += float(d_loss)

        with torch.no_grad():
            predictions = source_logits.detach().argmax(dim=1).cpu().numpy().reshape(-1)
            labels = source_labels.detach().cpu().numpy().reshape(-1)
        valid = (labels != IGNORE_INDEX) & (labels >= 0) & (labels < NUM_CLASSES)
        if bool(valid.any()):
            np.add.at(self.source_confusion_matrix, (labels[valid], predictions[valid]), 1)

        if target_logits is not None:
            with torch.no_grad():
                logits = target_logits.detach()
                probabilities = torch.softmax(logits, dim=1)
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
                normalized = entropy / float(np.log(NUM_CLASSES))
                predicted = probabilities.argmax(dim=1).cpu().numpy().reshape(-1)
                self.target_entropy_sum += float(entropy.sum().item())
                self.target_normalized_entropy_sum += float(normalized.sum().item())
                self.target_pixel_count += int(entropy.numel())
                self.target_predicted_class_counts += np.bincount(
                    predicted, minlength=NUM_CLASSES
                ).astype(np.int64)

        if d_accuracy is not None and int(d_updates) > 0:
            self.d_correct += float(d_accuracy) * int(d_updates)
            self.d_total += int(d_updates)

        if cls_grad_norm is not None and adv_grad_norm is not None:
            cls = float(cls_grad_norm)
            adv = float(adv_grad_norm)
            raw = adv / max(cls, 1e-12)
            effective = float(lambda_adv) * raw
            self.cls_grad_norms.append(cls)
            self.adv_grad_norms.append(adv)
            self.raw_adv_to_cls_ratios.append(raw)
            self.effective_gradient_ratios.append(effective)

        if target_features is not None and len(self.target_feature_vectors) < self.max_feature_rows:
            with torch.no_grad():
                compact = torch.nn.functional.adaptive_avg_pool2d(
                    target_features.detach(), 1
                ).flatten(1).cpu().numpy()
            remaining = self.max_feature_rows - len(self.target_feature_vectors)
            self.target_feature_vectors.extend(
                [row.copy() for row in compact[:remaining]]
            )

        if safety_event is not None:
            self.safety_events.append(dict(safety_event))

    def finalize(
        self,
        *,
        epoch: int,
        config: dict[str, Any],
        snd_fn: Callable[..., float] | None = None,
    ) -> dict[str, Any]:
        """Return compact, JSON-safe review metrics."""
        f1, macro_f1 = _f1_from_confusion(self.source_confusion_matrix)
        target_distribution = (
            (self.target_predicted_class_counts / float(self.target_pixel_count)).tolist()
            if self.target_pixel_count
            else [0.0] * NUM_CLASSES
        )
        if self.target_feature_vectors and snd_fn is not None:
            features = np.stack(self.target_feature_vectors, axis=0)
            snd = float(snd_fn(features, temperature=0.05))
            snd_basis = "review_epoch_training_target_features_first_128"
        else:
            snd = 0.0
            snd_basis = "unavailable_for_paused_training_path"

        def median(values: list[float]) -> float | None:
            return float(np.median(values)) if values else None

        def maximum(values: list[float]) -> float | None:
            return float(np.max(values)) if values else None

        gradient = {
            "source": "review_epoch_training_steps",
            "num_samples": len(self.effective_gradient_ratios),
            "cls_grad_median": median(self.cls_grad_norms),
            "adv_grad_median": median(self.adv_grad_norms),
            "raw_adv_to_cls_median": median(self.raw_adv_to_cls_ratios),
            "effective_adv_to_cls_median": median(self.effective_gradient_ratios),
            "effective_adv_to_cls_max": maximum(self.effective_gradient_ratios),
        }
        return {
            "epoch": int(epoch),
            "source": {
                "loss": self.source_loss_sum / max(1, self.num_steps),
                "f1_per_class": f1,
                "macro_f1": macro_f1,
                "confusion_matrix": self.source_confusion_matrix.tolist(),
                "pixels": int(self.source_confusion_matrix.sum()),
                "metric_basis": self.source_metric_basis,
            },
            "target_unsupervised": {
                "snd": snd,
                "snd_basis": snd_basis,
                "entropy": self.target_entropy_sum / max(1, self.target_pixel_count),
                "normalized_entropy": self.target_normalized_entropy_sum / max(1, self.target_pixel_count),
                "class_distribution": target_distribution,
                "predicted_class_counts": self.target_predicted_class_counts.tolist(),
                "pixel_count": int(self.target_pixel_count),
            },
            "discriminator": {
                "loss": self.d_loss_sum / max(1, self.num_steps),
                "marginal_accuracy": self.d_correct / max(1, self.d_total),
                "updates": int(self.d_total),
            },
            "gradients": gradient,
            "training": {
                "steps": int(self.num_steps),
                "lambda_adv": float(config.get("lambda_adv", 0.0)),
                "main_lr": float(config.get("main_lr", 0.0)),
                "D_lr": float(config.get("D_lr", 0.0)),
                "D_steps_per_G_step": int(config.get("D_steps_per_G_step", 0)),
                "temperature": float(config.get("class_knowledge", {}).get("temperature", 0.0)),
                "confidence_cap": float(config.get("class_knowledge", {}).get("confidence_cap", 0.0)),
                "prior_strength": float(config.get("prior_strength", 0.0)),
                "adversarial_state": config.get("adversarial_state", "active"),
            },
            "metric_provenance": {
                "source_training": "review_epoch_accumulator",
                "source_safety": "existing_fixed_source_validation",
                "target_training": "review_epoch_accumulator",
                "discriminator_training": "review_epoch_accumulator",
                "gradient": "review_epoch_training_steps",
                "snd": snd_basis,
                "extra_active_review_batch": False,
            },
            "safety_events": list(self.safety_events),
        }
