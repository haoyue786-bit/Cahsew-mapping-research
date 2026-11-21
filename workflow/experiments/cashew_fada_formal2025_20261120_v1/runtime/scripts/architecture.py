"""Immutable segmentation-architecture variants for formal AgentGuide jobs."""
from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


BASELINE = "a0_baseline"
A3_SPATIAL_2BLOCK = "a3_spatial_2block"
SUPPORTED = {BASELINE, A3_SPATIAL_2BLOCK}


class ResidualSpatial(nn.Module):
    """Zero-output 3×3 residual refiner, safe to add to an old checkpoint."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.conv2(F.gelu(self.norm(self.conv1(values))))


def normalize_architecture(value: Any) -> dict[str, Any]:
    """Return the complete immutable architecture record stored in job.json."""
    if value is None:
        # Legacy/isolated recipes may intentionally use A0. Formal validation
        # rejects omissions before this generic factory is called.
        name = BASELINE
    elif isinstance(value, str):
        name = value
    elif isinstance(value, dict):
        name = value.get("name", BASELINE)
    else:
        raise TypeError("architecture must be a string or object")
    name = str(name)
    if name not in SUPPORTED:
        raise ValueError({"unsupported_architecture": name, "supported": sorted(SUPPORTED)})
    return {
        "name": name,
        "post_fuse_spatial_blocks": 2 if name == A3_SPATIAL_2BLOCK else 0,
        "new_parameter_lr_multiplier": 5.0 if name == A3_SPATIAL_2BLOCK else 1.0,
        "second_conv_zero_initialized": bool(name == A3_SPATIAL_2BLOCK),
    }


def architecture_name(job: dict[str, Any]) -> str:
    return normalize_architecture(job.get("architecture"))["name"]


def make_model(gan, job: dict[str, Any], device: torch.device, *, resume_uninitialized: bool = False):
    """Load the canonical checkpoint then install the job-pinned A3 refinement."""
    initial_predictor=job.get('initial_transfer_weights_v3')
    if resume_uninitialized or initial_predictor:
        # All model and optimizer state is replaced by the verified latest
        # journal checkpoint before training or review is allowed.
        namespace = gan.make_model.__globals__
        model = namespace['base'].backbone_64swin2_convconnect.STC_DA(
            out_classes=namespace['NUM_CLASSES'], use_pretrained=False).to(device)
        load_stats = {'resume_uninitialized': True}
    else:
        model, load_stats = gan.make_model(device, Path(job["source"]["pretrained"]))
    architecture = normalize_architecture(job.get("architecture"))
    if architecture["name"] == BASELINE:
        return model, load_stats
    model.a3_spatial_refiner = nn.Sequential(ResidualSpatial(), ResidualSpatial()).to(device)

    def forward_fuse(net: nn.Module, inputs: torch.Tensor):
        delta = net.Swin(inputs)
        feature = net.fuse_conv(torch.cat([inputs, delta], dim=1))
        feature = net.a3_spatial_refiner(feature)
        return net.decoder(feature), feature

    # Every formal single-tile job has its own Python process.  Installing the
    # forward function on the protected runtime module therefore cannot leak
    # across jobs, while all FADA and review paths consistently see A3.
    gan.forward_fuse = forward_fuse
    # Source validation calls model(x); keep it identical to training/inference.
    def segmentation_forward(net, inputs):
        return forward_fuse(net, inputs)[0]
    model.forward = MethodType(segmentation_forward, model)
    from module_controls_v2 import enabled
    if enabled(job):
        from module_bn_v2 import install
        install(model)
    if initial_predictor and not resume_uninitialized:
        import hashlib,io
        raw=Path(initial_predictor['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=initial_predictor['sha256']:raise ValueError('initial inherited weights changed')
        value=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        if value.get('format')!='cashew_initial_transfer_weights_v3':raise ValueError('invalid inherited predictor')
        model.load_state_dict(value['state_dict'],strict=True)
    return model, load_stats


def model_parameter_groups(model: nn.Module, main_lr: float, weight_decay: float,
                           betas: tuple[float, float], eps: float, *, config=None) -> list[dict[str, Any]]:
    """Declared runs distinguish inherited/new A3; legacy groups keep 5x."""
    from module_controls_v2 import enabled, groups
    if config is not None and enabled(config):
        return groups(model, config, weight_decay, betas, eps)
    new_parameters = [parameter for name, parameter in model.named_parameters()
                      if name.startswith("a3_spatial_refiner.")]
    old_parameters = [parameter for parameter in model.parameters()
                      if not any(parameter is new for new in new_parameters)]
    groups = [{"params": old_parameters, "lr": main_lr, "lr_multiplier": 1.0,
               "weight_decay": weight_decay, "betas": betas, "eps": eps}]
    if new_parameters:
        from transfer_parameter_policy import bounds, value
        declared = bool(config is not None and bounds(config))
        multiplier = value(config) if declared else 5.0
        group = {"params": new_parameters, "lr": main_lr * multiplier, "lr_multiplier": multiplier,
                 "weight_decay": weight_decay, "betas": betas, "eps": eps}
        if declared:
            group.update(component='spatial_refiner',parameter_origin=(
                'inherited' if config['parameter_group_policy']['refiner_inherited'] else 'new'))
        groups.append(group)
    return groups


def feature_learning_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parameters whose gradients define the formal adaptation-pressure metric."""
    prefixes = ("Swin.", "fuse_conv", "a3_spatial_refiner.")
    return [parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith(prefixes)]
