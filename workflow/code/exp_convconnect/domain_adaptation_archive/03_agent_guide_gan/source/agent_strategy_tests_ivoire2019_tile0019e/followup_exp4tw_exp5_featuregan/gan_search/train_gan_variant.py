#!/usr/bin/env python3
"""Controlled Planet-free Stage4 GAN variants with held-out challenger reviews."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch import nn


FOLLOWUP = Path(__file__).resolve().parents[1]
PARENT = FOLLOWUP.parent
SHARED_CODE = PARENT / "shared/code"
EXP5_DISC = FOLLOWUP / "exp05_agent_feature_gan/discriminator"
AUDIT_CODE = FOLLOWUP / "gan_search/baseline_audit"
for path in (SHARED_CODE, EXP5_DISC, AUDIT_CODE):
    sys.path.insert(0, str(path))
from common import (BASE_LR, NUM_CLASSES, PRETRAINED, SOURCE_TRAIN, SOURCE_VALID,
                    TARGET_FULL, TARGET_ROOT, TARGET_TRAIN, StrategyDataset,
                    forward_fuse, make_model, read_lines, read_raster,
                    review_diagnostics, rng_state, seed_all, source_label_path,
                    weighted_source_loss)
from feature_gan import Stage4PatchDiscriminator
from strict_heldout_probe import (GlobalMLP, binary_report, patch_logits,
                                  split_names, train_torch_probe)


SOURCE_ROOT = Path("@ASSETS@/code/mydataset2020/newdataset/AlphaEarth_2019")
VARIANT_DIRS = {
    "v2a": "round_01_v2a_gap_control",
    "v2b": "round_02_v2b_global_local",
    "v2c": "round_03_v2c_conditional",
    "v2d_plain": "round_04_v2d_plain_multi",
    "v2d_diverse": "round_05_v2d_diverse_multi",
    "v2e_adaptive": "round_06_v2e_adaptive_multiview",
    "v2f_cdan": "round_07_v2f_cdan_global_local",
    "v2g_deep": "round_08_v2g_deep_capacity_control",
}


class GlobalDomainDiscriminator(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, stage4, probs=None, return_hidden=False):
        vector = stage4.mean(dim=(2, 3)) if stage4.ndim == 4 else stage4
        if probs is not None:
            vector = torch.cat([vector, probs], dim=1)
        hidden = F.leaky_relu(self.fc1(vector), 0.2)
        logits = self.fc2(hidden).flatten()
        return (logits, hidden) if return_hidden else logits


class CDANDomainDiscriminator(nn.Module):
    """Discriminate the multilinear joint of Stage4 feature and class prediction."""
    def __init__(self, feature_dim=768, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim * num_classes, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, stage4, probs, return_hidden=False):
        feature = stage4.mean(dim=(2, 3)) if stage4.ndim == 4 else stage4
        joint = torch.bmm(probs.unsqueeze(2), feature.unsqueeze(1)).flatten(1)
        hidden = F.leaky_relu(self.fc1(joint), 0.2)
        logits = self.fc2(hidden).flatten()
        return (logits, hidden) if return_hidden else logits


class DeepStage4PatchDiscriminator(nn.Module):
    """Higher-capacity local D used only for the registered capacity control."""
    def __init__(self, in_channels=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 512, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def forward(self, stage4):
        return self.net(stage4)


class DeepGlobalDomainDiscriminator(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, stage4, probs=None, return_hidden=False):
        vector = stage4.mean(dim=(2, 3)) if stage4.ndim == 4 else stage4
        hidden = F.leaky_relu(self.fc1(vector), 0.2)
        hidden = F.leaky_relu(self.fc2(hidden), 0.2)
        hidden = F.leaky_relu(self.fc3(hidden), 0.2)
        logits = self.fc4(hidden).flatten()
        return (logits, hidden) if return_hidden else logits


def stage4_map(model, x):
    tokens = model.Swin(x, return_bottleneck=True)
    if tokens.ndim == 3:
        tokens = model.norm(tokens)
        batch, length, channels = tokens.shape
        side = int(math.sqrt(length))
        if side * side != length:
            raise ValueError(f"non-square Stage4 token map: {tuple(tokens.shape)}")
        return tokens.transpose(1, 2).reshape(batch, channels, side, side)
    if tokens.ndim == 4 and tokens.shape[1] == 768:
        return tokens
    if tokens.ndim == 4 and tokens.shape[-1] == 768:
        return tokens.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"unsupported Stage4 shape: {tuple(tokens.shape)}")


def batch_from(dataset, source_indices, target_indices, device):
    source_items = [dataset[int(index)] for index in source_indices]
    target_items = [dataset[int(index)] for index in target_indices]
    source_batch = [torch.stack([torch.as_tensor(item[j]) for item in source_items]).to(device) for j in range(4)]
    target_batch = [torch.stack([torch.as_tensor(item[j]) for item in target_items]).to(device) for j in range(4)]
    return source_batch, target_batch


def set_requires_grad(module, flag):
    for parameter in module.parameters():
        parameter.requires_grad_(flag)


def gradient_norm(gradients):
    return math.sqrt(sum(float(gradient.detach().pow(2).sum().item()) for gradient in gradients if gradient is not None))


def build_discriminators(variant, device):
    if variant == "v2a":
        return nn.ModuleDict({"local": Stage4PatchDiscriminator()}).to(device), {"local": 1.0}
    if variant == "v2b":
        return nn.ModuleDict({"local": Stage4PatchDiscriminator(), "global": GlobalDomainDiscriminator()}).to(device), {"local": 1.0, "global": 0.5}
    if variant == "v2c":
        return nn.ModuleDict({"conditional": GlobalDomainDiscriminator(772)}).to(device), {"conditional": 1.0}
    if variant == "v2e_adaptive":
        modules = nn.ModuleDict({
            "local": Stage4PatchDiscriminator(),
            "global": GlobalDomainDiscriminator(),
            "conditional": GlobalDomainDiscriminator(772),
        }).to(device)
        return modules, {"local": 1.0, "global": 0.5, "conditional": 0.5}
    if variant == "v2f_cdan":
        modules = nn.ModuleDict({
            "local": Stage4PatchDiscriminator(),
            "global": GlobalDomainDiscriminator(),
            "cdan": CDANDomainDiscriminator(),
        }).to(device)
        return modules, {"local": 1.0, "global": 0.5, "cdan": 0.5}
    if variant == "v2g_deep":
        modules = nn.ModuleDict({
            "local": DeepStage4PatchDiscriminator(),
            "global": DeepGlobalDomainDiscriminator(),
        }).to(device)
        return modules, {"local": 1.0, "global": 0.5}
    modules = nn.ModuleDict({"local": Stage4PatchDiscriminator(), "global1": GlobalDomainDiscriminator(), "global2": GlobalDomainDiscriminator()}).to(device)
    return modules, {"local": 1.0, "global1": 0.5, "global2": 0.5}


def class_probs(model, x):
    return torch.softmax(model(x), dim=1).mean(dim=(2, 3))


def domain_outputs(discriminators, feature, probs, weights, return_hidden=False):
    outputs, hidden = {}, {}
    for name, discriminator in discriminators.items():
        if name == "local":
            outputs[name] = discriminator(feature)
        elif name in ("conditional", "cdan"):
            value = discriminator(feature, probs, return_hidden=return_hidden)
            outputs[name], hidden[name] = value if return_hidden else (value, None)
        else:
            value = discriminator(feature, return_hidden=return_hidden)
            outputs[name], hidden[name] = value if return_hidden else (value, None)
    return outputs, hidden


def domain_bce(outputs_source, outputs_target, weights):
    losses = {}
    for name in outputs_source:
        source_logits, target_logits = outputs_source[name], outputs_target[name]
        losses[name] = F.binary_cross_entropy_with_logits(source_logits, torch.zeros_like(source_logits)) + F.binary_cross_entropy_with_logits(target_logits, torch.ones_like(target_logits))
    return sum(weights[name] * losses[name] for name in losses), losses


def target_adversarial_loss(outputs, weights):
    losses = {name: F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)) for name, logits in outputs.items()}
    return sum(weights[name] * losses[name] for name in losses), losses


def diversity_penalty(hidden_source, hidden_target):
    names = [name for name in hidden_source if name.startswith("global") and hidden_source[name] is not None]
    if len(names) < 2:
        value = next(iter(hidden_source.values()), None)
        return value.sum() * 0 if value is not None else torch.tensor(0.0)
    penalties = []
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            left = torch.cat([hidden_source[names[first]], hidden_target[names[first]]])
            right = torch.cat([hidden_source[names[second]], hidden_target[names[second]]])
            penalties.append(F.cosine_similarity(left, right, dim=1).pow(2).mean())
    return torch.stack(penalties).mean()


def discriminator_accuracy(outputs_source, outputs_target):
    per_branch = {}
    for name in outputs_source:
        source = outputs_source[name].detach().flatten()
        target = outputs_target[name].detach().flatten()
        logits = torch.cat([source, target])
        labels = torch.cat([torch.zeros_like(source), torch.ones_like(target)])
        per_branch[name] = float(((logits > 0) == labels.bool()).float().mean().item())
    return float(np.mean(list(per_branch.values()))), per_branch


def read_image(relative, source):
    return read_raster((SOURCE_ROOT if source else TARGET_ROOT) / relative)


def collect_review_features(model, names, source, device, batch_size=16):
    maps, pooled, probabilities = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(names), batch_size):
            part = names[start:start + batch_size]
            x = torch.from_numpy(np.stack([read_image(name, source) for name in part])).to(device).float()
            fmap = stage4_map(model, x)
            maps.append(fmap.cpu().numpy().astype(np.float32))
            pooled.append(fmap.mean(dim=(2, 3)).cpu().numpy().astype(np.float32))
            probabilities.append(class_probs(model, x).cpu().numpy().astype(np.float32))
    return np.concatenate(maps), np.concatenate(pooled), np.concatenate(probabilities)


def balanced_indices(names, split, split_name):
    return [names.index(value) for value in split[split_name]]


def heldout_challengers(model, discriminators, variant, weights, device):
    source_names = read_lines(SOURCE_VALID)
    target_full = read_lines(TARGET_FULL)
    stride = max(1, len(target_full) // 220)
    target_names = target_full[::stride][:220]
    source_split, _ = split_names(source_names, "source", 4)
    target_split, _ = split_names(target_names, "target_review", 4)
    source_maps, source_global, source_probs = collect_review_features(model, source_names, True, device)
    target_maps, target_global, target_probs = collect_review_features(model, target_names, False, device)
    features = {
        "maps": {"source": source_maps, "target": target_maps},
        "global": {"source": source_global, "target": target_global},
        "conditional": {"source": np.concatenate([source_global, source_probs], axis=1),
                        "target": np.concatenate([target_global, target_probs], axis=1)},
        "cdan": {"source": np.einsum("bi,bj->bij", source_probs, source_global).reshape(len(source_global), -1),
                 "target": np.einsum("bi,bj->bij", target_probs, target_global).reshape(len(target_global), -1)},
        "probs": {"source": source_probs, "target": target_probs},
    }
    arrays, labels = {}, {}
    for split_name in ("train", "valid", "test"):
        source_idx = balanced_indices(source_names, source_split, split_name)
        target_idx = balanced_indices(target_names, target_split, split_name)
        count = min(len(source_idx), len(target_idx)); source_idx, target_idx = source_idx[:count], target_idx[:count]
        arrays[split_name] = {
            key: np.concatenate([features[key]["source"][source_idx], features[key]["target"][target_idx]])
            for key in ("maps", "global", "conditional", "cdan", "probs")
        }
        labels[split_name] = np.r_[np.zeros(count, dtype=np.int64), np.ones(count, dtype=np.int64)]

    torch.manual_seed(16)
    if variant == "v2c":
        same_model = GlobalMLP(772).to(device)
    elif variant == "v2g_deep":
        same_model = DeepStage4PatchDiscriminator().to(device)
    else:
        same_model = Stage4PatchDiscriminator().to(device)
    same_key = "maps" if variant != "v2c" else "conditional"
    same_train, same_valid = train_torch_probe(same_model, arrays["train"][same_key], labels["train"], arrays["valid"][same_key], labels["valid"], device)
    with torch.no_grad():
        same_logits = patch_logits(same_model, torch.from_numpy(arrays["test"][same_key]).to(device).float()).cpu().numpy()
    same_test = binary_report(labels["test"], same_logits)

    torch.manual_seed(17)
    global_model = GlobalMLP().to(device)
    global_train, global_valid = train_torch_probe(global_model, arrays["train"]["global"], labels["train"], arrays["valid"]["global"], labels["valid"], device)
    with torch.no_grad():
        global_logits = patch_logits(global_model, torch.from_numpy(arrays["test"]["global"]).to(device).float()).cpu().numpy()
    global_test = binary_report(labels["test"], global_logits)

    torch.manual_seed(18)
    conditional_model = GlobalMLP(772).to(device)
    conditional_train, conditional_valid = train_torch_probe(
        conditional_model, arrays["train"]["conditional"], labels["train"],
        arrays["valid"]["conditional"], labels["valid"], device)
    with torch.no_grad():
        conditional_logits = patch_logits(
            conditional_model,
            torch.from_numpy(arrays["test"]["conditional"]).to(device).float(),
        ).cpu().numpy()
    conditional_test = binary_report(labels["test"], conditional_logits)

    torch.manual_seed(19)
    cdan_model = GlobalMLP(3072).to(device)
    cdan_train, cdan_valid = train_torch_probe(
        cdan_model, arrays["train"]["cdan"], labels["train"],
        arrays["valid"]["cdan"], labels["valid"], device)
    with torch.no_grad():
        cdan_logits = patch_logits(
            cdan_model, torch.from_numpy(arrays["test"]["cdan"]).to(device).float()
        ).cpu().numpy()
    cdan_test = binary_report(labels["test"], cdan_logits)

    model.eval(); discriminators.eval()
    current_logits = []
    current_branch_logits = {name: [] for name in discriminators}
    with torch.no_grad():
        for domain_slice in (slice(0, len(labels["test"]) // 2), slice(len(labels["test"]) // 2, None)):
            fmap = torch.from_numpy(arrays["test"]["maps"][domain_slice]).to(device).float()
            probs = torch.from_numpy(arrays["test"]["probs"][domain_slice]).to(device).float()
            outputs, _ = domain_outputs(discriminators, fmap, probs, weights)
            branch_patch_logits = []
            for name, logits in outputs.items():
                reduced = patch_logits(nn.Identity(), logits)
                branch_patch_logits.append(reduced)
                current_branch_logits[name].append(reduced.cpu().numpy())
            current_logits.append(torch.stack(branch_patch_logits).mean(0).cpu().numpy())
    current = binary_report(labels["test"], np.concatenate(current_logits))
    current_per_branch = {
        name: binary_report(labels["test"], np.concatenate(parts))
        for name, parts in current_branch_logits.items()
    }

    linear = LogisticRegression(max_iter=500, solver="liblinear", random_state=16).fit(arrays["train"]["global"], labels["train"])
    linear_test = binary_report(labels["test"], linear.decision_function(arrays["test"]["global"]))
    return {
        "protocol": "patch-disjoint spatial-block train/valid/test; domain-balanced; target spatial subsample",
        "counts": {split: int(len(labels[split])) for split in labels},
        "current_D_test": current,
        "current_D_per_branch_test": current_per_branch,
        "fresh_same_arch": {"train": same_train, "valid": same_valid, "test": same_test},
        "fresh_global_mlp": {"train": global_train, "valid": global_valid, "test": global_test},
        "fresh_conditional_mlp": {
            "train": conditional_train, "valid": conditional_valid, "test": conditional_test,
        },
        "fresh_cdan_mlp": {"train": cdan_train, "valid": cdan_valid, "test": cdan_test},
        "fresh_linear_test": linear_test,
        "gap_same": same_test["accuracy"] - current["accuracy"],
        "gap_global": global_test["accuracy"] - current["accuracy"],
    }


def save_checkpoint(path, epoch, model, discriminators, model_optimizer, d_optimizer, config, metrics, decisions):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch), "state_dict": model.state_dict(),
        "discriminators_state_dict": discriminators.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "discriminator_optimizer_state_dict": d_optimizer.state_dict(),
        "scheduler_state_dict": {"epoch": int(epoch), "type": "constant within controlled 16-epoch run"},
        "rng_state": rng_state(), "config": config, "metrics": metrics,
        "decision_history": decisions, "planet_used_in_training": False,
    }, path)


def train_one_epoch(model, discriminators, model_optimizer, d_optimizer, dataset, source_count, target_count, config, device, rng):
    model.train(); discriminators.train()
    totals = {key: [] for key in ("D_loss", "D_acc", "diversity", "cls", "adv", "D_grad", "cls_grad", "adv_grad")}
    branch_acc = {name: [] for name in discriminators}
    for _ in range(config["steps_per_epoch"]):
        source_idx = [int(rng.randint(0, source_count)) for _ in range(config["batch_size"])]
        target_idx = [int(rng.randint(0, target_count)) for _ in range(config["batch_size"])]
        source_batch, target_batch = batch_from(dataset, source_idx, target_idx, device)
        source_aug, source_label, source_raw, _ = source_batch
        _, _, _, target_raw = target_batch
        source_label = source_label.long()

        set_requires_grad(model, False); set_requires_grad(discriminators, True)
        for _ in range(config["D_steps_per_G_step"]):
            with torch.no_grad():
                source_feature, target_feature = stage4_map(model, source_raw), stage4_map(model, target_raw)
                needs_probs = "conditional" in discriminators or "cdan" in discriminators
                source_probs = class_probs(model, source_raw) if needs_probs else None
                target_probs = class_probs(model, target_raw) if needs_probs else None
            d_optimizer.zero_grad()
            source_outputs, source_hidden = domain_outputs(discriminators, source_feature.detach(), source_probs, config["D_loss_weights"], True)
            target_outputs, target_hidden = domain_outputs(discriminators, target_feature.detach(), target_probs, config["D_loss_weights"], True)
            d_loss, _ = domain_bce(source_outputs, target_outputs, config["D_loss_weights"])
            diversity = diversity_penalty(source_hidden, target_hidden) if config["variant"] == "v2d_diverse" else d_loss * 0
            total_d = d_loss + config["lambda_diversity"] * diversity
            total_d.backward(); totals["D_grad"].append(gradient_norm([parameter.grad for parameter in discriminators.parameters()])); d_optimizer.step()
            accuracy, per_branch = discriminator_accuracy(source_outputs, target_outputs)
            totals["D_loss"].append(float(d_loss.item())); totals["D_acc"].append(accuracy); totals["diversity"].append(float(diversity.item()))
            for name, value in per_branch.items(): branch_acc[name].append(value)

        set_requires_grad(model, True); set_requires_grad(discriminators, False)
        model_optimizer.zero_grad()
        source_logits, _ = forward_fuse(model, source_aug.float())
        cls_loss = weighted_source_loss(source_logits, source_label)
        target_feature = stage4_map(model, target_raw)
        target_probs = class_probs(model, target_raw) if ("conditional" in discriminators or "cdan" in discriminators) else None
        target_outputs, _ = domain_outputs(discriminators, target_feature, target_probs, config["D_loss_weights"])
        adv_loss, _ = target_adversarial_loss(target_outputs, config["D_loss_weights"])
        encoder_params = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and name.startswith(("Swin.", "fuse_conv"))]
        cls_grads = torch.autograd.grad(cls_loss, encoder_params, retain_graph=True, allow_unused=True)
        adv_grads = torch.autograd.grad(adv_loss, encoder_params, retain_graph=True, allow_unused=True)
        (cls_loss + config["lambda_adv"] * adv_loss).backward(); model_optimizer.step()
        totals["cls"].append(float(cls_loss.item())); totals["adv"].append(float(adv_loss.item()))
        totals["cls_grad"].append(gradient_norm(cls_grads)); totals["adv_grad"].append(gradient_norm(adv_grads))
        set_requires_grad(discriminators, True)
    result = {key: float(np.mean(values)) for key, values in totals.items()}
    result["D_accuracy_per_branch"] = {name: float(np.mean(values)) for name, values in branch_acc.items()}
    result["effective_adv_to_cls_grad_ratio"] = config["lambda_adv"] * result["adv_grad"] / max(1e-12, result["cls_grad"])
    return result


def select_checkpoint(candidates):
    scores = {}
    baseline_c2 = candidates[0]["review"]["source"]["f1_per_class"][2]
    for item in candidates:
        review, challenger = item["review"], item["challenger"]
        source_c2 = review["source"]["f1_per_class"][2]
        macro = review["source"]["macro_f1"]
        snd = review["target_unsupervised"]["snd"]
        fresh = challenger["fresh_global_mlp"]["test"]["accuracy"]
        collapse_penalty = max(0.0, baseline_c2 - source_c2 - 0.05) * 5
        scores[item["epoch"]] = source_c2 + 0.25 * macro + 0.01 * snd - 0.02 * fresh - collapse_penalty
    selected = max(candidates, key=lambda item: scores[item["epoch"]])
    return selected, scores


def separability(report):
    """Threshold-free domain separability; reversed logits are still separable."""
    auc = float(report["auc"])
    return max(auc, 1.0 - auc)


def reset_discriminators(discriminators):
    def maybe_reset(module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    discriminators.apply(maybe_reset)


def agent_decision(variant, epoch, config, weights, review0, review, challenger, metrics):
    """Planet-free bounded policy. Returns next config, actions and refresh flag."""
    old = json.loads(json.dumps(config))
    new = json.loads(json.dumps(config))
    actions, reasons = [], []
    source_c2 = float(review["source"]["f1_per_class"][2])
    baseline_c2 = float(review0["source"]["f1_per_class"][2])
    grad_ratio = float(metrics["effective_adv_to_cls_grad_ratio"])

    if source_c2 < baseline_c2 - 0.05 or grad_ratio > 0.5:
        new["lambda_adv"] = max(0.001, float(new["lambda_adv"]) * 0.5)
        actions.append("REDUCE_ADVERSARIAL_PRESSURE")
        reasons.append("source class2 safety gate or effective adversarial-gradient gate fired")

    if variant == "v2a" and epoch == 8 and challenger["gap_global"] > 0.10:
        new["D_lr"] = float(new["D_lr"]) * 2.0
        new["D_steps_per_G_step"] = 4
        actions.append("D_ONLY_RECALIBRATE_BY_TIME_SCALE")
        reasons.append("fresh global challenger substantially exceeded current D")

    if variant == "v2b" and epoch == 8 and challenger["gap_global"] > 0.10:
        branch_reports = challenger["current_D_per_branch_test"]
        current_auc = max(float(report["auc"]) for report in branch_reports.values())
        current_accuracy = float(challenger["current_D_test"]["accuracy"])
        if current_auc >= 0.80 and current_accuracy < 0.65:
            # The discriminator ranks domains correctly but its operating point
            # is badly calibrated.  The old policy accidentally handled this
            # state only for v2a/v2e, leaving the selected v2b route on KEEP.
            new["D_lr"] = max(2e-5, float(new["D_lr"]) * 10.0)
            new["D_steps_per_G_step"] = 4
            actions.append("CALIBRATE_V2B_CAPABLE_D")
            reasons.append("fresh challenger gap plus high current AUC and low threshold accuracy")

    refresh = False
    if variant == "v2e_adaptive" and epoch == 8:
        branch_reports = challenger["current_D_per_branch_test"]
        current_sep = {name: separability(report) for name, report in branch_reports.items()}
        mean_sep = float(np.mean(list(current_sep.values())))
        fresh_sep = max(
            separability(challenger["fresh_same_arch"]["test"]),
            separability(challenger["fresh_global_mlp"]["test"]),
            separability(challenger["fresh_conditional_mlp"]["test"]),
        )

        # Weight each view by the residual domain signal it independently detects.
        updated_weights = dict(new["D_loss_weights"])
        for name in updated_weights:
            score = current_sep.get(name, mean_sep)
            if name == "local":
                updated_weights[name] = 1.0
            elif score >= 0.80:
                updated_weights[name] = 0.75
            elif score < 0.65:
                updated_weights[name] = 0.35
            else:
                updated_weights[name] = 0.5
        if updated_weights != new["D_loss_weights"]:
            new["D_loss_weights"] = updated_weights
            actions.append("REWEIGHT_DOMAIN_VIEWS")
            reasons.append("branch weights follow heldout threshold-free residual separability")

        current_accuracy = float(challenger["current_D_test"]["accuracy"])
        if fresh_sep >= 0.90 and mean_sep >= 0.75 and current_accuracy < 0.65:
            # A 50x jump to 1e-4 was observed to amplify encoder adversarial
            # gradients abruptly. Use a staged 10x calibration step instead.
            new["D_lr"] = 2e-5
            new["D_steps_per_G_step"] = 4
            actions.append("CALIBRATE_CAPABLE_D")
            reasons.append("current D ranks domains but is badly threshold-calibrated while fresh D is strong")
        elif fresh_sep - mean_sep > 0.15:
            new["D_lr"] = 1e-4
            new["D_steps_per_G_step"] = 4
            refresh = True
            actions.append("REFRESH_UNDERFIT_D")
            reasons.append("fresh challengers expose information current D does not capture")

        # Do not raise lambda_adv in the same review that changes D time scale.
        # The next observed gradient ratio must establish that the calibrated D
        # is safe before adversarial pressure can be reconsidered.

    changes = {
        key: {"old": old.get(key), "new": new.get(key)}
        for key in ("lambda_adv", "D_lr", "D_steps_per_G_step", "D_loss_weights")
        if old.get(key) != new.get(key)
    }
    return new, actions or ["KEEP"], reasons or ["all registered safety and adaptation gates stayed inactive"], refresh, changes


def apply_per_epoch_safety(config, metrics, d_optimizer, epoch, out, decisions):
    """Immediate Planet-free safety fuse for actions taken between full reviews."""
    ratio = float(metrics["effective_adv_to_cls_grad_ratio"])
    if not np.isfinite(ratio):
        raise RuntimeError(f"non-finite effective adversarial gradient ratio at epoch {epoch}")
    if ratio <= 0.5:
        return
    old = json.loads(json.dumps(config))
    # Aim below 0.25 on the next epoch and permit a low floor when D gradients
    # suddenly become much sharper after recalibration.
    config["lambda_adv"] = max(0.001, float(config["lambda_adv"]) * 0.25 / ratio)
    config["D_lr"] = max(BASE_LR, float(config["D_lr"]) * 0.5)
    config["D_steps_per_G_step"] = max(2, int(config["D_steps_per_G_step"]) - 1)
    for group in d_optimizer.param_groups:
        group["lr"] = config["D_lr"]
    decision = {
        "epoch": int(epoch),
        "action": ["PER_EPOCH_GRADIENT_SAFETY_FUSE"],
        "reasons": ["effective adversarial/classification gradient ratio exceeded 0.5"],
        "diagnosis": {"effective_grad_ratio": ratio},
        "parameter_changes": {
            key: {"old": old[key], "new": config[key]}
            for key in ("lambda_adv", "D_lr", "D_steps_per_G_step")
            if old[key] != config[key]
        },
        "planet_used": False,
    }
    decisions.append(decision)
    (out / f"decision_epoch_{epoch:03d}_safety.json").write_text(json.dumps(decision, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANT_DIRS), required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-epoch", type=int, default=16)
    ap.add_argument("--steps-per-epoch", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    if args.max_epoch != 16: raise ValueError("controlled comparison requires 16 epochs")
    seed_all(); random.seed(16); np.random.seed(16); torch.manual_seed(16)
    device = torch.device(args.device)
    out = FOLLOWUP / "gan_search" / VARIANT_DIRS[args.variant]
    config = {
        "variant": args.variant, "seed": 16, "lambda_adv": 0.05,
        "D_lr": BASE_LR, "main_lr": BASE_LR, "D_steps_per_G_step": 3,
        "steps_per_epoch": args.steps_per_epoch, "batch_size": args.batch_size,
        "lambda_diversity": 0.05 if args.variant == "v2d_diverse" else 0.0,
        "planet_used_in_training": False,
    }
    model, load_stats = make_model(device, PRETRAINED)
    discriminators, weights = build_discriminators(args.variant, device); config["D_loss_weights"] = weights
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=config["main_lr"], weight_decay=.01)
    d_optimizer = torch.optim.AdamW(discriminators.parameters(), lr=config["D_lr"], weight_decay=.01)
    source_names, target_names = read_lines(SOURCE_TRAIN), read_lines(TARGET_TRAIN)
    dataset = StrategyDataset(source_names, target_names, "none")
    rng = np.random.RandomState(16)
    dry = train_one_epoch(model, discriminators, model_optimizer, d_optimizer, dataset, len(source_names), len(target_names),
                          {**config, "steps_per_epoch": 1}, device, rng)
    dry.update({"update_performed": True, "finite": all(np.isfinite(value) for value in dry.values() if isinstance(value, float)),
                "load_stats": load_stats, "stage4_shape": list(stage4_map(model, torch.from_numpy(read_raster(TARGET_ROOT / target_names[0])[None]).to(device).float()).shape),
                "planet_used": False})
    out.mkdir(parents=True, exist_ok=True); (out / "dry_run.json").write_text(json.dumps(dry, indent=2))
    if args.dry_run:
        print(json.dumps(dry, indent=2)); return
    if (out / "metrics.jsonl").exists():
        raise FileExistsError(f"refuse to overwrite run: {out / 'metrics.jsonl'}")

    # Reinitialize after dry run so all full variants start from exactly the same source checkpoint and seed.
    seed_all(); np.random.seed(16); torch.manual_seed(16); rng = np.random.RandomState(16)
    model, load_stats = make_model(device, PRETRAINED)
    discriminators, weights = build_discriminators(args.variant, device); config["D_loss_weights"] = weights
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=config["main_lr"], weight_decay=.01)
    d_optimizer = torch.optim.AdamW(discriminators.parameters(), lr=config["D_lr"], weight_decay=.01)
    decisions = []
    review0 = review_diagnostics(model, read_lines(TARGET_FULL))
    challenger0 = heldout_challengers(model, discriminators, args.variant, weights, device)
    candidates = [{"epoch": 0, "checkpoint": str(out / "checkpoints/epoch_000_initial.pt"),
                   "review": review0, "challenger": challenger0, "metrics": None,
                   "config_at_checkpoint": json.loads(json.dumps(config))}]
    decisions.append({"epoch": 0, "action": "START", "diagnosis": "controlled Planet-free variant from common Ghana checkpoint", "config": config.copy()})
    save_checkpoint(out / "checkpoints/epoch_000_initial.pt", 0, model, discriminators, model_optimizer, d_optimizer, config, None, decisions)

    for epoch in range(1, args.max_epoch + 1):
        row = train_one_epoch(model, discriminators, model_optimizer, d_optimizer, dataset, len(source_names), len(target_names), config, device, rng)
        row.update({"epoch": epoch, "lambda_adv": config["lambda_adv"], "D_steps_per_G_step": config["D_steps_per_G_step"], "variant": args.variant})
        with (out / "metrics.jsonl").open("a") as handle: handle.write(json.dumps(row) + "\n")
        print(json.dumps({"variant": args.variant, "epoch": epoch, "D": row["D_acc"], "cls": row["cls"], "adv": row["adv"], "ratio": row["effective_adv_to_cls_grad_ratio"]}), flush=True)
        apply_per_epoch_safety(config, row, d_optimizer, epoch, out, decisions)
        if epoch in (8, 16):
            review = review_diagnostics(model, read_lines(TARGET_FULL))
            challenger = heldout_challengers(model, discriminators, args.variant, weights, device)
            source_c2 = review["source"]["f1_per_class"][2]
            config_at_checkpoint = json.loads(json.dumps(config))
            next_config, actions, reasons, refresh, changes = agent_decision(
                args.variant, epoch, config, weights, review0, review, challenger, row)
            decision = {"epoch": epoch, "action": actions, "reasons": reasons,
                        "diagnosis": {"current_D": challenger["current_D_test"]["accuracy"],
                                      "current_D_auc": challenger["current_D_test"]["auc"],
                                      "current_D_per_branch": challenger["current_D_per_branch_test"],
                                      "fresh_global": challenger["fresh_global_mlp"]["test"]["accuracy"],
                                      "fresh_conditional": challenger["fresh_conditional_mlp"]["test"]["accuracy"],
                                      "fresh_cdan": challenger["fresh_cdan_mlp"]["test"]["accuracy"],
                                      "gap_global": challenger["gap_global"], "source_class2_f1": source_c2,
                                      "effective_grad_ratio": row["effective_adv_to_cls_grad_ratio"]},
                        "D_G_time_scale": "D lagging" if challenger["gap_global"] > .1 else "balanced/capable",
                        "parameter_changes": changes, "planet_used": False,
                        "config_at_checkpoint": config_at_checkpoint, "next_config": next_config}
            decisions.append(decision); (out / f"decision_epoch_{epoch:03d}.json").write_text(json.dumps(decision, indent=2))
            checkpoint = out / f"checkpoints/accepted_epoch_{epoch:03d}.pt"
            save_checkpoint(checkpoint, epoch, model, discriminators, model_optimizer, d_optimizer, config_at_checkpoint, row, decisions)
            candidate_item = {"epoch": epoch, "checkpoint": str(checkpoint), "review": review,
                              "challenger": challenger, "metrics": row, "config_at_checkpoint": config_at_checkpoint}
            candidates.append(candidate_item)
            (out / f"review_epoch_{epoch:03d}.json").write_text(json.dumps(candidate_item, indent=2))

            config.clear(); config.update(next_config)
            weights.clear(); weights.update(config["D_loss_weights"])
            if refresh:
                reset_discriminators(discriminators)
                d_optimizer = torch.optim.AdamW(discriminators.parameters(), lr=config["D_lr"], weight_decay=.01)
            else:
                for group in d_optimizer.param_groups:
                    group["lr"] = config["D_lr"]

    selected, scores = select_checkpoint(candidates)
    final = out / "final"; final.mkdir(exist_ok=True)
    shutil.copy2(selected["checkpoint"], final / "selected_checkpoint.pt")
    selection = {
        "variant": args.variant, "selection_stage": "label-free before Planet post-hoc",
        "selected_epoch": selected["epoch"], "checkpoint": str(final / "selected_checkpoint.pt"),
        "candidate_scores": scores, "planet_used_for_selection": False,
        "selected_review": selected["review"], "selected_challenger": selected["challenger"],
        "mechanism": VARIANT_DIRS[args.variant],
        "config": selected.get("config_at_checkpoint", config),
        "terminal_training_config": config,
    }
    (final / "final_selection.json").write_text(json.dumps(selection, indent=2))
    print(json.dumps({"variant": args.variant, "selected_epoch": selected["epoch"], "scores": scores, "final": str(final / 'selected_checkpoint.pt')}, indent=2))


if __name__ == "__main__":
    main()
