#!/usr/bin/env python3
"""Held-out current-D and fresh challenger audit for a frozen Stage4 encoder."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
SHARED_CODE = ROOT.parent / "shared" / "code"
EXP5_CODE = ROOT / "exp05_agent_feature_gan"
sys.path.insert(0, str(SHARED_CODE))
sys.path.insert(0, str(EXP5_CODE / "discriminator"))
from common import (PRETRAINED, SOURCE_VALID, TARGET_FULL, TARGET_ROOT,
                    make_model, read_lines, read_raster, source_label_path)
from feature_gan import Stage4PatchDiscriminator


SOURCE_ROOT = Path("@ASSETS@/code/mydataset2020/newdataset/AlphaEarth_2019")
COORD_RE = re.compile(r"_(\d+)_(\d+)\.tif$", re.IGNORECASE)


class GlobalMLP(nn.Module):
    def __init__(self, channels=768):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(channels, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).flatten()


def stage4_map(model, x):
    tokens = model.Swin(x, return_bottleneck=True)
    if tokens.ndim == 3:
        tokens = model.norm(tokens)
        b, length, channels = tokens.shape
        side = int(math.sqrt(length))
        if side * side != length:
            raise ValueError(f"Stage4 tokens are not square: {tuple(tokens.shape)}")
        return tokens.transpose(1, 2).reshape(b, channels, side, side)
    if tokens.ndim == 4 and tokens.shape[1] == 768:
        return tokens
    if tokens.ndim == 4 and tokens.shape[-1] == 768:
        return tokens.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"unsupported Stage4 shape {tuple(tokens.shape)}")


def coord(rel):
    match = COORD_RE.search(Path(rel).name)
    return (int(match.group(1)), int(match.group(2))) if match else None


def split_names(names, domain, block_size=4):
    """Assign whole spatial blocks to train/valid/test deterministically."""
    blocks = {}
    for name in names:
        xy = coord(name)
        key = (xy[0] // block_size, xy[1] // block_size) if xy else (hash(name) % 100000, 0)
        blocks.setdefault(key, []).append(name)
    ordered = sorted(blocks)
    # Stable hash prevents Python hash randomization from affecting assignments.
    offset = int(hashlib.sha256(domain.encode()).hexdigest()[:8], 16) % 5
    assignment = {key: ("test" if (i + offset) % 5 == 0 else "valid" if (i + offset) % 5 == 1 else "train")
                  for i, key in enumerate(ordered)}
    out = {split: [] for split in ("train", "valid", "test")}
    for key, values in blocks.items():
        out[assignment[key]].extend(values)
    # Guarantee every split when the source set has very few blocks.  Tiny
    # partial target tiles can contain fewer than three spatial blocks (even
    # only one or two patches), so a distinct donor is impossible.  Reuse one
    # deterministic name for the read-only challenger probe; this does not
    # enter optimization, CORAL, or prediction and prevents ``max()`` from
    # failing on valid non-empty partial tiles.
    for split in ("train", "valid", "test"):
        if not out[split]:
            donors = [s for s in out if out[s]]
            if not donors:
                continue
            donor = max(donors, key=lambda s: len(out[s]))
            out[split].append(out[donor][0])
    return out, assignment


def read_image(rel, source):
    return read_raster((SOURCE_ROOT if source else TARGET_ROOT) / rel)


def extract_features(model, names, source, device, batch_size):
    maps, pooled = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(names), batch_size):
            part = names[start:start + batch_size]
            x = torch.from_numpy(np.stack([read_image(name, source) for name in part])).to(device).float()
            fmap = stage4_map(model, x).cpu().numpy().astype(np.float32)
            maps.append(fmap)
            pooled.append(fmap.mean(axis=(2, 3)))
            if start == 0 or (start // batch_size) % 50 == 0:
                print(f"extract {'source' if source else 'target'}: {min(start + batch_size, len(names))}/{len(names)}", flush=True)
    return np.concatenate(maps), np.concatenate(pooled)


def balanced_arrays(features, splits, source_names, target_names, split):
    source_idx = [source_names.index(x) for x in splits["source"][split]]
    target_idx = [target_names.index(x) for x in splits["target"][split]]
    n = min(len(source_idx), len(target_idx))
    source_idx, target_idx = source_idx[:n], target_idx[:n]
    return source_idx, target_idx


def binary_report(y_true, logits):
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
    pred = (prob >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else None,
        "n": int(len(y_true)),
        "positive_rate": float(pred.mean()),
    }


def patch_logits(model, x):
    logits = model(x)
    if logits.ndim > 1:
        logits = logits.mean(dim=tuple(range(1, logits.ndim)))
    return logits.flatten()


def train_torch_probe(model, train_x, train_y, valid_x, valid_y, device, epochs=120):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xtr = torch.from_numpy(train_x).to(device).float()
    ytr = torch.from_numpy(train_y).to(device).float()
    xv = torch.from_numpy(valid_x).to(device).float()
    yv = torch.from_numpy(valid_y).to(device).float()
    best_state, best_valid, patience = None, float("inf"), 0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(patch_logits(model, xtr), ytr)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(F.binary_cross_entropy_with_logits(patch_logits(model, xv), yv).item())
        if vloss < best_valid - 1e-5:
            best_valid, patience = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_logits = patch_logits(model, torch.from_numpy(train_x).to(device).float()).cpu().numpy()
        valid_logits = patch_logits(model, torch.from_numpy(valid_x).to(device).float()).cpu().numpy()
    return binary_report(train_y, train_logits), binary_report(valid_y, valid_logits)


def eval_current_d(disc, test_maps, test_y, device):
    disc.eval()
    with torch.no_grad():
        logits = patch_logits(disc, torch.from_numpy(test_maps).to(device).float()).cpu().numpy()
    return binary_report(test_y, logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--feature-cache", type=Path)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage4 extraction")
    torch.manual_seed(16); np.random.seed(16)
    device = torch.device(args.device)
    source_names = read_lines(SOURCE_VALID)
    target_names = read_lines(TARGET_FULL)
    source_split, source_blocks = split_names(source_names, "source", args.block_size)
    target_split, target_blocks = split_names(target_names, "target", args.block_size)
    splits = {"source": source_split, "target": target_split}
    model, load_stats = make_model(device, args.checkpoint)
    cache = args.feature_cache or args.output.with_suffix(".features.npz")
    if cache.exists():
        saved = np.load(cache)
        source_maps = saved["source_maps"].astype(np.float32)
        target_maps = saved["target_maps"].astype(np.float32)
        source_global = saved["source_global"].astype(np.float32)
        target_global = saved["target_global"].astype(np.float32)
        print(f"loaded feature cache: {cache}", flush=True)
    else:
        source_maps, source_global = extract_features(model, source_names, True, device, args.batch_size)
        target_maps, target_global = extract_features(model, target_names, False, device, args.batch_size)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, source_maps=source_maps.astype(np.float16), target_maps=target_maps.astype(np.float16),
                 source_global=source_global, target_global=target_global)
        print(f"saved feature cache: {cache}", flush=True)
    feature_maps = {"source": source_maps, "target": target_maps}
    feature_global = {"source": source_global, "target": target_global}

    train_si, train_ti = balanced_arrays(feature_global, splits, source_names, target_names, "train")
    valid_si, valid_ti = balanced_arrays(feature_global, splits, source_names, target_names, "valid")
    test_si, test_ti = balanced_arrays(feature_global, splits, source_names, target_names, "test")
    def join(arr, si, ti):
        return np.concatenate([arr["source"][si], arr["target"][ti]])
    y_train = np.r_[np.zeros(len(train_si), dtype=np.int64), np.ones(len(train_ti), dtype=np.int64)]
    y_valid = np.r_[np.zeros(len(valid_si), dtype=np.int64), np.ones(len(valid_ti), dtype=np.int64)]
    y_test = np.r_[np.zeros(len(test_si), dtype=np.int64), np.ones(len(test_ti), dtype=np.int64)]

    train_global = join(feature_global, train_si, train_ti)
    valid_global = join(feature_global, valid_si, valid_ti)
    test_global = join(feature_global, test_si, test_ti)
    train_maps = join(feature_maps, train_si, train_ti)
    valid_maps = join(feature_maps, valid_si, valid_ti)
    test_maps = join(feature_maps, test_si, test_ti)

    challenger_local = Stage4PatchDiscriminator().to(device)
    local_train, local_valid = train_torch_probe(challenger_local, train_maps, y_train, valid_maps, y_valid, device)
    with torch.no_grad():
        local_test_logits = patch_logits(challenger_local, torch.from_numpy(test_maps).to(device).float()).cpu().numpy()
    local_test = binary_report(y_test, local_test_logits)

    challenger_global = GlobalMLP().to(device)
    global_train, global_valid = train_torch_probe(challenger_global, train_global, y_train, valid_global, y_valid, device)
    with torch.no_grad():
        global_test_logits = challenger_global(torch.from_numpy(test_global).to(device).float()).cpu().numpy()
    global_test = binary_report(y_test, global_test_logits)

    linear = LogisticRegression(max_iter=500, solver="liblinear", random_state=16)
    linear.fit(train_global, y_train)
    linear_train = binary_report(y_train, linear.decision_function(train_global))
    linear_valid = binary_report(y_valid, linear.decision_function(valid_global))
    linear_test = binary_report(y_test, linear.decision_function(test_global))

    payload = torch.load(args.checkpoint, map_location="cpu")
    current = Stage4PatchDiscriminator().to(device)
    current_branch = "legacy_local"
    if isinstance(payload, dict) and "discriminator_state_dict" in payload:
        current_state = payload["discriminator_state_dict"]
    elif isinstance(payload, dict) and "discriminators_state_dict" in payload:
        packed = payload["discriminators_state_dict"]
        current_state = {
            key[len("local."):]: value for key, value in packed.items()
            if key.startswith("local.")
        }
        current_branch = "local"
        if not current_state:
            raise KeyError("checkpoint has no local discriminator branch for current-D comparison")
    else:
        raise KeyError("checkpoint has no supported discriminator state")
    current.load_state_dict(current_state)
    current_test = eval_current_d(current, test_maps, y_test, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(args.checkpoint), "selected_epoch": payload.get("epoch"),
        "protocol": "patch-disjoint spatial-block split; balanced source/target per split",
        "block_size": args.block_size, "seed": 16,
        "source_count": len(source_names), "target_count": len(target_names),
        "split_counts": {domain: {s: len(v) for s, v in values.items()} for domain, values in splits.items()},
        "balanced_counts": {"train_each_domain": len(train_si), "valid_each_domain": len(valid_si), "test_each_domain": len(test_si)},
        "feature_shape_source": list(source_maps.shape), "feature_shape_target": list(target_maps.shape),
        "load_stats": load_stats,
        "current_D_branch": current_branch,
        "current_D_heldout_test": current_test,
        "fresh_same_arch_local": {"train": local_train, "valid": local_valid, "test": local_test},
        "fresh_global_mlp": {"train": global_train, "valid": global_valid, "test": global_test},
        "fresh_linear_probe": {"train": linear_train, "valid": linear_valid, "test": linear_test},
    }
    result["gaps"] = {
        "same_arch_test_minus_current_D": local_test["accuracy"] - current_test["accuracy"],
        "global_test_minus_current_D": global_test["accuracy"] - current_test["accuracy"],
        "linear_test_minus_current_D": linear_test["accuracy"] - current_test["accuracy"],
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
