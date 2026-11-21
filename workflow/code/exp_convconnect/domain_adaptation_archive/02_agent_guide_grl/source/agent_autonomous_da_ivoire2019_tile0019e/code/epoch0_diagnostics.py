"""No-update epoch-0 diagnostics for the Ivoire 2019 autonomous run.

This file deliberately has no reference-label import.  It loads the verified
2019 source/target image lists, evaluates the pretrained model, measures an
independent frozen-feature domain probe, and computes unit gradient norms.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from osgeo import gdal

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import base_train_tile as base
from snd_utils import soft_neighborhood_density


SEED = 16
NUM_CLASSES = 4
DEVICE = torch.device("cuda:0")
SOURCE_ROOT = Path("@ASSETS@/code/mydataset2020/newdataset/AlphaEarth_2019")
LABEL_ROOT = Path("@ASSETS@/code/mydataset2020/newdataset/label_processed")
TARGET_ROOT = Path("@ASSETS@/alphaearth_data_split/2019")
EXP = Path(__file__).resolve().parents[1]
SOURCE_TRAIN = EXP / "data/source_train_2019.txt"
SOURCE_VALID = EXP / "data/source_valid_2019.txt"
TARGET_TRAIN = EXP / "data/target_train_Ivoire2019_0000000000000000019e.txt"
PRETRAINED = Path("@ASSETS@/logs_try_mutil/0226_mutil_GHA_pretrain_200epoch/epoch_weights/epoch_173.pt")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_lines(path):
    return [x.strip().split()[0] for x in Path(path).read_text().splitlines() if x.strip()]


def source_label_path(name):
    suffix = Path(name).name.replace("AlphaEarth_2019_", "", 1)
    return LABEL_ROOT / ("label_processed_" + suffix)


def read_array(path):
    ds = gdal.Open(str(path))
    if ds is None:
        raise FileNotFoundError(str(path))
    arr = ds.ReadAsArray()
    ds = None
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def source_sample(name):
    return read_array(SOURCE_ROOT / name), read_array(source_label_path(name))[0].astype(np.int64)


def target_sample(name):
    return read_array(TARGET_ROOT / name)


def batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def as_tensor(arrays):
    return torch.from_numpy(np.stack(arrays, axis=0)).to(DEVICE).float()


def weighted_seg_loss(logits, labels):
    weights = torch.tensor([3.0, 3.0, 10.0, 3.0], device=DEVICE)
    loss = F.cross_entropy(logits, labels, weight=weights, ignore_index=10, reduction="none")
    valid = labels != 10
    return loss[valid].mean()


def evaluate_source(model, names, batch_size=8):
    model.eval()
    pred_all, true_all, losses = [], [], []
    with torch.no_grad():
        for group in batches(names, batch_size):
            xs, ys = zip(*(source_sample(n) for n in group))
            x = as_tensor(xs)
            y = torch.from_numpy(np.stack(ys)).to(DEVICE).long()
            logits = model(x)
            losses.append(float(weighted_seg_loss(logits, y).item()))
            pred_all.append(logits.argmax(1).cpu().numpy().reshape(-1))
            true_all.append(y.cpu().numpy().reshape(-1))
    pred = np.concatenate(pred_all)
    true = np.concatenate(true_all)
    valid = true != 10
    pred, true = pred[valid], true[valid]
    f1 = []
    for cls in range(NUM_CLASSES):
        tp = int(np.sum((pred == cls) & (true == cls)))
        fp = int(np.sum((pred == cls) & (true != cls)))
        fn = int(np.sum((pred != cls) & (true == cls)))
        f1.append(float(2 * tp / max(1, 2 * tp + fp + fn)))
    counts = np.bincount(pred, minlength=NUM_CLASSES).astype(np.int64)
    true_counts = np.bincount(true, minlength=NUM_CLASSES).astype(np.int64)
    return {
        "loss": float(np.mean(losses)),
        "f1_per_class": f1,
        "macro_f1": float(np.mean(f1)),
        "predicted_class_counts": counts.tolist(),
        "true_class_counts": true_counts.tolist(),
        "pixels": int(len(true)),
    }


def collect_target_features(model, names, max_items=256, batch_size=8):
    names = names[:max_items]
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = model.fuse_conv.register_forward_hook(hook)
    probs, features = [], []
    model.eval()
    with torch.no_grad():
        for group in batches(names, batch_size):
            x = as_tensor([target_sample(n) for n in group])
            captured.clear()
            logits = model(x)
            feat = captured[-1]
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            features.append(feat.cpu().numpy())
            probs.append(torch.softmax(logits, dim=1).mean(dim=(2, 3)).cpu().numpy())
    handle.remove()
    return np.concatenate(features), np.concatenate(probs)


def collect_source_features(model, names, max_items=256, batch_size=8):
    names = names[:max_items]
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = model.fuse_conv.register_forward_hook(hook)
    features = []
    model.eval()
    with torch.no_grad():
        for group in batches(names, batch_size):
            x = as_tensor([source_sample(n)[0] for n in group])
            captured.clear()
            _ = model(x)
            feat = F.adaptive_avg_pool2d(captured[-1], 1).flatten(1)
            features.append(feat.cpu().numpy())
    handle.remove()
    return np.concatenate(features)


def domain_head_diagnostics(model, source_names, target_names, batch_size=8):
    source_names, target_names = source_names[:64], target_names[:64]
    logits_all = []
    model.eval()
    with torch.no_grad():
        for sgroup, tgroup in zip(batches(source_names, batch_size), batches(target_names, batch_size)):
            x = torch.cat([as_tensor([source_sample(n)[0] for n in sgroup]),
                           as_tensor([target_sample(n) for n in tgroup])], dim=0)
            logits_all.append(model(x, alpha=0.0, DA=True).cpu())
    logits = torch.cat(logits_all)
    labels = torch.cat([torch.zeros(len(source_names)), torch.ones(len(target_names))]).long()
    # The two loops above are paired and therefore equal in length for the
    # selected 64/64 samples.
    labels = torch.cat([torch.zeros(64), torch.ones(64)]).long()
    loss = F.cross_entropy(logits, labels)
    pred = logits.argmax(1)
    return {"loss": float(loss.item()), "accuracy": float((pred == labels).float().mean().item()),
            "n": int(len(labels))}


def independent_probe(source_features, target_features):
    n = min(len(source_features), len(target_features), 128)
    X = np.concatenate([source_features[:n], target_features[:n]], axis=0)
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    split = max(2, int(0.75 * len(y)))
    train_idx = np.r_[np.arange(0, int(0.75 * n)), np.arange(n, n + int(0.75 * n))]
    val_idx = np.setdiff1d(np.arange(len(y)), train_idx)
    clf = LogisticRegression(max_iter=200, random_state=SEED, solver="liblinear")
    clf.fit(X[train_idx], y[train_idx])
    probs = clf.predict_proba(X[val_idx])
    pred = clf.predict(X[val_idx])
    return {"accuracy": float(accuracy_score(y[val_idx], pred)),
            "loss": float(log_loss(y[val_idx], probs, labels=[0, 1])),
            "train_n": int(len(train_idx)), "validation_n": int(len(val_idx)),
            "feature_dim": int(X.shape[1]), "model": "sklearn.LogisticRegression",
            "source_target_n_each": int(n), "random_state": SEED}


def gradient_diagnostics(model, source_names, target_names):
    model.train()
    source_names, target_names = source_names[:4], target_names[:4]
    sx, sy = zip(*(source_sample(n) for n in source_names))
    tx = [target_sample(n) for n in target_names]
    x_s, y_s = as_tensor(sx), torch.from_numpy(np.stack(sy)).to(DEVICE).long()
    x_t = as_tensor(tx)
    enc = [p for n, p in model.named_parameters() if n.startswith("Swin.") and p.requires_grad]
    seg = weighted_seg_loss(model(x_s), y_s)
    # alpha=1 measures the unit adversarial gradient.  This is a diagnostic
    # only: no optimizer step is performed and the GRL is not used to update
    # the model here.
    da_logits = model(torch.cat([x_s, x_t], dim=0), alpha=1.0, DA=True)
    da_labels = torch.cat([torch.zeros(len(x_s)), torch.ones(len(x_t))]).long().to(DEVICE)
    da = F.cross_entropy(da_logits, da_labels)
    seg_g = torch.autograd.grad(seg, enc, retain_graph=True, allow_unused=True)
    da_g = torch.autograd.grad(da, enc, retain_graph=False, allow_unused=True)

    def norm(grads):
        value = sum(float(g.detach().pow(2).sum().item()) for g in grads if g is not None)
        return math.sqrt(value)

    gs, gd = norm(seg_g), norm(da_g)
    return {"seg_loss": float(seg.item()), "unit_da_loss": float(da.item()),
            "seg_gradient_norm": gs, "unit_da_gradient_norm": gd,
            "raw_gradient_ratio_da_over_seg": float(gd / max(gs, 1e-12)),
            "formal_optimizer_update": False, "encoder_parameter_count": len(enc)}


def rng_payload():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(EXP / "run"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU0 is not visible; refusing to run diagnostics on CPU")
    seed_all(SEED)
    base.SOURCE_IMAGE_ROOT = str(SOURCE_ROOT)
    base.SOURCE_LABEL_ROOT = str(LABEL_ROOT)
    base.SOURCE_IMAGE_PREFIX = "AlphaEarth_2019_"
    base.TARGET_IMAGE_ROOT = str(TARGET_ROOT)
    source_train, source_valid = read_lines(SOURCE_TRAIN), read_lines(SOURCE_VALID)
    target = read_lines(TARGET_TRAIN)

    model = base.backbone_64swin2_convconnect.STC_DA(out_classes=NUM_CLASSES, use_pretrained=False).to(DEVICE)
    load_stats = base.load_model_weights(model, str(PRETRAINED))
    source_eval = evaluate_source(model, source_valid)
    source_features = collect_source_features(model, source_valid, max_items=128)
    target_features, target_probs = collect_target_features(model, target, max_items=128)
    probe = independent_probe(source_features, target_features)
    domain_head = domain_head_diagnostics(model, source_valid, target)
    gradient = gradient_diagnostics(model, source_train, target)
    pred_counts = np.rint(target_probs.mean(axis=0) * len(target_probs) * 1).astype(float).tolist()
    class_distribution = target_probs.mean(axis=0).tolist()
    entropy = float(-(target_probs * np.log(np.clip(target_probs, 1e-8, 1.0))).sum(axis=1).mean())
    snd = soft_neighborhood_density(target_features, temperature=0.05)
    state = {
        "schema_version": 1,
        "state_type": "epoch_000_initial_no_update",
        "epoch": 0,
        "seed": SEED,
        "scope": {"country": "Ivoire", "tile": "0000000000000000019e", "target_year": 2019},
        "source": {"year": 2019, "train_n": len(source_train), "valid_n": len(source_valid), "validation": source_eval},
        "target_unsupervised": {"n": len(target), "sampled_n": len(target_features),
                                "mean_prediction_class_distribution": class_distribution,
                                "mean_prediction_entropy": entropy,
                                "soft_neighborhood_density": snd,
                                "feature_class_proxy_counts": pred_counts},
        "adversarial_domain_head": domain_head,
        "independent_frozen_feature_probe": probe,
        "unit_gradient_diagnostics": gradient,
        "pretrained_load": load_stats,
        "update_count": 0,
        "online_label_access": False,
        "rng_saved_in_checkpoint": True
    }
    out = Path(args.output)
    (out / "states").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "states/epoch_000_initial_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))
    main_params, domain_params = [], []
    for name, parameter in model.named_parameters():
        if name.startswith(("fc1_DA", "fc2_DA", "fc3_DA")):
            domain_params.append(parameter)
        else:
            main_params.append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": main_params, "lr": 2e-6},
        {"params": domain_params, "lr": 2e-6},
    ], betas=(0.9, 0.999), weight_decay=0.01)
    checkpoint = {"epoch": 0, "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                  "scheduler_state_dict": {"epoch": 0, "kind": "agent-controlled-review-scheduler"},
                  "rng_state": rng_payload(), "policy_state": {"accepted_path": [0]}, "diagnostic_state": state}
    torch.save(checkpoint, out / "checkpoints/epoch_000_initial.pt")
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
