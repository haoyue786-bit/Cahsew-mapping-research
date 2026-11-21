"""Shared, isolated utilities for the Ivoire 19e four-strategy experiment."""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from osgeo import gdal


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT.parent / "agent_autonomous_da_ivoire2019_tile0019e"
CASHEW = ROOT.parents[2]
EXP_CODE = CASHEW / "code/exp_convconnect"
for item in (OLD / "code", EXP_CODE, CASHEW / "code"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import base_train_tile as base  # noqa: E402
import epoch0_diagnostics as old_diag  # noqa: E402


SEED = 16
NUM_CLASSES = 4
IGNORE_INDEX = 10
CLASS_WEIGHTS = (3.0, 3.0, 10.0, 3.0)
BASE_LR = 2e-6
WEIGHT_DECAY = 0.01
MAX_EPOCH = 100
BATCH_SIZE = int(os.environ.get("SND_BATCH_SIZE", "16"))
PARTITION_SIZE = BATCH_SIZE * 10

SOURCE_IMAGE_ROOT = Path("@ASSETS@/code/mydataset2020/newdataset/AlphaEarth_2019")
SOURCE_LABEL_ROOT = Path("@ASSETS@/code/mydataset2020/newdataset/label_processed")
TARGET_ROOT = Path("@ASSETS@/alphaearth_data_split/2019")
PLANET_ROOT = Path("@WORKSPACE@/planetmap_split/Ivoire/Ivoire2017_0000000000000000019e")
PRETRAINED = Path("@ASSETS@/logs_try_mutil/0226_mutil_GHA_pretrain_200epoch/epoch_weights/epoch_173.pt")
TILE = "Ivoire2019_0000000000000000019e"
PLANET_TILE = "Ivoire2017_0000000000000000019e"
DATA = ROOT / "shared/data"
SOURCE_TRAIN = DATA / "source_train_2019.txt"
SOURCE_VALID = DATA / "source_valid_2019.txt"
TARGET_TRAIN = DATA / "target_train_Ivoire2019_0000000000000000019e.txt"
TARGET_FULL = DATA / "target_full_Ivoire2019_0000000000000000019e.txt"


def configure_old_base() -> None:
    base.SOURCE_IMAGE_ROOT = str(SOURCE_IMAGE_ROOT)
    base.SOURCE_LABEL_ROOT = str(SOURCE_LABEL_ROOT)
    base.SOURCE_IMAGE_PREFIX = "AlphaEarth_2019_"
    base.SOURCE_LABEL_PREFIX = "label_processed_"
    base.TARGET_IMAGE_ROOT = str(TARGET_ROOT)
    base.TARGET_LABEL_ROOT_BASE = "@WORKSPACE@/planetmap_split"
    base.TARGET_COUNTRY_NAMES = ["Ivoire"]
    base.TARGET_TILE_NAMES = [TILE]
    base.batch_size = BATCH_SIZE
    base.partition_size = PARTITION_SIZE
    base.n_epochs = MAX_EPOCH
    old_diag.DEVICE = torch.device("cuda:0")


configure_old_base()


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_lines(path: Path) -> List[str]:
    return [line.strip().split()[0] for line in path.read_text().splitlines() if line.strip()]


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, path)


def read_raster(path: Path, dtype=np.float32) -> np.ndarray:
    ds = gdal.Open(str(path))
    if ds is None:
        raise FileNotFoundError(path)
    arr = np.asarray(ds.ReadAsArray())
    ds = None
    if arr.ndim == 2:
        arr = arr[None, ...]
    if np.issubdtype(np.dtype(dtype), np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(dtype, copy=False)


def source_label_path(name: str) -> Path:
    stem = Path(name).name.replace("AlphaEarth_2019_", "", 1)
    return SOURCE_LABEL_ROOT / ("label_processed_" + stem)


def planet_path(target_rel: str) -> Path:
    mapped = Path(target_rel).name.replace(TILE + "_", PLANET_TILE + "_", 1)
    return PLANET_ROOT / mapped


def load_planet(target_rel: str) -> np.ndarray:
    arr = read_raster(planet_path(target_rel), dtype=np.int64)
    return arr[0]


def make_model(device: torch.device, checkpoint: Optional[Path] = None):
    model = base.backbone_64swin2_convconnect.STC_DA(out_classes=NUM_CLASSES, use_pretrained=False).to(device)
    stats = base.load_model_weights(model, str(checkpoint or PRETRAINED))
    return model, stats


def forward_fuse(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    delta = model.Swin(x)
    features = model.fuse_conv(torch.cat([x, delta], dim=1))
    return model.decoder(features), features


def domain_features(model, x: torch.Tensor) -> torch.Tensor:
    bottleneck = model.Swin(x, return_bottleneck=True)
    bottleneck = model.norm(bottleneck)
    return model.avgpool(bottleneck.transpose(1, 2)).flatten(1)


def class2_counts(pred: np.ndarray, true: np.ndarray) -> Dict[str, int]:
    valid = true != IGNORE_INDEX
    pred, true = pred[valid], true[valid]
    return {
        "tp2": int(np.sum((pred == 2) & (true == 2))),
        "fp2": int(np.sum((pred == 2) & (true != 2))),
        "fn2": int(np.sum((pred != 2) & (true == 2))),
        "valid_pixels": int(valid.sum()),
        "predicted_class2_pixels": int(np.sum(pred == 2)),
        "planet_class2_pixels": int(np.sum(true == 2)),
    }


def metrics_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn = counts["tp2"], counts["fp2"], counts["fn2"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return {"class2_precision": precision, "class2_recall": recall, "class2_f1": f1}


def make_optimizer(model, config: Dict[str, float], extras: Sequence[torch.nn.Module] = ()):
    main, domain = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (domain if name.startswith(("fc1_DA", "fc2_DA", "fc3_DA")) else main).append(param)
    for module in extras:
        domain.extend([p for p in module.parameters() if p.requires_grad])
    mult = float(config.get("main_lr_multiplier", 1.0))
    ratio = float(config.get("domain_lr_ratio", 1.0))
    return torch.optim.AdamW([
        {"params": main, "lr": BASE_LR * mult},
        {"params": domain, "lr": BASE_LR * mult * ratio},
    ], betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)


def set_optimizer_lr(optimizer, epoch: int, config: Dict[str, float]) -> float:
    progress = min(1.0, max(0.0, epoch / float(MAX_EPOCH)))
    cosine = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = BASE_LR * float(config.get("main_lr_multiplier", 1.0)) * cosine
    optimizer.param_groups[0]["lr"] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]["lr"] = lr * float(config.get("domain_lr_ratio", 1.0))
    return lr


def rng_state() -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: Optional[Dict[str, object]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def weighted_source_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor(CLASS_WEIGHTS, device=logits.device)
    losses = F.cross_entropy(logits, labels, weight=weights, ignore_index=IGNORE_INDEX, reduction="none")
    valid = labels != IGNORE_INDEX
    if not bool(valid.any()):
        return logits.sum() * 0
    label_valid = labels[valid]
    return losses[valid].sum() / weights[label_valid].sum().clamp_min(1e-8)


class StrategyDataset(torch.utils.data.Dataset):
    """Wrap the original source augmentation and append method target labels."""

    def __init__(self, source: Sequence[str], target: Sequence[str], mode: str,
                 pseudo_dir: Optional[Path] = None):
        self.source = list(source)
        self.target = list(target)
        self.mode = mode
        self.pseudo_dir = pseudo_dir
        self.inner = base.SegmentationDataset_AEF_SourceMapped(
            self.source, self.target, base.input_shape, NUM_CLASSES, True, base.VOCdevkit_path
        )
        base.configure_source_dataset(self.inner)
        self.inner.img_target = str(TARGET_ROOT)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        img_aug, src_label, src_raw, tgt_raw = self.inner[index]
        rel = self.target[index % len(self.target)]
        if self.mode == "conditional":
            return img_aug, src_label, src_raw, tgt_raw, load_planet(rel)
        if self.mode == "cluster":
            if self.pseudo_dir is None:
                raise RuntimeError("pseudo_dir required")
            item = np.load(self.pseudo_dir / (Path(rel).stem + ".npz"))
            return img_aug, src_label, src_raw, tgt_raw, item["labels"].astype(np.int64), item["weights"].astype(np.float32)
        return img_aug, src_label, src_raw, tgt_raw


def paired_partitions(source: Sequence[str], target: Sequence[str]):
    src_parts, _ = base.partitions(list(source), PARTITION_SIZE, rand=True)
    tgt_parts, _ = base.partitions(list(target), PARTITION_SIZE, rand=True)
    count = max(len(src_parts), len(tgt_parts))
    for idx in range(count):
        yield src_parts[idx % len(src_parts)], tgt_parts[idx % len(tgt_parts)]


def loader_for(source: Sequence[str], target: Sequence[str], mode: str,
               pseudo_dir: Optional[Path] = None):
    dataset = StrategyDataset(source, target, mode, pseudo_dir=pseudo_dir)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                                       num_workers=0, generator=torch.Generator())


def review_diagnostics(model, target_lines: Sequence[str]) -> Dict[str, object]:
    source_valid = read_lines(SOURCE_VALID)
    source = old_diag.evaluate_source(model, source_valid)
    source_features = old_diag.collect_source_features(model, source_valid, max_items=128)
    target_features, probs = old_diag.collect_target_features(model, list(target_lines), max_items=128)
    probe = old_diag.independent_probe(source_features, target_features)
    head = old_diag.domain_head_diagnostics(model, source_valid, list(target_lines))
    snd = old_diag.soft_neighborhood_density(target_features, temperature=0.05)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1).mean())
    return {
        "source": source,
        "target_unsupervised": {
            "snd": float(snd),
            "entropy": entropy,
            "class_distribution": probs.mean(axis=0).tolist(),
        },
        "independent_frozen_feature_probe": probe,
        "adversarial_domain_head": head,
    }
