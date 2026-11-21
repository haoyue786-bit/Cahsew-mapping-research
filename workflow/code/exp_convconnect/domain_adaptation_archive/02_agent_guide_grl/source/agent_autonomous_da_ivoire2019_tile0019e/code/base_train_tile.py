import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import traceback
import warnings

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from osgeo import gdal
from sklearn.metrics import f1_score
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(CURRENT_DIR)
CODE_DIR = os.path.dirname(EXPERIMENT_DIR)
if EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENT_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
CASHEW_CODE_ROOT = os.path.dirname(CODE_DIR)
if CASHEW_CODE_ROOT not in sys.path:
    sys.path.insert(0, CASHEW_CODE_ROOT)

from utils.training import get_lr_scheduler, set_optimizer_lr
from utils.utils import seed_everything
from reffer.utils import *
import backbone_64swin2_convconnect
from tool.gradient import *
from reffer.run_info import print_run_info

warnings.filterwarnings("ignore", category=FutureWarning)

batch_size = int(os.environ.get("SND_BATCH_SIZE", "32"))
partition_size = batch_size * 10
n_epochs = 100
VOCdevkit_path = "@ASSETS@/code/mydataset2020/newdataset"
SOURCE_IMAGE_ROOT = os.path.join(VOCdevkit_path, "AlphaEarth_2019")
SOURCE_LABEL_ROOT = os.path.join(VOCdevkit_path, "label_processed")
SOURCE_IMAGE_PREFIX = "AlphaEarth_2019_"
SOURCE_LABEL_PREFIX = "label_processed_"
num_classes = 4
cashew_class_id = 2

TARGET_IMAGE_ROOT = os.environ.get(
    "CASHEW_TARGET_IMAGE_ROOT",
    "@ASSETS@/alphaearth_data_split/2018",
)
TARGET_LABEL_ROOT_BASE = "@WORKSPACE@/planetmap_split"
PRETRAINED_WEIGHT_PATH = "@ASSETS@/logs_try_mutil/0226_mutil_GHA_pretrain_200epoch/epoch_weights/epoch_173.pt"
PREPARED_TXT_VERSION = 3

TARGET_SAMPLES_PER_TILE = 1000
DA_VAL_EVERY_PRE_50 = 5
DA_VAL_EVERY_POST_50 = 3
EARLY_STOP_MIN_EPOCH = 50
EARLY_STOP_EVAL_PATIENCE = 4
MIN_DA_VAL_IMPROVEMENT = 1e-8
TARGET_COUNTRY_NAMES = ["Benin", "Ghana", "Nigeria"]
TARGET_TILE_NAMES = [
    "Benin2018_00000000000000000248",
    "Ghana2018_0000000000000000012d",
    "Nigeria2018_000000000000000000b7",
]

optimizer_type = "adamw"
momentum = 0.9
weight_decay = 0.01
Init_lr = 2e-6
Min_lr = Init_lr * 0.1
lr_decay_type = "cos"
start_epoch = 1


def _read_bool_env(env_name, default):
    value = os.environ.get(env_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


resume_training = _read_bool_env("CASHEW_RESUME_TRAINING", True)
resume_unfinished_only = _read_bool_env("CASHEW_RESUME_UNFINISHED_ONLY", True)
reuse_prepared_txt = _read_bool_env("CASHEW_REUSE_PREPARED_TXT", True)

EPOCH_CKPT_RE = re.compile(r"^epoch_(\d+)\.pt$")


def _default_metric_array():
    return np.full(num_classes, np.nan, dtype=np.float32)


def _safe_float(value, default):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_metric_array(values):
    arr = _default_metric_array()
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        return arr
    for idx in range(min(len(values), num_classes)):
        arr[idx] = _safe_float(values[idx], float("nan"))
    return arr


def _default_training_state():
    return {
        "best_da_val_loss": float("inf"),
        "best_da_val_f1": _default_metric_array(),
        "best_epoch": None,
        "no_improve_eval_count": 0,
        "last_da_val_loss": float("nan"),
        "last_da_val_f1": _default_metric_array(),
    }


def _normalize_training_state(training_state):
    if not isinstance(training_state, dict):
        return _default_training_state()

    normalized = _default_training_state()
    normalized["best_da_val_loss"] = _safe_float(training_state.get("best_da_val_loss"), float("inf"))
    normalized["best_da_val_f1"] = _normalize_metric_array(training_state.get("best_da_val_f1"))
    normalized["best_epoch"] = _safe_int(training_state.get("best_epoch"), None)
    normalized["no_improve_eval_count"] = _safe_int(training_state.get("no_improve_eval_count"), 0) or 0
    normalized["last_da_val_loss"] = _safe_float(training_state.get("last_da_val_loss"), float("nan"))
    normalized["last_da_val_f1"] = _normalize_metric_array(training_state.get("last_da_val_f1"))
    return normalized


def _clean_train_log_df(df):
    if df.empty or "epoch" not in df.columns:
        return df.iloc[0:0].copy()

    clean_df = df.copy()
    clean_df["epoch"] = pd.to_numeric(clean_df["epoch"], errors="coerce")
    clean_df = clean_df[clean_df["epoch"].notna()].copy()
    if clean_df.empty:
        return clean_df

    clean_df["epoch"] = clean_df["epoch"].astype(int)
    clean_df = clean_df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last")
    clean_df.reset_index(drop=True, inplace=True)
    return clean_df


def load_training_state_from_csv(csv_path, upto_epoch=None):
    state = _default_training_state()
    state["csv_rows"] = 0

    if not os.path.exists(csv_path):
        return state

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return state

    df = _clean_train_log_df(df)
    if upto_epoch is not None:
        df = df[df["epoch"] <= int(upto_epoch)].copy()

    state["csv_rows"] = len(df)
    if df.empty:
        return state

    last_row = df.iloc[-1]
    state["last_da_val_loss"] = _safe_float(last_row.get("da_val_loss"), float("nan"))
    state["last_da_val_f1"] = _normalize_metric_array(
        [last_row.get(f"da_val_score_{idx}") for idx in range(num_classes)]
    )
    state["best_da_val_loss"] = _safe_float(last_row.get("best_da_val_loss_min"), float("inf"))
    state["best_epoch"] = _safe_int(last_row.get("best_epoch"), None)
    state["no_improve_eval_count"] = _safe_int(last_row.get("no_improve_eval_count"), 0) or 0

    if state["best_epoch"] is not None:
        best_rows = df[df["epoch"] == state["best_epoch"]]
        if not best_rows.empty:
            best_row = best_rows.iloc[-1]
            state["best_da_val_f1"] = _normalize_metric_array(
                [best_row.get(f"da_val_score_{idx}") for idx in range(num_classes)]
            )
        else:
            fallback_best = _default_metric_array()
            fallback_best[cashew_class_id] = _safe_float(last_row.get("best_da_val_score_2"), float("nan"))
            state["best_da_val_f1"] = fallback_best

    return state


def prepare_train_log_for_resume(csv_path, next_epoch):
    if not os.path.exists(csv_path):
        return 0

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        os.remove(csv_path)
        return 0

    df = _clean_train_log_df(df)
    if next_epoch <= start_epoch:
        df = df.iloc[0:0].copy()
    else:
        df = df[df["epoch"] < int(next_epoch)].copy()

    if df.empty:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        return 0

    df.to_csv(csv_path, index=False)
    return len(df)


def is_raw_state_dict(payload):
    return isinstance(payload, dict) and payload and all(torch.is_tensor(v) for v in payload.values())


def normalize_checkpoint_payload(payload):
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        return payload
    if is_raw_state_dict(payload):
        return {"state_dict": payload}
    raise ValueError("unsupported checkpoint format")


def load_state_dict_compatible(model, state_dict):
    model_dict = model.state_dict()

    new_dict = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            new_dict[k] = v
        elif k.startswith("module.") and k[7:] in model_dict and model_dict[k[7:]].shape == v.shape:
            new_dict[k[7:]] = v
        elif ("module." + k) in model_dict and model_dict["module." + k].shape == v.shape:
            new_dict["module." + k] = v
        else:
            skipped.append(k)

    model_dict.update(new_dict)
    model.load_state_dict(model_dict, strict=False)

    return {
        "loaded_params": len(new_dict),
        "ckpt_params": len(state_dict),
        "skipped_keys_count": len(skipped),
        "skipped_keys_head": skipped[:30],
    }


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def extract_epoch_from_checkpoint_name(path):
    match = EPOCH_CKPT_RE.match(os.path.basename(path))
    if not match:
        return None
    return int(match.group(1))


def find_latest_epoch_checkpoint(epoch_ckpt_dir):
    if not os.path.isdir(epoch_ckpt_dir):
        return None

    latest_epoch = None
    latest_path = None
    for file_name in os.listdir(epoch_ckpt_dir):
        match = EPOCH_CKPT_RE.match(file_name)
        if not match:
            continue
        epoch = int(match.group(1))
        if latest_epoch is None or epoch > latest_epoch:
            latest_epoch = epoch
            latest_path = os.path.join(epoch_ckpt_dir, file_name)

    if latest_epoch is None:
        return None

    return {
        "epoch": latest_epoch,
        "path": latest_path,
    }


def load_json_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        try:
            obj, end = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            raise exc
        trailing = text[end:].strip()
        if trailing:
            print(f"[WARN] ignored trailing data in json: {json_path}")
        return obj


def write_json_atomic(json_path, data):
    tmp_path = f"{json_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, json_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_existing_tile_summary(tile_log_dir):
    summary_path = os.path.join(tile_log_dir, "tile_summary.json")
    if not os.path.exists(summary_path):
        return None
    return load_json_file(summary_path)


def load_existing_tile_meta(tile_log_dir):
    meta_path = os.path.join(tile_log_dir, "tile_meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        return load_json_file(meta_path)
    except json.JSONDecodeError as exc:
        print(f"[WARN] ignore invalid tile_meta.json: {meta_path} ({exc})")
        return None


def is_tile_summary_complete(tile_summary):
    if not isinstance(tile_summary, dict):
        return False
    if tile_summary.get("training_complete") is True:
        return True

    stop_reason = str(tile_summary.get("stop_reason") or "")
    epochs_completed = _safe_int(tile_summary.get("epochs_completed"), 0) or 0
    return stop_reason.startswith("early_stop_no_improve_") or epochs_completed >= n_epochs


def save_training_checkpoint(model, optimizer, epoch, path, training_state=None, extra_state=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_state": {
            "best_da_val_loss": float(training_state["best_da_val_loss"]),
            "best_da_val_f1": [float(x) for x in training_state["best_da_val_f1"]],
            "best_epoch": training_state["best_epoch"],
            "no_improve_eval_count": int(training_state["no_improve_eval_count"]),
            "last_da_val_loss": float(training_state["last_da_val_loss"]),
            "last_da_val_f1": [float(x) for x in training_state["last_da_val_f1"]],
        } if training_state is not None else None,
    }
    if extra_state:
        checkpoint["extra_state"] = extra_state
    torch.save(checkpoint, path)


def load_model_weights(model, weight_path):
    checkpoint = normalize_checkpoint_payload(torch.load(weight_path, map_location="cpu"))
    stats = load_state_dict_compatible(model, checkpoint["state_dict"])
    stats["checkpoint_path"] = weight_path
    stats["checkpoint_epoch"] = _safe_int(checkpoint.get("epoch"), None)
    stats["has_optimizer_state"] = isinstance(checkpoint.get("optimizer_state_dict"), dict)
    return stats


def load_lines(file_path):
    with open(file_path, "r") as f:
        return f.readlines()


def write_lines(file_path, lines):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.writelines(lines)


def map_source_label_name(name):
    normalized_name = os.path.basename(os.path.normpath(name))
    stem, ext = os.path.splitext(normalized_name)
    if stem.startswith(SOURCE_IMAGE_PREFIX):
        suffix = stem[len(SOURCE_IMAGE_PREFIX):]
    else:
        suffix = stem
    return f"{SOURCE_LABEL_PREFIX}{suffix}{ext}"


def map_da_val_label_relpath(name, country_name):
    normalized_name = os.path.normpath(name)
    base_name = os.path.basename(normalized_name)
    stem, ext = os.path.splitext(base_name)
    parts = stem.split("_")
    if len(parts) < 4:
        return normalized_name

    # Parse from the end so country names like "Burkina_Faso" remain intact.
    tile_id = parts[-3]
    patch_suffix = "_".join(parts[-2:])
    label_tile_dir = f"{country_name}2017_{tile_id}"
    label_file_name = f"{label_tile_dir}_{patch_suffix}{ext}"
    return os.path.join(label_tile_dir, label_file_name)


def get_target_label_root(country_name):
    return os.path.join(TARGET_LABEL_ROOT_BASE, country_name)


def count_class_pixels(label_path, class_id):
    ds = gdal.Open(label_path)
    if ds is None:
        raise FileNotFoundError(f"failed to open label: {label_path}")
    arr = ds.ReadAsArray()
    ds = None
    if arr.ndim > 2:
        arr = arr[0]
    return int(np.sum(arr == class_id))


def is_all_zero_image(image_path):
    ds = gdal.Open(image_path)
    if ds is None:
        raise FileNotFoundError(f"failed to open image: {image_path}")
    band = ds.GetRasterBand(1)
    if band is None:
        ds = None
        raise RuntimeError(f"missing first band: {image_path}")
    arr = band.ReadAsArray()
    ds = None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return not bool(np.any(arr != 0))


def discover_target_tile_groups():
    country_names = sorted(
        name for name in os.listdir(TARGET_IMAGE_ROOT)
        if os.path.isdir(os.path.join(TARGET_IMAGE_ROOT, name))
    )
    if TARGET_COUNTRY_NAMES:
        wanted_countries = set(TARGET_COUNTRY_NAMES)
        country_names = [name for name in country_names if name in wanted_countries]

    env_tile = os.environ.get("SND_TARGET_TILE_NAME", "").strip()
    wanted_tiles = {env_tile} if env_tile else (set(TARGET_TILE_NAMES) if TARGET_TILE_NAMES else None)
    tile_groups = []
    for country_name in country_names:
        country_root = os.path.join(TARGET_IMAGE_ROOT, country_name)
        tile_names = sorted(
            name for name in os.listdir(country_root)
            if os.path.isdir(os.path.join(country_root, name))
        )
        for tile_name in tile_names:
            tile_key = f"{country_name}/{tile_name}"
            if wanted_tiles and tile_name not in wanted_tiles and tile_key not in wanted_tiles:
                continue
            tile_groups.append(
                {
                    "country_name": country_name,
                    "tile_name": tile_name,
                    "tile_key": tile_key,
                }
            )
    return tile_groups


def build_full_tile_lines(country_name, tile_name):
    tile_dir = os.path.join(TARGET_IMAGE_ROOT, country_name, tile_name)
    files = sorted(
        name for name in os.listdir(tile_dir)
        if os.path.isfile(os.path.join(tile_dir, name)) and name.lower().endswith((".tif", ".tiff"))
    )
    if not files:
        raise FileNotFoundError(f"no tif files found in tile dir: {tile_dir}")
    filtered_lines = []
    all_zero_filtered_count = 0
    for file_name in files:
        image_path = os.path.join(tile_dir, file_name)
        if is_all_zero_image(image_path):
            all_zero_filtered_count += 1
            continue
        filtered_lines.append(f"{country_name}/{tile_name}/{file_name}\n")
    return filtered_lines, {
        "raw_full_count": len(files),
        "all_zero_filtered_count": all_zero_filtered_count,
        "nonzero_full_count": len(filtered_lines),
    }


def select_tile_train_lines(country_name, tile_name, full_lines, sample_size, class_id=2, seed=16):
    positives = []
    negatives = []
    target_label_root = get_target_label_root(country_name)

    for line in full_lines:
        rel_path = line.strip().split()[0]
        label_rel = map_da_val_label_relpath(rel_path, country_name)
        label_path = os.path.join(target_label_root, label_rel)
        class_pixels = count_class_pixels(label_path, class_id)
        if class_pixels > 0:
            positives.append(rel_path)
        else:
            negatives.append(rel_path)

    rng = random.Random(f"{seed}:{tile_name}")
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positive_set = set(positives)

    if sample_size <= 0 or len(full_lines) <= sample_size:
        selected = positives + negatives
    else:
        selected = positives[:sample_size]
        if len(selected) < sample_size:
            selected.extend(negatives[: sample_size - len(selected)])

    rng.shuffle(selected)
    selected_lines = [path + "\n" for path in selected]
    return selected_lines, {
        "country_name": country_name,
        "tile_name": tile_name,
        "tile_key": f"{country_name}/{tile_name}",
        "full_count": len(full_lines),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "selected_count": len(selected),
        "selected_positive_count": sum(1 for path in selected if path in positive_set),
    }


def get_visible_gpu_tokens():
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_value:
        return [token.strip() for token in env_value.split(",") if token.strip()]

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        raise RuntimeError("no CUDA device found")
    return [str(i) for i in range(gpu_count)]


def format_float_tag(value):
    text = f"{float(value):.10g}"
    text = text.replace("-", "m").replace(".", "p")
    return text


def get_da_eval_interval(epoch):
    if epoch <= EARLY_STOP_MIN_EPOCH:
        return DA_VAL_EVERY_PRE_50
    return DA_VAL_EVERY_POST_50


def should_run_da_eval(epoch):
    if epoch <= EARLY_STOP_MIN_EPOCH:
        return epoch % DA_VAL_EVERY_PRE_50 == 0
    return (epoch - EARLY_STOP_MIN_EPOCH) % DA_VAL_EVERY_POST_50 == 0


def is_early_stop_active_epoch(epoch):
    return epoch > EARLY_STOP_MIN_EPOCH


class SegmentationDataset_AEF_SourceMapped(SegmentationDataset_AEF):
    def __getitem__(self, index):
        src_index = index % len(self.annotation_lines)
        annotation_line = self.annotation_lines[src_index]
        name = annotation_line.split()[0]

        img_path = os.path.join(self.img_source, name)
        img = gdal.Open(img_path)
        if img is None:
            raise FileNotFoundError(f"Source image not found: {img_path}")

        img_width = img.RasterXSize
        img_height = img.RasterYSize
        img_data = img.ReadAsArray(0, 0, img_width, img_height)
        del img

        label_name_mapper = getattr(self, "source_label_name_mapper", None)
        label_name = label_name_mapper(name) if label_name_mapper is not None else name
        seg_path = os.path.join(self.label_dir, label_name)
        seg = gdal.Open(seg_path)
        if seg is None:
            raise FileNotFoundError(f"Label image not found: {seg_path}")

        seg_data = seg.ReadAsArray(0, 0, seg.RasterXSize, seg.RasterYSize)
        del seg

        if self.DA and self.train:
            img_data_aug, seg_data = self.get_random_data(img_data, seg_data)

            tgt_index = index % len(self.annotation_lines_tgr)
            annotation_line_tgr = self.annotation_lines_tgr[tgt_index]
            name_DA = annotation_line_tgr.split()[0]
            tgr_path = os.path.join(self.img_target, name_DA)
            tgr = gdal.Open(tgr_path)
            if tgr is None:
                raise FileNotFoundError(f"Target-domain image not found: {tgr_path}")
            tgr_data = tgr.ReadAsArray()
            del tgr

            return img_data_aug, seg_data, img_data, tgr_data

        return img_data, seg_data


def configure_source_dataset(dataset):
    dataset.img_source = SOURCE_IMAGE_ROOT
    dataset.label_dir = SOURCE_LABEL_ROOT
    dataset.source_label_name_mapper = map_source_label_name
    return dataset


def partitions(data_lines, partition_size, rand=False):
    samples = data_lines.copy()
    if rand:
        random.shuffle(samples)
    num_partitions = math.ceil(len(samples) / partition_size)
    parts = []
    for i in range(num_partitions):
        start_idx = i * partition_size
        end_idx = start_idx + partition_size
        parts.append(samples[start_idx:end_idx])
    return parts, num_partitions


def train_epoch(
    model,
    train_lines,
    train_tgr_lines,
    optimizer,
    epoch,
    criterion,
    criterion_DA,
    weights,
    device,
    domain_loss_weight,
    worker_tag,
    country_name,
    tile_name,
):
    model.train()
    epoch_loss = 0.0
    epoch_loss_DA = 0.0
    epoch_grad_norm = 0.0
    alpha = 0.0
    train_steps = 0
    step_in_epoch = 0
    da_batches = 0

    train_parts, _ = partitions(train_lines, partition_size, rand=True)
    tgr_parts, _ = partitions(train_tgr_lines, partition_size, rand=True)
    num_train_parts = len(train_parts)
    num_tgr_parts = len(tgr_parts)
    num_partitions = max(num_train_parts, num_tgr_parts)

    if num_partitions == 0:
        return 0.0, 0.0, 0.0

    steps_per_epoch = 0
    for part_idx in range(num_partitions):
        partition = train_parts[part_idx % num_train_parts]
        tgr_partition = tgr_parts[part_idx % num_tgr_parts]
        dataset_len = max(len(partition), len(tgr_partition))
        steps_per_epoch += max(1, math.ceil(dataset_len / batch_size))

    progress_bar = tqdm(
        total=steps_per_epoch,
        desc=f"{worker_tag} {tile_name} E{epoch:03d}",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
    )

    for part_idx in range(num_partitions):
        partition = train_parts[part_idx % num_train_parts]
        tgr_partition = tgr_parts[part_idx % num_tgr_parts]
        dataset = SegmentationDataset_AEF_SourceMapped(
            partition, tgr_partition, input_shape, num_classes, True, VOCdevkit_path
        )
        configure_source_dataset(dataset)
        dataset.img_target = TARGET_IMAGE_ROOT
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            generator=torch.Generator(),
        )

        for _, (image_batch, label_batch, src_raw_da, tgt_raw_da) in enumerate(data_loader):
            image_batch = image_batch.to(device).float()
            label_batch = label_batch.long().to(device)

            src_raw_da = src_raw_da.to(device).float()
            tgt_raw_da = tgt_raw_da.to(device).float()
            da_images = torch.cat([src_raw_da, tgt_raw_da], dim=0)
            da_labels = torch.cat(
                [
                    torch.zeros(src_raw_da.size(0), dtype=torch.long, device=device),
                    torch.ones(tgt_raw_da.size(0), dtype=torch.long, device=device),
                ],
                dim=0,
            )

            p = float((epoch - 1) * steps_per_epoch + step_in_epoch) / max(1, (n_epochs * steps_per_epoch))
            alpha_max = float(os.environ.get("TRIAL_GRL_ALPHA_MAX", "0.6"))
            alpha = (2.0 / (1.0 + np.exp(-10 * p)) - 1) * alpha_max

            out_DA = model(da_images, alpha=alpha, DA=True)
            loss_DA = criterion_DA(out_DA, da_labels)
            batch_loss_DA = torch.mean(loss_DA)
            epoch_loss_DA += batch_loss_DA.item()
            da_batches += 1
            step_in_epoch += 1

            optimizer.zero_grad()
            out = model(image_batch)
            loss_main = criterion(out, label_batch)
            loss_main = torch.masked_select(loss_main, label_batch != 10)
            label_left = torch.masked_select(label_batch, label_batch != 10)
            loss_main = loss_main.sum() / weights[label_left].sum()

            total_loss = loss_main + domain_loss_weight * batch_loss_DA
            total_loss.backward()
            grad_sq = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad_sq += float(parameter.grad.detach().pow(2).sum().item())
            epoch_grad_norm += math.sqrt(grad_sq)
            optimizer.step()

            epoch_loss += loss_main.item()
            train_steps += 1

            progress_bar.update(1)
            progress_bar.set_postfix(
                seg=f"{loss_main.item():.4f}",
                da=f"{batch_loss_DA.item():.4f}",
                w=f"{domain_loss_weight:.3f}",
            )

    progress_bar.close()
    return (
        epoch_loss / max(1, train_steps),
        epoch_loss_DA / max(1, da_batches),
        alpha,
        epoch_grad_norm / max(1, train_steps),
    )


def evaluate_epoch_DA(model, lines, criterion, weights, device, country_name):
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    pred_all = []
    true_all = []

    parts, _ = partitions(lines, partition_size, rand=False)
    with torch.no_grad():
        for partition in parts:
            dataset = SegmentationDataset_AEF_val(
                partition,
                partition,
                input_shape,
                num_classes,
                False,
                VOCdevkit_path,
            )
            dataset.img_source = TARGET_IMAGE_ROOT
            dataset.label_dir = get_target_label_root(country_name)
            dataset.label_name_mapper = lambda name, country_name=country_name: map_da_val_label_relpath(name, country_name)

            data_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )

            for image_batch, label_batch in data_loader:
                image_batch = image_batch.to(device)
                label_batch = label_batch.long().to(device)

                out = model(image_batch)
                loss = criterion(out, label_batch)
                loss = torch.masked_select(loss, label_batch != 10)
                label_left = torch.masked_select(label_batch, label_batch != 10)

                total_loss += loss.sum().item()
                total_weight += weights[label_left].sum().item()

                out = torch.argmax(torch.nn.functional.softmax(out, dim=1), dim=1)
                pred_all.append(out.cpu().numpy())
                true_all.append(label_batch.cpu().numpy())

    pred_all = np.concatenate([p.reshape(-1) for p in pred_all])
    true_all = np.concatenate([t.reshape(-1) for t in true_all])

    mask = true_all != 10
    pred_all = pred_all[mask]
    true_all = true_all[mask]

    f1_array = f1_score(
        true_all,
        pred_all,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    val_loss = total_loss / (total_weight + 1e-8)
    return val_loss, f1_array


def build_tile_summary(
    tile_group,
    best_model_path,
    best_da_val_loss,
    best_da_val_f1,
    best_epoch,
    stop_reason,
    epochs_completed,
    domain_loss_weight,
    resume_info=None,
):
    tile_stats = tile_group["tile_stats"]
    summary = {
        "country_name": tile_group["country_name"],
        "tile_name": tile_group["tile_name"],
        "tile_key": tile_group["tile_key"],
        "best_model_path": best_model_path,
        "best_da_val_loss": best_da_val_loss,
        "best_da_val_f1_class2": float(best_da_val_f1[cashew_class_id]) if best_epoch is not None else None,
        "best_epoch": best_epoch,
        "stop_reason": stop_reason,
        "epochs_completed": int(epochs_completed),
        "tile_train_samples": len(load_lines(tile_group["train_txt_path"])),
        "tile_val_samples": len(load_lines(tile_group["full_txt_path"])),
        "tile_positive_count": tile_stats["positive_count"],
        "tile_selected_positive_count": tile_stats["selected_positive_count"],
        "pretrained_weight_path": PRETRAINED_WEIGHT_PATH,
        "domain_loss_weight": domain_loss_weight,
        "train_txt_path": tile_group["train_txt_path"],
        "full_txt_path": tile_group["full_txt_path"],
        "training_complete": stop_reason.startswith("early_stop_no_improve_") or int(epochs_completed) >= n_epochs,
    }
    if resume_info is not None:
        summary.update(
            {
                "resume_enabled": bool(resume_info.get("resume_enabled")),
                "resumed": bool(resume_info.get("resumed")),
                "resume_checkpoint_path": resume_info.get("checkpoint_path"),
                "resume_checkpoint_epoch": resume_info.get("checkpoint_epoch"),
                "resume_optimizer_restored": bool(resume_info.get("optimizer_restored")),
                "resume_training_state_source": resume_info.get("training_state_source"),
                "resume_log_rows_kept": resume_info.get("log_rows_kept", 0),
            }
        )
    return summary


def load_resume_checkpoint(model, optimizer, device, epoch_ckpt_dir, csv_path):
    resume_info = {
        "resume_enabled": bool(resume_training),
        "resumed": False,
        "checkpoint_path": None,
        "checkpoint_epoch": None,
        "next_epoch": start_epoch,
        "optimizer_restored": False,
        "training_state_source": None,
        "log_rows_kept": 0,
    }
    resume_info.update(_default_training_state())

    latest_checkpoint = find_latest_epoch_checkpoint(epoch_ckpt_dir)
    if latest_checkpoint is None:
        return resume_info

    payload = normalize_checkpoint_payload(torch.load(latest_checkpoint["path"], map_location="cpu"))
    load_state_dict_compatible(model, payload["state_dict"])

    checkpoint_epoch = _safe_int(payload.get("epoch"), latest_checkpoint["epoch"])
    resume_info["resumed"] = True
    resume_info["checkpoint_path"] = latest_checkpoint["path"]
    resume_info["checkpoint_epoch"] = checkpoint_epoch
    resume_info["next_epoch"] = checkpoint_epoch + 1

    optimizer_state = payload.get("optimizer_state_dict")
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)
        move_optimizer_state_to_device(optimizer, device)
        resume_info["optimizer_restored"] = True

    training_state = payload.get("training_state")
    if isinstance(training_state, dict):
        resume_info.update(_normalize_training_state(training_state))
        resume_info["training_state_source"] = "checkpoint"
    else:
        resume_info.update(load_training_state_from_csv(csv_path, upto_epoch=checkpoint_epoch))
        resume_info["training_state_source"] = "csv"

    return resume_info


def train_single_tile(tile_group, train_lines, device, worker_tag, domain_loss_weight):
    country_name = tile_group["country_name"]
    tile_name = tile_group["tile_name"]
    tile_key = tile_group["tile_key"]
    tile_log_dir = tile_group["tile_log_dir"]
    epoch_ckpt_dir = os.path.join(tile_log_dir, "epoch_weights")
    best_model_path = os.path.join(tile_log_dir, "best_epoch_weights.pt")
    csv_path = os.path.join(tile_log_dir, f"train_log_{tile_name}.csv")
    os.makedirs(epoch_ckpt_dir, exist_ok=True)

    existing_summary = load_existing_tile_summary(tile_log_dir)
    if resume_training and is_tile_summary_complete(existing_summary):
        return existing_summary

    tile_train_lines = load_lines(tile_group["train_txt_path"])
    tile_val_lines = load_lines(tile_group["full_txt_path"])
    tile_stats = tile_group["tile_stats"]

    model = backbone_64swin2_convconnect.STC_DA(out_classes=num_classes, use_pretrained=False)
    model = model.to(device)
    pretrained_stats = load_model_weights(model, PRETRAINED_WEIGHT_PATH)

    weights = torch.tensor([3.0, 3.0, 10.0, 3.0], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=10, reduction="none")
    criterion_DA = torch.nn.CrossEntropyLoss(reduction="none")

    optimizer = {
        "adam": optim.Adam(model.parameters(), lr=Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay),
        "adamw": optim.AdamW(model.parameters(), lr=Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay),
        "sgd": optim.SGD(model.parameters(), lr=Init_lr, momentum=momentum, nesterov=True, weight_decay=weight_decay),
    }[optimizer_type]
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, n_epochs)
    resume_info = load_resume_checkpoint(
        model=model,
        optimizer=optimizer,
        device=device,
        epoch_ckpt_dir=epoch_ckpt_dir,
        csv_path=csv_path,
    ) if resume_training else {
        "resume_enabled": False,
        "resumed": False,
        "checkpoint_path": None,
        "checkpoint_epoch": None,
        "next_epoch": start_epoch,
        "optimizer_restored": False,
        "training_state_source": None,
        "log_rows_kept": 0,
        **_default_training_state(),
    }
    resume_info["log_rows_kept"] = prepare_train_log_for_resume(csv_path, resume_info["next_epoch"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    run_info = {
        "Env": {
            "worker_tag": worker_tag,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": str(device),
        },
        "Tile": {
            "country_name": country_name,
            "tile_name": tile_name,
            "tile_key": tile_key,
            "tile_log_dir": tile_log_dir,
            "tile_train_samples": len(tile_train_lines),
            "tile_val_samples": len(tile_val_lines),
            "tile_positive_count": tile_stats["positive_count"],
            "tile_selected_positive_count": tile_stats["selected_positive_count"],
        },
        "Data": {
            "target_image_root": TARGET_IMAGE_ROOT,
            "target_label_root_base": TARGET_LABEL_ROOT_BASE,
            "target_country_label_root": get_target_label_root(country_name),
        },
        "Train": {
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "partition_size": partition_size,
            "da_val_every_pre_50": DA_VAL_EVERY_PRE_50,
            "da_val_every_post_50": DA_VAL_EVERY_POST_50,
            "early_stop_min_epoch": EARLY_STOP_MIN_EPOCH,
            "early_stop_eval_patience": EARLY_STOP_EVAL_PATIENCE,
            "domain_loss_weight": domain_loss_weight,
            "start_epoch": resume_info["next_epoch"],
        },
        "Pretrained": pretrained_stats,
        "Resume": {
            "resume_training": bool(resume_training),
            "resumed": bool(resume_info["resumed"]),
            "resume_unfinished_only": bool(resume_unfinished_only),
            "checkpoint_path": resume_info["checkpoint_path"],
            "checkpoint_epoch": resume_info["checkpoint_epoch"],
            "optimizer_restored": bool(resume_info["optimizer_restored"]),
            "training_state_source": resume_info["training_state_source"],
            "log_rows_kept": resume_info["log_rows_kept"],
        },
        "Model": {
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
    }
    print_run_info(
        run_info,
        title=f"Tile Training Config: {tile_key}",
        save_path=os.path.join(tile_log_dir, "run_config_table.txt"),
        sort_keys=False,
    )

    best_da_val_loss = resume_info["best_da_val_loss"]
    best_da_val_f1 = resume_info["best_da_val_f1"]
    best_epoch = resume_info["best_epoch"]
    no_improve_eval_count = resume_info["no_improve_eval_count"]
    last_da_val_loss = resume_info["last_da_val_loss"]
    last_da_val_f1 = resume_info["last_da_val_f1"]

    if isinstance(existing_summary, dict) and existing_summary.get("resume_reset_for_new_schedule"):
        best_da_val_loss = float("inf")
        best_da_val_f1 = _default_metric_array()
        best_epoch = None
        no_improve_eval_count = 0
        print(
            f"[{worker_tag}] [Tile {tile_key}] reset legacy best/no-improve state for resumed training "
            f"under the new validation schedule."
        )

    stop_reason = "max_epochs_reached"
    completed_without_training = False

    if resume_info["resumed"]:
        checkpoint_epoch = resume_info["checkpoint_epoch"]
        early_stop_reached = (
            checkpoint_epoch is not None
            and should_run_da_eval(checkpoint_epoch)
            and is_early_stop_active_epoch(checkpoint_epoch)
            and no_improve_eval_count >= EARLY_STOP_EVAL_PATIENCE
        )
        if checkpoint_epoch is not None and (checkpoint_epoch >= n_epochs or early_stop_reached):
            stop_reason = (
                f"early_stop_no_improve_{EARLY_STOP_EVAL_PATIENCE}_evals"
                if early_stop_reached
                else "max_epochs_reached"
            )
            completed_without_training = True

    if completed_without_training:
        epochs_completed = resume_info["checkpoint_epoch"]
        tile_summary = build_tile_summary(
            tile_group=tile_group,
            best_model_path=best_model_path,
            best_da_val_loss=best_da_val_loss,
            best_da_val_f1=best_da_val_f1,
            best_epoch=best_epoch,
            stop_reason=stop_reason,
            epochs_completed=epochs_completed,
            domain_loss_weight=domain_loss_weight,
            resume_info=resume_info,
        )
        write_json_atomic(os.path.join(tile_log_dir, "tile_summary.json"), tile_summary)

        del model, optimizer, criterion, criterion_DA, weights
        torch.cuda.empty_cache()
        return tile_summary

    epochs_completed = resume_info["checkpoint_epoch"] or 0
    for epoch in range(resume_info["next_epoch"], n_epochs + 1):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        train_loss, train_loss_DA, epoch_alpha = train_epoch(
            model=model,
            train_lines=train_lines,
            train_tgr_lines=tile_train_lines,
            optimizer=optimizer,
            epoch=epoch,
            criterion=criterion,
            criterion_DA=criterion_DA,
            weights=weights,
            device=device,
            domain_loss_weight=domain_loss_weight,
            worker_tag=worker_tag,
            country_name=country_name,
            tile_name=tile_name,
        )

        improved = False
        if should_run_da_eval(epoch):
            da_val_loss, da_val_f1 = evaluate_epoch_DA(
                model=model,
                lines=tile_val_lines,
                criterion=criterion,
                weights=weights,
                device=device,
                country_name=country_name,
            )
            last_da_val_loss = da_val_loss
            last_da_val_f1 = da_val_f1

            print(
                f"[{worker_tag}] [Tile {tile_key}] [DA Eval @ Epoch {epoch}] "
                f"DA Val Loss: {da_val_loss:.6f}, "
                f"F1: Class0={da_val_f1[0]:.4f}, "
                f"Class1={da_val_f1[1]:.4f}, "
                f"Class2={da_val_f1[2]:.4f}, "
                f"Class3={da_val_f1[3]:.4f}, "
                f"DA Train Loss: {train_loss_DA:.4f}"
            )

            if da_val_loss < best_da_val_loss - MIN_DA_VAL_IMPROVEMENT:
                best_da_val_loss = da_val_loss
                best_da_val_f1 = da_val_f1
                best_epoch = epoch
                improved = True

            if is_early_stop_active_epoch(epoch):
                if improved:
                    no_improve_eval_count = 0
                else:
                    no_improve_eval_count += 1
            else:
                no_improve_eval_count = 0

            if improved:
                save_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    path=best_model_path,
                    training_state={
                        "best_da_val_loss": best_da_val_loss,
                        "best_da_val_f1": best_da_val_f1,
                        "best_epoch": best_epoch,
                        "no_improve_eval_count": no_improve_eval_count,
                        "last_da_val_loss": last_da_val_loss,
                        "last_da_val_f1": last_da_val_f1,
                    },
                )
        elif epoch <= EARLY_STOP_MIN_EPOCH:
            no_improve_eval_count = 0

        epoch_model_path = os.path.join(epoch_ckpt_dir, f"epoch_{epoch:03d}.pt")
        save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            path=epoch_model_path,
            training_state={
                "best_da_val_loss": best_da_val_loss,
                "best_da_val_f1": best_da_val_f1,
                "best_epoch": best_epoch,
                "no_improve_eval_count": no_improve_eval_count,
                "last_da_val_loss": last_da_val_loss,
                "last_da_val_f1": last_da_val_f1,
            },
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_DA": train_loss_DA,
            "epoch_alpha": epoch_alpha,
            "da_val_loss": last_da_val_loss,
            "da_val_score_0": last_da_val_f1[0],
            "da_val_score_1": last_da_val_f1[1],
            "da_val_score_2": last_da_val_f1[2],
            "da_val_score_3": last_da_val_f1[3],
            "best_da_val_loss_min": best_da_val_loss,
            "best_da_val_score_2": best_da_val_f1[2],
            "best_epoch": best_epoch,
            "improved": improved,
            "no_improve_eval_count": no_improve_eval_count,
            "domain_loss_weight": domain_loss_weight,
        }
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
        epochs_completed = epoch

        if (
            should_run_da_eval(epoch)
            and is_early_stop_active_epoch(epoch)
            and no_improve_eval_count >= EARLY_STOP_EVAL_PATIENCE
        ):
            stop_reason = (
                f"early_stop_no_improve_{EARLY_STOP_EVAL_PATIENCE}_evals_after_epoch_{EARLY_STOP_MIN_EPOCH}"
            )
            print(
                f"[{worker_tag}] [Tile {tile_key}] early stop at epoch {epoch}: "
                f"DA Val Loss has not improved for {EARLY_STOP_EVAL_PATIENCE} validations after epoch "
                f"{EARLY_STOP_MIN_EPOCH}."
            )
            break

    tile_summary = build_tile_summary(
        tile_group=tile_group,
        best_model_path=best_model_path,
        best_da_val_loss=best_da_val_loss,
        best_da_val_f1=best_da_val_f1,
        best_epoch=best_epoch,
        stop_reason=stop_reason,
        epochs_completed=epochs_completed,
        domain_loss_weight=domain_loss_weight,
        resume_info=resume_info,
    )
    write_json_atomic(os.path.join(tile_log_dir, "tile_summary.json"), tile_summary)

    del model, optimizer, criterion, criterion_DA, weights
    torch.cuda.empty_cache()
    return tile_summary


def train_single_tile(tile_group, train_lines, device, worker_tag, domain_loss_weight):
    """Train one tile without reading PlanetMap labels.

    The earlier baseline implementation above is intentionally retained as a
    reference snapshot, but this later definition is the one used by the
    worker.  It saves every epoch and never evaluates, early-stops, or selects
    a checkpoint from target labels.  Selection is performed post-hoc by the
    analysis scripts.
    """
    tile_key = tile_group["tile_key"]
    tile_name = tile_group["tile_name"]
    tile_log_dir = tile_group["tile_log_dir"]
    epoch_ckpt_dir = os.path.join(tile_log_dir, "epoch_weights")
    csv_path = os.path.join(tile_log_dir, f"train_log_{tile_name}.csv")
    os.makedirs(epoch_ckpt_dir, exist_ok=True)

    target_lines = load_lines(tile_group["train_txt_path"])
    model = backbone_64swin2_convconnect.STC_DA(out_classes=num_classes, use_pretrained=False).to(device)
    pretrained_stats = load_model_weights(model, PRETRAINED_WEIGHT_PATH)
    weights = torch.tensor([3.0, 3.0, 10.0, 3.0], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=10, reduction="none")
    criterion_da = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.AdamW(
        model.parameters(), lr=Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay
    )
    scheduler = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, n_epochs)

    start = 1
    latest = find_latest_epoch_checkpoint(epoch_ckpt_dir) if resume_training else None
    if latest is not None:
        payload = normalize_checkpoint_payload(torch.load(latest["path"], map_location="cpu"))
        load_state_dict_compatible(model, payload["state_dict"])
        if isinstance(payload.get("optimizer_state_dict"), dict):
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
        start = int(payload.get("epoch", latest["epoch"])) + 1

    run_info = {
        "worker_tag": worker_tag,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tile_key": tile_key,
        "source_train_txt": os.environ.get("SND_SOURCE_TRAIN_TXT"),
        "source_train_samples": len(train_lines),
        "target_train_samples": len(target_lines),
        "pretrained": pretrained_stats,
        "domain_loss_weight": domain_loss_weight,
        "max_epoch": n_epochs,
        "batch_size": batch_size,
        "planet_labels_used_by_training": False,
        "early_stopping": False,
    }
    write_json_atomic(os.path.join(tile_log_dir, "snd_train_config.json"), run_info)

    for epoch in range(start, n_epochs + 1):
        set_optimizer_lr(optimizer, scheduler, epoch)
        train_loss, da_loss, alpha = train_epoch(
            model=model,
            train_lines=train_lines,
            train_tgr_lines=target_lines,
            optimizer=optimizer,
            epoch=epoch,
            criterion=criterion,
            criterion_DA=criterion_da,
            weights=weights,
            device=device,
            domain_loss_weight=domain_loss_weight,
            worker_tag=worker_tag,
            country_name=tile_group["country_name"],
            tile_name=tile_name,
        )
        checkpoint_path = os.path.join(epoch_ckpt_dir, f"epoch_{epoch:03d}.pt")
        save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            path=checkpoint_path,
            training_state={
                "best_da_val_loss": float("inf"),
                "best_da_val_f1": _default_metric_array(),
                "best_epoch": None,
                "no_improve_eval_count": 0,
                "last_da_val_loss": float("nan"),
                "last_da_val_f1": _default_metric_array(),
            },
        )
        pd.DataFrame([{
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_DA": da_loss,
            "epoch_alpha": alpha,
            "domain_loss_weight": domain_loss_weight,
            "planet_labels_used": False,
        }]).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    summary = {
        "country_name": tile_group["country_name"],
        "tile_name": tile_name,
        "tile_key": tile_key,
        "best_model_path": None,
        "best_da_val_loss": float("nan"),
        "best_da_val_f1_class2": float("nan"),
        "best_epoch": None,
        "epochs_completed": n_epochs,
        "stop_reason": "max_epochs_reached",
        "domain_loss_weight": domain_loss_weight,
        "planet_labels_used_by_training": False,
        "early_stopping": False,
        "epoch_checkpoint_dir": epoch_ckpt_dir,
    }
    write_json_atomic(os.path.join(tile_log_dir, "tile_summary.json"), summary)
    del model, optimizer, criterion, criterion_da, weights
    torch.cuda.empty_cache()
    return summary


def worker_loop(worker_idx, local_gpu_idx, gpu_token, task_queue, result_queue, domain_loss_weight):
    worker_tag = f"GPU{gpu_token}/W{worker_idx}"
    try:
        torch.cuda.set_device(local_gpu_idx)
        device = torch.device(f"cuda:{local_gpu_idx}")
        torch.set_num_threads(1)
        seed_everything(16 + worker_idx)
        source_train_txt = os.environ.get(
            "SND_SOURCE_TRAIN_TXT",
            os.path.join(EXPERIMENT_DIR, "source_split", "source_train.txt"),
        )
        train_lines = load_lines(source_train_txt)
    except Exception as exc:
        result_queue.put({
            "status": "worker_init_error",
            "worker_tag": worker_tag,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        return

    while True:
        tile_group = task_queue.get()
        if tile_group is None:
            result_queue.put({"status": "worker_done", "worker_tag": worker_tag})
            return

        tile_name = tile_group["tile_name"]
        tile_key = tile_group["tile_key"]
        try:
            print(f"[{worker_tag}] start tile={tile_key}")
            tile_summary = train_single_tile(
                tile_group=tile_group,
                train_lines=train_lines,
                device=device,
                worker_tag=worker_tag,
                domain_loss_weight=domain_loss_weight,
            )
            print(f"[{worker_tag}] done tile={tile_key} best_da_val_loss={tile_summary['best_da_val_loss']:.6f}")
            result_queue.put({
                "status": "ok",
                "worker_tag": worker_tag,
                "country_name": tile_group["country_name"],
                "tile_name": tile_name,
                "tile_key": tile_key,
                "tile_summary": tile_summary,
            })
        except Exception as exc:
            error_payload = {
                "status": "error",
                "worker_tag": worker_tag,
                "country_name": tile_group["country_name"],
                "tile_name": tile_name,
                "tile_key": tile_key,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            os.makedirs(tile_group["tile_log_dir"], exist_ok=True)
            with open(os.path.join(tile_group["tile_log_dir"], "worker_error.json"), "w") as f:
                json.dump(error_payload, f, indent=2, ensure_ascii=False)
            result_queue.put(error_payload)


def prepare_tile_groups(target_txt_dir, auto_log_root):
    target_tiles = discover_target_tile_groups()
    if not target_tiles:
        raise RuntimeError(f"no target tiles found under: {TARGET_IMAGE_ROOT}")

    tile_groups = []
    for original_index, tile_info in enumerate(target_tiles):
        country_name = tile_info["country_name"]
        tile_name = tile_info["tile_name"]
        tile_key = tile_info["tile_key"]

        country_txt_dir = os.path.join(target_txt_dir, country_name)
        full_txt_path = os.path.join(country_txt_dir, f"{tile_name}_full.txt")
        train_txt_path = os.path.join(country_txt_dir, f"{tile_name}_train1000_cashewfirst.txt")

        tile_log_dir = os.path.join(auto_log_root, country_name, tile_name)
        os.makedirs(tile_log_dir, exist_ok=True)
        cached_tile_meta = load_existing_tile_meta(tile_log_dir)
        skip_cached_tile = (
            reuse_prepared_txt
            and isinstance(cached_tile_meta, dict)
            and cached_tile_meta.get("prepared_txt_version") == PREPARED_TXT_VERSION
            and cached_tile_meta.get("skip_reason") == "all_zero_tile"
        )
        if skip_cached_tile:
            print(f"[TXT Cache] skip cached all-zero tile for {tile_key}")
            continue

        use_cached_txt = (
            reuse_prepared_txt
            and os.path.exists(full_txt_path)
            and os.path.exists(train_txt_path)
            and isinstance(cached_tile_meta, dict)
            and cached_tile_meta.get("prepared_txt_version") == PREPARED_TXT_VERSION
            and cached_tile_meta.get("country_name") == country_name
            and cached_tile_meta.get("tile_name") == tile_name
            and cached_tile_meta.get("full_txt_path") == full_txt_path
            and cached_tile_meta.get("train_txt_path") == train_txt_path
            and isinstance(cached_tile_meta.get("tile_stats"), dict)
        )

        if use_cached_txt:
            tile_stats = cached_tile_meta["tile_stats"]
            print(f"[TXT Cache] reuse prepared txt for {tile_key}")
        else:
            missing_paths = []
            if not os.path.exists(full_txt_path):
                missing_paths.append(full_txt_path)
            if not os.path.exists(train_txt_path):
                missing_paths.append(train_txt_path)

            if missing_paths:
                print(f"[TXT Cache] skip {tile_key} because prepared txt is missing: {missing_paths}")
                continue

            full_tile_lines = load_lines(full_txt_path)
            train_tile_lines = load_lines(train_txt_path)
            if not full_tile_lines or not train_tile_lines:
                print(
                    f"[TXT Cache] skip {tile_key} because prepared txt is empty "
                    f"(full={len(full_tile_lines)}, train={len(train_tile_lines)})"
                )
                continue

            if isinstance(cached_tile_meta, dict) and isinstance(cached_tile_meta.get("tile_stats"), dict):
                tile_stats = cached_tile_meta["tile_stats"]
            else:
                tile_stats = {
                    "country_name": country_name,
                    "tile_name": tile_name,
                    "tile_key": tile_key,
                    "full_count": len(full_tile_lines),
                    "raw_full_count": len(full_tile_lines),
                    "nonzero_full_count": len(full_tile_lines),
                    "all_zero_filtered_count": 0,
                    "positive_count": 0,
                    "negative_count": len(full_tile_lines),
                    "selected_count": len(train_tile_lines),
                    "selected_positive_count": 0,
                }
            print(f"[TXT Cache] use prepared txt without fresh metadata for {tile_key}")

        tile_group = {
            "original_index": original_index,
            "country_name": country_name,
            "tile_name": tile_name,
            "tile_key": tile_key,
            "tile_log_dir": tile_log_dir,
            "train_txt_path": train_txt_path,
            "full_txt_path": full_txt_path,
            "tile_stats": tile_stats,
            "prepared_txt_cached": use_cached_txt,
            "prepared_txt_version": PREPARED_TXT_VERSION,
        }
        write_json_atomic(os.path.join(tile_log_dir, "tile_meta.json"), tile_group)
        tile_groups.append(tile_group)

    tile_groups.sort(
        key=lambda item: (
            0 if item["tile_stats"]["positive_count"] > 0 else 1,
            item["original_index"],
        )
    )
    return tile_groups


def filter_unfinished_tile_groups(tile_groups):
    pending_groups = []
    skipped_groups = []
    for tile_group in tile_groups:
        tile_summary = load_existing_tile_summary(tile_group["tile_log_dir"])
        if is_tile_summary_complete(tile_summary):
            skipped_groups.append(
                {
                    "tile_key": tile_group["tile_key"],
                    "stop_reason": tile_summary.get("stop_reason"),
                    "epochs_completed": tile_summary.get("epochs_completed"),
                }
            )
        else:
            pending_groups.append(tile_group)
    return pending_groups, skipped_groups


def collect_tile_summaries(auto_log_root):
    tile_summaries = []
    if not os.path.isdir(auto_log_root):
        return tile_summaries

    for root, _, files in os.walk(auto_log_root):
        if "tile_summary.json" not in files:
            continue
        summary_path = os.path.join(root, "tile_summary.json")
        tile_summaries.append(load_json_file(summary_path))

    tile_summaries.sort(key=lambda item: (item["country_name"], item["tile_name"]))
    return tile_summaries


def run(domain_loss_weight, run_tag):
    seed_everything(16)

    if not os.path.exists(PRETRAINED_WEIGHT_PATH):
        raise FileNotFoundError(f"pretrained weight not found: {PRETRAINED_WEIGHT_PATH}")

    auto_log_root = os.path.join(EXPERIMENT_DIR, f"tile_logs_{run_tag}")
    target_txt_dir = os.path.join(EXPERIMENT_DIR, f"target_txt_auto_{run_tag}")
    os.makedirs(auto_log_root, exist_ok=True)
    os.makedirs(target_txt_dir, exist_ok=True)

    tile_groups = prepare_tile_groups(target_txt_dir=target_txt_dir, auto_log_root=auto_log_root)
    skipped_finished_tiles = []
    if resume_training and resume_unfinished_only:
        tile_groups, skipped_finished_tiles = filter_unfinished_tile_groups(tile_groups)

    if not tile_groups:
        summary_df = pd.DataFrame(collect_tile_summaries(auto_log_root))
        summary_csv_path = os.path.join(auto_log_root, "tile_training_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        if not summary_df.empty:
            best_row = summary_df.sort_values("best_da_val_loss", ascending=True).iloc[0].to_dict()
            with open(os.path.join(auto_log_root, "best_tile_overall.json"), "w") as f:
                json.dump(best_row, f, indent=2, ensure_ascii=False)
        print(
            f"[Auto Train {run_tag}] no pending tiles; "
            f"skipped_finished_tiles={len(skipped_finished_tiles)}"
        )
        return

    gpu_tokens = get_visible_gpu_tokens()
    print(
        f"[Auto Train {run_tag}] tiles={len(tile_groups)}, "
        f"gpus={gpu_tokens}, domain_loss_weight={domain_loss_weight}, "
        f"pretrained={PRETRAINED_WEIGHT_PATH}, "
        f"resume_training={resume_training}, "
        f"resume_unfinished_only={resume_unfinished_only}, "
        f"skipped_finished_tiles={len(skipped_finished_tiles)}"
    )

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for tile_group in tile_groups:
        task_queue.put(tile_group)
    for _ in gpu_tokens:
        task_queue.put(None)

    workers = []
    for worker_idx, gpu_token in enumerate(gpu_tokens):
        proc = ctx.Process(
            target=worker_loop,
            args=(worker_idx, worker_idx, gpu_token, task_queue, result_queue, domain_loss_weight),
        )
        proc.start()
        workers.append(proc)

    ok_results = []
    error_results = []
    worker_done_count = 0
    target_done_count = len(gpu_tokens)

    while worker_done_count < target_done_count:
        message = result_queue.get()
        status = message["status"]
        if status == "ok":
            ok_results.append(message["tile_summary"])
        elif status == "worker_done":
            worker_done_count += 1
        else:
            error_results.append(message)
            if status == "worker_init_error":
                worker_done_count += 1

    for proc in workers:
        proc.join()

    summary_df = pd.DataFrame(collect_tile_summaries(auto_log_root))
    summary_csv_path = os.path.join(auto_log_root, "tile_training_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    if not summary_df.empty:
        best_row = summary_df.sort_values("best_da_val_loss", ascending=True).iloc[0].to_dict()
        with open(os.path.join(auto_log_root, "best_tile_overall.json"), "w") as f:
            json.dump(best_row, f, indent=2, ensure_ascii=False)
        print(
            f"[Overall Best {run_tag}] tile={best_row['tile_name']} "
            f"best_da_val_loss={best_row['best_da_val_loss']:.6f} "
            f"best_epoch={best_row['best_epoch']}"
        )

    if error_results:
        with open(os.path.join(auto_log_root, "worker_errors.json"), "w") as f:
            json.dump(error_results, f, indent=2, ensure_ascii=False)
        raise RuntimeError(f"{len(error_results)} tile worker job(s) failed; see worker_errors.json")
