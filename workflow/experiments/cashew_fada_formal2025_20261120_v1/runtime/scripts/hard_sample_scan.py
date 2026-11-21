"""Evaluate a fixed, geometry-screened acquisition frame with the delivered predictor."""

from collections import Counter, defaultdict
from pathlib import Path
import math

import numpy as np
import torch
from scipy.ndimage import label

from hard_sample_training import PER_REVIEW, read, round_dir, save, sha

CATEGORIES = ('competing', 'diffuse', 'miss_probe', 'extra_probe', 'random_probe')
MAX_NEW = 16


def masks(prob, support):
    cashew = prob[2]
    strongest = prob[[0, 1, 3]].max(axis=0)
    peak = prob.max(axis=0)
    top_two = (prob > cashew).sum(axis=0) <= 1
    competing = support & top_two & (abs(cashew - strongest) <= .10) & (cashew + strongest >= .55)
    diffuse = support & ~competing & (peak <= .45) & (peak - cashew <= .12)
    diffuse &= ((peak[None] - prob) <= .12).sum(axis=0) >= 3
    return competing, diffuse


def eligible(mask, support):
    if mask.sum() < .05 * support.sum():
        return False
    components, n = label(mask, structure=np.ones((3, 3), dtype='uint8'))
    return bool(n and np.bincount(components.ravel())[1:].max() >= 64)


def image_support(image):
    return np.isfinite(image).all(0) & (image != 0).any(0)


def prototypes(job, common):
    """A morphology-independent spectral clue drawn only from source training labels."""
    if not Path(job['source']['source_train']).is_file():
        return {}
    from train_tile import source_label_path
    source = job['source']; names = Path(source['source_train']).read_text().splitlines()
    rng = np.random.RandomState(job['seed'])
    counts = Counter(); totals = defaultdict(lambda: np.zeros(64, dtype='float64'))
    for i in rng.permutation(len(names))[:48]:
        name = names[i]
        image = common.read_raster(Path(source['source_root']) / name, dtype=np.float32)
        labels = common.read_raster(source_label_path(Path(source['source_label_root']), name)).squeeze()
        if image.shape != (64, *labels.shape):
            raise ValueError('Source probe label grid differs')
        valid = image_support(image)
        for klass in (0, 1, 2, 3):
            candidates = np.flatnonzero((labels == klass) & valid)
            if len(candidates):
                ixs = rng.choice(candidates, min(128, len(candidates)), replace=False)
                vectors = image.reshape(64,-1)[:, ixs]
                vectors /= np.linalg.norm(vectors, axis=0).clip(1e-8)
                totals[klass] += vectors.sum(axis=1); counts[klass] += len(ixs)
    return {k:np.asarray(v / np.linalg.norm(v), dtype='float32')
            for k,v in totals.items() if counts[k] and np.linalg.norm(v) > 0}


def audit_masks(image, support, prob, refs):
    if 2 not in refs or len(refs) < 2:
        return None, None
    # A vote discrepancy is a reason to inspect imagery. No labels are derived here.
    feature = np.where(support[None], image, 0)
    feature /= np.linalg.norm(feature, axis=0).clip(1e-8)
    margin = np.einsum('c,chw->hw', refs[2], feature) - np.maximum.reduce([
        np.einsum('c,chw->hw', v, feature) for k,v in refs.items() if k != 2])
    misses = support & (prob[2] < .18) & (prob.max(0) >= .65) & (margin > .04)
    extras = support & (prob[2] >= .65) & (margin < -.04)
    return misses, extras


def encode_mask(mask):
    return np.packbits(mask.ravel()).tobytes().hex()


def decode_mask(value):
    return np.unpackbits(np.frombuffer(bytes.fromhex(value), dtype='uint8')).reshape(128,128).astype(bool)


def scan(job, handles, epoch, checkpoint_sha):
    from module_bn_v2 import domain
    root = round_dir(job, epoch); scan_path = root / 'screen.json'
    if scan_path.is_file():
        frozen = read(scan_path)
        if frozen['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError('Hard-example screen belongs to another checkpoint')
        return frozen
    from hard_sample_training import manifest
    tree = job['hard_sample_frame']
    if sha(tree['path']) != tree['sha256']:
        raise ValueError('Acquisition frame identity changed')
    frame = read(tree['path']); h = handles
    previous_epoch = epoch - PER_REVIEW
    known = manifest(job, previous_epoch)
    prior_screen = read(round_dir(job, previous_epoch) / 'screen.json') if previous_epoch else None
    previous_masks = prior_screen['masks'] if prior_screen else {}
    used = {row['path'] for row in known['entries']}
    unavailable = {attempt['path'] for e in range(PER_REVIEW,epoch,PER_REVIEW)
                   for attempt in manifest(job,e)['report']['attempts']
                   if attempt['status'] in ('no_imagery','no_trainable_pixels')}
    refs = prototypes(job, h.common) if previous_epoch else {}
    picks = defaultdict(list); all_masks = {}; counts = Counter(); valid_pixels = 0
    h.model.eval()
    with torch.inference_mode(), domain('target'):
        for i in range(0, len(frame), 8):
            batch = frame[i:i+8]
            images = np.stack([h.common.read_raster(Path(r['aef']), dtype=np.float32) for r in batch])
            clean = np.where((np.isfinite(images).all(1))[:,None], images, 0)
            logits, _ = h.gan.forward_fuse(h.model, torch.from_numpy(clean).to(h.device))
            probabilities = torch.softmax(logits, 1).cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError('Nonfinite acquisition probabilities')
            for row, image, prob in zip(batch, images, probabilities):
                support = image_support(image); n = int(support.sum())
                if n < 0.60 * support.size:
                    continue
                valid_pixels += n; counts['valid_patches'] += 1
                path = row['path']
                competing, diffuse = masks(prob, support)
                chip = {'competing': competing, 'diffuse': diffuse}
                for key, m in zip(('miss_probe', 'extra_probe'), audit_masks(image, support, prob, refs)):
                    if m is not None:
                        chip[key] = m
                if path in used:
                    counts['already_in_run'] += 1
                    continue
                if path in unavailable:
                    counts['previously_unusable'] += 1
                    continue
                all_masks[path] = {k:encode_mask(v) for k,v in chip.items() if eligible(v, support)}
                if not all_masks[path]:
                    del all_masks[path]
                for key, value in chip.items():
                    if path not in all_masks or key not in all_masks[path]:
                        continue
                    persistent = previous_masks.get(path, {}).get(key)
                    persistence = float((value & decode_mask(persistent)).sum()) / n if persistent else 0
                    share = float(value.sum())/n
                    trees = float((value & (prob[[0,1,3]].argmax(0) == 1)).sum())/n if key == 'competing' else 0
                    score = share + .6*trees + .6*persistence
                    picks[key].append(dict(**row, share=share, score=score, persistent_fraction=persistence))
                # An unweighted, geographically spread check of the frame.
                import hashlib
                score = int(hashlib.sha256((str(epoch) + path).encode()).hexdigest()[:12], 16) / 16**12
                picks['random_probe'].append(dict(**row, share=None, score=score, persistent_fraction=0))
    result = dict(epoch=epoch, checkpoint_sha256=checkpoint_sha, scanned=len(frame),
                  valid_patches=counts['valid_patches'], valid_pixels=valid_pixels,
                  existing_patches=len(known['entries']), previously_unusable=counts['previously_unusable'],
                  candidates={k:spread(picks[k]) for k in CATEGORIES},
                  masks=all_masks)
    save(scan_path, result)
    return result


def spread(rows):
    """Favor different large tiles and discourage adjacent chip centers."""
    buckets = defaultdict(list)
    for row in sorted(rows, key=lambda r: (-r['score'], r['path'])):
        buckets[row['tile']].append(row)
    order = []
    visited = defaultdict(int)
    while buckets:
        tile = min(buckets, key=lambda k: (visited[k], -buckets[k][0]['score'], k))
        champion = buckets[tile].pop(0); visited[tile] += 1
        if not buckets[tile]:
            del buckets[tile]
        nearby = any(math.hypot(champion['lon']-p['lon'],champion['lat']-p['lat']) < .017
                     for p in order)
        if not nearby:
            order.append(champion)
        if len(order) >= MAX_NEW * 4:
            break
    return order


def inventory(screen):
    return dict(scanned=screen['scanned'], valid_patches=screen['valid_patches'],
                valid_pixels=screen['valid_pixels'], existing_patches=screen['existing_patches'],
                previously_unusable=screen.get('previously_unusable',0),
                **{'available_'+k:len(screen['candidates'][k]) for k in CATEGORIES})
