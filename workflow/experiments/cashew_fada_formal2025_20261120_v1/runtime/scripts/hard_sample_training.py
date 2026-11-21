"""Four-class supervision from the run's accepted image annotations."""

from collections import Counter
from pathlib import Path
import hashlib
import json
import random

import torch
from torch.nn import functional as F

VERSION = 'hard_samples_v1'
WEIGHT = 'hard_sample_weight'
REPEATS = 'hard_sample_repeats'
BOUNDS = {WEIGHT: [0.0, 1.0], REPEATS: [1, 3]}
PER_REVIEW = 5
IGNORE = 10


class NoTrainableAnnotations(RuntimeError):
    """An annotation-dependent trajectory cannot advance from this boundary."""


def require_trainable_annotations(document):
    if document.get('backend') != 'sample_assist' or document['epoch'] == 0:
        return
    report = document['report']
    if not document['entries']:
        raise NoTrainableAnnotations('No verified training labels are available at the annotation boundary')
    if any(report['requested'].values()) and report['added'] == 0:
        raise NoTrainableAnnotations('The requested batch produced no verified new training label')


def enabled(record):
    policy = record.get('hard_sample_policy', {})
    return policy.get('version') == VERSION and policy.get('annotation_backend') != 'disabled'


def initialize(config, job):
    config['hard_sample_policy'] = dict(job['hard_sample_policy'])
    if enabled(job):
        config.update({WEIGHT: 0.5, REPEATS: 1})


def bounds(config):
    return BOUNDS if enabled(config) else {}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def save(path, value):
    from sealed_dispatch_v1 import publish
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    publish(path, value)


def mode(job):
    value = job['hard_sample_policy']['annotation_backend']
    if value not in ('simulated', 'sample_assist', 'disabled'):
        raise ValueError('Unknown annotation backend')
    return value


def work_root(job):
    return Path(job['output_root']) / 'hard_samples'


def round_dir(job, epoch):
    return work_root(job) / ('epoch_%06d' % epoch)


def manifest(job, epoch):
    if not epoch:
        return dict(epoch=0, backend=mode(job), entries=[], revision=identity([]))
    document = read(round_dir(job, epoch) / 'ready.json')
    if ((document['epoch'], document['backend']) != (epoch, mode(job))
            or document['revision'] != identity(document['entries'])
            or len({r['key'] for r in document['entries']}) != len(document['entries'])):
        raise ValueError('Invalid run sample revision')
    return document


def sample_tensors(entries, common, device):
    import numpy as np
    import rasterio
    images, labels = [], []
    for sample in entries:
        if sha(sample['label']) != sample['label_sha256']:
            raise ValueError('Accepted annotation bytes changed')
        image = common.read_raster(Path(sample['aef']), dtype=np.float32)
        support = np.isfinite(image).all(axis=0) & (image != 0).any(axis=0)
        with rasterio.open(sample['label']) as source:
            target = source.read(1).astype(np.int64)
        if target.shape != support.shape or not np.isin(target, [0, 1, 2, 3, IGNORE]).all():
            raise ValueError('Invalid four-class annotation on the original grid')
        target[~support] = IGNORE
        images.append(np.where(support[None], image, 0))
        labels.append(target)
    return (torch.from_numpy(np.stack(images)).to(device),
            torch.from_numpy(np.stack(labels)).to(device))


def pixel_loss(logits, labels):
    """Use the same four-class interpretation for training and comparisons."""
    if logits.shape[1] != 4 or logits.shape[2:] != labels.shape[1:]:
        raise ValueError('Four logits per original AEF pixel are required')
    return F.cross_entropy(logits.float(), labels, ignore_index=IGNORE, reduction='none')


def training_loss(logits, labels):
    losses = pixel_loss(logits, labels).flatten(1)
    count = (labels != IGNORE).flatten(1).sum(1)
    valid = count > 0
    # A large, easy homogeneous region does not outweigh another entire chip.
    return (losses.sum(1)[valid] / count[valid]).mean() if valid.any() else logits.sum() * 0


def comparison(gan, common, model, device, entries):
    """Fixed masks, delivered target BN/argmax; counts describe training fit."""
    from module_bn_v2 import domain
    from hard_sample_scan import decode_mask
    was_training = model.training
    model.eval()
    rows = []
    try:
        with torch.inference_mode(), domain('target'):
            for offset in range(0, len(entries), 8):
                samples = entries[offset:offset + 8]
                images, labels = sample_tensors(samples, common, device)
                logits, _ = gan.forward_fuse(model, images)
                predicted = logits.argmax(1) == 2
                losses = pixel_loss(logits, labels)
                for i, sample in enumerate(samples):
                    accepted = labels[i] != IGNORE
                    overall = _fit(accepted, labels[i], predicted[i], losses[i])
                    if overall is None:
                        raise ValueError('Accepted annotation has no trainable pixels')
                    focused = None
                    if sample.get('hard_area'):
                        mask = torch.from_numpy(decode_mask(sample['hard_area'])).to(device)
                        focused = _fit(accepted & mask, labels[i], predicted[i], losses[i])
                    rows.append(dict(key=sample['key'], hard_area_fit=focused, **overall))
    finally:
        model.train(was_training)
    return rows


def _fit(mask, labels, predicted_cashew, losses):
    count = int(mask.sum())
    if not count:
        return None
    label = labels[mask]; predicted = predicted_cashew[mask]
    return dict(valid=count,pixels_by_class=[int((label == k).sum()) for k in range(4)],
                loss=float(losses[mask].mean()),missed_cashew=int(((label == 2) & ~predicted).sum()),
                extra_cashew=int(((label != 2) & predicted).sum()))


def _summary(rows):
    n = len(rows)
    return dict(patches=n, valid_pixels=sum(row['valid'] for row in rows),
                class_pixels=[sum(row['pixels_by_class'][i] for row in rows) for i in range(4)],
                mean_patch_loss=sum(row['loss'] for row in rows) / n if n else None,
                missed_cashew=sum(row['missed_cashew'] for row in rows),
                extra_cashew=sum(row['extra_cashew'] for row in rows))


def summary(rows):
    return dict(_summary(rows),hard_area_fit=_summary([r['hard_area_fit'] for r in rows if r.get('hard_area_fit')]))


class EpochSamples:
    """Deterministic exposure for a pinned round. Commit only successful steps."""

    def __init__(self, job, epoch, config, steps, common, device, *, current_round=None):
        self.epoch = epoch
        boundary = (epoch - 1) // PER_REVIEW * PER_REVIEW if current_round is None else current_round
        self.manifest = manifest(job, boundary)
        self.weight = float(config[WEIGHT]); self.device = device; self.common = common
        candidates = list(self.manifest['entries'])
        rng = random.Random(job['seed'] + boundary * 173)
        rng.shuffle(candidates)
        groups = [[s for s in candidates if bool(s['counts'][2]) == sign] for sign in (True, False)]
        # Mix patches with Cashew support and lookalikes when both are available.
        from itertools import zip_longest
        ordered = [s for pair in zip_longest(*groups) for s in pair if s is not None]
        ordered *= int(config[REPEATS])
        self.schedule = {}
        block = steps * 4
        if self.weight and ordered:
            start = ((epoch - 1 - boundary) * block) % len(ordered)
            order = (ordered[start:] + ordered[:start])[:block]
            for i, sample in enumerate(order):
                self.schedule.setdefault(i * steps // len(order), []).append(sample)
        self.updated = Counter()
        self.pixels = [0] * 4
        self.completed_steps = 0
        self.loss_sum = 0.0
        self.probes = []

    def forward(self, step, gan, model, source_loss):
        from human_polygon_v1 import frozen_auxiliary_bn
        from module_bn_v2 import domain
        samples = self.schedule.get(step)
        if not samples:
            return None, None
        images, labels = sample_tensors(samples, self.common, self.device)
        with frozen_auxiliary_bn(model), domain('target'):
            logits, _ = gan.forward_fuse(model, images)
        raw = training_loss(logits, labels)
        if step in (min(self.schedule), max(self.schedule)):
            params = [p for name,p in model.named_parameters() if p.requires_grad
                      and name.startswith(('decoder.', 'fuse_conv.'))]
            if params:
                sg = torch.autograd.grad(source_loss, params, retain_graph=True, allow_unused=True)
                ag = torch.autograd.grad(raw, params, retain_graph=True, allow_unused=True)
                sn = sum(float(x.detach().float().square().sum()) for x in sg if x is not None)**0.5
                an = sum(float(x.detach().float().square().sum()) for x in ag if x is not None)**0.5
                dot = sum(float((a.detach().float()*s.detach().float()).sum())
                          for a,s in zip(ag,sg) if a is not None and s is not None)
                self.probes.append(dict(step=step, source_norm=sn, annotation_norm=an,
                                        weighted_annotation_to_source=self.weight*an/sn if sn else None,
                                        cosine=max(-1.0, min(1.0, dot/(sn*an))) if sn and an else None))
        counts = [int((labels == k).sum()) for k in range(4)]
        return self.weight * raw, (samples, counts, float(raw.detach()))

    def commit(self, exposure):
        if exposure is None:
            return
        samples, counts, loss = exposure
        self.updated.update(s['key'] for s in samples)
        self.pixels = [a+b for a,b in zip(self.pixels, counts)]
        self.completed_steps += 1
        self.loss_sum += loss

    def metrics(self):
        return dict(epoch=self.epoch, backend=self.manifest['backend'],
                    revision=self.manifest['revision'], weight=self.weight,
                    completed_steps=self.completed_steps, sampled_keys=dict(self.updated),
                    pixels_by_class=self.pixels, raw_loss=self.loss_sum / self.completed_steps
                    if self.completed_steps else None, gradient_probes=self.probes)
