"""Measure a fixed, bounded human TRAINING panel even when its loss weight is zero.

No target-point files are accepted. Full inventory is scanned once; subsequent
reviews read only the fixed panel and check the inventory's file identities.
The local inventory is a view of the declared, hashed training-library manifest.
"""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import rasterio
import torch
from human_negative_label_paths import resolve_label_path
from human_training_evidence_contract import POLICY, COUNT_KEYS, enabled, validate
from complete_history_store import _canonical, _write_immutable, _csv_view, _read_json
from transfer_record_contract_v1 import model_digest


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _signature(path):
    p = Path(path).resolve(strict=True); s = p.stat()
    return [str(p), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def _paths(job, sample):
    supp = job['supplementary']
    return Path(supp['image_root']) / sample['name'], resolve_label_path(supp['label_root'], sample['name'])


def _read_pair(job, sample):
    paths = _paths(job, sample)
    signatures = [_signature(p) for p in paths]
    for path, key in zip(paths, ('image_sha256', 'mask_sha256')):
        if _sha(path.read_bytes()) != sample[key]:
            raise ValueError('human training image/mask changed')
    with rasterio.open(paths[0]) as image, rasterio.open(paths[1]) as mask:
        if (image.count != 64 or mask.count != 1 or image.shape != mask.shape or
                image.crs != mask.crs or not image.transform.almost_equals(mask.transform, precision=1e-9)):
            raise ValueError('human image/mask grid mismatch')
        x = image.read().astype(np.float32); y = mask.read(1)
    if [_signature(p) for p in paths] != signatures:
        raise ValueError('human pair changed while reading')
    if not np.isin(y, (0, 2, 10)).all():
        raise ValueError('unexpected human training label encoding')
    valid = np.isfinite(x).all(axis=0) & (x != 0).any(axis=0)
    return x, y, valid, signatures


def inventory(job):
    if not enabled(job):
        return None
    policy = job.get('polygon_supervision_policy', {})
    if (job.get('supplementary', {}).get('mode') != 'human_polygon_binary'
            or policy.get('version') != 'human_polygon_group_formal_v1'
            or policy.get('user_authorized') is not True
            or policy.get('old_supplementary_labels_allowed') is not False):
        raise ValueError('declared user human training polygon library required')
    spec = job['period_library_manifest']; raw = Path(spec['path']).read_bytes()
    if _sha(raw) != spec['sha256']:
        raise ValueError('human library manifest changed')
    manifest = json.loads(raw)
    if (not isinstance(manifest, dict) or set(manifest) != {'version', 'target_year', 'samples'}
            or manifest['version'] != 'two_period_human_library_v1'
            or manifest['target_year'] != job['target_year'] or not isinstance(manifest['samples'], list)):
        raise ValueError('human training manifest schema/year mismatch')
    samples = manifest['samples']
    names = [s.strip() for s in Path(job['supplementary']['names_file']).read_text().splitlines() if s.strip()]
    if names != [s['name'] for s in samples] or len(names) != job['supplementary']['count']:
        raise ValueError('human training pool membership differs')
    root = Path(job['output_root']) / 'evidence_store/human_inventory'
    path = root / 'inventory.json'
    if path.exists():
        saved = _read_json(path)
        if (saved['job_id'], saved['run_id'], saved['manifest_sha256'], saved['policy']) != (
                job['job_id'], job['run_id'], spec['sha256'], POLICY):
            raise ValueError('human inventory owner/policy changed')
        if len(saved['samples']) != len(samples):
            raise ValueError('human inventory length changed')
        for sample, row in zip(samples, saved['samples']):
            if sample != row['sample'] or [_signature(p) for p in _paths(job, sample)] != row['signatures']:
                raise ValueError('human inventory files changed; do not reuse observations')
        return saved
    from two_period_human_library_guard_v1 import validate as verify_library
    receipt = verify_library(job, names, job['supplementary']['image_root'], job['supplementary']['label_root'])
    counts = dict.fromkeys(COUNT_KEYS, 0); years = {}; rows = []
    for sample in samples:
        x, y, valid, signatures = _read_pair(job, sample)
        n, p = int(((y == 0) & valid).sum()), int(((y == 2) & valid).sum())
        row = {'sample': sample, 'sample_id': _sha(sample['name'].encode()), 'signatures': signatures,
               'negative_pixels': n, 'positive_pixels': p,
               'ignored_pixels': int(((y == 10) & valid).sum()), 'invalid_pixels': int((~valid).sum())}
        rows.append(row)
        for key in ('negative_pixels', 'positive_pixels', 'ignored_pixels', 'invalid_pixels'):
            counts[key] += row[key]
        counts['patches'] += 1; counts['negative_patches'] += int(n > 0)
        counts['positive_patches'] += int(p > 0); counts['mixed_patches'] += int(n > 0 and p > 0)
        year = str(sample['image_year']); years[year] = years.get(year, 0) + 1
    selected = set()
    for sign in ('negative', 'positive'):
        eligible = [r for r in rows if r[sign + '_pixels']]
        eligible.sort(key=lambda r: _sha(('%s:%s:%s' % (POLICY['seed'], sign, r['sample_id'])).encode()))
        selected.update(r['sample_id'] for r in eligible[:POLICY['max_patches_per_sign']])
    result = {'job_id': job['job_id'], 'run_id': job['run_id'], 'manifest_sha256': spec['sha256'],
              'policy': POLICY, 'receipt': receipt, 'counts': counts, 'image_year_counts': years,
              'panel': sorted(selected), 'samples': rows}
    _write_immutable(path, _canonical(result)); _csv_view(root / 'inventory.csv', result)
    return result


def collect(job, gan, model, device, epoch, *, persist=True):
    """Measure this predictor; only formal producers use the formal epoch cache.

    Trial branches must set persist=False: they reuse the fixed inventory but
    never read or overwrite another predictor's epoch-bound measurement.
    """
    if type(persist) is not bool:
        raise ValueError('human probe persistence must be a boolean')
    if not enabled(job):
        return None
    inv = inventory(job); before = model_digest(model.state_dict())
    stem = Path(job['output_root']) / 'evidence_store/human_training_fit' / ('epoch_%06d' % epoch)
    if persist and stem.with_suffix('.json').exists():
        from human_training_evidence_contract import bind
        saved = _read_json(stem.with_suffix('.json'))
        bind(job, saved, epoch, model.state_dict())
        if saved['inventory']['inventory_sha256'] != _sha(_canonical(inv)):
            raise ValueError('cached human measurement inventory changed')
        _csv_view(stem.with_suffix('.csv'), saved)
        return saved
    # Preserve individual submodule flags, all random streams, and buffers. No
    # optimizer is passed or touched. A probe is never a supplementary update.
    device = torch.device(device); flags = [(m, m.training) for m in model.modules()]
    py = random.getstate(); nr = np.random.get_state(); cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    fit = dict(patches=0, negative_pixels=0, positive_pixels=0,
               negative_predicted_cashew=0, positive_predicted_other=0)
    sums = {'negative': 0., 'positive': 0.}; selected = set(inv['panel'])
    try:
        model.eval()
        with torch.no_grad():
            for row in inv['samples']:
                if row['sample_id'] not in selected:
                    continue
                x, y, valid, signatures = _read_pair(job, row['sample'])
                if signatures != row['signatures']:
                    raise ValueError('human probe library changed')
                # Non-finite embeddings cannot enter a convolution; those pixels
                # are excluded using the original validity mask above.
                x = np.where(np.isfinite(x), x, 0.)
                tensor = torch.from_numpy(x).unsqueeze(0).to(device)
                logits, _ = gan.forward_fuse(model, tensor)
                if logits.shape != (1, 4, *y.shape) or not torch.isfinite(logits).all():
                    raise ValueError('invalid human probe predictions')
                prob = torch.softmax(logits, dim=1)[0, 2].cpu().numpy()
                pred = logits.argmax(dim=1)[0].cpu().numpy()
                fit['patches'] += 1
                for sign, label in (('negative', 0), ('positive', 2)):
                    support = valid & (y == label); count = int(support.sum())
                    fit[sign + '_pixels'] += count
                    sums[sign] += float(prob[support].sum(dtype=np.float64))
                fit['negative_predicted_cashew'] += int((valid & (y == 0) & (pred == 2)).sum())
                fit['positive_predicted_other'] += int((valid & (y == 2) & (pred != 2)).sum())
        if model_digest(model.state_dict()) != before:
            raise ValueError('human probe mutated model state')
    finally:
        with torch.no_grad():
            for key, value in model.named_buffers():
                value.copy_(buffers[key])
        for module, training in flags:
            module.training = training
        random.setstate(py); np.random.set_state(nr); torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, device)
    for sign in sums:
        n = fit[sign + '_pixels']; fit[sign + '_mean_p2'] = sums[sign] / n if n else None
    result = {'version': POLICY['version'], 'role': 'human_training_polygons', 'independent': False,
              'job_id': job['job_id'], 'run_id': job['run_id'], 'epoch': epoch,
              'model_state_sha256': before,
              'inventory': {'manifest_sha256': inv['manifest_sha256'], 'inventory_sha256': _sha(_canonical(inv)),
                  'pairing_verified': True, 'target_year': job['target_year'], 'counts': inv['counts'],
                  'image_year_counts': inv['image_year_counts'], 'panel_sha256': _sha(_canonical(inv['panel']))},
              'probe': {'rule': 'argmax_class_2', 'max_patches_per_sign': POLICY['max_patches_per_sign'],
                        'seed': POLICY['seed'], 'summary': fit}}
    validate(result)
    # Ensure every review is about exactly the inventory registered at E0.
    first = Path(job['output_root']) / 'evidence_store/human_training_fit/epoch_000000.json'
    if epoch > 0 and (not first.is_file() or json.loads(first.read_bytes())['inventory'] != result['inventory']):
        raise ValueError('human training inventory differs from initialization')
    if persist:
        _write_immutable(stem.with_suffix('.json'), _canonical(result))
        _csv_view(stem.with_suffix('.csv'), result)
    return result
