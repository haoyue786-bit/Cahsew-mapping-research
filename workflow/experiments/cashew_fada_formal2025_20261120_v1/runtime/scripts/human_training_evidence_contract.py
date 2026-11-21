"""Typed supplementary TRAINING polygon evidence, never target point validation."""
import math
import re
from sealed_packet import values_digest

VERSION = 'human_training_evidence_v1'
POLICY = {'version': VERSION, 'max_patches_per_sign': 32, 'seed': 16}
COUNT_KEYS = {'patches', 'negative_patches', 'positive_patches', 'mixed_patches',
              'negative_pixels', 'positive_pixels', 'ignored_pixels', 'invalid_pixels'}
FIT_KEYS = {'patches', 'negative_pixels', 'positive_pixels', 'negative_predicted_cashew',
            'positive_predicted_other', 'negative_mean_p2', 'positive_mean_p2'}


def enabled(job):
    policy = job.get('human_training_evidence_policy')
    if policy is None:
        return False
    if policy != POLICY:
        raise ValueError('unknown human training evidence policy')
    return True


def _fields(value, fields):
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('unrecognized human training evidence fields')


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('invalid human training evidence digest')


def _counts(value, fields):
    _fields(value, fields)
    for key in fields:
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError('invalid human training count')


def validate(value):
    _fields(value, {'version', 'role', 'independent', 'job_id', 'run_id', 'epoch',
                    'model_state_sha256', 'inventory', 'probe'})
    if (value['version'] != VERSION or value['role'] != 'human_training_polygons'
            or value['independent'] is not False):
        raise ValueError('human evidence is training fit only')
    if type(value['epoch']) is not int or not 0 <= value['epoch'] <= 200:
        raise ValueError('invalid human evidence epoch')
    for key in ('job_id', 'run_id'):
        v = value[key]
        if (not isinstance(v, str) or len(v) > 180 or
                not re.fullmatch('[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*', v)
                or any(x in ('.', '..') for x in v.split('/'))):
            raise ValueError('invalid human evidence owner')
    _digest(value['model_state_sha256'])
    inv = value['inventory']
    _fields(inv, {'manifest_sha256', 'inventory_sha256', 'pairing_verified', 'target_year',
                  'counts', 'image_year_counts', 'panel_sha256'})
    for key in ('manifest_sha256', 'inventory_sha256', 'panel_sha256'):
        _digest(inv[key])
    if inv['pairing_verified'] is not True or inv['target_year'] not in (2019, 2025):
        raise ValueError('unverified human pairing or year')
    _counts(inv['counts'], COUNT_KEYS)
    c = inv['counts']
    if not (c['mixed_patches'] <= min(c['positive_patches'], c['negative_patches'])
            and c['positive_patches'] + c['negative_patches'] - c['mixed_patches'] <= c['patches']):
        raise ValueError('inconsistent human patch counts')
    years = inv['image_year_counts']
    if not isinstance(years, dict) or set(years) - ({'2018', '2019'} if inv['target_year'] == 2019 else {'2024', '2025'}):
        raise ValueError('human imagery is from another period')
    if any(type(n) is not int or n < 0 for n in years.values()) or sum(years.values()) != c['patches']:
        raise ValueError('inconsistent human image years')
    probe = value['probe']
    _fields(probe, {'rule', 'max_patches_per_sign', 'seed', 'summary'})
    if (probe['rule'] != 'argmax_class_2' or probe['max_patches_per_sign'] != POLICY['max_patches_per_sign']
            or probe['seed'] != POLICY['seed']):
        raise ValueError('human probe policy mismatch')
    fit = probe['summary']
    _fields(fit, FIT_KEYS)
    _counts({k: fit[k] for k in FIT_KEYS if not k.endswith('_mean_p2')},
            FIT_KEYS - {'negative_mean_p2', 'positive_mean_p2'})
    if fit['patches'] > min(c['patches'], 2 * POLICY['max_patches_per_sign']):
        raise ValueError('human probe exceeds its panel')
    for sign, error in (('negative', 'negative_predicted_cashew'), ('positive', 'positive_predicted_other')):
        n = fit[sign + '_pixels']; mean = fit[sign + '_mean_p2']
        if n > c[sign + '_pixels'] or fit[error] > n:
            raise ValueError('human probe support mismatch')
        if (n == 0 and mean is not None) or (n and
                (type(mean) not in (int, float) or not math.isfinite(mean) or not 0 <= mean <= 1)):
            raise ValueError('missing human signs must remain unknown')
    return value


def artifact(value):
    validate(value)
    epoch = value['epoch']
    return {'kind': 'human_training', 'epoch': epoch, 'values': value,
            'sha256': values_digest('human_training', epoch, value)}


def checked(record, epoch, trusted):
    _fields(record, {'kind', 'epoch', 'values', 'sha256'})
    expected = artifact(record['values'])
    if record != expected or record['epoch'] != epoch or ('human_training', epoch, record['sha256']) not in trusted:
        raise ValueError('untrusted human training evidence')
    return {k: record[k] for k in ('epoch', 'sha256', 'values')}


def bind(job, record, epoch, state):
    """Independent worker binding; hashes alone cannot authorize a producer."""
    if not enabled(job):
        if record is not None:
            raise ValueError('human evidence not declared by job')
        return
    validate(record)
    from transfer_record_contract_v1 import model_digest
    if ((record['job_id'], record['run_id'], record['epoch']) !=
            (job['job_id'], job['run_id'], epoch)
            or record['model_state_sha256'] != model_digest(state)
            or record['inventory']['target_year'] != job['target_year']
            or record['inventory']['manifest_sha256'] != job['period_library_manifest']['sha256']):
        raise ValueError('human evidence differs from live job/checkpoint')
