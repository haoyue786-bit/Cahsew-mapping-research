"""Unified, user-selected area alarms; never modifies logits, losses or checkpoints."""
from __future__ import annotations
import copy
import math

VERSION = 'unified_area_alert_v1'


def classify(fraction, policy):
    if policy.get('version') != VERSION:
        raise ValueError('unknown area alert policy')
    thresholds = policy['thresholds']
    if len(thresholds) != 3 or not all(type(x) in (int, float) and math.isfinite(x) and 0 < x < 1 for x in thresholds) or thresholds != sorted(set(thresholds)):
        raise ValueError('area thresholds must be three increasing fractions')
    if fraction is None:
        return {'level': 'unmeasured', 'fraction': None, 'exceeded_thresholds': []}
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Class2 area must be a finite fraction, not a percent')
    count = sum(fraction > x for x in thresholds)
    return {'level': ['below_warning', 'warning', 'high', 'severe'][count],
            'fraction': fraction, 'exceeded_thresholds': thresholds[:count]}


def late_start(policy, maximum_epoch, review_interval=5):
    fraction = policy.get('minimum_activation_fraction', 0)
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('invalid late activation fraction')
    if type(maximum_epoch) is not int or maximum_epoch <= 0 or type(review_interval) is not int or review_interval <= 0:
        raise ValueError('invalid training budget')
    absolute = policy.get('minimum_activation_epoch')
    if absolute is not None:
        if type(absolute) is not int or absolute < 0:
            raise ValueError('invalid absolute late activation epoch')
        # A short bounded recovery may end before this boundary; it then
        # legally keeps negatives off for its entire run.
        return int(math.ceil(absolute / review_interval) * review_interval)
    return int(math.ceil(fraction * maximum_epoch / review_interval) * review_interval)


def packet_evidence(job, current_review, candidates, initialization=False):
    policy = job.get('area_alert_policy')
    if not policy:
        return None
    start = late_start(job.get('conditional_negative_policy', {}), int(job['max_epoch']), int(job.get('review_interval', 5)))
    if current_review is None:
        fraction = None
        epoch = int(job.get('stopped_after_epoch', 0))
    elif initialization:
        fraction = current_review['target_unsupervised']['class_distribution'][2]
        epoch = 0
    else:
        fraction = current_review['target_class2_fraction']
        epoch = int(current_review['epoch'])
    result = {'policy': copy.deepcopy(policy), 'current': classify(fraction, policy),
        'minimum_negative_activation_review': start, 'late_activation_eligible': epoch >= start,
        'measurement': 'eval-mode hard argmax Class2 fraction on target_full_list valid pixels',
        'candidate_alerts': [{'epoch': int(x['epoch']), **classify(x['target_class2_fraction'], policy)} for x in candidates],
        'instruction': 'Area alarms prioritize diagnosis and late reversible dose tests; they do not prove false positives. Before the late boundary keep formal negatives off. After it, persistent alarms require investigating candidate-background mistakes and an informative trial or a concrete blocker, not perpetual KEEP solely because weak labels are uncertain. A weak response with confirmed exposure and source safety motivates a larger single-factor dose test. Area rebound alone does not justify disabling. Compare background suppression, retained cashew, source safety and exposure at all endpoints. At SELECT discuss alarms but do not select solely by minimum area or eliminate candidates for crossing a threshold.'}
    return result
