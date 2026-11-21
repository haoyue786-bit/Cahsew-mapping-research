"""Bounded validation repair for completed automatic-negative responses.

Never normalize, edit, or accept an invalid scientific response in Python.
Archive responses and ask the same durable group thread to correct its format.
Unknown transport outcomes fail closed until their durable turn is reconciled.
"""
import hashlib
import json
import math
from pathlib import Path

from persistent_luna import PersistentLunaRecoveryRequired

FIELDS = {'action', 'selected_count', 'automatic_weight', 'ranking_weights',
          'rationale', 'expected_response', 'reversal_condition'}
MIN_REGIONAL_PRIOR_WEIGHT = 0.25
RANKING = {'source_cashew_distance', 'p2_increase', 'regional_prior'}


def validate_selection(value, candidate_count):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError('Response must contain exactly the seven advertised fields')
    count, dose, weights = value['selected_count'], value['automatic_weight'], value['ranking_weights']
    if type(count) is not int or not 0 <= count <= candidate_count:
        raise ValueError('selected_count must be an integer between 0 and candidate_count')
    if type(dose) not in (int, float) or not math.isfinite(dose) or not 0 <= dose <= .005:
        raise ValueError('automatic_weight must be finite and within [0, 0.005]')
    if not isinstance(weights, dict) or set(weights) != RANKING:
        raise ValueError('ranking_weights must contain exactly the three advertised keys')
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
           for x in weights.values()) or abs(sum(weights.values()) - 1) >= 1e-5:
        raise ValueError('ranking_weights must be finite within [0, 1] and sum to 1, including ABSTAIN')
    if weights['regional_prior'] < MIN_REGIONAL_PRIOR_WEIGHT:
        raise ValueError('regional_prior ranking weight must be at least 0.25 for this user-authorized fresh cohort')
    if not ((value['action'] == 'ACTIVATE' and count > 0 and dose > 0) or
            (value['action'] == 'ABSTAIN' and count == 0 and dose == 0)):
        raise ValueError('ACTIVATE requires positive count and dose; ABSTAIN requires count=0 AND automatic_weight=0')
    if any(not isinstance(value[k], str) for k in ('rationale', 'expected_response', 'reversal_condition')):
        raise ValueError('Rationale, expected_response and reversal_condition must be strings')
    return value


def _dump_once(path, value):
    content = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError('Immutable validation artifact already exists: ' + str(path))
        return
    with path.open('x') as handle:
        handle.write(content)


def _completed(result, thread_id):
    if (result.get('thread_id') != thread_id or not result.get('turn_id') or
            result.get('event_tracking', {}).get('turn_status') != 'completed'):
        raise PersistentLunaRecoveryRequired({'automatic_selection_outcome_unknown': True,
                                             'thread_id': thread_id, 'turn_id': result.get('turn_id')})


def validated_turn(client, destination, candidate_count, **kwargs):
    """At most three new calls per selection; reuse all completed raw results."""
    destination = Path(destination)
    archive = destination / 'selection_validation'
    archive.mkdir(exist_ok=True)
    thread_id = kwargs['thread_id']
    result = None
    legacy = destination / 'selection_raw_result.json'
    if legacy.exists():
        result = json.loads(legacy.read_text())
        digest = hashlib.sha256(legacy.read_bytes()).hexdigest()
        _dump_once(archive / ('legacy_' + digest[:16] + '.json'), result)
    requests = sorted(archive.glob('attempt_*_request.json'))
    for request in requests:
        raw = request.with_name(request.name.replace('_request.json', '_raw.json'))
        if not raw.exists():
            raise PersistentLunaRecoveryRequired({'automatic_selection_unresolved_request': str(request),
                                                 'thread_id': thread_id})
        result = json.loads(raw.read_text())
    previous = kwargs.get('previous_turn_id')
    error = None
    rejected = None
    if result is not None:
        _completed(result, thread_id)
        try:
            value = validate_selection(json.loads(result['final_text']), candidate_count)
            _dump_once(archive / 'accepted_result.json', result)
            return value, result
        except (ValueError, KeyError, TypeError) as exc:
            error, rejected = str(exc), result['final_text']
            previous = result['turn_id']
            _dump_once(archive / ('rejected_' + previous + '.json'),
                       {'turn_id': previous, 'error_type': type(exc).__name__, 'error': error})
            _dump_once(archive / 'first_validation_error.json',
                       json.loads((archive / 'first_validation_error.json').read_text())
                       if (archive / 'first_validation_error.json').exists() else
                       {'turn_id': previous, 'error_type': type(exc).__name__, 'error': error})
        reconciled = client.reconcile_thread(thread_id=thread_id, timeout_seconds=60)
        latest = reconciled.get('turn') or {}
        if latest.get('id') != previous or latest.get('status') != 'completed':
            raise PersistentLunaRecoveryRequired({'automatic_selection_thread_advanced_or_unknown': latest,
                                                 'expected_turn_id': previous, 'thread_id': thread_id})
    for attempt in range(len(requests) + 1, 4):
        call = dict(kwargs, previous_turn_id=previous)
        if error is not None:
            call['text'] = (kwargs['text'] + '\nVALIDATION REPAIR ONLY: The previous completed response was rejected. '
                            'Keep its scientific action/intent and the same observation cutoff; correct only the inconsistent '
                            'fields. Python will not edit or normalize your response. Return the full valid object. '
                            'ABSTAIN requires selected_count=0 and automatic_weight=0. Ranking weights must sum to 1 '
                            'even when unused during ABSTAIN. Error: ' + error + '\nRejected response: ' + rejected)
        prefix = archive / ('attempt_%02d' % attempt)
        _dump_once(prefix.with_name(prefix.name + '_request.json'),
                   {'thread_id': thread_id, 'previous_turn_id': previous, 'text': call['text'],
                    'candidate_count': candidate_count, 'attempt': attempt})
        # Transport exceptions propagate; they never consume another validation call.
        result = client.run_structured_turn(**call)
        _dump_once(prefix.with_name(prefix.name + '_raw.json'), result)
        _completed(result, thread_id)
        try:
            value = validate_selection(json.loads(result['final_text']), candidate_count)
        except (ValueError, KeyError, TypeError) as exc:
            error, rejected, previous = str(exc), result['final_text'], result['turn_id']
            failure = {'turn_id': previous, 'error_type': type(exc).__name__, 'error': error}
            _dump_once(prefix.with_name(prefix.name + '_validation_error.json'), failure)
            if not (archive / 'first_validation_error.json').exists():
                _dump_once(archive / 'first_validation_error.json', failure)
            continue
        if not legacy.exists():
            _dump_once(legacy, result)
        _dump_once(archive / 'accepted_result.json', result)
        return value, result
    raise ValueError('Automatic-negative selection exhausted three validation attempts: ' + str(error))
