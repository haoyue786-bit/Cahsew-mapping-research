"""CPU-only measured negative-use accounting, with explicit epoch scope.

Selection manifests and schedules never count as executed training. Independent
point labels and accuracy reports are not inputs to this module.
"""
import copy

COUNTS = ('updates', 'patch_occurrences', 'pixel_occurrences')


def counts(value):
    if not isinstance(value, dict):
        return None
    result = {}
    for key in COUNTS:
        x = value.get(key)
        if type(x) is not int or x < 0:
            return None
        result[key] = x
    return result


def collect_epoch_usage(cutoff, log_rows, observed, reviews, current):
    """Prefer explicit epoch facts over inferred log order; retain conflicts."""
    rows = {}
    conflicts = set()

    def add(epoch, value, source, priority, generation=None):
        if type(epoch) is not int or not 0 < epoch <= cutoff:
            return
        measured = counts(value)
        if measured is None:
            return
        old = rows.get(epoch)
        if old and old['priority'] > priority:
            return
        if old and old['priority'] == priority and old['counts'] != measured:
            conflicts.add(epoch)
            return
        if not old or priority > old['priority']:
            conflicts.discard(epoch)
        rows[epoch] = dict(epoch=epoch, generation_epoch=generation,
                           counts=measured, source=source, priority=priority)

    for row in log_rows:
        value = row.get('record', {})
        add(row.get('epoch'), dict(updates=value.get('updates'),
            patch_occurrences=value.get('unique_patches'), pixel_occurrences=value.get('pixels')),
            'training.log; epoch inferred from commit and sequential ordinary records', 1,
            row.get('generation_epoch'))
    for item in reviews:
        record = item.get('record', {})
        review = record.get('review', record)
        add(item.get('epoch'), (review.get('metrics') or {}).get('automatic_negative'),
            'explicit completed review epoch', 2)
    for item in observed:
        add(item.get('epoch'), (item.get('record', {}).get('metrics') or {}).get('automatic_negative'),
            'explicit completed epoch fact', 3)
    if isinstance(current, dict):
        add(current.get('epoch', cutoff), (current.get('metrics') or {}).get('automatic_negative'),
            'current completed review epoch only', 2)
    result = []
    for epoch, row in sorted(rows.items()):
        row = copy.deepcopy(row)
        row.pop('priority')
        if epoch in conflicts:
            row.update(counts=None, conflict=True,
                       meaning='Conflicting same-priority records; never infer zero usage.')
        result.append(row)
    return result


def interval(rows, start, end):
    selected = [r for r in rows if start <= r['epoch'] <= end]
    valid = [r for r in selected if r['counts'] is not None]
    available = {r['epoch'] for r in valid}
    missing = sorted(set(range(start, end + 1)) - available)
    totals = {k: sum(r['counts'][k] for r in valid) for k in COUNTS}
    used = [r['epoch'] for r in valid if r['counts']['updates'] > 0 or r['counts']['patch_occurrences'] > 0]
    status = 'observed_nonzero' if used else ('unknown' if missing else 'observed_zero')
    return dict(start_epoch=start, end_epoch=end, observed_epochs=len(valid),
                missing_or_conflicting_epochs=missing, complete=not missing,
                measured_totals=totals, counts_are_lower_bounds=bool(missing),
                epochs_with_actual_use=used, status=status,
                scope='Ordinary main-training uses in this interval; patch occurrences are not cross-epoch unique patches.')


def build_usage(cutoff, review_epochs, log_rows, observed, reviews, current):
    rows = collect_epoch_usage(cutoff, log_rows, observed, reviews, current)
    previous = max((e for e in review_epochs if e < cutoff), default=0)
    latest = next((r for r in rows if r['epoch'] == cutoff), None)
    recent = interval(rows, previous + 1, cutoff)
    overall = interval(rows, 1, cutoff)
    return dict(version='measured_automatic_negative_usage_v1', cutoff_epoch=cutoff,
        current_epoch_only=latest, since_previous_review=recent, whole_run=overall,
        by_epoch=rows,
        interpretation='Current-epoch zero is not interval zero. observed_nonzero proves actual use, not benefit. Missing records are unknown, never zero. Selected count/dose and a committed manifest do not prove execution. Earlier narrative claims must yield to these measured facts.')


def compact_usage(document):
    result = {k: copy.deepcopy(document[k]) for k in
              ['version', 'cutoff_epoch', 'current_epoch_only', 'since_previous_review', 'interpretation']}
    overall = document['whole_run']
    used = overall['epochs_with_actual_use']
    result['whole_run_observed'] = dict(
        through_epoch=overall['end_epoch'], measured_totals=overall['measured_totals'],
        complete=overall['complete'], counts_are_lower_bounds=overall['counts_are_lower_bounds'],
        missing_or_conflicting_epoch_count=len(overall['missing_or_conflicting_epochs']),
        first_epoch_with_actual_use=min(used) if used else None,
        last_epoch_with_actual_use=max(used) if used else None, status=overall['status'],
        detail_query='automatic_negative_usage /by_epoch with epoch range')
    return result
