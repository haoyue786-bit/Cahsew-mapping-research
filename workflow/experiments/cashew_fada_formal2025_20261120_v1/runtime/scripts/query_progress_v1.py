"""Detect non-progressing tool loops, never limit useful evidence retrieval."""
import json
from pathlib import Path


class QueryStalled(RuntimeError):
    """The exact looping turn has been interrupted and reconciled."""


def numbered_files(stage, prefix):
    return sorted(Path(stage).glob(prefix + '*.json'),
                  key=lambda p: int(p.stem.rsplit('_', 1)[1]))


def signature(tool, arguments):
    return json.dumps([tool, arguments], sort_keys=True, separators=(',', ':'), allow_nan=False)


def error_for(repeat, result):
    if repeat < 3:
        return None
    message = ('Repeated identical evidence request without progress. In code mode, parse the tool reply '
               'before accessing its fields: if typeof raw === "string", use JSON.parse(raw); '
               'if raw.content exists, parse its text block. Inspect the decoded status. '
               'Do not use undefined next_offset as a page position. '
               'Use next_request, or choose a relevant JSON pointer/history interval. '
               'No evidence has been removed and no page was advanced silently.')
    if result.get('next_request'):
        message += ' Next request: ' + json.dumps(result['next_request'], ensure_ascii=False)
    return {'code': -32602, 'message': message}


def annotate(record, previous, repeat):
    params = record['request']['params']
    current = signature(params['tool'], params.get('arguments'))
    repeat = repeat + 1 if current == previous else 1
    record['query_progress'] = {'version': 1, 'repeat': repeat}
    error = error_for(repeat, record.get('delivered_result', record['result']))
    if error is not None:
        record['rpc_error'] = error
    return current, repeat


def verify_progress(records):
    previous, repeat = None, 0
    for record in records:
        if 'query_progress' not in record:
            if 'rpc_error' in record:
                raise ValueError('unaudited tool error')
            previous, repeat = None, 0
            continue
        expected = {k: v for k, v in record.items() if k not in ('query_progress', 'rpc_error')}
        previous, repeat = annotate(expected, previous, repeat)
        if expected != record:
            raise ValueError('query progress receipt differs from actual requests')


def interrupt_stalled(client, params):
    """Stop this exact code loop; uncertain remote state remains blocked."""
    import time
    from complete_history_store import _write_immutable
    client._query_interrupting = True
    tid, turn = params['threadId'], params['turnId']
    client._request('turn/interrupt', {'threadId': tid, 'turnId': turn}, 60)
    for _ in range(10):
        evidence = client.reconcile_thread(thread_id=tid, turn_id=turn, timeout_seconds=60)
        actual = evidence.get('turn') or {}
        if actual.get('id') == turn and actual.get('status') in ('interrupted', 'failed'):
            if not evidence.get('turns') or evidence['turns'][-1].get('id') != turn:
                raise RuntimeError('stalled turn is not the last turn; reconciliation required')
            marker = {'version': 'query_stall_recovery_v1', 'thread_id': tid, 'turn_id': turn,
                      'status': actual['status'], 'remote_reconciled': True,
                      'reason': 'repeated_identical_query', 'checked_at': time.time()}
            _write_immutable(client.history_audit / 'query_stall_reconciled.json',
                             (json.dumps(marker, sort_keys=True) + '\n').encode())
            raise QueryStalled('Repeated query loop interrupted; checkpoint and full evidence preserved')
        if actual.get('status') == 'completed':
            raise RuntimeError('turn completed during interruption; reconcile its answer before recovery')
        time.sleep(.2)
    raise RuntimeError('query-loop interruption is unresolved; never resend')
