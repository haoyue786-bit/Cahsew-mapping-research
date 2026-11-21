"""Bounded trial comparison plus authenticated paging of complete tool results.

The native measured results remain immutable. Delivery views are separate and
verified, and the reader can inspect any JSON subtree of a prior tool result in
this exact thread, including a live trial before the epoch is committed.
"""
import copy
import hashlib
import json
from pathlib import Path

PAGE_CHARS = 6000
TOOLS = {'query_training_history': 'tool_call_',
         'run_matched_training_trial': 'trial_tool_call_'}
SPEC = {'type': 'function', 'name': 'read_training_tool_result',
        'description': 'Read complete details of an already returned history or matched-trial tool result in this thread. Use this if a large history result was truncated or a compact trial needs more evidence. A live trial is readable here before epoch commit; do not query the current epoch through history. Choose the source tool and its call_index from _full_result. pointer is an RFC6901 JSON pointer (empty means the root); offset starts at 0. Small subtrees return value; large subtrees return JSON text pages of 6000 characters with next_offset and child_keys. All original evidence remains available, including both trial arms, every module and gradient. This only reads saved results; it never repeats training.',
        'inputSchema': {'type': 'object', 'additionalProperties': False,
            'required': ['source_tool', 'call_index', 'pointer', 'offset'],
            'properties': {'source_tool': {'type': 'string', 'enum': sorted(TOOLS)},
                'call_index': {'type': 'integer', 'minimum': 1, 'maximum': 64},
                'pointer': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 0}}}}


LEGACY_SPEC = copy.deepcopy(SPEC)
SPEC['inputSchema']['properties']['call_index'].pop('maximum')
SPEC['description'] += (' Code-mode replies may be JSON strings or content envelopes: decode before reading fields. '
    'Read status before using value/json_text. Follow the complete next_request exactly; next_request=null means done. '
    'Never loop on next_offset!==null without checking it is an integer greater than the current offset. '
    'Identical requests are idempotent, not an automatic next-page command. A missing offset is invalid.')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def locator(source_tool, index):
    return {'reader_tool': SPEC['name'], 'source_tool': source_tool, 'call_index': index,
            'pointer': '', 'offset': 0,
            'note': 'Complete original result is saved. Read any subtree or successive pages; no training is rerun.'}


def delivery(result, source_tool, index, *, annotation_summary=True):
    """Put retrieval instructions first; give both trial endpoints before detail."""
    out = {'_full_result': locator(source_tool, index)}
    if source_tool == 'query_training_history' or result.get('status') != 'completed':
        out.update(result)
        return out
    for key in ('version', 'trial_id', 'status', 'epoch', 'formal_state_adopted',
                'all_initial_states_matched', 'branch_checkpoint_selection_allowed',
                'checkpoint_sha256', 'subset_sha256', 'source_target_minibatch_seed',
                'elapsed_seconds', 'scope'):
        if key in result: out[key] = result[key]
    measured_annotation = annotation_summary and any(
        item.get('annotated_training_updates') is not None
        for arm in result['arms'].values() for item in arm.get('observations',[]))
    out['view'] = 'compact_trial_comparison_v2' if measured_annotation else 'compact_trial_comparison_v1'
    def endpoint(value):
        # Endpoint metadata includes scalar BN mode and a textual scope. They
        # are evidence too; only measured mapping fields have functional keys.
        return {kind: ({k: v for k, v in fields.items() if not k.startswith('functional_')}
                       if isinstance(fields, dict) else copy.deepcopy(fields))
                for kind, fields in value.items()}
    out['baseline'] = endpoint(result['baseline'])
    out['arms'] = {}
    for role in ('control', 'treatment'):
        arm = result['arms'][role]
        out['arms'][role] = {k: arm[k] for k in ('completed_generator_updates',
                            'source_target_batch_indices_sha256', 'effective_controls') if k in arm}
        out['arms'][role]['end'] = endpoint(arm['end'])
        if measured_annotation:
            updates = [row['annotated_training_updates'] for row in arm['observations']
                       if row.get('annotated_training_updates') is not None]
            probes = [probe for row in updates for probe in row['gradient_probes']]
            out['arms'][role]['annotation_exposure'] = {
                'optimizer_steps':sum(row['completed_steps'] for row in updates),
                'class_pixel_occurrences':[sum(row['pixels_by_class'][i] for row in updates) for i in range(4)],
                'nonzero_annotation_gradient_probes':sum(probe['annotation_norm']>0 for probe in probes),
                'by_epoch':[dict(epoch=row['epoch'],optimizer_steps=row['completed_steps'],
                                 raw_four_class_loss=row['raw_loss']) for row in updates],
                'first_observation_pointer':'/arms/'+role+'/observations/0/annotated_training_updates',
                'last_observation_pointer':'/arms/'+role+'/observations/'+str(len(updates)-1)+'/annotated_training_updates',
            }
    out['detail_paths'] = {
        role: {kind: '/arms/' + role + '/' + kind for kind in ('training', 'end', 'effective_controls')}
        for role in ('control', 'treatment')}
    out['detail_note'] = 'Module motion, BN, per-component gradients, actual human exposure and functional activations are in the complete result. Read relevant subtrees before making claims about them. Source and unlabelled subset endpoints are not target accuracy.'
    if measured_annotation:
        out['detail_note'] = ('The compact view includes both arms\' actual annotation dose and loss. '
                              'Read individual epoch subtrees for further gradient detail, or the training subtree '
                              'for module observations. Fit describes the shared annotation masks.')
    if len(canonical(out)) > 12000:
        raise ValueError('trial comparison exceeds bounded delivery budget')
    return out


def verify_delivery(record, source_tool, index):
    presented = record.get('delivered_result')
    if presented is not None and presented != delivery(record['result'],source_tool,index,
                            annotation_summary=presented.get('view')!='compact_trial_comparison_v1'):
        raise ValueError('delivered tool evidence differs from measured result')


def thread_matches(recorded, current):
    """Recognize exact previous tool contracts without weakening other fields."""
    if recorded == current: return True
    for legacy_reader in (LEGACY_SPEC, None):
        legacy = copy.deepcopy(current)
        legacy['dynamicTools'] = [legacy_reader if item == SPEC else item
                                  for item in current['dynamicTools'] if item != SPEC or legacy_reader is not None]
        if recorded == legacy: return True
    return False


def read(stage, arguments, thread_id, *, version=1):
    if version not in (1, 2): raise ValueError("unknown detail reader version")
    # A missing offset means the first page; keep original arguments in the receipt.
    if version == 1 and isinstance(arguments, dict) and 'offset' not in arguments:
        arguments = dict(arguments, offset=0)
    if (not isinstance(arguments, dict) or set(arguments) != {'source_tool', 'call_index', 'pointer', 'offset'}
            or arguments.get('source_tool') not in TOOLS
            or type(arguments.get('call_index')) is not int or (arguments['call_index'] < 1 or (version == 1 and arguments['call_index'] > 64))
            or not isinstance(arguments.get('pointer'), str) or len(arguments['pointer']) > 1000
            or type(arguments.get('offset')) is not int or arguments['offset'] < 0):
        return {'status': 'invalid_arguments', 'message': ('Use an advertised source tool, call_index 1..64, JSON pointer and nonnegative character offset.' if version == 1 else 'Decode the tool reply first. Supply source_tool, positive call_index, JSON pointer and explicit integer offset. Start at 0; follow next_request only after checking status=ok. Missing/undefined offset is invalid.')}
    path = Path(stage) / (TOOLS[arguments['source_tool']] + '%02d.json' % arguments['call_index'])
    if path.is_symlink() or not path.is_file():
        return {'status': 'unavailable', 'message': 'That tool result has not been returned in this thread.'}
    raw = path.read_bytes(); record = json.loads(raw); params = record['request']['params']
    if params.get('threadId') != thread_id or params.get('tool') != arguments['source_tool']:
        raise ValueError('detail result belongs to another thread')
    value = record['result']; pointer = arguments['pointer']
    try:
        if pointer and not pointer.startswith('/'): raise ValueError('invalid pointer')
        for token in pointer.split('/')[1:] if pointer else []:
            token = token.replace('~1', '/').replace('~0', '~')
            if isinstance(value, list):
                if not token.isdecimal(): raise ValueError('invalid array index')
                value = value[int(token)]
            else: value = value[token]
    except (KeyError, IndexError, TypeError, ValueError):
        return {'status': 'invalid_pointer', 'message': 'Read the parent pointer and use its child_keys or array indices.'}
    encoded = canonical(value); offset = arguments['offset']
    if offset > len(encoded):
        return {'status': 'invalid_offset', 'total_characters': len(encoded)}
    out = {'status': 'ok', **arguments, 'source_record_sha256': hashlib.sha256(raw).hexdigest(),
           'total_characters': len(encoded), 'next_offset': None}
    if offset == 0 and len(encoded) <= PAGE_CHARS:
        out['value'] = value
    else:
        out['json_text'] = encoded[offset:offset + PAGE_CHARS]
        if offset + PAGE_CHARS < len(encoded): out['next_offset'] = offset + PAGE_CHARS
        if isinstance(value, dict):
            out['child_keys'] = list(value)[:32]
            out['child_key_count'] = len(value)
        elif isinstance(value, list): out['array_length'] = len(value)
    if version == 2:
        out['reader_version'] = 2
        out['complete'] = out['next_offset'] is None
        out['next_request'] = None if out['complete'] else dict(arguments, offset=out['next_offset'])
    from online_boundary import verify_online
    verify_online(out)
    return out


def verify_read_transcripts(stage, thread_id, turn_ids):
    hashes = []
    from query_progress_v1 import numbered_files, verify_progress
    files = numbered_files(stage, 'detail_tool_call_')
    records = []
    if files:
        thread = json.loads((Path(stage) / 'thread.json').read_text())
        if SPEC not in thread.get('dynamicTools', []) and LEGACY_SPEC not in thread.get('dynamicTools', []):
            raise ValueError('detail reader was not advertised in this thread')
    for i, path in enumerate(files, 1):
        if path.is_symlink() or path.name != 'detail_tool_call_%02d.json' % i:
            raise ValueError('invalid detail transcript')
        raw = path.read_bytes(); record = json.loads(raw); call = record['request']; params = call['params']
        if (call.get('method') != 'item/tool/call' or params.get('tool') != SPEC['name']
                or params.get('threadId') != thread_id or params.get('turnId') not in turn_ids):
            raise ValueError('detail query belongs to another turn')
        version = record.get('reader_version', 1)
        if version == 2 and (SPEC not in thread.get('dynamicTools', []) or 'query_progress' not in record):
            raise ValueError('new reader contract or progress receipt missing')
        if record['result'] != read(stage, params['arguments'], thread_id, version=version):
            raise ValueError('detail query differs from authenticated evidence')
        hashes.append(hashlib.sha256(raw).hexdigest())
        records.append(record)
    verify_progress(records)
    return hashes
