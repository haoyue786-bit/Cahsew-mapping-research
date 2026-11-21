"""Point-isolated, same-run training records and human-readable CSV views.

Records are projected from the worker's own committed checkpoints and sealed
transactions.  The CSV files are views; authenticated journal and transaction
files remain the source of truth.  Never open target point evaluation data.
"""
from query_progress_v1 import numbered_files
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path

from native_safe_checkpoint_codec_v1 import decode


def _canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _project(value, path=()):
    """Copy authenticated training records without keyword/text censorship.

    Evaluation files are excluded by the history producer/ownership boundary.
    Explicit forbidden point fields fail rather than silently changing evidence.
    """
    from online_boundary import BLOCK, TARGET_CONTEXT, POINT_RESULTS
    if value is None or type(value) in (bool, str, int, float):
        if type(value) is float and not math.isfinite(value):
            raise ValueError('non-finite historical measurement')
        return value
    if isinstance(value, (list, tuple)):
        return [_project(x, path) for x in value]
    if isinstance(value, dict):
        result = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError('history keys must be strings')
            child = value[key]
            if key == 'target_point_evidence' and child is False:
                result[key] = False
                continue
            name = key.lower()
            target_context = any(p.lower() in TARGET_CONTEXT for p in path)
            if name in BLOCK or (target_context and name in POINT_RESULTS):
                raise ValueError('Evaluation data is not a current-run training producer: '+'.'.join(path+(key,)))
            result[key] = _project(child, path + (key,))
        return result
    raise ValueError('unsupported value in worker record: ' + type(value).__name__)


def _write_immutable(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError('unsafe evidence path')
    fd, temp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != raw:
                raise ValueError('immutable evidence record differs: ' + path.name)
        dfd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        os.unlink(temp)


def _flat(value, pointer=''):
    if isinstance(value, dict):
        if not value:
            yield pointer, '{}'
        for key in sorted(value):
            yield from _flat(value[key], pointer + '/' + key.replace('~', '~0').replace('/', '~1'))
    elif isinstance(value, list):
        if not value:
            yield pointer, '[]'
        for index, item in enumerate(value):
            yield from _flat(item, pointer + '/' + str(index))
    else:
        yield pointer or '/', json.dumps(value, ensure_ascii=False, allow_nan=False)


def _csv_view(path, value):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(('field', 'value_json'))
    writer.writerows(_flat(value))
    _write_immutable(path, stream.getvalue().encode('utf-8'))


def _read_json(path, *, transcript=False):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or unsafe history file: ' + path.name)
    # Full authenticated dialogue/tool transcripts grow with actual queries.
    # Keep the small-metadata bound, but never truncate valid transcripts.
    if transcript:
        is_result = path.name == 'raw_result.json' and path.parent.name.startswith('history_attempt_')
        is_tool = bool(re.fullmatch(r'(trial|detail)_tool_call_[0-9]+\.json', path.name))
        if not (is_result or is_tool):
            raise ValueError('unexpected transcript path')
        with path.open('r', encoding='utf-8') as stream:
            return json.load(stream)
    raw = path.read_bytes()
    if len(raw) > 5_000_000:
        raise ValueError('oversized history record')
    return json.loads(raw)


def _identity(root, job_id, run_id):
    identity = _read_json(Path(root) / 'journal/identity.json')
    if identity.get('job_id') != job_id or identity.get('run_id') != run_id:
        raise ValueError('cross-run history access')
    return identity


def save_epoch(root, job_id, run_id, epoch, checkpoint):
    """Create an immutable classified snapshot from the actual trained bytes."""
    root = Path(root)
    _identity(root, job_id, run_id)
    if type(epoch) is not int or epoch < 1 or not isinstance(checkpoint, bytes):
        raise ValueError('invalid trained epoch')
    directory = root / 'journal' / ('epoch_%06d' % epoch)
    metadata = _read_json(directory / 'trained.json')
    digest = _sha(checkpoint)
    if metadata.get('epoch') != epoch or metadata.get('checkpoint_sha256') != digest:
        raise ValueError('training snapshot checkpoint mismatch')
    payload = decode(checkpoint)
    if payload.get('epoch') != epoch or not isinstance(payload.get('metrics'), dict):
        raise ValueError('invalid trained checkpoint')
    from predictor_artifacts_v3 import publish as publish_predictor
    if (root/'job_candidate.json').exists():
        publish_predictor(root,_read_json(root/'job_candidate.json'),payload,digest,str((directory/'trained_checkpoint.bin').relative_to(root)))
    metrics = _project(payload['metrics'], ('training',))
    config = _project(payload['config'], ('controls',))
    record = {'version': 'point_isolated_own_run_epoch_v1', 'job_id': job_id,
              'run_id': run_id, 'epoch': epoch, 'checkpoint_sha256': digest,
              'training': metrics, 'controls': config}
    store = root / 'evidence_store'
    stem = 'epoch_%06d' % epoch
    _write_immutable(store / 'snapshots' / (stem + '.json'), _canonical(record))
    _csv_view(store / 'training' / (stem + '.csv'), metrics)
    _csv_view(store / 'controls' / (stem + '.csv'), config)
    _csv_view(store / 'human_exposure' / (stem + '.csv'), human_response(record))
    if 'module_observation' in metrics:
        _csv_view(store / 'module_detail' / (stem + '.csv'), metrics['module_observation']['components'])
        _csv_view(store / 'batchnorm_detail' / (stem + '.csv'), metrics['module_observation']['batchnorm'])
    if 'module_gradient_probes' in metrics:
        _csv_view(store / 'module_gradients' / (stem + '.csv'), [dict(step=p['step'], component=c, **v) for p in metrics['module_gradient_probes'] for c,v in p['components'].items()])
    if 'module_dynamics' in metrics:
        _csv_view(store / 'model_dynamics' / (stem + '.csv'), metrics['module_dynamics']['components'])
        _csv_view(store / 'batchnorm' / (stem + '.csv'), metrics['module_dynamics']['batchnorm'])
    return record


def human_response(snapshot):
    """Same-epoch execution, not scheduled availability or independent accuracy."""
    metrics, config = snapshot['training'], snapshot['controls']
    return {'epoch': snapshot['epoch'], 'role': 'human_training_polygons', 'independent': False,
            'controls': {k: config[k] for k in ('supplementary_loss_weight', 'supplementary_positive_scale',
                'supplementary_interval_steps', 'supplementary_occurrence_cap', 'supplementary_horizon_epochs') if k in config},
            'actual_exposure': metrics.get('supplementary_actual_exposure'),
            'component_response': metrics.get('negative_controls'),
            'gradient_direction': metrics.get('gradient_direction'),
            'measurement_semantics': 'Execution belongs to this epoch. Missing measurements are unknown. '
                'Gradient probes are sampled component diagnostics, not optimizer update ratios or causal effects.'}


def human_history(root, job_id, run_id, epoch, snapshot=None, commit=None):
    from human_training_evidence_contract import validate, artifact
    root = Path(root)
    if epoch == 0:
        initial = save_initialization(root, job_id, run_id)
        evidence = initial['request_evidence'].get('human_training')
        result = {'epoch': 0, 'actual_exposure': None, 'measurement': None}
    else:
        if snapshot is None or commit is None or snapshot['epoch'] != epoch:
            raise ValueError('human history requires authenticated epoch snapshot')
        result = human_response(snapshot); result['measurement'] = None
        evidence = None
        if commit.get('route') == 'FULL_REVIEW' and commit.get('decision_kind') == 'agent':
            values, _ = _transaction(root, job_id, run_id, epoch, 'FULL_REVIEW')
            evidence = values['request'].get('evidence', {}).get('human_training')
            obs = _read_json(root / 'journal' / ('epoch_%06d' % epoch) / 'review_observation.json')
            if (obs.get('epoch') != epoch or obs.get('checkpoint_sha256') != snapshot['checkpoint_sha256']
                    or (evidence or {}).get('values') != obs.get('human_training')):
                raise ValueError('human history differs from committed review')
    if evidence is not None:
        value = validate(evidence['values'])
        expected = artifact(value); expected.pop('kind')
        if (evidence != expected or (value['job_id'], value['run_id'], value['epoch']) != (job_id, run_id, epoch)):
            raise ValueError('human history ownership/digest differs')
        result['measurement'] = value
    result['probe_status'] = 'measured_at_this_epoch' if evidence is not None else 'not_collected_at_this_epoch'
    _csv_view(root / 'evidence_store/human_history' / ('epoch_%06d.csv' % epoch), result)
    return result


def load_epoch(root, job_id, run_id, epoch):
    root = Path(root)
    identity = _identity(root, job_id, run_id)
    if type(epoch) is not int or not 1 <= epoch <= identity.get('max_epoch', 0):
        raise ValueError('invalid history epoch')
    directory = root / 'journal' / ('epoch_%06d' % epoch)
    commit = _read_json(directory / 'commit.json')
    trained = _read_json(directory / 'trained.json')
    if (commit.get('epoch') != epoch or trained.get('epoch') != epoch or
        commit.get('trained_checkpoint_sha256') != trained.get('checkpoint_sha256') or
        commit.get('route') != trained.get('route')):
        raise ValueError('history is not an authenticated committed epoch')
    path = root / 'evidence_store/snapshots' / ('epoch_%06d.json' % epoch)
    if not path.exists():
        if os.environ.get('CASHEW_FAST_RESUME') == '1':
            raise ValueError('history snapshot is missing; reconcile rather than read an old checkpoint')
        # Recovery for a crash between writing the trained checkpoint and its
        # derivative snapshot; the checkpoint is rehashed before projection.
        checkpoint = directory / 'trained_checkpoint.bin'
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise ValueError('trained checkpoint missing')
        save_epoch(root, job_id, run_id, epoch, checkpoint.read_bytes())
    record = _read_json(path)
    if (record.get('job_id') != job_id or record.get('run_id') != run_id or
        record.get('epoch') != epoch or record.get('checkpoint_sha256') != trained.get('checkpoint_sha256')):
        raise ValueError('history snapshot identity mismatch')
    # A second projection makes tampering with forbidden keys fail closed.
    if _project(record['training'], ('training',)) != record['training'] or _project(record['controls'], ('controls',)) != record['controls']:
        raise ValueError('history snapshot contains excluded information')
    return record, commit


def _transaction(root, job_id, run_id, epoch, kind):
    if kind == 'initialization':
        base = Path(root) / 'agent/sealed_v1/initialization/transaction'
        expected_kind = 'initialization'
    else:
        folder = 'light' if kind == 'LIGHT_REVIEW' else 'full'
        base = Path(root) / 'agent/sealed_v1' / ('epoch_%03d' % epoch) / folder / 'transaction'
        expected_kind = 'light_review' if kind == 'LIGHT_REVIEW' else 'full_review'
    commit = _read_json(base / 'commit.json')
    if commit.get('identity') != {'job_id': job_id, 'run_id': run_id, 'epoch': epoch, 'kind': expected_kind}:
        raise ValueError('history transaction identity mismatch')
    values = {}
    for name, key in (('request.json', 'request_sha256'), ('response.json', 'response_sha256'),
                      ('decision.json', 'decision_sha256')):
        path = base / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('missing history transaction artifact')
        raw = path.read_bytes()
        if len(raw) > 5_000_000 or _sha(raw) != commit.get(key):
            raise ValueError('history transaction changed')
        values[name[:-5]] = json.loads(raw)
    if values['request'].get('capabilities', {}).get('target_point_evidence') is not False:
        raise ValueError('history request is not target-point isolated')
    if values['response'].get('action') != values['decision'].get('action'):
        raise ValueError('history decision does not match response')
    return values, commit


def save_initialization(root, job_id, run_id):
    root = Path(root)
    _identity(root, job_id, run_id)
    values, txn = _transaction(root, job_id, run_id, 0, 'initialization')
    initial = _read_json(root / 'journal/initial.json')
    if initial.get('epoch') != 0 or initial.get('checkpoint_sha256') != txn.get('post_checkpoint_sha256'):
        raise ValueError('initialization decision was not committed to this journal')
    safe = {'epoch': 0, 'action': values['decision']['action'],
            'request_evidence': _project(values['request'].get('evidence', {}), ('evidence',)),
            'response': _project(values['response'], ('response',)),
            'decision': _project(values['decision'], ('decision',)),
            'transaction_sha256': txn['decision_sha256']}
    _csv_view(root / 'evidence_store/initialization' / 'epoch_000000.csv', safe)
    return safe


def review_data(root, job_id, run_id, epoch, commit, include_queries=True):
    root = Path(root)
    route = commit.get('route')
    action = {'epoch': epoch, 'route': route, 'decision_kind': commit.get('decision_kind'),
              'action': commit.get('action')}
    default_file = root / 'journal' / ('epoch_%06d' % epoch) / (
        'default_skip.json' if route == 'SKIP' else
        'default_before_agent.json' if route == 'FULL_REVIEW' else 'default_after_agent.json')
    if default_file.exists():
        default = _read_json(default_file)
        if default.get('epoch') != epoch:
            raise ValueError('default transition epoch mismatch')
        action['default_event'] = _project(default.get('event'), ('default_event',))
    review = reasoning = queries = None
    if commit.get('decision_kind') == 'agent':
        if route not in ('FULL_REVIEW', 'LIGHT_REVIEW'):
            raise ValueError('unknown committed review kind')
        values, txn = _transaction(root, job_id, run_id, epoch, route)
        if values['decision'].get('action') != action['action']:
            raise ValueError('review action does not match journal')
        action['intervention'] = _project(values['decision'].get('intervention', {}), ('intervention',))
        action['decision_sha256'] = txn['decision_sha256']
        reasoning = _project(values['response'], ('response',))
        transport = (root / 'agent/sealed_v1' / ('epoch_%03d' % epoch) /
                     ('full' if route == 'FULL_REVIEW' else 'light') / 'transport')
        queries = []
        if include_queries and (transport / 'receipt.json').exists():
            receipt = _read_json(transport / 'receipt.json')
            files = numbered_files(transport, 'tool_call_')
            hashes = []
            for index, path in enumerate(files, 1):
                if path.name != 'tool_call_%02d.json' % index or path.is_symlink():
                    raise ValueError('invalid saved history query')
                raw = path.read_bytes()
                record = json.loads(raw)
                from training_tool_output_v1 import verify_delivery
                verify_delivery(record,'query_training_history',index)
                params = record.get('request', {}).get('params', {})
                if params.get('tool') != 'query_training_history':
                    raise ValueError('unexpected saved history tool')
                hashes.append(_sha(raw))
                queries.append({'arguments': _project(params.get('arguments', {}), ('arguments',)),
                                'result_sha256': _sha(_canonical(record.get('result'))),
                                'recorded_epochs': [x.get('epoch') for x in record.get('result', {}).get('records', [])]})
            trial_files = sorted(transport.glob('trial_tool_call_*.json'))
            if trial_files:
                from paired_trial_v1 import verify_transcripts
                turn_ids = [receipt['turn_id']]
                for old in sorted(transport.glob('history_attempt_*/raw_result.json')):
                    tid = _read_json(old, transcript=True).get('turn_id')
                    if tid and tid not in turn_ids: turn_ids.append(tid)
                hashes += verify_transcripts(transport,root,epoch,values['request'],receipt['thread_id'],turn_ids)
                for path in trial_files:
                    record = _read_json(path, transcript=True)
                    queries.append({'tool':'run_matched_training_trial',
                                    'arguments':_project(record['request']['params']['arguments'],('arguments',)),
                                    'status':record['result']['status'],'trial_id':record['result'].get('trial_id')})
            from training_tool_output_v1 import verify_read_transcripts
            detail_turn_ids = [receipt['turn_id']]
            for old in sorted(transport.glob('history_attempt_*/raw_result.json')):
                tid = _read_json(old, transcript=True).get('turn_id')
                if tid and tid not in detail_turn_ids: detail_turn_ids.append(tid)
            hashes += verify_read_transcripts(transport,receipt['thread_id'],detail_turn_ids)
            for path in numbered_files(transport, 'detail_tool_call_'):
                record = _read_json(path, transcript=True)
                queries.append({'tool':'read_training_tool_result',
                                'arguments':_project(record['request']['params']['arguments'],('arguments',)),
                                'status':record['result']['status']})
            audit = hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode()).hexdigest()
            if audit != receipt.get('history_audit_sha256'):
                raise ValueError('saved history tool transcript changed')
        review = {'request_evidence': _project(values['request'].get('evidence', {}), ('evidence',)),
                  'decision': _project(values['decision'], ('decision',)),
                  'history_queries': queries}
        store = root / 'evidence_store'
        stem = 'epoch_%06d.csv' % epoch
        if include_queries:
            _csv_view(store / 'reviews' / stem, review)
            _csv_view(store / 'agent_queries' / stem, queries)
        _csv_view(store / 'agent_reasoning' / stem, reasoning)
    _csv_view(root / 'evidence_store/actions' / ('epoch_%06d.csv' % epoch), action)
    return action, reasoning, review, queries


def candidate(root, job_id, run_id, epoch, checkpoint_sha256, cutoff):
    root = Path(root)
    path = root / 'frozen_candidates.json'
    if not path.exists():
        return None
    manifest = _read_json(path)
    if (manifest.get('job_id') != job_id or manifest.get('run_id') != run_id or
        type(manifest.get('stop_epoch')) is not int):
        raise ValueError('candidate freeze identity mismatch')
    if manifest['stop_epoch'] >= cutoff:
        # A later freeze must never change an earlier decision's view.
        return None
    stop = _read_json(root / 'stop_receipt.json')
    if (
        stop.get('epoch') != manifest['stop_epoch'] or
        manifest.get('stop_receipt_sha256') != _sha((root / 'stop_receipt.json').read_bytes())):
        raise ValueError('candidate freeze identity or cutoff mismatch')
    matches = [x for x in manifest.get('candidates', []) if x.get('epoch') == epoch]
    if not matches:
        return None
    if len(matches) != 1 or matches[0].get('sha256') != checkpoint_sha256:
        raise ValueError('candidate differs from trained checkpoint')
    result = {'epoch': epoch, 'checkpoint_sha256': checkpoint_sha256, 'frozen_candidate': True,
              'stop_epoch': manifest['stop_epoch']}
    _csv_view(root / 'evidence_store/candidates' / ('epoch_%06d.csv' % epoch), result)
    return result


def observations(root, epoch, checkpoint_sha256, *, job_id=None, run_id=None, commit=None):
    path = Path(root) / 'journal' / ('epoch_%06d' % epoch) / 'review_observation.json'
    if not path.exists():
        return None, None
    obs = _read_json(path)
    if obs.get('epoch') != epoch or obs.get('checkpoint_sha256') != checkpoint_sha256:
        raise ValueError('review observation checkpoint mismatch')
    if commit is not None:
        if commit.get('route') != 'FULL_REVIEW' or commit.get('decision_kind') != 'agent':
            raise ValueError('review observation lacks committed full review')
        values, _ = _transaction(root, job_id, run_id, epoch, 'FULL_REVIEW')
        evidence = values['request'].get('evidence') or {}
        if ((evidence.get('source_validation') or {}).get('values') != obs.get('source') or
            (evidence.get('unlabelled_panel') or {}).get('values') != obs.get('panel')):
            raise ValueError('review observations differ from committed request')
    source = _project(obs.get('source', {}), ('source_validation',))
    panel = _project(obs.get('panel', {}), ('unlabelled_panel',))
    _csv_view(Path(root) / 'evidence_store/source_validation' / ('epoch_%06d.csv' % epoch), source)
    _csv_view(Path(root) / 'evidence_store/unlabelled_panel' / ('epoch_%06d.csv' % epoch), panel)
    return source, panel
