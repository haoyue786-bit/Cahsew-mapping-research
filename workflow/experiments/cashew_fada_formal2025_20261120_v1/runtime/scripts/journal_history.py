"""Query complete point-isolated evidence from the current run only."""
import hashlib
import json
from pathlib import Path

from complete_history_store import (_identity, candidate, load_epoch, observations,
                                    review_data, save_initialization, human_history)

KINDS = frozenset(('initialization', 'training', 'controls', 'source',
                   'unlabelled_panel', 'action', 'reasoning', 'review', 'queries', 'candidates', 'candidate_details', 'human_training', 'annotated_training', 'trials', 'module_detail', 'module_gradients','interventions'))


def fetch(root, *, job_id, run_id, cutoff_epoch, start_epoch, end_epoch, kinds, limit=40):
    if any(type(v) is not int for v in (cutoff_epoch, start_epoch, end_epoch, limit)):
        raise ValueError('integer history bounds required')
    if not (0 <= start_epoch <= end_epoch < cutoff_epoch and end_epoch - start_epoch <= 199 and 1 <= limit <= 40):
        raise ValueError('invalid or future history range')
    if not isinstance(kinds, list) or not kinds or len(kinds) != len(set(kinds)) or set(kinds) - KINDS:
        raise ValueError('invalid evidence kinds')
    root = Path(root)
    identity = _identity(root, job_id, run_id)
    if end_epoch > identity.get('max_epoch', 0):
        raise ValueError('history exceeds this run')
    rows = []
    if start_epoch == 0 and any(k in kinds for k in ('initialization', 'human_training', 'candidate_details')):
        row = {'epoch': 0}
        if 'initialization' in kinds:
            row['initialization'] = save_initialization(root, job_id, run_id)
        if 'human_training' in kinds:
            row['human_training'] = human_history(root, job_id, run_id, 0)
        if 'candidate_details' in kinds:
            from checkpoint_selection_evidence_v2 import collect
            row['candidate_details'] = collect(root, job_id, run_id, 0)
        rows.append(row)
    for epoch in range(max(1, start_epoch), end_epoch + 1):
        if not (root / 'journal' / ('epoch_%06d' % epoch) / 'commit.json').is_file():
            continue
        snapshot, commit = load_epoch(root, job_id, run_id, epoch)
        row = {'epoch': epoch, 'checkpoint_sha256': snapshot['checkpoint_sha256']}
        if 'trials' in kinds:
            from complete_history_store import _project
            entries=[]
            for path in sorted((root/'trial_evidence'/('epoch_%06d'%epoch)).glob('*/intent.json')):
                intent=json.loads(path.read_text()); result_path=path.parent/'result.json'
                if intent['job_id']!=job_id or intent['run_id']!=run_id or intent['epoch']!=epoch:
                    raise ValueError('trial history identity mismatch')
                entries.append({'arguments':intent['arguments'],'result':json.loads(result_path.read_text()) if result_path.is_file() else {'status':'incomplete'}})
            row['trials']=_project(entries)
        if 'candidate_details' in kinds and commit.get('route') == 'FULL_REVIEW' and commit.get('decision_kind') == 'agent':
            from checkpoint_selection_evidence_v2 import collect
            row['candidate_details'] = collect(root, job_id, run_id, epoch)
        for kind in ('training', 'controls'):
            if kind in kinds:
                row[kind] = dict(snapshot[kind])
                if kind=='training' and 'module_gradient_probes' in row[kind]:
                    from module_observer_v2 import project as project_modules
                    raw_probes=row[kind].pop('module_gradient_probes')
                    row[kind]['module_gradient_summary']={k:v for k,v in project_modules({'module_gradient_probes':raw_probes}).items()}
                    row[kind]['module_gradient_detail_available']='query kind module_gradients for each sampled update'
        if 'module_detail' in kinds:row['module_detail']=snapshot['training'].get('module_observation')
        if 'module_gradients' in kinds:row['module_gradients']=snapshot['training'].get('module_gradient_probes')
        if 'interventions' in kinds:
            from training_recall_v3 import ledger
            row['interventions']=ledger(root,job_id,run_id,epoch,cutoff_epoch)
        if 'human_training' in kinds:
            row['human_training'] = human_history(root, job_id, run_id, epoch, snapshot, commit)
        if 'annotated_training' in kinds:
            from complete_history_store import _project
            obs = root/'journal'/('epoch_%06d'%epoch)/'review_observation.json'
            fit = None
            if obs.is_file() and commit.get('route') == 'FULL_REVIEW':
                measured = json.loads(obs.read_text())
                if measured['checkpoint_sha256'] != snapshot['checkpoint_sha256']:
                    raise ValueError('Annotated training review differs from committed epoch')
                fit = measured.get('annotated_training')
            row['annotated_training'] = _project(dict(exposure=snapshot['training'].get('annotated_training_updates'), fit=fit))
        if any(k in kinds for k in ('action', 'reasoning', 'review', 'queries')):
            action, reasoning, review, queries = review_data(root, job_id, run_id, epoch, commit,include_queries=bool(set(kinds)&{'review','queries'}))
            if 'action' in kinds:
                row['action'] = action
            if 'reasoning' in kinds and reasoning is not None:
                row['reasoning'] = reasoning
            if 'review' in kinds and review is not None:
                row['review'] = review
            if 'queries' in kinds and queries is not None:
                row['queries'] = queries
        if 'candidates' in kinds:
            frozen = candidate(root, job_id, run_id, epoch, snapshot['checkpoint_sha256'], cutoff_epoch)
            if frozen is not None:
                row['candidate'] = frozen
        if 'source' in kinds or 'unlabelled_panel' in kinds:
            source, panel = observations(root, epoch, snapshot['checkpoint_sha256'],
                                         job_id=job_id, run_id=run_id, commit=commit)
            if 'source' in kinds and source is not None:
                row['source'] = source
            if 'unlabelled_panel' in kinds and panel is not None:
                row['unlabelled_panel'] = panel
        rows.append(row)
    if len(rows) > limit:
        raise ValueError('more than 40 records; choose a narrower epoch range')
    result = {'version': 'same_run_complete_history_v2', 'job_id': job_id, 'run_id': run_id,
              'cutoff_epoch': cutoff_epoch, 'start_epoch': start_epoch, 'end_epoch': end_epoch,
              'semantics': 'Only committed same-run records. Missing review measurements were not collected at that epoch. Target point evidence is excluded.',
              'records': rows}
    result['sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return result
