"""Candidate same-run STOP/SELECT workflow; never reads target points."""
from pathlib import Path
from native_safe_checkpoint_codec_v1 import decode
from review_contract import build_checkpoint_selection
from sealed_packet import values_digest
from selection_freeze import hash_file

def finalize(worker,*,skill_sha256,verifier_sha256,reference_hashes):
    worker._authorize();worker._authorize_review_sources(None);worker._authorize_stop(None);worker._authorize_selection()
    journal=worker.journal;journal._held();state=journal.recover()
    if state['state'] not in ('STOPPED','STOPPED_AT_LIMIT'):raise ValueError('a committed stop is required')
    from checkpoint_selection_contract_v2 import enabled
    if enabled(worker.job):
        return finalize_complete(worker, state, skill_sha256=skill_sha256,
                                 verifier_sha256=verifier_sha256, reference_hashes=reference_hashes)
    stop=state['epoch'];candidates=[];observations=[];trusted=set()
    for e in range(1,stop+1):
        prefix='epoch_%06d/'%e
        meta=journal._fs.json(prefix+'trained.json');commit=journal._fs.json(prefix+'commit.json')
        if meta.get('route')!='FULL_REVIEW' or commit.get('decision_kind')!='agent':continue
        obs=journal._fs.json(prefix+'review_observation.json')
        if obs.get('epoch')!=e or obs.get('checkpoint_sha256')!=meta.get('checkpoint_sha256'):
            raise ValueError('checkpoint-bound review evidence is missing')
        cp=decode(journal._fs.read(prefix+'trained_checkpoint.bin'))
        source=obs.get('source') or {};panel=obs.get('panel') or {}
        vals={'source_class2_f1':source.get('class2_f1'),'target_unlabelled_class2_fraction':panel.get('predicted_class2_fraction')}
        if any(x is None for x in vals.values()):raise ValueError('source or unlabelled panel evidence missing')
        if cp['metrics'].get('cls') is not None:vals['training_cls_loss']=cp['metrics']['cls']
        art={'kind':'candidate','epoch':e,'values':vals,'sha256':values_digest('candidate',e,vals)}
        candidates.append({'epoch':e,'path':str(journal._dir(e)/'trained_checkpoint.bin')});observations.append(art);trusted.add(('candidate',e,art['sha256']))
    if not candidates or candidates[-1]['epoch']!=stop:raise ValueError('stopped full-review checkpoint is not a candidate')
    txn='agent/sealed_v1/epoch_%03d/full/transaction'%stop
    worker.freeze_stop_and_candidates(txn,candidates)
    signature=hash_file(Path(worker.job['output_root'])/'frozen_candidates.json')
    import os
    history_required=os.environ.get('CASHEW_HISTORY_REQUIRED','0')=='1'
    request=build_checkpoint_selection(stop,observations,trusted,history_required=history_required)
    expected={'stop_epoch':stop}
    if history_required:expected['history_required']=True
    worker.select_checkpoint(request,trusted,expected,manifest_sha256=signature,skill_sha256=skill_sha256,verifier_sha256=verifier_sha256,reference_hashes=reference_hashes)
    return worker.commit_selection('agent/sealed_v1/selection/transaction',signature)


def finalize_complete(worker, state, *, skill_sha256, verifier_sha256, reference_hashes):
    """E0 + every committed full review; no checkpoint scan to build the table."""
    import os
    from checkpoint_selection_contract_v2 import VERSION, artifact, summary
    from checkpoint_selection_evidence_v2 import collect, checkpoint_relative
    if os.environ.get('CASHEW_HISTORY_REQUIRED') != '1':
        raise ValueError('complete selector requires audited history')
    root = Path(worker.job['output_root']); stop = state['epoch']; epochs = [0]
    for e in range(1, stop + 1):
        commit = worker.journal._fs.json('epoch_%06d/commit.json' % e)
        if commit.get('route') == 'FULL_REVIEW' and commit.get('decision_kind') == 'agent':
            epochs.append(e)
    if epochs[-1] != stop:
        raise ValueError('STOP must be a committed full-review candidate')
    from hard_sample_training import enabled as annotated_training
    if annotated_training(worker.job):
        from hard_sample_evidence import freeze_library, fit_candidates
        from json import loads
        pairs = [{'epoch':e,'path':str(root/checkpoint_relative(e))} for e in epochs]
        worker.freeze_stop_and_candidates('agent/sealed_v1/epoch_%03d/full/transaction' % stop,pairs)
        frozen = loads((root/'frozen_candidates.json').read_text())
        library = freeze_library(worker.job,stop)
        h = worker.handles
        fit_candidates(worker.job,h.gan,h.common,h.model,h.device,frozen['candidates'],library)
    candidates = []; observations = []; details = []; trusted = set()
    for epoch in epochs:
        detail = collect(root, worker.job['job_id'], worker.job['run_id'], epoch)
        d = artifact(detail); values = summary(detail)
        c = {'kind': 'candidate', 'epoch': epoch, 'values': values,
             'sha256': values_digest('candidate', epoch, values)}
        candidates.append({'epoch': epoch, 'path': str(root / checkpoint_relative(epoch))})
        observations.append(c); details.append(d)
        trusted.update((x['kind'], x['epoch'], x['sha256']) for x in (c, d))
    if not annotated_training(worker.job):
        worker.freeze_stop_and_candidates('agent/sealed_v1/epoch_%03d/full/transaction' % stop, candidates)
    signature = hash_file(root / 'frozen_candidates.json')
    request = build_checkpoint_selection(stop, observations, trusted, history_required=True,
                                         selection_policy=VERSION, candidate_details=details)
    expected = {'stop_epoch': stop, 'history_required': True,
                'selection_policy': VERSION, 'candidate_details': details}
    worker.select_checkpoint(request, trusted, expected, manifest_sha256=signature,
        skill_sha256=skill_sha256, verifier_sha256=verifier_sha256, reference_hashes=reference_hashes)
    return worker.commit_selection('agent/sealed_v1/selection/transaction', signature)
