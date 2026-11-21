"""Build candidate evidence from committed requests, not unbound CSV summaries."""
from pathlib import Path
from complete_history_store import _identity, _transaction, _read_json, _write_immutable, _csv_view, _canonical
from checkpoint_selection_contract_v2 import validate_detail, TRAIN_FIELDS
from sealed_packet import CONTROL


def checkpoint_relative(epoch):
    return 'journal/initial_checkpoint.bin' if epoch == 0 else 'journal/epoch_%06d/trained_checkpoint.bin' % epoch


def collect(root, job_id, run_id, epoch, *, include_annotation_fit=True):
    root = Path(root); _identity(root, job_id, run_id)
    specification = root / 'job_candidate.json'
    job = _read_json(specification) if specification.is_file() else {}
    if job and ((job.get('job_id'),job.get('run_id')) != (job_id,run_id)
                or Path(job.get('output_root','')).resolve() != root.resolve()):
        raise ValueError('Candidate job specification differs from its journal owner')
    if type(epoch) is not int or not 0 <= epoch <= 200:
        raise ValueError('invalid candidate epoch')
    if epoch == 0:
        values, txn = _transaction(root, job_id, run_id, 0, 'initialization')
        meta = _read_json(root / 'journal/initial.json')
        if meta != {'epoch': 0, 'checkpoint_sha256': txn['post_checkpoint_sha256']}:
            raise ValueError('E0 is not the committed initialization')
        evidence = values['request']['evidence']; training = {}
        controls = {k: v for k, v in values['decision']['intervention']['parameters_after'].items() if k in CONTROL}
        checkpoint_sha = txn['post_checkpoint_sha256']
    else:
        base = root / 'journal' / ('epoch_%06d' % epoch)
        meta = _read_json(base / 'trained.json'); commit = _read_json(base / 'commit.json')
        if (meta.get('epoch') != epoch or meta.get('route') != 'FULL_REVIEW'
                or commit.get('epoch') != epoch or commit.get('route') != 'FULL_REVIEW'
                or commit.get('decision_kind') != 'agent'
                or commit.get('trained_checkpoint_sha256') != meta.get('checkpoint_sha256')):
            raise ValueError('candidate lacks a committed full review')
        values, txn = _transaction(root, job_id, run_id, epoch, 'FULL_REVIEW')
        evidence = values['request']['evidence']
        obs = _read_json(base / 'review_observation.json')
        if (obs.get('epoch') != epoch or obs.get('checkpoint_sha256') != meta['checkpoint_sha256']
                or obs.get('source') != evidence['source_validation']['values']
                or obs.get('panel') != evidence['unlabelled_panel']['values']
                or obs.get('annotated_training') != (evidence.get('annotated_training') or {}).get('values')
                or obs.get('human_training') != (evidence.get('human_training') or {}).get('values')):
            raise ValueError('candidate observation differs from committed request')
        rows = [r for r in evidence['training_window'] if r['training']['epoch'] == epoch and r['controls']['epoch'] == epoch]
        if len(rows) != 1:
            raise ValueError('candidate lacks its own measured training interval')
        training = {k: v for k, v in rows[0]['training']['values'].items() if k in TRAIN_FIELDS}
        controls = rows[0]['controls']['values']; checkpoint_sha = meta['checkpoint_sha256']
    from hard_sample_training import enabled as annotated_training, work_root
    from hard_sample_evidence import selection_fit
    active = annotated_training(job)
    human = (evidence.get('human_training') or {}).get('values')
    if human is None and not active:
        raise ValueError('new selection policy requires measured human training evidence')
    if human is not None and (human.get('job_id'), human.get('run_id')) != (job_id, run_id):
        raise ValueError('candidate human evidence belongs to another run')
    if active:
        from predictor_artifacts_v3 import verify
        predictor = verify(root,job,epoch,checkpoint_sha)
        model_sha = predictor['model_state_sha256']
    else:
        model_sha = human['model_state_sha256']
    detail = {'epoch': epoch, 'checkpoint_sha256': checkpoint_sha,
        'model_state_sha256': model_sha, 'producer_request_sha256': txn['request_sha256'],
        'source_validation': evidence['source_validation']['values'],
        'unlabelled_panel': evidence['unlabelled_panel']['values'], 'human_training': human,
        'training': training, 'controls': controls,
        'controls_timing': 'after_initialization_before_training' if epoch == 0 else 'used_during_completed_epoch',
        'review_action': values['decision']['action'], 'review_action_applies_after_epoch': epoch}
    frozen = active and include_annotation_fit and (work_root(job)/'selection_library.json').is_file()
    if active:
        detail['annotated_training'] = selection_fit(job,epoch,checkpoint_sha,model_sha) if frozen else None
    validate_detail(detail)
    folder = 'selection_candidates_final' if frozen else 'selection_candidates'
    path = root / 'evidence_store' / folder / ('epoch_%06d' % epoch)
    _write_immutable(path.with_suffix('.json'), _canonical(detail)); _csv_view(path.with_suffix('.csv'), detail)
    return detail


def verify_checkpoint(detail, raw):
    """Bind all tensor state, including BN, before freeze and final publication."""
    import hashlib
    from native_safe_checkpoint_codec_v1 import decode
    from transfer_record_contract_v1 import model_digest
    if hashlib.sha256(raw).hexdigest() != detail['checkpoint_sha256']:
        raise ValueError('candidate checkpoint bytes differ')
    payload = decode(raw)
    if payload['epoch'] != detail['epoch'] or model_digest(payload['state_dict']) != detail['model_state_sha256']:
        raise ValueError('candidate model or normalization state differs')
    return payload


def predictor_identity(detail, payload):
    import hashlib
    from transfer_record_contract_v1 import model_digest
    from checkpoint_selection_contract_v2 import DECODER
    bn = {k: v for k, v in payload['state_dict'].items()
          if k.rsplit('.', 1)[-1] in ('running_mean', 'running_var', 'num_batches_tracked')}
    bn_sha = model_digest(bn) if bn else hashlib.sha256(b'no_batchnorm_buffers').hexdigest()
    return {**DECODER, 'model_state_sha256': detail['model_state_sha256'],
            'batchnorm_buffers_sha256': bn_sha, 'checkpoint_sha256': detail['checkpoint_sha256']}


def verify_frozen_candidates(job, journal, request, manifest, *, verify_models=False, selected_epoch=None):
    from checkpoint_selection_contract_v2 import VERSION, artifact, summary
    from sealed_safe_final_selection_v1 import SafeRoot
    import hashlib
    if request.get('selection_policy') != VERSION or manifest.get('selection_policy') != VERSION:
        raise ValueError('selection policy differs from frozen manifest')
    expected_epochs = [0]
    for epoch in range(1, manifest['stop_epoch'] + 1):
        entry = journal._fs.json('epoch_%06d/commit.json' % epoch)
        if entry.get('route') == 'FULL_REVIEW' and entry.get('decision_kind') == 'agent':
            expected_epochs.append(epoch)
    if ([x.get('epoch') for x in manifest['candidates']] != expected_epochs or
            [x.get('epoch') for x in request['candidate_details']] != expected_epochs or
            [x.get('epoch') for x in request['candidate_observations']] != expected_epochs):
        raise ValueError('selector omitted E0 or a committed full-review candidate')
    fs = SafeRoot(job['output_root']); selected = None
    try:
        for item, supplied, coarse in zip(manifest['candidates'], request['candidate_details'], request['candidate_observations']):
            epoch = item['epoch']; relative = checkpoint_relative(epoch)
            if fs.rel(item.get('path', '')) != relative:
                raise ValueError('candidate is not the owned initial/trained checkpoint')
            detail = collect(job['output_root'], job['job_id'], job['run_id'], epoch)
            projected = artifact(detail); projected.pop('kind')
            if (projected != supplied or summary(detail) != coarse['values']
                    or detail['checkpoint_sha256'] != item['sha256']):
                raise ValueError('candidate evidence differs from committed producer')
            from predictor_artifacts_v3 import enabled as compact_enabled,verify as verify_compact,predictor_identity as compact_identity
            if compact_enabled(job):
                receipt=verify_compact(job['output_root'],job,epoch,item['sha256'],load_predictor=verify_models and epoch==selected_epoch)
                if receipt['model_state_sha256']!=detail['model_state_sha256']:raise ValueError('predictor and source evidence differ')
                if verify_models and epoch==selected_epoch:
                    selected={'candidate':item,'predictor':compact_identity(receipt),'selected_evidence_sha256':supplied['sha256']}
                continue
            raw = fs.read(relative)
            if hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('frozen candidate changed')
            if verify_models:
                payload = verify_checkpoint(detail, raw)
                if epoch == selected_epoch:
                    selected = {'candidate': item, 'predictor': predictor_identity(detail, payload),
                                'selected_evidence_sha256': supplied['sha256']}
                del payload
            del raw
        if verify_models and selected is None:
            raise ValueError('selected candidate is not present')
        return selected
    finally:
        fs.close()
