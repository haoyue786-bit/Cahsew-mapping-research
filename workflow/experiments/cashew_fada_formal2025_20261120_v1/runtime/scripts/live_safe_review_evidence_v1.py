"""Candidate weights-only live, point-sealed source and target-panel review producer.

This adapter performs the measurement rather than accepting caller-supplied
metric dictionaries. It invokes only the configured source evaluator and fixed
unlabelled imagery evaluator. The caller must serialize model access, establish
clean training lineage, pin/review *all* runtime code and supply a registered
fixed panel independently of target point outcomes. Metadata snapshots cannot
prove those properties or prevent hostile in-flight file replacement.

Comparison/stability and SND are intentionally absent: the legacy functions do
not establish the pixel-mask/subset identities needed to compare them safely.
This module does not authorize training, dispatch an Agent, or open points.
"""
from pathlib import Path
import math

import torch
from panel_projection_v1 import count, value, vector
from sealed_packet import SOURCE, PANEL, nums

def _certified_panel_parameters(job):
    """Load a run-owned registered panel; caller still certifies lineage.

    The digest binds the exact pre-created registration, not its historical
    scientific provenance. This function cannot authorize a training run.
    """
    import hashlib,json
    from sealed_safe_final_selection_v1 import SafeRoot
    if not isinstance(job,dict):raise ValueError('job required')
    root=job.get('output_root');expected=job.get('panel_registration_sha256')
    if not isinstance(root,str) or not Path(root).is_absolute() or not isinstance(expected,str) or len(expected)!=64:
        raise ValueError('run-owned panel registration and digest required')
    fs=SafeRoot(root)
    try:raw=fs.read('panel_registration_candidate.json')
    finally:fs.close()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('panel registration digest changed')
    registration=json.loads(raw);domain=job.get('target_domain') or {};prior=job.get('geographic_prior') or {}
    return {'run_root':root,'image_root':job.get('target_root'),'job_id':job.get('job_id'),'run_id':job.get('run_id'),
            'target_year':job.get('target_year'),'domain_id':domain.get('gid'),'domain_name':domain.get('name'),
            'source_csv_path':domain.get('patch_csv'),'prior_card_path':prior.get('card'),'registration':registration}

def _registered_panel(job):
    from evaluation_sample_v1 import EvaluationNames
    from point_sealed_panel_loader_candidate_v1 import load_for_candidate
    args=_certified_panel_parameters(job)
    if not isinstance(args,dict):raise ValueError('missing certified panel registration')
    if (str(args.get('run_root'))!=str(job.get('output_root')) or str(args.get('image_root'))!=str(job.get('target_root'))
        or args.get('job_id')!=job.get('job_id') or args.get('run_id')!=job.get('run_id')
        or args.get('target_year')!=job.get('target_year')):raise ValueError('panel registration differs from worker identity')
    domain=job.get('target_domain') or {};prior=job.get('geographic_prior') or {}
    if (args.get('domain_id')!=domain.get('gid') or args.get('domain_name')!=domain.get('name')
        or args.get('source_csv_path')!=domain.get('patch_csv') or args.get('prior_card_path')!=prior.get('card')):
        raise ValueError('panel registration differs from target frame')
    registration=args.get('registration');artifacts=(registration or {}).get('artifacts') or {};policy=job.get('target_evaluation') or {}
    root=Path(job['output_root'])
    for role,path_key,hash_key in (('names','names_file','names_sha256'),('locations','locations_csv','locations_csv_sha256'),('report','report','report_sha256')):
        rec=artifacts.get(role) or {}
        if str(root/rec.get('path',''))!=str(policy.get(path_key)) or rec.get('sha256')!=policy.get(hash_key):
            raise ValueError('panel job paths or hashes differ from registration')
    for role,key in (('full','target_full_list'),('training','target_train_list')):
        if str(root/(artifacts.get(role) or {}).get('path',''))!=str(job.get(key)):
            raise ValueError('panel frame differs from registration')
    panel=load_for_candidate(**args)
    result=EvaluationNames(list(panel.names),dict(panel.policy),[dict(x) for x in panel.locations],dict(panel.report))
    result.candidate_audit=dict(panel.audit)
    return result


def _model_matches(model, state):
    live = model.state_dict()
    if not isinstance(state, dict) or set(live) != set(state):
        raise ValueError('checkpoint/model state keys differ')
    for name, tensor in live.items():
        saved = state[name]
        if not isinstance(tensor, torch.Tensor) or not isinstance(saved, torch.Tensor):
            raise ValueError('unsupported non-tensor model state')
        if tensor.shape != saved.shape or tensor.dtype != saved.dtype:
            raise ValueError('checkpoint/model tensor shape or dtype differs')
        if not torch.equal(tensor.detach().cpu(), saved.detach().cpu()):
            raise ValueError('live model differs from review checkpoint')


def _project(source, target, roster):
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError('invalid producer result')
    if not isinstance(roster, (tuple, list)) or not 0 < len(roster) <= 64 or len(set(roster)) != len(roster):
        raise ValueError('invalid panel tile roster')
    f1 = vector(source.get('f1_per_class'))
    fractions = vector(target.get('class_distribution'), normalized=True)
    source_out = {'source_loss': value(source.get('loss'), low=0),
                  'validation_pixels': count(source.get('pixels')),
                  **{'class%d_f1' % i: f1[i] for i in range(4)}}
    panel_out = {'predicted_class2_fraction': fractions[2],
                 'class0_fraction': fractions[0], 'class1_fraction': fractions[1],
                 'class3_fraction': fractions[3],
                 'mean_entropy': value(target.get('entropy'), low=0, high=math.log(4)+1e-6),
                 'normalized_entropy': value(target.get('normalized_entropy'), low=0, high=1),
                 'sample_patch_count': count(target.get('patch_count')),
                 'sample_valid_pixel_count': count(target.get('pixel_count')),
                 'probe_snd': None}
    tiles = target.get('tile_estimates')
    if not isinstance(tiles, list):
        raise ValueError('missing fixed-panel tile estimates')
    seen = set()
    total_patches = 0; total_pixels = 0
    for tile in tiles:
        if not isinstance(tile, dict) or tile.get('tile_id') not in roster or tile['tile_id'] in seen:
            raise ValueError('unknown or duplicate panel tile')
        tile_id = tile['tile_id']; seen.add(tile_id); i = roster.index(tile_id)
        patches = count(tile.get('sample_patch_count')); pixels = count(tile.get('sampled_valid_pixels'))
        fraction = value(tile.get('class2_fraction'), low=0, high=1)
        if patches is None or pixels is None or (pixels == 0 and fraction is not None):
            raise ValueError('invalid tile support')
        total_patches += patches; total_pixels += pixels
        panel_out.update({'tile_%d_class2_fraction' % i: fraction,
                          'tile_%d_sample_patch_count' % i: patches,
                          'tile_%d_sampled_valid_pixels' % i: pixels})
    panel_out['missing_tile_count'] = len(roster) - len(seen)
    if seen != set(roster):
        raise ValueError('incomplete panel tile coverage')
    if (panel_out['sample_patch_count'] is None or panel_out['sample_patch_count'] != total_patches
            or panel_out['sample_valid_pixel_count'] is None or panel_out['sample_valid_pixel_count'] != total_pixels
            or total_patches < 1 or total_pixels < 1):
        raise ValueError('inconsistent panel support')
    if source_out['validation_pixels'] is None or source_out['validation_pixels'] < 1:
        raise ValueError('empty source validation support')
    return nums(source_out, SOURCE), nums(panel_out, PANEL)


def collect(job, gan, common, model, device, checkpoint, epoch, *, checkpoint_reader):
    """Evaluate current checkpoint against source split and fixed imagery panel.

    Returns two strictly numeric Agent-facing records and a separate audit
    object. The caller must verify that the checkpoint is the owning run's
    current committed epoch, and check/lock its environment. User-supplied
    dictionaries cannot act as measurements. No independent test results,
    sample paths, point IDs, plots, free text or human-polygon metrics are sent.
    """
    from source_panel_attestation_safe_candidate_v1 import check, input_snapshot, assert_unchanged
    from train_expert_v4 import full_target_prediction_metrics
    from native_safe_checkpoint_codec_v1 import decode
    import hashlib
    if not callable(checkpoint_reader):raise ValueError('owned checkpoint reader required')
    if type(epoch) is not int or epoch < 0:
        raise ValueError('invalid epoch')
    raw=checkpoint_reader()
    if not isinstance(raw,bytes):raise ValueError('owned checkpoint bytes required')
    cp_hash=hashlib.sha256(raw).hexdigest();payload=decode(raw)
    if type(payload.get('epoch')) is not int or payload['epoch'] != epoch:
        raise ValueError('checkpoint epoch mismatch')
    source_names = [x.strip() for x in Path(job['source']['source_valid']).read_text().splitlines() if x.strip()]
    panel = _registered_panel(job)
    # sorted roster follows the real weighted producer's iteration order.
    roster = sorted({r['tile_id'] for r in panel.locations})
    before = input_snapshot(job, source_names, panel)
    identity = check(job, gan, common, source_names, panel, epoch=epoch, expected_epoch=epoch)
    saved = payload['state_dict']
    _model_matches(model, saved)
    was_training = bool(model.training)
    try:
        from module_bn_v2 import domain as bn_domain
        from module_functional_v2 import FunctionalObserver
        from contextlib import nullcontext
        detailed = job.get('module_dynamics_policy') == 'module_observation_v2'
        source_obs = FunctionalObserver(model) if detailed else nullcontext()
        panel_obs = FunctionalObserver(model) if detailed else nullcontext()
        with bn_domain('source'), source_obs:
            source = common.old_diag.evaluate_source(model, source_names)
        with panel_obs:
            target = full_target_prediction_metrics(gan, common, model, Path(job['target_root']), panel, device,
                                                    batch_size=16, return_argmax=False)
        _model_matches(model, saved)
        if hashlib.sha256(checkpoint_reader()).hexdigest() != cp_hash:
            raise ValueError('checkpoint changed during review')
        after = input_snapshot(job, source_names, panel)
        assert_unchanged(before, after)
        source_out, panel_out = _project(source, target, roster)
        if detailed:
            source_out.update(source_obs.result()); panel_out.update(panel_obs.result())
        if panel_out['sample_patch_count'] != len(panel):
            raise ValueError('not all registered panel patches were measured')
    finally:
        model.train(was_training)
    audit = {'version': 'live_review_evidence_v1', 'epoch': epoch,
             'checkpoint_sha256': cp_hash, 'input_snapshot_sha256': before['sha256'],
             'input_snapshot_count': before['file_count'], 'source_identity': identity,
             'panel_comparison_emitted': False, 'probe_snd_emitted': False,
             'training_authorized': False}
    return source_out, panel_out, audit
