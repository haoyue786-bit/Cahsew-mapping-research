"""Candidate read-only transfer of a frozen second-period model to a first-period job.

Requires the existing selection transaction to reverify its complete chain; the
caller must already hold the source journal lease. No optimizer, discriminator,
RNG state, training history, thresholds or target evaluation are transferred.
This is not proof of blind provenance and is not a training launcher.
"""
import hashlib
import json
from pathlib import Path
from native_safe_checkpoint_codec_v1 import decode
from sealed_safe_final_selection_v1 import SafeRoot, commit


def prepare(source_job, target_job, journal, manifest_sha256):
    if not isinstance(source_job,dict) or not isinstance(target_job,dict):
        raise ValueError('both jobs must be mappings')
    if source_job.get('target_year') not in (2024,2025) or target_job.get('target_year') not in (2018,2019):
        raise ValueError('expected second-period source and first-period destination')
    if source_job.get('point_validation_online') is not False or target_job.get('point_validation_online') is not False:
        raise ValueError('online target point validation must be disabled for both jobs')
    sg=(source_job.get('target_domain') or {}).get('gid');tg=(target_job.get('target_domain') or {}).get('gid')
    if not isinstance(sg,str) or not sg or sg!=tg:
        raise ValueError('source and destination target group must match')
    architecture=source_job.get('architecture')
    if not isinstance(architecture,str) or not architecture or architecture!=target_job.get('architecture'):
        raise ValueError('source and destination model architecture must match')
    if (not source_job.get('job_id') or not source_job.get('run_id') or not target_job.get('job_id')
        or not target_job.get('run_id') or (source_job['job_id'],source_job['run_id'])==(target_job['job_id'],target_job['run_id'])):
        raise ValueError('distinct identified jobs are required')
    if type(journal.max_epoch) is not int or not 1<=journal.max_epoch<=200:
        raise ValueError('second-period training budget exceeds 200 epochs')
    if type(manifest_sha256) is not str or len(manifest_sha256)!=64 or any(c not in '0123456789abcdef' for c in manifest_sha256):
        raise ValueError('invalid frozen candidate manifest digest')
    journal._held()
    root=Path(source_job['output_root'])
    fs=SafeRoot(root)
    try:
        existing=json.loads(fs.read('frozen_selection.json'))
    finally:
        fs.close()
    # commit replays and verifies the entire STOP/SELECT and candidate chain;
    # the file must already exist so this operation cannot choose a model.
    frozen=commit(root,'agent/sealed_v1/selection/transaction',manifest_sha256,job=source_job,journal=journal)
    if existing!=frozen:
        raise ValueError('previously frozen selection changed')
    fs=SafeRoot(root)
    try:
        raw=fs.read(frozen['checkpoint'])
    finally:
        fs.close()
    if hashlib.sha256(raw).hexdigest()!=frozen['checkpoint_sha256']:
        raise ValueError('selected checkpoint changed during transfer')
    state=decode(raw)
    if state['epoch']!=frozen['selected_epoch']:
        raise ValueError('selected epoch mismatch')
    from transfer_record_contract_v1 import model_digest
    return {'model_state_dict':state['state_dict'], 'transfer_record':{
        'version':'two_period_frozen_handoff_candidate_v1','source_job_id':source_job['job_id'],
        'source_run_id':source_job['run_id'],'source_period':'2024-2025','source_epoch':frozen['selected_epoch'],
        'source_checkpoint_sha256':frozen['checkpoint_sha256'],'selection_decision_sha256':frozen['selection_decision_sha256'],
        'model_state_sha256':model_digest(state['state_dict']),
        'target_job_id':target_job['job_id'],'target_run_id':target_job['run_id'],
        'target_period':'2018-2019','target_gid':sg,'architecture':architecture,
        'optimizer_and_discriminators_transferred':False,'provenance_certified':False}}


def initialize(source_job, target_job, journal, manifest_sha256, construct):
    """Build a new epoch-zero state; caller must supply fresh factories.

    This only constructs a checkpoint in memory. It does not initialize a
    journal, start an Agent, prepare imagery or authorize a training job.
    """
    import torch
    import train_expert_v4 as train
    from native_safe_checkpoint_codec_v1 import encode
    handoff=prepare(source_job,target_job,journal,manifest_sha256)
    parts=construct()
    required={'model','local_discriminator','global_discriminator','model_optimizer','local_optimizer','global_optimizer','config'}
    if not isinstance(parts,dict) or set(parts)!=required:
        raise ValueError('factory must return the exact model, discriminators, optimizers and config')
    models=[parts['model'],parts['local_discriminator'],parts['global_discriminator']]
    optimizers=[parts['model_optimizer'],parts['local_optimizer'],parts['global_optimizer']]
    if any(not isinstance(x,torch.nn.Module) for x in models) or len({id(x) for x in models})!=3:
        raise ValueError('fresh distinct model and discriminators required')
    if any(not isinstance(x,torch.optim.Optimizer) or len(x.state)!=0 for x in optimizers) or len({id(x) for x in optimizers})!=3:
        raise ValueError('fresh distinct optimizers with empty state required')
    for model,optimizer in zip(models,optimizers):
        expected={id(p) for p in model.parameters() if p.requires_grad}
        actual=[id(p) for group in optimizer.param_groups for p in group['params']]
        if not expected or len(actual)!=len(set(actual)) or set(actual)!=expected:
            raise ValueError('optimizer parameters do not match their new model')
    incoming=handoff['model_state_dict'];initial=parts['model'].state_dict()
    if set(incoming)!=set(initial) or any(incoming[k].shape!=initial[k].shape or incoming[k].dtype!=initial[k].dtype for k in initial):
        raise ValueError('frozen model weights do not match destination architecture')
    parts['model'].load_state_dict(incoming,strict=True)
    metrics={'two_period_transfer':handoff['transfer_record']}
    payload=train.checkpoint_payload(0,*models,*optimizers,parts['config'],metrics)
    return encode(payload),handoff['transfer_record']
