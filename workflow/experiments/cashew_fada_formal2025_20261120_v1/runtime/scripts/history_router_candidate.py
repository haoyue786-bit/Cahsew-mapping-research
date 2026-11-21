"""Owned LIGHT/FULL decision transactions with audited same-run tools.

The worker authenticates evidence and holds its journal lease. Trial tools run
inside the full-review turn on isolated copies; only the final ordinary control
transition is committed here. Unknown submissions are never automatically resent.
"""
from contextlib import nullcontext
import copy
from native_safe_checkpoint_codec_v1 import decode,encode
from pathlib import Path
from fd_safe_decision_transaction_candidate_v1 import SafeTransaction
from history_bridge_candidate import attach_dispatch_receipt
from request_reconstruction import verify_request
from decision_bridge import normalize_review,normalize_anomaly_review
from healthy_recovery_runtime_v4 import verify_decision_checkpoint_controls
from checkpoint_transition_candidate_v1 import _OptimizerProxy
from agent_service_slots import call_in_slot
from unified_epoch_pool_bridge_sim_v1 import _equal
import history_dispatch_candidate as dispatcher

KINDS={'safe_anomaly_review_candidate_v1':'light_review','safe_full_review_candidate_v1':'full_review'}
ALLOWED={'KEEP','ADJUST','CONTINUE','PAUSE_ADV','RESUME_ADV'}

def _checkpoint(raw,epoch):
    cp=decode(raw)
    if cp['epoch']!=epoch:raise ValueError('safe checkpoint epoch mismatch')
    if '_automatic_negative_state' in cp['config'] or 'unified_automatic_pool_v1' in cp['config']:
        raise ValueError('automatic pool not supported')
    return cp

def _decision(job,request,answer,pre,*,allow_stop=False):
    actions=ALLOWED|({'STOP'} if allow_stop and request.get('contract_version')=='safe_full_review_candidate_v1' else set())
    if answer.get('action') not in actions:raise ValueError('unsupported or terminal review decision')
    cp=_checkpoint(pre,request['epoch'])
    if request['contract_version']=='safe_anomaly_review_candidate_v1':
        decision=normalize_anomaly_review(request,answer,cp['config'])
    else:
        reconstruction_job=dict(job)
        if answer.get('action')=='STOP':reconstruction_job['research_stopselect_policy']='trend_judgment_v2'
        decision=normalize_review(request,answer,cp['config'],reconstruction_job,minimum_stop_epoch=0,stop_allowed=allow_stop,
            source_collapsed=False,experiment_attempts=0,experiment_limit=0)
    if (decision.get('action') not in actions or
        decision.get('continue_training') is not (False if decision.get('action')=='STOP' else True)):
        raise ValueError('terminal or unsupported review decision')
    return decision

def _post(pre,decision):
    from train_expert_v4 import apply_config
    cp=_checkpoint(pre,decision['epoch']);config=cp['config']
    verify_decision_checkpoint_controls(decision,config,'before')
    if decision.get('action') not in ALLOWED|{'STOP'}:raise ValueError('unsupported review action')
    if decision.get('action') in ('KEEP','CONTINUE','STOP'):
        if (decision['intervention'].get('changed_parameters') or decision['intervention'].get('schedule_updates')
            or decision['intervention']['parameters_before']!=decision['intervention']['parameters_after']):
            raise ValueError('KEEP cannot change training controls')
        verify_decision_checkpoint_controls(decision,config,'after')
        return pre
    old_optimizers={k:copy.deepcopy(cp[k]) for k in ('model_optimizer','local_optimizer','global_optimizer')}
    optimizers=[_OptimizerProxy(cp[k]) for k in ('model_optimizer','local_optimizer','global_optimizer')]
    apply_config(config,decision,*optimizers)
    verify_decision_checkpoint_controls(decision,config,'after')
    changed=set(decision['intervention'].get('changed_parameters') or [])
    schedules={x.get('component') for x in decision['intervention'].get('schedule_updates') or []}
    for key in old_optimizers:
        old=old_optimizers[key];new=cp[key]
        if not _equal(old['state'],new['state']) or len(old['param_groups'])!=len(new['param_groups']):
            raise ValueError('review changed optimizer moments or topology')
        lr_key='main_lr' if key=='model_optimizer' else 'D_lr'
        wd_key='main_weight_decay' if key=='model_optimizer' else 'D_weight_decay'
        may_lr=lr_key in changed or lr_key in schedules or (key=='model_optimizer' and 'main_lr_finish_epochs' in changed)
        for before,after in zip(old['param_groups'],new['param_groups']):
            if set(before)!=set(after):raise ValueError('review changed optimizer group structure')
            module_lr = key=='model_optimizer' and before.get('module_control') in changed
            legacy_refiner_lr = key=='model_optimizer' and before.get('component')=='spatial_refiner' and 'refiner_lr_multiplier' in changed
            for name in before:
                if name=='lr_multiplier' and (module_lr or legacy_refiner_lr):continue
                if name=='lr' and (may_lr or module_lr or legacy_refiner_lr):continue
                if name=='weight_decay' and wd_key in changed:continue
                if not _equal(before[name],after[name]):raise ValueError('review changed unapproved optimizer group field')
    return encode(cp)

def _route(job,request,trusted,expected,pre,*,codex_binary,skill_sha256,verifier_sha256,
          reference_hashes,client_class=None,previous=None,timeout=7200,allow_stop=False):
    """Route one review; owning worker must enforce release and provenance.

    Supports the real Codex client and test clients. The caller is responsible
    for authorization, source isolation and the held run lease; this helper
    does not authorize a run or determine artifact provenance.
    """
    if not isinstance(job,dict) or not isinstance(request,dict):raise ValueError('job/request required')
    if job.get('point_validation_online') is not False:raise ValueError('target point isolation required')
    policy=job.get('experimental_gate')
    if (not isinstance(policy,dict) or policy.get('version')!='experimental_gate_v1'
        or policy.get('experimental_only') is not True or policy.get('mode')!='enforce'):
        raise ValueError('explicit enforced experimental policy required')
    if client_class is None:client_class=dispatcher.PersistentLunaClient
    if request.get('contract_version') not in KINDS:raise ValueError('unsupported review stage')
    schema=request.get('schema') or {};actions=(schema.get('properties') or {}).get('action',{}).get('enum')
    allowed=ALLOWED|({'STOP'} if allow_stop and request['contract_version']=='safe_full_review_candidate_v1' else set())
    if not isinstance(actions,list) or not actions or not set(actions).issubset(allowed):
        raise ValueError('unsafe or unsupported allowed actions')
    if request.get('automatic_selection_enabled') or (request.get('evidence') or {}).get('active_automatic_pool') is not None:
        raise ValueError('automatic pool not supported')
    clean=verify_request(request,trusted,expected=expected,previous=previous);epoch=clean.get('epoch')
    if type(epoch) is not int or epoch<1:raise ValueError('invalid review epoch')
    _checkpoint(pre,epoch)
    prefix='agent/sealed_v1/epoch_%03d/%s'%(epoch,'light' if KINDS[clean['contract_version']]=='light_review' else 'full')
    from sealed_safe_final_selection_v1 import SafeRoot
    root=SafeRoot(job['output_root'])
    try:root.mkdirs(prefix)
    finally:root.close()
    stage=Path(job['output_root'])/prefix
    txn=SafeTransaction(stage/'transaction',job['job_id'],job['run_id'],epoch,KINDS[clean['contract_version']])
    with nullcontext(txn):
        def check(raw,e):_checkpoint(raw,e)
        txn.prepare(clean,pre,check)
        reply=call_in_slot(lambda:dispatcher.dispatch(clean,trusted,expected,stage/'transport',codex_binary=codex_binary,
            skill_sha256=skill_sha256,verifier_sha256=verifier_sha256,reference_hashes=reference_hashes,
            previous=previous,before_turn=txn.claim_submission,client_class=client_class,timeout=timeout,
            history_root=job['output_root'],history_job_id=job['job_id'],history_run_id=job['run_id']))
        attach_dispatch_receipt(txn,stage/'transport',trusted=trusted,expected=expected,skill_path=dispatcher.SKILL,
            skill_sha256=skill_sha256,verifier_sha256=verifier_sha256,reference_hashes=reference_hashes,previous=previous,
            history_root=job['output_root'],history_job_id=job['job_id'],history_run_id=job['run_id'])
        decision=txn.normalize(lambda q,a,p:_decision(job,q,a,p,allow_stop=allow_stop))
        def verify_locked(q,a,p,d,post):
            if d!=_decision(job,q,a,p,allow_stop=allow_stop):raise ValueError('decision differs from answer')
            if not _equal(decode(post),decode(_post(p,d))):raise ValueError('post differs from authorized transition')
        commit,post=txn.commit(check,_post,post_validate=verify_locked)
        return {'decision':decision,'commit':commit,'post_checkpoint':post,'replayed':reply['replayed']}


def route(job, request, *args, **kwargs):
    from query_recovery_v1 import with_recovery
    kind = 'light' if request.get('contract_version') == 'safe_anomaly_review_candidate_v1' else 'full'
    stage = Path(job['output_root']) / 'agent/sealed_v1' / ('epoch_%03d' % request['epoch']) / kind
    return with_recovery(stage, lambda: _route(job, request, *args, **kwargs))
