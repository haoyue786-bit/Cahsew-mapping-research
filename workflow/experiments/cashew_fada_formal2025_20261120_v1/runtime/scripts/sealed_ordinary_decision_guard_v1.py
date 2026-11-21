"""Independent raw-answer consistency guard, not a provenance attestor.

Stopping and experiment eligibility must already have been independently
verified by the worker. Permissive reconstruction inputs below are used only
to reconstruct the semantics of a received answer, never to authorize it.
"""
import copy,io
import torch
from decision_bridge import (normalize_initialization,normalize_review,
    normalize_anomaly_review,normalize_selection,normalize_automatic_selection)

def verify(job,request,answer,pre,actual):
    checkpoint=torch.load(io.BytesIO(pre),map_location='cpu',weights_only=False)
    if not isinstance(checkpoint,dict) or not isinstance(checkpoint.get('config'),dict):
        raise ValueError('decision checkpoint lacks native configuration')
    config=copy.deepcopy(checkpoint['config']);kind=request.get('contract_version')
    if not isinstance(actual,dict):raise ValueError('decision must be a mapping')
    extras={}
    if kind=='safe_initialization_candidate_v1':
        wanted=normalize_initialization(request,answer,config,job)
    elif kind in ('safe_full_review_candidate_v1','safe_branch_review_candidate_v1'):
        # This is only an answer/decision consistency reconstruction. STOP
        # permission itself must be checked against authenticated worker state.
        reconstruction_job=dict(job)
        if answer.get('action')=='STOP':reconstruction_job['research_stopselect_policy']='trend_judgment_v2'
        try:
            wanted=normalize_review(request,answer,config,reconstruction_job,minimum_stop_epoch=0,stop_allowed=True,
                source_collapsed=False,experiment_attempts=0,experiment_limit=1)
        except ValueError as exc:
            raise ValueError('cannot independently reconstruct review: '+str(exc)+'; action='+str(answer.get('action'))) from exc
        if request.get('automatic_selection_enabled'):
            extras['automatic_negative']=normalize_automatic_selection(request,answer,
                actual_candidate_count=request.get('automatic_candidate_limit'),actual_max_weight=request.get('automatic_max_weight'))
    elif kind=='safe_anomaly_review_candidate_v1':
        if answer.get('action')=='DISABLE_AUTOMATIC_NEGATIVE':
            from sealed_joint_negative_decision_v1 import _pool_state
            pool=_pool_state(pre,checkpoint)
            if not isinstance(pool,dict):raise ValueError('cannot disable absent pool')
            legacy=config.get('_automatic_negative_state')
            if legacy is None:config['_automatic_negative_state']={'weight':pool['weight']}
            elif legacy.get('weight')!=pool['weight']:raise ValueError('pool configurations disagree')
            extras['automatic_negative']={'action':'DISABLE'}
        wanted=normalize_anomaly_review(request,answer,config)
    elif kind=='safe_checkpoint_selection_candidate_v1':
        wanted=normalize_selection(request,answer,config,job,frozen_stop_epoch=request.get('stop_epoch'))
    else:raise ValueError('unsupported decision guard contract')
    wanted.update(extras)
    if actual!=wanted:raise ValueError('normalized decision differs from raw answer and checkpoint')
    return actual
