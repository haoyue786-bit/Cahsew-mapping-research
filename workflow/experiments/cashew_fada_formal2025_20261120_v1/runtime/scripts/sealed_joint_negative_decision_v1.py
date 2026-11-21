"""Synthetic candidate joint review/automatic-negative normalization.

The original sealed router still verifies request, skill binding and exact turn
receipt before calling these functions. Origin attestation is a mandatory
independent callback; a callback returning True is not itself proof. Dynamic
pool assets/real data and production launch remain outside this module.
"""
import copy,io
import torch
from decision_transaction import digest
from decision_bridge import normalize_review,normalize_anomaly_review,normalize_automatic_selection
from checkpoint_transition_candidate_v1 import native_transition


def _checkpoint(raw,epoch):
    if not isinstance(raw,bytes):raise ValueError('checkpoint bytes required')
    state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    if not isinstance(state,dict) or state.get('epoch')!=epoch or not isinstance(state.get('config'),dict):
        raise ValueError('checkpoint/decision epoch mismatch')
    return state

def _pool_state(raw,state):
    """Decode exactly one supported pool representation; reject ambiguity."""
    config=state['config'];unified='unified_automatic_pool_v1' in config
    if unified:
        if 'pool' in state or '_automatic_negative_state' in config:raise ValueError('duplicate pool state')
        from native_pool_checkpoint_adapter_sim_v1 import decode
        return decode(raw)['pool']
    return state.get('pool')

def _attest(attestor,**facts):
    if not callable(attestor) or attestor(**facts) is not True:raise ValueError('independent pool source attestation required')

def full(request,answer,raw,job,*,candidate_count=None,max_weight=None,source_attest=None,
         minimum_stop_epoch,stop_allowed,source_collapsed,experiment_attempts=0,experiment_limit=0):
    """Bind ordinary and optional auto decision from exactly the same answer."""
    state=_checkpoint(raw,request.get('epoch'))
    decision=normalize_review(request,answer,state['config'],job,minimum_stop_epoch=minimum_stop_epoch,
        stop_allowed=stop_allowed,source_collapsed=source_collapsed,
        experiment_attempts=experiment_attempts,experiment_limit=experiment_limit)
    if request.get('automatic_selection_enabled'):
        _attest(source_attest,request=request,checkpoint_sha256=digest(raw),
                candidate_count=candidate_count,max_weight=max_weight)
        selection=normalize_automatic_selection(request,answer,actual_candidate_count=candidate_count,actual_max_weight=max_weight)
        if selection['action']=='ACTIVATE' and _pool_state(raw,state) is not None:
            raise ValueError('cannot publish while a pool is already active')
        decision['automatic_negative']=selection
    elif answer.get('automatic_negative') is not None:
        raise ValueError('automatic-negative selection was not advertised')
    return decision

def light(request,answer,raw,*,source_attest=None):
    """Map only a validated light DISABLE to the bridge's pool action."""
    state=_checkpoint(raw,request.get('epoch'));pool=_pool_state(raw,state)
    config=copy.deepcopy(state['config'])
    if answer.get('action')=='DISABLE_AUTOMATIC_NEGATIVE' and pool is not None:
        existing=config.get('_automatic_negative_state')
        if existing is None:config['_automatic_negative_state']={'weight':pool.get('weight')}
        elif not isinstance(existing,dict) or existing.get('weight')!=pool.get('weight'):
            raise ValueError('checkpoint pool representations disagree')
    decision=normalize_anomaly_review(request,answer,config)
    action=decision.get('automatic_negative_action')
    if action=='DISABLE':
        advertised=request['evidence'].get('active_automatic_pool')
        if not isinstance(pool,dict) or not isinstance(advertised,dict):raise ValueError('cannot disable a missing pool')
        value=(advertised.get('values') or {}).get('weight')
        if pool.get('weight')!=value:raise ValueError('pool checkpoint differs from advertised active dose')
        _attest(source_attest,request=request,checkpoint_sha256=digest(raw),pool=pool)
        decision['automatic_negative']={'action':'DISABLE'}
    elif action not in (None,'KEEP'):raise ValueError('unsupported light automatic action')
    return decision

def ordinary_transition(pre,decision):
    """Pure hyperparameter portion; the pool action remains separately bound."""
    ordinary=copy.deepcopy(decision);ordinary.pop('automatic_negative',None)
    ordinary.pop('automatic_negative_action',None)
    disable=ordinary.get('action')=='DISABLE_AUTOMATIC_NEGATIVE'
    if disable:ordinary['action']='KEEP'
    base=native_transition(pre,ordinary)
    if not disable:return base
    state=_checkpoint(base,decision.get('epoch'));pool_state=state['config'].get('_automatic_negative_state')
    if pool_state is None and 'unified_automatic_pool_v1' in state['config']:
        if _pool_state(base,state) is None:raise ValueError('disable requires active unified pool')
        return base
    if not isinstance(pool_state,dict) or not pool_state.get('weight',0)>0:
        raise ValueError('disable requires active checkpoint pool configuration')
    pool_state['weight']=0.0;pool_state['disabled_by_light_review_epoch']=decision['epoch']
    out=io.BytesIO();torch.save(state,out);return out.getvalue()

def transition(pre,decision,*,pool_transition=None,verify_pool_post=None):
    """Apply both decisions to one independently deserialized checkpoint.

    Pool callbacks are test-only until the caller proves asset/label exclusion,
    correct selected catalog rows and single-authority durability. They must
    return only bytes and verify all non-pool model/optimizer/RNG state.
    """
    selection=decision.get('automatic_negative')
    if selection is None or selection.get('action')=='ABSTAIN':return native_transition(pre,decision)
    if selection.get('action') not in {'ACTIVATE','DISABLE'}:raise ValueError('unsupported pool transition')
    if selection['action']=='DISABLE':
        if selection!={'action':'DISABLE'} or decision.get('action')!='DISABLE_AUTOMATIC_NEGATIVE':
            raise ValueError('pool disable requires the validated light action')
    else:
        from automatic_negative_selection_validation_v1 import validate_selection
        count=selection.get('selected_count')
        if type(count) is not int:raise ValueError('invalid pool count')
        validate_selection(selection,count)
    if decision.get('action') in {'STOP','SELECT_CHECKPOINT','REQUEST_EXPERIMENT'} or decision.get('continue_training') is not True:
        raise ValueError('terminal or branch decision cannot change pool')
    # The light disable is a pool instruction, not a hyperparameter update.
    base=ordinary_transition(pre,decision)
    if not callable(pool_transition) or not callable(verify_pool_post):raise ValueError('verified pool transition required')
    post=pool_transition(base,selection)
    if not isinstance(post,bytes) or verify_pool_post(pre=pre,base=base,post=post,selection=selection) is not True:
        raise ValueError('pool transition output not independently verified')
    return post


def dispatch_full(job,request,trusted,expected,pre,*,candidate_count,max_weight,source_attest,
                  checkpoint_validate,pool_transition,verify_pool_post,minimum_stop_epoch,
                  stop_allowed,source_collapsed,**router_options):
    """Preflight source identity before sending, then commit one joint turn.

    This candidate accepts a worker-owned attestor only. It does not certify the
    attestor, the candidate catalog or target-point isolation. The real launch
    remains blocked; a plain True callback is suitable only for synthetic tests.
    """
    if (not isinstance(job,dict) or not isinstance(job.get('experimental_gate'),dict)
            or job['experimental_gate'].get('simulation_only') is not True):
        raise ValueError('joint dispatcher is synthetic-only')
    from request_reconstruction import verify_request
    from sealed_transaction_router_v1 import route_prepared
    verify_request(request,trusted,expected=expected,previous=router_options.get('previous'))
    _checkpoint(pre,request.get('epoch'))
    if request.get('automatic_selection_enabled'):
        if (type(candidate_count) is not int or candidate_count!=request.get('automatic_candidate_limit')
                or type(max_weight) not in (int,float) or max_weight!=request.get('automatic_max_weight')):
            raise ValueError('candidate catalog or job dose cap changed')
        _attest(source_attest,request=request,checkpoint_sha256=digest(pre),candidate_count=candidate_count,max_weight=max_weight)
    def normalize(req,answer,raw):
        return full(req,answer,raw,job,candidate_count=candidate_count,max_weight=max_weight,source_attest=source_attest,
                    minimum_stop_epoch=minimum_stop_epoch,stop_allowed=stop_allowed,source_collapsed=source_collapsed)
    def apply(raw,decision):
        return transition(raw,decision,pool_transition=pool_transition,verify_pool_post=verify_pool_post)
    return route_prepared(job,request,trusted,expected,pre,checkpoint_validate=checkpoint_validate,
                          normalize_decision=normalize,transition=apply,**router_options)
