"""Versioned per-tile negative dose/exposure controls; no automatic quality proxy."""
from __future__ import annotations
from contextlib import contextmanager
import math
import random
import numpy as np

VERSION='tile_negative_controls_v1'
WEIGHT='supplementary_loss_weight'
CAP='supplementary_occurrence_cap'
INTERVAL='supplementary_interval_steps'
HORIZON='supplementary_horizon_epochs'
BOUNDS={WEIGHT:[0.,1.],CAP:[0,8],INTERVAL:[1,16],HORIZON:[1,200]}
INTEGER_CONTROLS={CAP,INTERVAL,HORIZON}
CURVES={WEIGHT:{'constant','cosine_zero'}}

def enabled(config):
    return config.get('negative_control_policy',{}).get('version')==VERSION

def initialize(config,job):
    policy=job.get('negative_control_policy')
    if not policy:return
    if policy.get('version')!=VERSION:raise ValueError('unsupported negative control policy')
    config['negative_control_policy']=dict(policy)
    supp=config['supplementary_training']
    config.update({WEIGHT:float(supp['loss_weight']),CAP:0,INTERVAL:1,HORIZON:100})
    config['schedule_state'][WEIGHT]={'curve':'constant','mode':'advance','anchor_epoch':0,'phase_at_anchor':0.}
    from conditional_negatives_v2 import initialize as initialize_conditional
    initialize_conditional(config, job)
    if job.get('negative_response_policy'):
        config['negative_response_policy']=dict(job['negative_response_policy'])
    if job.get('polygon_supervision_policy'):
        config['polygon_supervision_policy']=dict(job['polygon_supervision_policy'])
        config['supplementary_positive_scale']=.25

def component_probe_steps(schedule, steps, config, epoch_start, full_epoch_steps):
    """Include an actually exposed update when a new sparse-pool job requests v3."""
    result={0,steps-1}
    if config.get('negative_response_policy',{}).get('version')=='late_negative_response_v3':
        active=[i for i in sorted(schedule) if schedule[i] and weight(config,epoch_start+(i+1)/full_epoch_steps)>0]
        if active:result.update((active[0],active[-1]))
    return result

def bounds(config):
    result={k:list(v) for k,v in BOUNDS.items()} if enabled(config) else {}
    if result and config.get('negative_pool_count',config.get('supplementary_training',{}).get('count',0))==0:result[WEIGHT]=[0.,0.]
    if result and config.get('polygon_supervision_policy'):
        result[CAP]=[1,4]
        result['supplementary_positive_scale']=[0.,2.]
    return result

def schedule_phase(config,epoch):
    state=config['schedule_state'][WEIGHT]
    phase=float(state['phase_at_anchor'])
    if state['mode']=='advance':phase+=max(0.,float(epoch)-float(state['anchor_epoch']))/float(config[HORIZON])
    return min(1.,max(0.,phase))

def weight(config,epoch):
    if not enabled(config):return float(config['supplementary_training'].get('loss_weight',1.))
    if not config['supplementary_training'].get('count'):return 0.
    curve=config['schedule_state'][WEIGHT]['curve']
    if curve=='constant':factor=1.
    elif curve=='cosine_zero':factor=.5*(1.+math.cos(math.pi*schedule_phase(config,epoch)))
    else:raise ValueError('invalid supplementary curve')
    return float(config[WEIGHT])*factor

def update_state(config,item,epoch):
    old=config['schedule_state'][WEIGHT];operation=item['operation']
    phase=float(item['phase']) if operation in {'set_phase','set_phase_hold'} else schedule_phase(config,epoch)
    config['schedule_state'][WEIGHT]={'curve':old['curve'],'mode':'hold' if operation in {'hold','set_phase_hold'} else 'advance',
                                     'anchor_epoch':float(epoch),'phase_at_anchor':phase}

def exposure_schedule(count,steps,rng,cap,interval):
    """Cap=0 means replacement/unlimited; positive cap is a per-epoch maximum.

    A limited pass never repeats a pool item inside its batch. Updates are
    spread across eligible slots. A final partial batch keeps mean reduction;
    dose is controlled explicitly, never guessed from the smaller batch size.
    """
    if type(cap) is not int or cap<0 or type(interval) is not int or interval<1:raise ValueError('invalid negative exposure')
    if not count:return {}
    slots=np.arange(0,steps,interval,dtype=int);size=min(16,count)
    if cap==0:return {int(i):rng.randint(0,count,size=size).tolist() for i in slots}
    batches=[]
    for _ in range(cap):
        items=rng.permutation(count)
        batches.extend(items[i:i+size].tolist() for i in range(0,count,size))
    batches=batches[:len(slots)]
    indices=np.rint(np.linspace(0,len(slots)-1,len(batches))).astype(int) if len(batches)>1 else np.asarray([len(slots)//2])
    return {int(slots[i]):batch for i,batch in zip(indices,batches)}

@contextmanager
def rng_scope(rng,seed,device):
    import torch
    branch=rng.get_state();python=random.getstate();numpy=np.random.get_state();cpu=torch.get_rng_state()
    gpu=torch.cuda.get_rng_state(device) if device.type=='cuda' else None
    try:
        rng.seed(seed);random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
        if gpu is not None:
            with torch.cuda.device(device):torch.cuda.manual_seed(seed)
        yield
    finally:
        rng.set_state(branch);random.setstate(python);np.random.set_state(numpy);torch.set_rng_state(cpu)
        if gpu is not None:torch.cuda.set_rng_state(gpu,device)

def component_gradients(model,source,negative,adversarial,coefficient):
    """Read-only first/last-update diagnostics; norms/cosines are not accuracy."""
    import torch
    result={}
    for name,prefixes in [('fusion_and_spatial_refiner',('fuse_conv.','a3_spatial_refiner.')),('decoder',('decoder.',))]:
        params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(prefixes)]
        if not params:continue
        gradients=[]
        for loss in (source,negative,adversarial):
            gradients.append(torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True) if loss.requires_grad else tuple(None for _ in params))
        def norm(values):
            return math.sqrt(sum(float(v.detach().square().sum()) for v in values if v is not None))
        sn,nn,an=map(norm,gradients)
        dot=sum(float((a.detach()*b.detach()).sum()) for a,b in zip(gradients[0],gradients[1]) if a is not None and b is not None)
        result[name]={'source_norm':sn,'weighted_negative_norm':nn,'raw_adversarial_norm':an,
                      'weighted_adversarial_norm':float(coefficient)*an,
                      'negative_to_source_norm_ratio':nn/sn if sn>0 else None,
                      'source_negative_cosine':dot/(sn*nn) if sn>0 and nn>0 else None,
                      'opposition_is_not_a_quality_or_disable_gate':True}
    return result
