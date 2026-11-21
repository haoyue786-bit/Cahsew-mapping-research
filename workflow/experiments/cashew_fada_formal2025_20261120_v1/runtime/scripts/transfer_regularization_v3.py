"""Optional L2-SP anchors to the verified inherited 2025 parameters.

No target labels, pseudo labels, teacher-map quotas, BN-stat penalties or
automatic activation. Each coefficient is independently reversible to zero.
"""
import hashlib,io,math
from pathlib import Path
import torch
from module_controls_v2 import component

VERSION='optional_transfer_anchor_v3'
BOUNDS={k:[0.,.1] for k in ('representation_anchor_weight','refiner_anchor_weight','decoder_anchor_weight')}

def enabled(config):return config.get('transfer_regularization_policy')==VERSION
def bounds(config):return dict(BOUNDS) if enabled(config) else {}
def initialize(config,job):
    if job.get('transfer_regularization_policy')!=VERSION:return
    config['transfer_regularization_policy']=VERSION
    for k in BOUNDS:config[k]=0.

def control(name):
    c=component(name)
    if c=='other':return None
    if c.startswith('refiner'):return 'refiner_anchor_weight'
    if c=='decoder':return 'decoder_anchor_weight'
    return 'representation_anchor_weight'

def bind(model,path,sha256):
    """Read the immutable initial predictor once per fresh/resumed process."""
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha256:raise ValueError('initial predictor anchor changed')
    value=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if value.get('format')!='cashew_initial_transfer_weights_v3':raise ValueError('invalid transfer anchor')
    state=value['state_dict'];refs={}
    for name,p in model.named_parameters():
        if control(name) is None:continue
        v=state[name]
        if v.shape!=p.shape or not torch.isfinite(v).all().item():raise ValueError('anchor parameter mismatch')
        refs[name]=v.detach().to(device=p.device,dtype=p.dtype).clone()
    model._regularization_anchor=refs
    return refs

def terms(model,config):
    if not enabled(config):return {}
    for k in BOUNDS:
        if type(config[k]) not in (int,float) or not math.isfinite(config[k]) or not 0<=config[k]<=.1:raise ValueError('invalid anchor coefficient')
    active={k:config[k] for k in BOUNDS if config[k]>0}
    if not active:return {}
    refs=getattr(model,'_regularization_anchor',None)
    if refs is None:raise ValueError('regularization anchor was not bound')
    result={}
    for name,p in model.named_parameters():
        key=control(name)
        if key in active and p.requires_grad:
            loss=.5*active[key]*(p-refs[name]).square().sum()
            result[key]=result.get(key,0.)+loss
    return result
