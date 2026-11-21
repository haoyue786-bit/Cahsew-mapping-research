"""Candidate weights-only codec for newly created native checkpoints.

``encode`` accepts an already constructed, trusted in-memory checkpoint, never
an old pickle. ``decode`` performs only weights-only deserialization. It does
not authenticate provenance, prevent resource-exhaustion attacks, authorize a
run, or make historical checkpoints safe to load. Legacy migration is separate.
"""
import copy,io,math,random
from collections import OrderedDict
import numpy as np
import torch

VERSION='cashew_native_weights_only_v1'
FIELDS={'epoch','state_dict','local_discriminator','global_discriminator','model_optimizer',
        'local_optimizer','global_optimizer','config','metrics','rng_state'}
RNG_FIELDS={'python','numpy','torch','cuda','active_cuda_ordinal'}
NUMPY_FIELDS={'algorithm','keys','position','has_gauss','cached_gaussian'}

def _tree(value,active=None):
    if active is None:active=set()
    if type(value) in (str,int,bool,type(None),bytes):return
    if type(value) is float:
        if not math.isfinite(value):raise ValueError('nonfinite scalar in checkpoint')
        return
    if type(value) is torch.Tensor:
        if value.layout!=torch.strided or value.is_quantized:raise ValueError('unsupported tensor layout')
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            raise ValueError('nonfinite checkpoint tensor')
        return
    if type(value) not in (dict,OrderedDict,list,tuple):raise ValueError('unsupported checkpoint value type')
    identity=id(value)
    if identity in active:raise ValueError('cyclic checkpoint structure')
    active.add(identity)
    try:
        if isinstance(value,dict):
            for k,v in value.items():
                if type(k) not in (str,int):raise ValueError('unsupported checkpoint mapping key')
                _tree(v,active)
            if isinstance(value,OrderedDict) and hasattr(value,'_metadata'):
                _tree(value._metadata,active)
        else:
            for item in value:_tree(item,active)
    finally:active.remove(identity)

def _numpy_native(state):
    if type(state) is not tuple or len(state)!=5:raise ValueError('invalid NumPy RNG state')
    name,keys,pos,has_gauss,cached=state
    if name!='MT19937' or type(keys) is not np.ndarray or keys.dtype!=np.uint32 or keys.shape!=(624,):
        raise ValueError('unsupported NumPy RNG generator')
    if type(pos) is not int or not 0<=pos<=624 or type(has_gauss) is not int or has_gauss not in (0,1):
        raise ValueError('invalid NumPy RNG position or Gaussian flag')
    if type(cached) is not float or not math.isfinite(cached):raise ValueError('invalid NumPy Gaussian cache')
    return {'algorithm':name,'keys':torch.from_numpy(keys.astype(np.int64)),'position':pos,
            'has_gauss':has_gauss,'cached_gaussian':cached}

def _numpy_decode(state):
    if type(state) is not dict or set(state)!=NUMPY_FIELDS or state['algorithm']!='MT19937':raise ValueError('invalid encoded NumPy RNG')
    keys=state['keys'];pos=state['position'];has_gauss=state['has_gauss'];cached=state['cached_gaussian']
    if (type(keys) is not torch.Tensor or keys.dtype!=torch.int64 or keys.shape!=(624,)
        or keys.device.type!='cpu' or torch.any(keys<0).item() or torch.any(keys>4294967295).item()):
        raise ValueError('invalid encoded NumPy RNG keys')
    if (type(pos) is not int or not 0<=pos<=624 or type(has_gauss) is not int or has_gauss not in (0,1)
        or type(cached) is not float or not math.isfinite(cached)):raise ValueError('invalid encoded NumPy RNG scalar')
    native=('MT19937',keys.numpy().astype(np.uint32),pos,has_gauss,cached)
    try:np.random.RandomState().set_state(native)
    except Exception as exc:raise ValueError('invalid NumPy RNG state') from exc
    return native

def _validate(payload,*,encoded):
    if type(payload) is not dict or set(payload)!=FIELDS:raise ValueError('invalid native checkpoint fields')
    if type(payload['epoch']) is not int or payload['epoch']<0:raise ValueError('invalid checkpoint epoch')
    for k in ('state_dict','local_discriminator','global_discriminator','model_optimizer','local_optimizer','global_optimizer','config','metrics'):
        if type(payload[k]) not in (dict,OrderedDict):raise ValueError('invalid checkpoint mapping: '+k)
    for k in ('state_dict','local_discriminator','global_discriminator'):
        if not payload[k] or any(type(v) is not torch.Tensor for v in payload[k].values()):raise ValueError('invalid model state: '+k)
    for k in ('model_optimizer','local_optimizer','global_optimizer'):
        obj=payload[k]
        if type(obj.get('state')) is not dict or type(obj.get('param_groups')) is not list or not obj['param_groups']:
            raise ValueError('invalid optimizer state: '+k)
    rng=payload['rng_state']
    if type(rng) is not dict or set(rng)!=RNG_FIELDS:raise ValueError('invalid RNG fields')
    try:random.Random().setstate(rng['python'])
    except Exception as exc:raise ValueError('invalid Python RNG') from exc
    t=rng['torch']
    if type(t) is not torch.Tensor or t.device.type!='cpu' or t.dtype!=torch.uint8 or t.ndim!=1:raise ValueError('invalid Torch RNG')
    try:torch.Generator(device='cpu').set_state(t)
    except Exception as exc:raise ValueError('invalid Torch generator state') from exc
    cuda=rng['cuda'];ordinal=rng['active_cuda_ordinal']
    if cuda is not None and (type(cuda) not in (list,tuple) or any(type(x) is not torch.Tensor or x.dtype!=torch.uint8 or x.ndim!=1 or x.device.type!='cpu' or not x.numel() for x in cuda)):
        raise ValueError('invalid saved CUDA generators')
    if ordinal is not None and (type(ordinal) is not int or ordinal<0 or cuda is None or ordinal>=len(cuda)):
        raise ValueError('invalid active CUDA ordinal')
    if encoded:_numpy_decode(rng['numpy']);_tree(payload)
    else:
        _numpy_native(rng['numpy'])
        # NumPy MT19937 keys are the sole permitted non-weights-only object.
        for key,value in payload.items():
            if key!='rng_state':_tree(value)
        for key,value in rng.items():
            if key!='numpy':_tree(value)

def encode(payload):
    """Encode a *trusted in-memory* checkpoint; never opens checkpoint files."""
    _validate(payload,encoded=False)
    encoded=dict(payload);encoded['rng_state']=dict(payload['rng_state'])
    encoded['rng_state']['numpy']=_numpy_native(payload['rng_state']['numpy'])
    buf=io.BytesIO();torch.save({'format':VERSION,'checkpoint':encoded},buf)
    return buf.getvalue()

def decode(blob):
    """Safely parse this format, rejecting legacy/unknown pickle by default."""
    if type(blob) is not bytes or not blob:raise ValueError('checkpoint bytes required')
    try:envelope=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
    except Exception as exc:raise ValueError('weights-only checkpoint rejected; legacy migration requires separate certification') from exc
    if type(envelope) is not dict or set(envelope)!={'format','checkpoint'} or envelope['format']!=VERSION:
        raise ValueError('unknown or legacy checkpoint format')
    payload=envelope['checkpoint'];_validate(payload,encoded=True)
    result=dict(payload);result['rng_state']=dict(payload['rng_state'])
    result['rng_state']['numpy']=_numpy_decode(payload['rng_state']['numpy'])
    return result
