"""Synthetic-only native checkpoint codec for the unified epoch/pool bridge.

Authoritative pool state is encoded solely in config[FIELD]. The bridge sees a
logical top-level pool/retired projection; these keys are never saved top-level.
The embedded receipt binds the state and measured aggregate exposure in this
same checkpoint. It is not authenticated provenance. A production worker must
verify lineage, target isolation, exposure and file integrity independently.
Only load trusted locally generated torch checkpoints; torch deserialization of
untrusted data may execute code. No model, training or publishing entry point.
"""
import copy
import io
import torch
from decision_transaction import canonical,digest
from automatic_pool_transaction_v1 import _state

FIELD='unified_automatic_pool_v1'
VERSION='native_pool_checkpoint_adapter_sim_v1'
KEYS={'epoch','state_dict','local_discriminator','global_discriminator','model_optimizer',
      'local_optimizer','global_optimizer','config','metrics','rng_state'}
EXPOSURE=('processed_contiguous_positions','completed_G_updates','completed_G_auxiliary_steps',
          'completed_G_patch_occurrences','completed_G_pixel_occurrences')

def unpack(raw):
    if not isinstance(raw,bytes):raise ValueError('checkpoint bytes required')
    obj=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    if not isinstance(obj,dict) or not KEYS.issubset(obj) or not isinstance(obj['config'],dict) or not isinstance(obj['metrics'],dict):
        raise ValueError('incomplete native checkpoint')
    if type(obj['epoch']) is not int or obj['epoch']<0:raise ValueError('invalid native epoch')
    if not isinstance(obj['rng_state'],dict) or not {'python','numpy','torch','cuda','active_cuda_ordinal'}.issubset(obj['rng_state']):
        raise ValueError('incomplete RNG state')
    return obj

def pack(obj):
    buf=io.BytesIO();torch.save(obj,buf);return buf.getvalue()

def _evidence(metrics):
    auto=metrics.get('automatic_negative')
    if auto is None:return None
    if not isinstance(auto,dict) or any(type(auto.get(k)) is not int or auto[k]<0 for k in EXPOSURE):
        raise ValueError('missing completed pool exposure')
    values={k:auto[k] for k in EXPOSURE}
    if values['completed_G_auxiliary_steps']>values['completed_G_updates'] or values['processed_contiguous_positions']>values['completed_G_updates']:
        raise ValueError('inconsistent pool exposure')
    steps=(metrics.get('experimental_actual_updates') or {}).get('steps')
    if not isinstance(steps,dict) or type(steps.get('G')) is not int or steps['G']!=values['completed_G_updates']:
        raise ValueError('pool exposure does not match actual G updates')
    return values

def _slot(epoch,pool,retired,metrics):
    state=_state({'epoch':epoch,'pool':copy.deepcopy(pool),'retired':copy.deepcopy(retired)})
    evidence=_evidence(metrics)
    return {'version':VERSION,'pool':state['pool'],'retired':state['retired'],
            'receipt':{'epoch':epoch,'state_sha256':digest(canonical(state)),
                       'training_exposure':evidence,'training_exposure_sha256':digest(canonical(evidence))}}

def decode(raw):
    native=unpack(raw)
    if 'pool' in native or 'retired' in native:raise ValueError('top-level pool state forbidden')
    cfg=native['config']
    if '_automatic_negative_state' in cfg:raise ValueError('legacy automatic state cannot coexist')
    slot=cfg.get(FIELD)
    if not isinstance(slot,dict) or set(slot)!={'version','pool','retired','receipt'} or slot.get('version')!=VERSION:
        raise ValueError('missing native pool receipt')
    expected=_slot(native['epoch'],slot['pool'],slot['retired'],native['metrics'])
    if slot!=expected:raise ValueError('native pool state or exposure receipt changed')
    logical=dict(native);logical['config']=dict(cfg);del logical['config'][FIELD]
    logical['pool']=copy.deepcopy(slot['pool']);logical['retired']=copy.deepcopy(slot['retired'])
    return logical

def encode(logical):
    if not isinstance(logical,dict) or not KEYS.issubset(logical) or 'pool' not in logical or 'retired' not in logical:
        raise ValueError('missing logical checkpoint state')
    if not isinstance(logical['config'],dict) or FIELD in logical['config'] or '_automatic_negative_state' in logical['config']:
        raise ValueError('duplicate or legacy pool state forbidden')
    out=dict(logical);out['config']=dict(logical['config']);out.pop('pool');out.pop('retired')
    out['config'][FIELD]=_slot(logical['epoch'],logical['pool'],logical['retired'],logical['metrics'])
    return pack(out)

def initialize(raw,*,attest):
    """Explicitly seed an epoch-zero checkpoint with no automatic pool history."""
    native=unpack(raw)
    if native['epoch']!=0 or 'pool' in native or 'retired' in native or FIELD in native['config'] or '_automatic_negative_state' in native['config']:
        raise ValueError('only clean epoch-zero native checkpoint supported')
    if not callable(attest) or attest(operation='native_initialization',checkpoint=raw) is not True:
        raise ValueError('checkpoint provenance attestation required')
    native['pool']=None;native['retired']=[]
    return encode(native)

def working_config(previous_checkpoint):
    """Return the checked native training config without duplicating pool state.

    The caller must not pass a stale embedded receipt to checkpoint_payload:
    pool state is supplied through the single-authority bridge instead.
    """
    return copy.deepcopy(decode(previous_checkpoint)['config'])

def seal_trained(previous_checkpoint,trained_checkpoint,*,attest):
    """Bind a newly trained native checkpoint to its immediately prior state.

    The caller must build it using working_config(previous_checkpoint) and
    must independently attest that full model/optimizer/RNG state and exposure
    really arose from this exact prior checkpoint. This codec cannot prove it.
    """
    previous=decode(previous_checkpoint);trained=unpack(trained_checkpoint)
    if trained['epoch']!=previous['epoch']+1 or 'pool' in trained or 'retired' in trained:
        raise ValueError('trained checkpoint is not the next native epoch')
    if FIELD in trained['config'] or '_automatic_negative_state' in trained['config']:
        raise ValueError('trained config contains a stale or duplicate pool state')
    exposure=trained['metrics'].get('automatic_negative')
    if previous['pool'] is not None and exposure is None:
        raise ValueError('active pool exposure missing from trained checkpoint')
    if previous['pool'] is None and exposure is not None:
        raise ValueError('pool exposure exists without an active prior pool')
    if not callable(attest) or attest(operation='native_epoch',previous_checkpoint=previous_checkpoint,
                                      trained_checkpoint=trained_checkpoint) is not True:
        raise ValueError('trained checkpoint lineage attestation required')
    trained['pool']=copy.deepcopy(previous['pool']);trained['retired']=copy.deepcopy(previous['retired'])
    return encode(trained)
