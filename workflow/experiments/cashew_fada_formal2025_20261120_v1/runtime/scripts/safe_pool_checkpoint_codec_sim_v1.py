"""Simulation-only unified pool state in a weights-only native checkpoint.

No legacy pickle, target point data, migration, or production authorization.
This is a bridge view: on-disk pool state exists ONLY in config; the top-level
projection is ephemeral. Evidence origin must be independently authenticated.
"""
import copy
import native_safe_checkpoint_codec_v1 as safe
from automatic_pool_transaction_v1 import _state
from decision_transaction import canonical,digest

FIELD='unified_automatic_pool_v1'
VERSION='safe_pool_checkpoint_sim_v1'
EXPOSURE=('processed_contiguous_positions','completed_G_updates','completed_G_auxiliary_steps','completed_G_patch_occurrences','completed_G_pixel_occurrences')

def _slot(epoch,pool,retired,metrics):
 state=_state({'epoch':epoch,'pool':copy.deepcopy(pool),'retired':copy.deepcopy(retired)})
 auto=metrics.get('automatic_negative')
 if auto is None:exp=None
 else:
  if not isinstance(auto,dict) or any(type(auto.get(k)) is not int or auto[k]<0 for k in EXPOSURE):raise ValueError('missing pool exposure')
  exp={k:auto[k] for k in EXPOSURE};steps=(metrics.get('experimental_actual_updates') or {}).get('steps')
  if (not isinstance(steps,dict) or type(steps.get('G')) is not int or steps['G']!=exp['completed_G_updates']
      or exp['processed_contiguous_positions']>exp['completed_G_updates'] or exp['completed_G_auxiliary_steps']>exp['completed_G_updates']):
   raise ValueError('inconsistent pool exposure')
 return {'version':VERSION,'pool':state['pool'],'retired':state['retired'],
         'receipt':{'epoch':epoch,'state_sha256':digest(canonical(state)),'training_exposure':exp,'training_exposure_sha256':digest(canonical(exp))}}

def decode(raw):
 p=safe.decode(raw);cfg=p['config']
 if '_automatic_negative_state' in cfg:raise ValueError('legacy pool state forbidden')
 slot=cfg.get(FIELD)
 if not isinstance(slot,dict) or set(slot)!={'version','pool','retired','receipt'} or slot.get('version')!=VERSION:raise ValueError('unknown safe pool slot')
 if slot!=_slot(p['epoch'],slot['pool'],slot['retired'],p['metrics']):raise ValueError('safe pool receipt mismatch')
 out=dict(p);out['config']=dict(cfg);del out['config'][FIELD];out['pool']=copy.deepcopy(slot['pool']);out['retired']=copy.deepcopy(slot['retired']);return out

def encode(logical):
 if not isinstance(logical,dict) or 'pool' not in logical or 'retired' not in logical or not isinstance(logical.get('config'),dict):raise ValueError('missing logical pool state')
 if FIELD in logical['config'] or '_automatic_negative_state' in logical['config']:raise ValueError('duplicate or legacy pool state')
 p=dict(logical);p['config']=dict(logical['config']);p.pop('pool');p.pop('retired')
 p['config'][FIELD]=_slot(logical['epoch'],logical['pool'],logical['retired'],logical['metrics'])
 return safe.encode(p)

def initial(payload):
 if not isinstance(payload,dict) or payload.get('epoch')!=0 or 'pool' in payload or 'retired' in payload or FIELD in payload.get('config',{}) or '_automatic_negative_state' in payload.get('config',{}):raise ValueError('clean epoch-zero state required')
 q=dict(payload);q['pool']=None;q['retired']=[];return encode(q)
