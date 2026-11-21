"""Synthetic weights-only joint pool router and journal; production hard-disabled.

All checkpoint decoding goes through native_safe_checkpoint_codec_v1. Agent
responses are reconstructed against point-sealed requests; pool mutations go
through the existing single-authority UnifiedEpochPool. No caller-supplied
attestor proves clean training, input provenance, or authorization.
"""
import copy,json
import tempfile
from pathlib import Path
from decision_transaction import digest,read_json
from fd_safe_decision_transaction_candidate_v1 import SafeTransaction
from sealed_safe_final_selection_v1 import SafeRoot
from sealed_epoch_journal_v1 import EpochJournal
from sealed_safe_epoch_journal_v1 import SafeTrainingJournal
from sealed_worker_adapter_v1 import stage_for
from dispatch_transaction_bridge import attach_dispatch_receipt
from request_reconstruction import verify_request
from decision_bridge import normalize_review,normalize_anomaly_review,normalize_automatic_selection
from sealed_safe_router_sim_v1 import _post as safe_ordinary_post
from unified_epoch_pool_bridge_sim_v1 import _equal
import native_safe_checkpoint_codec_v1 as safe
import safe_pool_checkpoint_codec_sim_v1 as poolcodec
import sealed_dispatch_v1 as dispatcher

KINDS={'safe_anomaly_review_candidate_v1':'light_review','safe_full_review_candidate_v1':'full_review'}

def _production_gate():
 raise RuntimeError('trusted pool and training lineage unavailable; production disabled')

def _decision(job,request,answer,raw):
 p=poolcodec.decode(raw)
 if p['epoch']!=request.get('epoch'):raise ValueError('decision checkpoint epoch mismatch')
 cfg=copy.deepcopy(p['config']);kind=request.get('contract_version');action=answer.get('action')
 if kind=='safe_full_review_candidate_v1':
  if action not in ('KEEP','ADJUST','CONTINUE','PAUSE_ADV','RESUME_ADV'):raise ValueError('terminal/unknown action unsupported')
  d=normalize_review(request,answer,cfg,job,minimum_stop_epoch=0,stop_allowed=False,source_collapsed=False,experiment_limit=0)
  if request.get('automatic_selection_enabled'):
   d['automatic_negative']=normalize_automatic_selection(request,answer,actual_candidate_count=request.get('automatic_candidate_limit'),actual_max_weight=request.get('automatic_max_weight'))
   if d['automatic_negative']['action']=='ACTIVATE' and p['pool'] is not None:raise ValueError('pool already active')
  elif answer.get('automatic_negative') is not None:raise ValueError('unadvertised pool response')
 elif kind=='safe_anomaly_review_candidate_v1':
  if action=='DISABLE_AUTOMATIC_NEGATIVE':
   pool=p['pool'];advertised=((request.get('evidence') or {}).get('active_automatic_pool') or {}).get('values') or {}
   if not isinstance(pool,dict) or advertised.get('weight')!=pool['weight']:raise ValueError('unverified active pool')
   cfg['_automatic_negative_state']={'weight':pool['weight']}
  elif action not in ('KEEP','ADJUST'):raise ValueError('unsupported light action')
  d=normalize_anomaly_review(request,answer,cfg)
  if action=='DISABLE_AUTOMATIC_NEGATIVE':d['automatic_negative']={'action':'DISABLE'}
 else:raise ValueError('unsupported review contract')
 if d.get('continue_training') is not True:raise ValueError('terminal decision unsupported')
 return d

def ordinary_post(pre,decision):
 logical=poolcodec.decode(pre);oldpool=copy.deepcopy(logical['pool']);retired=copy.deepcopy(logical['retired'])
 ordinary=copy.deepcopy(decision);ordinary.pop('automatic_negative',None);ordinary.pop('automatic_negative_action',None)
 if ordinary.get('action')=='DISABLE_AUTOMATIC_NEGATIVE':ordinary['action']='KEEP'
 native=dict(logical);native.pop('pool');native.pop('retired')
 plain=safe.encode(native);updated=safe.decode(safe_ordinary_post(plain,ordinary))
 updated['pool']=oldpool;updated['retired']=retired;return poolcodec.encode(updated)

def route(journal,request,trusted,expected,pre,*,transition,pool_receipt_verify=None,codex_binary,skill_sha256,verifier_sha256,reference_hashes,previous=None,client_class=None,timeout=1800):
 job=journal.job;policy=job.get('experimental_gate') or {}
 if not isinstance(policy,dict) or policy.get('simulation_only') is not True or policy.get('experimental_only') is not True:raise ValueError('synthetic policy required')
 if client_class is None:client_class=dispatcher.PersistentLunaClient
 clean=verify_request(request,trusted,expected=expected,previous=previous);epoch=clean.get('epoch');kind=KINDS.get(clean.get('contract_version'))
 if kind is None or type(epoch) is not int or epoch<1:raise ValueError('unsupported request')
 poolcodec.decode(pre)
 prefix='agent/sealed_v1/epoch_%03d/%s'%(epoch,'light' if kind=='light_review' else 'full')
 owner=SafeRoot(job['output_root'])
 try:owner.mkdirs(prefix)
 finally:owner.close()
 stage=Path(job['output_root'])/prefix;txn=SafeTransaction(stage/'transaction',job['job_id'],job['run_id'],epoch,kind)
 txn.prepare(clean,pre,journal.check)
 reply=dispatcher.dispatch(clean,trusted,expected,stage/'transport',codex_binary=codex_binary,skill_sha256=skill_sha256,
   verifier_sha256=verifier_sha256,reference_hashes=reference_hashes,previous=previous,before_turn=txn.claim_submission,client_class=client_class,timeout=timeout)
 attach_dispatch_receipt(txn,stage/'transport',trusted=trusted,expected=expected,skill_path=dispatcher.SKILL,skill_sha256=skill_sha256,
   verifier_sha256=verifier_sha256,reference_hashes=reference_hashes,previous=previous)
 decision=txn.normalize(lambda q,a,p:_decision(job,q,a,p))
 def verify(q,a,p,d,post):
  if d!=_decision(job,q,a,p):raise ValueError('decision differs from response')
  action=(d.get('automatic_negative') or {}).get('action')
  if action in ('ACTIVATE','DISABLE'):
   base_path=journal._dir(epoch)/'pool_action/native_post.bin'
   if base_path.is_symlink() or not base_path.is_file() or not _equal(poolcodec.decode(base_path.read_bytes()),poolcodec.decode(ordinary_post(p,d))):
    raise ValueError('pool action native controls differ from approved decision')
   if not callable(pool_receipt_verify) or pool_receipt_verify(epoch,d,post) is not True:raise ValueError('pool receipt not verified')
  elif not _equal(poolcodec.decode(post),poolcodec.decode(ordinary_post(p,d))):raise ValueError('unauthorized post checkpoint')
 commit,post=txn.commit(journal.check,transition,post_validate=verify)
 return {'decision':decision,'post_checkpoint':post,'commit':commit,'replayed':reply['replayed']}

class SafePoolJournal(SafeTrainingJournal):
 """Pool-aware simulation journal; only bridge ownership can advance it."""
 def __init__(self,root,job,max_epoch):
  _production_gate()
  if job.get('experimental_gate',{}).get('simulation_only') is not True:raise ValueError('synthetic-only journal')
  try:Path(root).resolve().relative_to(Path(tempfile.gettempdir()).resolve())
  except ValueError:raise ValueError('synthetic journal must be temporary')
  super().__init__(root,job,max_epoch)
 def recover(self):
  self._held();bridge=getattr(self,'_active_pool_bridge',None)
  if bridge is None:raise RuntimeError('safe pool bridge lease required')
  return SafeTrainingJournal.recover(self)

 def _verify_committed_agent(self,epoch,request,answer,pre,decision,post):
  bridge=getattr(self,'_active_pool_bridge',None)
  if bridge is None or decision!=_decision(self.job,request,answer,pre):raise ValueError('stored review decision differs from response')
  action=(decision.get('automatic_negative') or {}).get('action')
  if action in ('ACTIVATE','DISABLE'):
   base=self._fs.read('epoch_%06d/pool_action/native_post.bin'%epoch)
   if not _equal(poolcodec.decode(base),poolcodec.decode(ordinary_post(pre,decision))):
    raise ValueError('stored pool native controls differ from approved decision')
   receipt=self._fs.json('epoch_%06d/pool_receipt.json'%epoch)
   bridge._audit_pool_action(epoch,receipt['pool_after'],post)
  elif not _equal(poolcodec.decode(post),poolcodec.decode(ordinary_post(pre,decision))):raise ValueError('stored ordinary post differs')

 def active_pool_evidence(self,epoch):
  import math
  self._held();bridge=getattr(self,'_active_pool_bridge',None)
  if bridge is None:return None
  state=bridge.recover()
  if state['state']!='LIGHT_REVIEW' or state['epoch']!=epoch:raise RuntimeError('pool evidence requires owning light review')
  checkpoint=state['checkpoint'];logical=bridge.decode(checkpoint);receipt=self._fs.json('epoch_%06d/pool_receipt.json'%epoch)
  if logical.get('epoch')!=epoch or receipt.get('post_sha256')!=digest(checkpoint) or receipt.get('epoch')!=epoch:
   raise ValueError('pool receipt is not bound to current checkpoint')
  after=receipt.get('pool_after');before=receipt.get('pool_before');exposure=receipt.get('exposure')
  if not isinstance(after,dict) or not isinstance(before,dict) or not isinstance(exposure,dict):raise ValueError('incomplete pool receipt')
  pool=after.get('pool')
  if pool is None:
   if logical.get('pool') is not None:raise ValueError('checkpoint and pool mismatch')
   return None
  if logical.get('pool')!=pool or not isinstance(before.get('pool'),dict):raise ValueError('checkpoint and pool mismatch')
  auto=(logical.get('metrics') or {}).get('automatic_negative')
  if not isinstance(auto,dict):raise ValueError('active pool exposure not measured')
  for a,b in (('processed_contiguous_positions','positions'),('completed_G_updates','completed_g'),
              ('completed_G_auxiliary_steps','auxiliary_steps'),('completed_G_patch_occurrences','patch_occurrences'),
              ('completed_G_pixel_occurrences','pixel_occurrences')):
   if type(auto.get(a)) is not int or type(exposure.get(b)) is not int or auto[a]!=exposure[b]:raise ValueError('pool exposure mismatch')
  updates=auto.get('updates');weight=pool.get('weight');effective=auto.get('effective_weight')
  if type(updates) is not int or updates<0 or updates!=exposure['auxiliary_steps']:raise ValueError('pool update mismatch')
  if any(type(v) not in (int,float) or not math.isfinite(v) for v in (weight,effective)) or not 0<weight<=.005 or effective not in (0,weight):
   raise ValueError('invalid pool weight')
  if auto.get('configured_weight')!=weight:raise ValueError('configured pool weight mismatch')
  cursor=pool.get('cursor');horizon=pool.get('horizon');start=before['pool'].get('cursor')
  if (type(cursor) is not int or type(horizon) is not int or type(start) is not int or not 0<=start<=cursor<=horizon
      or auto.get('generation_start_cursor')!=start or auto.get('generation_end_cursor')!=cursor
      or auto.get('generation_horizon_updates')!=horizon or cursor-start!=exposure['positions']):raise ValueError('pool cursor mismatch')
  return {'weight':weight,'effective_weight':effective,'updates':updates,'cursor':cursor,'horizon':horizon}

 def finish_agent(self,epoch,request,trusted,expected,**options):
  self._held();s=self.recover();kind={'LIGHT_REVIEW':'safe_anomaly_review_candidate_v1','FULL_REVIEW':'safe_full_review_candidate_v1'}.get(s['state'])
  if s['epoch']!=epoch or kind is None or request.get('contract_version')!=kind or request.get('epoch')!=epoch:raise RuntimeError('review mismatch')
  self._verify_review_request(epoch,request,s['state'],expected,trusted,options.get('previous'))
  pre=s['checkpoint']
  if s['state']=='FULL_REVIEW':pre,_=self._default_transition(epoch,pre,'before_agent')
  result=route(self,request,trusted,expected,pre,**options);post=result['post_checkpoint']
  if s['state']=='LIGHT_REVIEW':post,_=self._default_transition(epoch,post,'after_agent')
  return self._commit(epoch,post,'agent',result['decision']['action'])
