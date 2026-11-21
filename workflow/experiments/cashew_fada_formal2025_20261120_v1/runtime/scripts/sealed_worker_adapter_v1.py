"""Experimental worker adapter: sealed model response -> native decision.

This adapter does not start training, attest evidence provenance, execute a
branch, apply a decision, or freeze a checkpoint. The worker launch remains
blocked until all those boundaries and trajectory identities are verified.
"""
import hashlib,json
from pathlib import Path
import sealed_dispatch_v1 as dispatcher
from decision_bridge import normalize_initialization,normalize_anomaly_review,normalize_review,normalize_selection,normalize_automatic_selection

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def stage_for(job,request):
 if not isinstance(job,dict) or not isinstance(job.get('job_id'),str) or not isinstance(job.get('run_id'),str):raise ValueError('missing worker identity')
 root=Path(job['output_root']).resolve();kind=request['contract_version'];epoch=request.get('epoch')
 if kind=='safe_checkpoint_selection_candidate_v1':stage=root/'agent'/'sealed_v1'/'selection'
 elif kind=='safe_initialization_candidate_v1' and epoch==0:stage=root/'agent'/'sealed_v1'/'initialization'
 elif kind in ('safe_anomaly_review_candidate_v1','safe_full_review_candidate_v1','safe_branch_review_candidate_v1') and type(epoch) is int and epoch>0:
  stage=root/'agent'/'sealed_v1'/('epoch_%03d'%epoch)/('light' if kind=='safe_anomaly_review_candidate_v1' else 'branch' if kind=='safe_branch_review_candidate_v1' else 'full')
 else:raise ValueError('invalid decision stage')
 stage=stage.resolve()
 try:stage.relative_to(root)
 except ValueError:raise ValueError('decision stage escapes worker root')
 return stage

def request_decision(job,config,request,trusted,expected,*,codex_binary,skill_sha256,verifier_sha256,reference_hashes,
 previous=None,minimum_stop_epoch=None,stop_allowed=None,source_collapsed=None,experiment_attempts=0,experiment_limit=0,
 frozen_stop_epoch=None,automatic_candidate_count=None,automatic_max_weight=None,before_turn=None,client_class=None,timeout=1800):
 """Construct immutable context, dispatch, then invoke worker's native guard."""
 if request.get('contract_version')=='safe_anomaly_review_candidate_v1' and request.get('evidence',{}).get('active_automatic_pool') is not None:
  raise ValueError('active pool reviews require the checkpoint-bound transaction router')
 kind=request['contract_version'];stage=stage_for(job,request);stage.mkdir(parents=True,exist_ok=True)
 context={'job_id':job['job_id'],'run_id':job['run_id'],'request_sha256':digest(request),
  'config_sha256':digest(config),'job_sha256':digest(job),'expected_sha256':digest(expected),
  'minimum_stop_epoch':minimum_stop_epoch,'stop_allowed':stop_allowed,'source_collapsed':source_collapsed,
  'experiment_attempts':experiment_attempts,'experiment_limit':experiment_limit,'frozen_stop_epoch':frozen_stop_epoch,
  'automatic_candidate_count':automatic_candidate_count,'automatic_max_weight':automatic_max_weight}
 # Immutable context makes a replay against altered configuration fail closed.
 dispatcher.publish(stage/'normalization_context.json',context)
 opts={'codex_binary':codex_binary,'skill_sha256':skill_sha256,'verifier_sha256':verifier_sha256,
       'reference_hashes':reference_hashes,'previous':previous,'before_turn':before_turn,'timeout':timeout}
 if client_class is not None:opts['client_class']=client_class
 result=dispatcher.dispatch(request,trusted,expected,stage,**opts)
 if digest(config)!=context['config_sha256'] or digest(job)!=context['job_sha256'] or digest(expected)!=context['expected_sha256']:
  raise RuntimeError('worker state changed during model request; reconcile before applying')
 answer=result['answer'];auto=None
 if kind=='safe_initialization_candidate_v1':decision=normalize_initialization(request,answer,config,job)
 elif kind=='safe_anomaly_review_candidate_v1':decision=normalize_anomaly_review(request,answer,config)
 elif kind in ('safe_full_review_candidate_v1','safe_branch_review_candidate_v1'):
  decision=normalize_review(request,answer,config,job,minimum_stop_epoch=minimum_stop_epoch,stop_allowed=stop_allowed,
    source_collapsed=source_collapsed,experiment_attempts=experiment_attempts,experiment_limit=experiment_limit)
  if request.get('automatic_selection_enabled'):
   auto=normalize_automatic_selection(request,answer,actual_candidate_count=automatic_candidate_count,actual_max_weight=automatic_max_weight)
 elif kind=='safe_checkpoint_selection_candidate_v1':decision=normalize_selection(request,answer,config,job,frozen_stop_epoch=frozen_stop_epoch)
 else:raise ValueError('unknown decision stage')
 envelope={'decision':decision,'automatic_negative':auto,'dispatch_receipt':result['receipt'],
           'normalization_context_sha256':digest(context)}
 dispatcher.publish(stage/'native_decision.json',envelope)
 return envelope
