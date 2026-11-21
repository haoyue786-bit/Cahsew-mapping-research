"""Same original catalog/miner; selection is supplied by the joint review."""
from pathlib import Path
import numpy as np
from on_demand_memory import read,write,sha
from automatic_negative_selection_validation_v1 import validate_selection

def summarize(rows,identity,config):
 fields=['source_cashew_distance_mean','p2_increase_mean','regional_context_rank']
 distribution={}
 for k in fields:
  a=np.array([r[k] for r in rows],dtype=float)
  distribution[k]=dict(count=len(a),quantiles={str(q):float(np.quantile(a,q)) for q in [0,.1,.25,.5,.75,.9,1]}) if len(a) else dict(count=0,quantiles={})
 # Representatives are display examples, never a mining/selection count limit.
 chosen={}
 for k in fields:
  ordered=sorted(rows,key=lambda r:r[k])
  if ordered:
   for pos in [0,len(ordered)//2,len(ordered)-1]:chosen[ordered[pos]['patch_id']]=ordered[pos]
 keep=['patch_id','source_cashew_distance_mean','p2_increase_mean','regional_context_rank','valid_pixels']
 return dict(candidate_count=len(rows),distribution=distribution,examples=[{k:r[k] for k in keep if k in r} for r in chosen.values()],examples_are_not_candidate_limits=True,full_catalog_query='get_training_evidence(topic=negative_candidates,pointer=/rows,offset=...,limit=...)',previous_pool=config.get('_automatic_negative_state'),checkpoint_sha256=identity['source_checkpoint_sha256'],semantics='Distance/probability increase/prior are soft evidence; no new hard gate, geographic quota, count cap or maximum-p2 requirement; all protected samples unchanged. Negative loss separate, one occurrence per patch over128normal updates. Weight0..0.005, regional ranking>=0.25 and weights sum1.')

def make_controller(Base):
 class JointController(Base):
  def prepare_joint(self,epoch,checkpoint,review,config):
   from legacy_agent_guard_v1 import block_legacy
   block_legacy(self.job, 'direct_joint_negative_prepare')
   destination=self.out/('epoch_%03d'%epoch);destination.mkdir(exist_ok=True)
   if (destination/'candidate_catalog.json').exists():
    rows=read(destination/'candidate_catalog.json');identity=read(destination/'catalog_identity.json')
    assert identity['source_checkpoint_sha256']==sha(checkpoint)
    for row in rows:assert sha(row['cache'])==row['cache_sha256']
   else:rows,identity=self.catalog(epoch,checkpoint,destination)
   summary=summarize(rows,identity,config)
   request=dict(epoch=epoch,rows=rows,identity=identity,summary=summary)
   write(destination/'joint_candidates.json',request)
   self.job['_joint_negative_candidates']=dict(epoch=epoch,path=str(destination/'joint_candidates.json'),sha256=sha(destination/'joint_candidates.json'),summary=summary)
   return summary

  def select(self,epoch,destination,rows,identity,review,config):
   from legacy_agent_guard_v1 import block_legacy
   block_legacy(self.job, 'direct_joint_negative_select')
   # This override deliberately has no model/client. Refresh can only consume
   # a selection already produced by the same scientific review.
   p=destination/'joint_selection.json'
   if not p.is_file():raise RuntimeError('Joint negative decision missing; cannot issue a hidden second model call')
   result=read(p);assert result['epoch']==epoch and result['checkpoint_sha256']==identity['source_checkpoint_sha256']
   decision=validate_selection(result['automatic_negative'],len(rows))
   assert sha(result['review_decision'])==result['review_decision_sha256']
   write(destination/'selection_decision.json',decision)
   write(destination/'selection_provenance.json',dict(joint_training_review=True,new_model_calls=0,thread_id=result['thread_id'],turn_id=result['turn_id'],review_decision=result['review_decision'],review_decision_sha256=result['review_decision_sha256']))
   return decision
 return JointController
