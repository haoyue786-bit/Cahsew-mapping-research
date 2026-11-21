"""Side-effect-free native config application on trusted local checkpoint bytes.

Experimental only: this is not a worker launch authorization, lineage check,
or a deserializer for externally supplied files. The caller authenticates bytes
and fully validates the state, provenance, and decision before calling it.
Pool publication and branches are deliberately unsupported here.
"""
import io
import torch


class _OptimizerProxy:
 def __init__(self,state):
  groups=state.get('param_groups') if isinstance(state,dict) else None
  if not isinstance(groups,list) or not groups or any(not isinstance(g,dict) for g in groups):
   raise ValueError('missing optimizer groups')
  self.param_groups=groups


def native_transition(pre,decision):
 """Return an independently serialized post state, never touch a live model."""
 if not isinstance(pre,bytes) or not isinstance(decision,dict):raise ValueError('missing checkpoint or decision')
 payload=torch.load(io.BytesIO(pre),map_location='cpu',weights_only=False)
 if not isinstance(payload,dict) or payload.get('epoch')!=decision.get('epoch'):raise ValueError('checkpoint epoch mismatch')
 config=payload.get('config')
 if not isinstance(config,dict):raise ValueError('missing checkpoint configuration')
 from healthy_recovery_runtime_v4 import verify_decision_checkpoint_controls
 verify_decision_checkpoint_controls(decision,config,'before')
 action=decision.get('action')
 if action not in {'KEEP','ADJUST','CONTINUE','PAUSE_ADV','RESUME_ADV','STOP','SELECT_CHECKPOINT','REQUEST_EXPERIMENT','DISABLE_AUTOMATIC_NEGATIVE'}:
  raise ValueError('unsupported native action')
 auto=decision.get('automatic_negative')
 if auto is not None and (not isinstance(auto,dict) or auto.get('action')!='ABSTAIN'):
  raise ValueError('automatic pool changes require separate verified transaction')
 if action=='DISABLE_AUTOMATIC_NEGATIVE':raise ValueError('pool disable publication is not implemented')
 if action in {'STOP','SELECT_CHECKPOINT','REQUEST_EXPERIMENT'}:
  if decision.get('intervention',{}).get('changed_parameters'):
   raise ValueError('terminal or branch decision must not alter controls')
  verify_decision_checkpoint_controls(decision,config,'after')
  return pre
 from train_expert_v4 import apply_config
 optimizers=[_OptimizerProxy(payload.get(name)) for name in ('model_optimizer','local_optimizer','global_optimizer')]
 apply_config(config,decision,*optimizers)
 verify_decision_checkpoint_controls(decision,config,'after')
 out=io.BytesIO();torch.save(payload,out);return out.getvalue()
