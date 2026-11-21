"""Opt-in deterministic, bounded optimizer sampling; not a launch permit."""
import torch
from experimental_gate_v1 import StepRecorder

class BoundedRecorder(StepRecorder):
 def bind_model(self,model):
  from module_controls_v2 import component
  self.module_names={id(p):component(n) for n,p in model.named_parameters()}
  self.full_module_updates=[]
 def __init__(self,count=8,max_tensors=16):
  if type(count) is not int or count<1 or type(max_tensors) is not int or max_tensors<1:raise ValueError('sample budgets must be positive integers')
  super().__init__(count=count);self.max_tensors=max_tensors;self.coverage=[]
 def step(self,name,optimizer,sample):
  if name not in self.counts or type(sample) is not bool:raise ValueError('unknown optimizer or sample state')
  self.attempted_counts[name]+=1
  if not any(p.grad is not None for g in optimizer.param_groups for p in g['params']):raise RuntimeError('optimizer has no connected gradients: update prevented')
  self.groups[name]=[{'lr':float(g['lr']),'lr_multiplier':float(g.get('lr_multiplier',1.)), 'weight_decay':float(g.get('weight_decay',0)), 'betas':list(g.get('betas',())), 'eps':g.get('eps')} for g in optimizer.param_groups]
  before=[];full=[]
  if name=='G' and sample and hasattr(self,'module_names'):
   full=[(self.module_names[id(p)],p,p.detach().clone()) for g in optimizer.param_groups for p in g['params'] if p.grad is not None]
  if sample:
   with torch.no_grad():
    for gi,group in enumerate(optimizer.param_groups):
     eligible=[p for p in group['params'] if p.grad is not None and p.numel()]
     n=min(self.max_tensors,len(eligible));positions=[round(i*(len(eligible)-1)/max(n-1,1)) for i in range(n)]
     self.coverage.append({'optimizer':name,'group':gi,'eligible_tensors':len(eligible),'sampled_tensors':n})
     for position in positions:
      p=eligible[position];count=min(self.count,p.numel());idx=torch.linspace(0,p.numel()-1,count,device=p.device).round().long();before.append((gi,p,idx,p.detach().reshape(-1)[idx].clone()))
  optimizer.step();self.counts[name]+=1;self.with_grad_counts[name]+=1
  if full:
   from collections import defaultdict
   rows=defaultdict(lambda:{'delta2':0.,'before2':0.,'elements':0})
   with torch.no_grad():
    for c,p,old in full:
     r=rows[c];r['delta2']+=float((p-old).double().square().sum());r['before2']+=float(old.double().square().sum());r['elements']+=p.numel()
   self.full_module_updates.extend(dict(optimizer=name,optimizer_step=self.counts[name],component=c,
       update_l2=r['delta2']**.5,parameter_l2=r['before2']**.5,
       update_to_parameter_l2=(r['delta2']/r['before2'])**.5 if r['before2'] else None,
       parameter_elements=r['elements'],scope='Exact full active module at this executed optimizer step; sampled steps, not every update.') for c,r in rows.items())
  if sample:
   with torch.no_grad():
    for gi,p,idx,old in before:
     after=p.detach().reshape(-1)[idx];delta=(after-old).float();finite=torch.isfinite(after).all()&torch.isfinite(old).all()&torch.isfinite(delta).all()
     self.samples.append((name,gi,torch.stack((delta.square().sum(),old.float().square().sum(),finite.float(),delta.new_tensor(float(idx.numel()))))))
 def finish(self):
  result=super().finish();result['sample_tensor_coverage']=self.coverage
  result['sample_budget']={'max_tensors_per_group':self.max_tensors,'elements_per_tensor':self.count,'selection':'fixed evenly spaced; no RNG'}
  result['meaning']='sampled tensors and elements only; not full norms, causality or population representativeness; no samples means missing, not zero'
  if hasattr(self,'full_module_updates'):result['full_module_update_samples']=self.full_module_updates
  return result
