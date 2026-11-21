"""Opt-in actual supplementary-exposure telemetry for a completed epoch.

No extra data reads or RNG; masks use already loaded, augmented training tensors.
Valid-pixel counts and nonzero coefficients show exposure, not gradient magnitude,
causality, target-map correctness or held-out accuracy. Sample IDs stay in memory.
"""
import math
import torch

class Exposure:
 def __init__(self,schedule):
  self.planned_occurrences=0;self.planned_steps=0;self.planned_ids=set()
  for ids in schedule.values():
   converted=self._ids(ids);self.planned_occurrences+=len(converted);self.planned_steps+=bool(converted);self.planned_ids.update(converted)
  self.attempted=0;self.attempted_steps=0;self.attempted_ids=set()
  self.loaded=0;self.loaded_steps=0;self.loaded_ids=set();self.forward_steps=0;self.loss_steps=0
  self.completed_steps=0;self.completed_occurrences=0;self.completed_ids=set()
  self.pos_steps=0;self.neg_steps=0;self.pos_pixels=0;self.neg_pixels=0;self.pos_ids=set();self.neg_ids=set();self.unknown_sign_steps=0
  self._current=None
 @staticmethod
 def _ids(ids):
  try:
   out=[int(x) for x in ids]
   if any(isinstance(x,(bool,float)) or int(x)!=x for x in ids):raise ValueError('noninteger index')
  except (TypeError,ValueError,OverflowError):raise ValueError('invalid supplementary sample indices')
  if any(i<0 for i in out):raise ValueError('negative supplementary sample index')
  return out
 def start(self,ids):
  if self._current is not None:raise RuntimeError('supplementary step pending')
  ids=self._ids(ids);self._current={'ids':ids,'loaded':False,'forward':False,'loss':False,'signs':None}
  if not ids:raise ValueError('empty supplementary batch cannot be executed')
  self.attempted+=len(ids);self.attempted_steps+=bool(ids);self.attempted_ids.update(ids)
 def loaded_batch(self,images,labels):
  s=self._current
  if s is None or s['loaded'] or images.shape[0]!=len(s['ids']) or labels.shape[0]!=len(s['ids']):raise ValueError('loaded batch does not match planned indices')
  s['loaded']=True;self.loaded+=len(s['ids']);self.loaded_steps+=bool(s['ids']);self.loaded_ids.update(s['ids'])
 def forward_done(self):
  if self._current is None or not self._current['loaded'] or self._current['forward']:raise RuntimeError('invalid supplementary forward state')
  self._current['forward']=True;self.forward_steps+=1
 def observed_loss(self,images,labels,*,human_mode,weight,positive_scale=None,expected_stats=None):
  s=self._current
  if s is None or not s['forward'] or s['loss']:raise RuntimeError('invalid supplementary loss state')
  if type(weight) not in (int,float) or not math.isfinite(weight) or weight<0:raise ValueError('invalid supplementary loss weight')
  s['loss']=True;self.loss_steps+=1
  if not human_mode:return
  if type(positive_scale) not in (int,float) or not math.isfinite(positive_scale) or positive_scale<0:raise ValueError('invalid positive scale')
  if images.ndim!=4 or labels.ndim!=3 or images.shape[0]!=labels.shape[0] or images.shape[2:]!=labels.shape[1:]:raise ValueError('invalid human mask shape')
  with torch.no_grad():
   valid=torch.isfinite(images).all(dim=1)&(images!=0).any(dim=1)
   pos=((labels==2)&valid).flatten(1).sum(1);neg=((labels==0)&valid).flatten(1).sum(1)
   by_patch=torch.stack((neg,pos),dim=1).cpu().tolist()
  totalneg=sum(int(x[0]) for x in by_patch);totalpos=sum(int(x[1]) for x in by_patch)
  if not isinstance(expected_stats,dict) or totalneg!=expected_stats.get('negative_pixels') or totalpos!=expected_stats.get('positive_pixels'):raise ValueError('human exposure masks disagree with loss stats')
  s['signs']={'neg':[(i,int(v[0])) for i,v in zip(s['ids'],by_patch) if v[0]>0 and weight>0],
              'pos':[(i,int(v[1])) for i,v in zip(s['ids'],by_patch) if v[1]>0 and weight>0 and positive_scale>0]}
 def optimizer_succeeded(self):
  s=self._current
  if s is None or not s['loss']:raise RuntimeError('optimizer result before supplementary loss')
  self.completed_steps+=1;self.completed_occurrences+=len(s['ids']);self.completed_ids.update(s['ids'])
  signs=s['signs']
  if signs is None:self.unknown_sign_steps+=1
  else:
   self.neg_steps+=bool(signs['neg']);self.pos_steps+=bool(signs['pos'])
   self.neg_pixels+=sum(n for _,n in signs['neg']);self.pos_pixels+=sum(n for _,n in signs['pos'])
   self.neg_ids.update(i for i,_ in signs['neg']);self.pos_ids.update(i for i,_ in signs['pos'])
  self._current=None
 def snapshot(self):
  if self._current is not None:raise RuntimeError('supplementary step was not committed')
  return {'version':'supplementary_actual_exposure_v1','planned_steps':self.planned_steps,'planned_occurrences':self.planned_occurrences,'planned_unique_patches':len(self.planned_ids),
   'attempted_steps':self.attempted_steps,'attempted_occurrences':self.attempted,'attempted_unique_patches':len(self.attempted_ids),
   'loaded_steps':self.loaded_steps,'loaded_occurrences':self.loaded,'loaded_unique_patches':len(self.loaded_ids),'forward_steps':self.forward_steps,'loss_steps':self.loss_steps,
   'completed_optimizer_steps':self.completed_steps,'completed_occurrences':self.completed_occurrences,'completed_unique_patches':len(self.completed_ids),
   'human_negative_effective_steps':self.neg_steps,'human_positive_effective_steps':self.pos_steps,'human_negative_effective_pixel_occurrences':self.neg_pixels,'human_positive_effective_pixel_occurrences':self.pos_pixels,
   'human_negative_effective_unique_patches':len(self.neg_ids),'human_positive_effective_unique_patches':len(self.pos_ids),'unknown_sign_completed_steps':self.unknown_sign_steps,
   'semantics':'per-patch valid pixels and nonzero loss coefficient after a completed G step; this does not establish nonzero gradient, causal effect, or target accuracy'}
