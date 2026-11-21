"""In-memory paired source/target exposure counters; no extra data reads.

Record batch indices (never emitted), confirm already-loaded tensor batch sizes,
and mark only after the G optimizer step returns. For a paused adversary the
unlabelled target batch may be loaded but it is not used in G/D loss. Counts do
not establish valid labelled pixels, nonzero gradient, or target accuracy.
"""
import math,numbers
class Exposure:
 def __init__(self,source_count,target_count):
  if any(type(x) is not int or x<1 for x in (source_count,target_count)):raise ValueError('invalid domain population')
  self.limits={'source':source_count,'target':target_count};self._current=None
  self._count={phase:{d:0 for d in self.limits} for phase in ('planned','loaded','completed','target_used','target_generator')}
  self._ids={phase:{d:set() for d in self.limits} for phase in ('planned','loaded','completed','target_used','target_generator')}
  self.steps={phase:0 for phase in ('planned','loaded','completed','target_used','target_generator')}
 def start(self,source_indices,target_indices,*,adversarial_active,generator_adversarial_weight,discriminator_active=None):
  if discriminator_active is None:discriminator_active=adversarial_active
  if self._current is not None or type(adversarial_active) is not bool or type(discriminator_active) is not bool or (discriminator_active and not adversarial_active):raise ValueError('pending step or unknown adversarial state')
  if type(generator_adversarial_weight) not in (int,float) or not math.isfinite(generator_adversarial_weight) or generator_adversarial_weight<0:raise ValueError('invalid adversarial weight')
  batch={}
  for d,indices in (('source',source_indices),('target',target_indices)):
   try:raw=list(indices)
   except TypeError:raise ValueError('non-iterable index sequence')
   if not raw or any(isinstance(x,bool) or not isinstance(x,numbers.Integral) or not 0<=x<self.limits[d] for x in raw):raise ValueError('invalid sample index')
   batch[d]=tuple(int(x) for x in raw)
  self._current={'batch':batch,'active':discriminator_active,'generator_active':adversarial_active and generator_adversarial_weight>0,'loaded':False}
  for d in self.limits:self._count['planned'][d]+=len(batch[d]);self._ids['planned'][d].update(batch[d])
  self.steps['planned']+=1
 def loaded(self,source_size,target_size):
  c=self._current
  if c is None or c['loaded'] or type(source_size) is not int or type(target_size) is not int:raise ValueError('invalid load')
  if source_size!=len(c['batch']['source']) or target_size!=len(c['batch']['target']):raise ValueError('loaded batch differs from planned')
  c['loaded']=True
  for d in self.limits:self._count['loaded'][d]+=len(c['batch'][d]);self._ids['loaded'][d].update(c['batch'][d])
  self.steps['loaded']+=1
 def optimizer_succeeded(self):
  c=self._current
  if c is None or not c['loaded']:raise RuntimeError('incomplete paired batch')
  for d in self.limits:self._count['completed'][d]+=len(c['batch'][d]);self._ids['completed'][d].update(c['batch'][d])
  self.steps['completed']+=1
  if c['active']:
   self._count['target_used']['target']+=len(c['batch']['target']);self._ids['target_used']['target'].update(c['batch']['target']);self.steps['target_used']+=1
  if c['generator_active']:
   self._count['target_generator']['target']+=len(c['batch']['target']);self._ids['target_generator']['target'].update(c['batch']['target']);self.steps['target_generator']+=1
  self._current=None
 def snapshot(self):
  if self._current is not None:raise RuntimeError('partial epoch exposure must not be published as completed')
  out={'version':'source_target_exposure_v1','planned_steps':self.steps['planned'],'loaded_steps':self.steps['loaded'],'completed_g_steps':self.steps['completed'],'target_loss_active_completed_steps':self.steps['target_used']}
  for phase in ('planned','loaded','completed'):
   for domain in self.limits:
    out[f'{domain}_{phase}_occurrences']=self._count[phase][domain];out[f'{domain}_{phase}_unique']=len(self._ids[phase][domain])
  out['target_loss_active_occurrences']=self._count['target_used']['target'];out['target_loss_active_unique']=len(self._ids['target_used']['target'])
  out['target_generator_nonzero_weight_steps']=self.steps['target_generator'];out['target_generator_nonzero_weight_occurrences']=self._count['target_generator']['target'];out['target_generator_nonzero_weight_unique']=len(self._ids['target_generator']['target'])
  out['meaning']='indices of successfully loaded batches and completed G steps; target_loss_active means D target loss evaluated; target_generator_nonzero_weight means G adversarial loss has nonzero coefficient, not nonzero gradient, target labels, or model quality; source valid label pixels unmeasured'
  return out
