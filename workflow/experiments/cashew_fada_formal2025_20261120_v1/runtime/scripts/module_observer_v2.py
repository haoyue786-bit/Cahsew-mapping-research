"""Deterministic, bounded module observations; no historical checkpoint scans."""
import hashlib
import json
from pathlib import Path
import torch
from module_controls_v2 import COMPONENTS, component
from module_dynamics_v1 import positions

VERSION = 'module_observation_v2'
FIELDS = ('sample_before_l2', 'sample_delta_l2', 'sample_relative_l2', 'sample_elements',
          'sample_tensors', 'eligible_tensors', 'lr_min', 'lr_max', 'optimizer_step_min',
          'optimizer_step_max', 'initial_delta_l2', 'initial_relative_l2',
          'intervention_delta_l2', 'intervention_relative_l2', 'intervention_epoch',
          'bn_delta_l2', 'bn_count_delta', 'full_parameter_elements', 'full_parameter_l2',
          'full_initial_relative_l2', 'full_epoch_relative_l2', 'adam_first_moment_l2', 'adam_second_moment_mean')
GRAD_FIELDS = ('source_norm', 'human_negative_norm', 'human_positive_norm',
               'local_adversarial_norm', 'global_adversarial_norm',
               'source_negative_cosine', 'source_positive_cosine',
               'source_local_cosine', 'source_global_cosine', 'local_global_cosine',
               'source_tensors', 'negative_tensors', 'positive_tensors', 'local_tensors', 'global_tensors',
               'regularization_norm','regularization_tensors','source_regularization_cosine','negative_regularization_cosine')
KEYS = {'detail_'+c+'_'+f for c in COMPONENTS for f in FIELDS}
KEYS |= {'gradient_'+c+'_'+f for c in COMPONENTS for f in GRAD_FIELDS}
KEYS |= {'module_gradient_probe_count', 'module_observation_seconds'}
KEYS |= {'actual_update_'+c+'_'+f for c in COMPONENTS for f in ('relative_l2_median','observations')}

def project(metrics):
    result = {}
    samples=(metrics.get('experimental_actual_updates') or {}).get('full_module_update_samples',[])
    for c in COMPONENTS:
        values=sorted(x['update_to_parameter_l2'] for x in samples if x['component']==c and x['update_to_parameter_l2'] is not None)
        if values:
            result['actual_update_'+c+'_relative_l2_median']=(values[(len(values)-1)//2]+values[len(values)//2])/2
            result['actual_update_'+c+'_observations']=len(values)
    obs = metrics.get('module_observation')
    if obs is not None:
        if obs.get('version') != VERSION: raise ValueError('invalid module observation')
        for c, row in obs['components'].items():
            if c not in COMPONENTS or set(row) != set(FIELDS): raise ValueError('unknown module field')
            result.update({'detail_'+c+'_'+f: row[f] for f in FIELDS})
        result['module_observation_seconds'] = obs['observation_seconds']
    probes = metrics.get('module_gradient_probes')
    if probes is not None:
        result['module_gradient_probe_count'] = len(probes)
        for c in COMPONENTS:
            for f in GRAD_FIELDS:
                values = sorted(p['components'].get(c, {}).get(f) for p in probes
                                if p['components'].get(c, {}).get(f) is not None)
                result['gradient_'+c+'_'+f] = (values[(len(values)-1)//2]+values[len(values)//2])/2 if values else None
    return result

def _named(models):
    for role, model in models.items():
        for name, p in model.named_parameters():
            yield role+'/'+name, p, component(name) if role == 'G' else role

@torch.no_grad()
def snapshot(models):
    candidates = list(_named(models)); entries = {}
    for c in COMPONENTS:
        eligible = [(name, p) for name, p, k in candidates if k == c and p.requires_grad]
        for index in positions(len(eligible), 32):
            name, p = eligible[index]
            idx = torch.tensor(positions(p.numel(), 128), device=p.device)
            entries[name] = p.detach().reshape(-1)[idx].cpu().tolist()
    return entries

def _raw(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()

def anchor(root, models, epoch, config):
    """One small immutable sample per actual control boundary, cached by caller."""
    from expert_policy_v4 import flat
    from complete_history_store import _write_immutable
    dest = Path(root)/'module_anchors'; dest.mkdir(exist_ok=True)
    signature = hashlib.sha256(_raw({**flat(config), 'schedule_state': config.get('schedule_state')})).hexdigest()
    records = []
    for p in sorted(dest.glob('anchor_*.json')):
        # These are only bounded samples, never full model checkpoints.
        meta = json.loads(p.read_text())
        if meta['epoch'] <= epoch: records.append(meta)
    latest = max(records, key=lambda x: x['epoch']) if records else None
    if latest is None or latest['control_sha256'] != signature:
        latest = {'version': VERSION, 'epoch': epoch, 'control_sha256': signature,
                  'samples': snapshot(models)}
        _write_immutable(dest/('anchor_%06d.json'%epoch), _raw(latest))
        records.append(latest)
    initial = min(records, key=lambda x: x['epoch'])
    if initial['epoch'] != 0: raise ValueError('missing initialized module anchor')
    return initial, latest

class Observer:
    @torch.no_grad()
    def __init__(self, models, optimizers, initial, intervention):
        import time
        start = time.perf_counter()
        self.models = models; self.opts = optimizers
        self.initial = initial; self.intervention = intervention
        self.before = snapshot(models); self.bn = {}; self.done = False
        self.full_before={role+'/'+name:p.detach().clone() for role,model in models.items() for name,p in model.named_parameters() if p.requires_grad}
        for role, model in models.items():
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    for bank in ('', 'domain_source_'):
                        mean = getattr(module, bank+'running_mean', None)
                        var = getattr(module, bank+'running_var', None)
                        count = getattr(module, bank+'num_batches_tracked', None)
                        if mean is not None:
                            self.bn[(role, name, bank)] = (torch.cat((mean, var)).detach().clone(), int(count))
        self.capture_seconds = time.perf_counter()-start

    @torch.no_grad()
    def finish(self):
        import time
        if self.done: raise RuntimeError('observer already finished')
        self.done = True; start = time.perf_counter()
        after = snapshot(self.models); rows = {}; rates = {}; steps = {}
        for opt in self.opts.values():
            for group in opt.param_groups:
                for p in group['params']:
                    rates[id(p)] = float(group['lr'])
                    steps[id(p)] = int(opt.state.get(p, {}).get('step', 0))
        named = list(_named(self.models))
        for c in COMPONENTS:
            params = [(n, p) for n, p, k in named if k == c and p.requires_grad]
            names = [n for n, p in params if n in after]
            def norm(which):
                base = delta = 0.
                for n in names:
                    a = torch.tensor(which[n], dtype=torch.float64)
                    b = torch.tensor(after[n], dtype=torch.float64)
                    base += float(a.square().sum()); delta += float((a-b).square().sum())
                return base**.5, delta**.5, (delta/base)**.5 if base else None
            base, delta, relative = norm(self.before)
            _, init_delta, init_rel = norm(self.initial['samples'])
            _, action_delta, action_rel = norm(self.intervention['samples'])
            lr = [rates[id(p)] for n,p in params if id(p) in rates]
            st = [steps[id(p)] for n,p in params if id(p) in steps]
            rows[c] = dict(sample_before_l2=base if names else None, sample_delta_l2=delta if names else None,
                sample_relative_l2=relative, sample_elements=sum(len(after[n]) for n in names),
                sample_tensors=len(names), eligible_tensors=len(params), lr_min=min(lr) if lr else None,
                lr_max=max(lr) if lr else None, optimizer_step_min=min(st) if st else None,
                optimizer_step_max=max(st) if st else None, initial_delta_l2=init_delta if names else None,
                initial_relative_l2=init_rel, intervention_delta_l2=action_delta if names else None,
                intervention_relative_l2=action_rel, intervention_epoch=self.intervention['epoch'],
                bn_delta_l2=None, bn_count_delta=0,
                full_parameter_elements=sum(p.numel() for n,p in params),full_parameter_l2=None,
                full_initial_relative_l2=None,full_epoch_relative_l2=None,adam_first_moment_l2=None,adam_second_moment_mean=None)
            if params:
                sq=lambda v:float(v.detach().double().square().sum())
                rows[c]['full_parameter_l2']=sum(sq(p) for n,p in params)**.5
                den=sum(sq(self.full_before[n]) for n,p in params)
                rows[c]['full_epoch_relative_l2']=(sum(sq(p-self.full_before[n]) for n,p in params)/den)**.5 if den else None
                refs=getattr(self.models['G'],'_regularization_anchor',{})
                inherited=[(n,p) for n,p in params if n.startswith('G/') and n[2:] in refs]
                den0=sum(sq(refs[n[2:]]) for n,p in inherited)
                rows[c]['full_initial_relative_l2']=(sum(sq(p-refs[n[2:]]) for n,p in inherited)/den0)**.5 if den0 else None
                ids={id(p) for n,p in params};states=[opt.state[p] for opt in self.opts.values() for group in opt.param_groups for p in group['params'] if id(p) in ids and 'exp_avg' in opt.state.get(p,{})]
                if states:
                    rows[c]['adam_first_moment_l2']=sum(sq(s['exp_avg']) for s in states)**.5
                    rows[c]['adam_second_moment_mean']=sum(float(s['exp_avg_sq'].double().sum()) for s in states)/sum(s['exp_avg_sq'].numel() for s in states)
        bn_rows = []; squared = {}
        for (role, name, bank), (old, old_count) in self.bn.items():
            m = dict(self.models[role].named_modules())[name]
            new = torch.cat((getattr(m, bank+'running_mean'), getattr(m, bank+'running_var')))
            delta = float((new.double()-old.double()).square().sum())
            count = int(getattr(m, bank+'num_batches_tracked'))-old_count
            c = component(name+'.') if role == 'G' else role
            squared[c] = squared.get(c, 0.)+delta; rows[c]['bn_count_delta'] += count
            bn_rows.append(dict(module=role+'/'+name, bank=bank or 'target_or_shared',
                                delta_l2=delta**.5, count_delta=count))
        for c, sq in squared.items(): rows[c]['bn_delta_l2'] = sq**.5
        return dict(version=VERSION, components=rows, batchnorm=bn_rows,
                    initial_epoch=0, intervention_epoch=self.intervention['epoch'],
                    observation_seconds=self.capture_seconds+time.perf_counter()-start,
                    scope='deterministic at most 32 tensors/module and 128 elements/tensor; exact BN buffers; not full weight norms or accuracy')

def gradients(model, losses, step):
    """Full gradient norms on each module at a few executed updates, not Adam deltas."""
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    params = [p for n,p in named]; gs = {}
    keys = ('source', 'human_negative', 'human_positive', 'local_adversarial', 'global_adversarial','regularization')
    for key in keys:
        loss = losses.get(key)
        gs[key] = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True) if loss is not None and loss.requires_grad else (None,)*len(params)
    rows = {}
    for c in COMPONENTS:
        if c.startswith('D_'): continue
        inds = [i for i,(n,p) in enumerate(named) if component(n)==c]
        row = {}
        for key in keys:
            vals = [gs[key][i].detach().float() for i in inds if gs[key][i] is not None]
            row[key+'_norm'] = float(torch.stack([v.square().sum() for v in vals]).sum().sqrt()) if vals else None
        for a,b,label in [('source','human_negative','source_negative'),('source','human_positive','source_positive'),
                          ('source','local_adversarial','source_local'),('source','global_adversarial','source_global'),
                          ('local_adversarial','global_adversarial','local_global'),
                          ('source','regularization','source_regularization'),('human_negative','regularization','negative_regularization')]:
            vals = [(gs[a][i].detach().float(),gs[b][i].detach().float()) for i in inds if gs[a][i] is not None and gs[b][i] is not None]
            if vals:
                dot,aa,bb = (float(x) for x in torch.stack([torch.stack(((x*y).sum(),x.square().sum(),y.square().sum())) for x,y in vals]).sum(0))
                row[label+'_cosine'] = dot/(aa*bb)**.5 if aa and bb else None
            else: row[label+'_cosine'] = None
        for key,label in zip(keys, ('source','negative','positive','local','global','regularization')):
            row[label+'_tensors'] = sum(gs[key][i] is not None for i in inds)
        rows[c] = row
    return {'step':step, 'components':rows, 'scope':'weighted loss-component raw gradients before optional projection; not actual optimizer updates or causal effects'}

def human_terms(logits, labels, images, weight, positive_scale):
    valid = torch.isfinite(images).all(1) & (images != 0).any(1)
    neg = (labels == 0)&valid; pos = (labels == 2)&valid
    z = torch.logsumexp(logits,1)
    return {'human_negative': weight*(z-torch.logsumexp(logits[:,[0,1,3]],1))[neg].mean() if weight and neg.any() else None,
            'human_positive': weight*positive_scale*(z-logits[:,2])[pos].mean() if weight and positive_scale and pos.any() else None}
