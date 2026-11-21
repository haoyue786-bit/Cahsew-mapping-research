"""Bounded epoch-displacement and exact BN-moment observations, no checkpoint IO.

Weight norms describe a deterministic sample, never a full-model estimate.
Observation does not run a forward pass, consume RNG or change BN/parameters.
"""
import time
import torch

from module_dynamics_contract import VERSION, COMPONENTS, FIELDS

def component(name):
    if name.startswith(('Swin.', 'norm.')):return 'backbone'
    if name.startswith('a3_spatial_refiner.'):return 'refiner'
    if name.startswith('fuse_conv.'):return 'fusion'
    if name.startswith('decoder.'):return 'decoder'
    return 'other'

def positions(total, count):
    n = min(total, count)
    return [round(i * (total - 1) / max(n - 1, 1)) for i in range(n)]

class EpochDynamics:
    @torch.no_grad()
    def __init__(self, model, optimizer, *, tensors_per_component=16, elements_per_tensor=256):
        if any(type(x) is not int or x < 1 for x in (tensors_per_component, elements_per_tensor)):
            raise ValueError('positive sample budgets required')
        start = time.perf_counter()
        self.model = model; self.optimizer = optimizer
        self.budget = {'tensors_per_component': tensors_per_component, 'elements_per_tensor': elements_per_tensor}
        self.groups = {}; self.bn = []; self.used = False
        lrs = {id(p):float(g['lr']) for g in optimizer.param_groups for p in g['params']}
        for key in COMPONENTS:
            eligible = [(name,p) for name,p in model.named_parameters()
                        if component(name)==key and p.requires_grad and p.numel() and id(p) in lrs]
            entries = []
            for i in positions(len(eligible), tensors_per_component):
                name,p = eligible[i]
                idx = torch.tensor(positions(p.numel(), elements_per_tensor),device=p.device,dtype=torch.long)
                entries.append((name,p,idx,p.detach().reshape(-1)[idx].clone()))
            rates = [lrs[id(p)] for _,p in eligible]
            self.groups[key] = {'eligible':len(eligible), 'entries':entries,
                                'lr_min':min(rates) if rates else None,'lr_max':max(rates) if rates else None}
        for name,m in model.named_modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and m.track_running_stats:
                self.bn.append((name,m,m.running_mean.detach().clone(),m.running_var.detach().clone(),int(m.num_batches_tracked)))
        self.capture_seconds = time.perf_counter()-start

    @torch.no_grad()
    def finish(self):
        if self.used:raise RuntimeError('epoch recorder already finalized')
        self.used=True; start=time.perf_counter(); rows={}; names={}; bn_rows=[]
        for key,g in self.groups.items():
            before=delta=0.; count=0
            for name,p,idx,old in g['entries']:
                now=p.detach().reshape(-1)[idx]
                if not torch.isfinite(now).all() or not torch.isfinite(old).all():raise ValueError('nonfinite parameter sample')
                before+=float(old.double().square().sum());delta+=float((now.double()-old.double()).square().sum());count+=old.numel()
            rows[key]={'sample_before_l2':before**.5 if count else None,'sample_delta_l2':delta**.5 if count else None,
                       'sample_relative_l2':(delta/before)**.5 if before else None,'sample_elements':count,
                       'sample_tensors':len(g['entries']),'eligible_tensors':g['eligible'],
                       'lr_min':g['lr_min'],'lr_max':g['lr_max'],
                       'bn_before_l2':None,'bn_delta_l2':None,'bn_relative_l2':None,
                       'bn_elements':0,'bn_modules':0,'bn_forward_count_delta':0}
            names[key]=[name for name,_,_,_ in g['entries']]
        bn_sums={key:[0.,0.] for key in COMPONENTS}
        for name,m,old_mean,old_var,old_steps in self.bn:
            key=component(name+'.'); a=torch.cat((old_mean,old_var)).double(); b=torch.cat((m.running_mean,m.running_var)).double()
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('nonfinite BN moments')
            before=float(a.square().sum());delta=float((b-a).square().sum());steps=int(m.num_batches_tracked)-old_steps
            bn_sums[key][0]+=before;bn_sums[key][1]+=delta
            rows[key]['bn_elements']+=a.numel();rows[key]['bn_modules']+=1;rows[key]['bn_forward_count_delta']+=steps
            bn_rows.append({'name':name,'component':key,'before_l2':before**.5,'delta_l2':delta**.5,
                            'relative_l2':(delta/before)**.5 if before else None,'forward_count_delta':steps})
        for key,(a,b) in bn_sums.items():
            if rows[key]['bn_modules']:
                rows[key].update(bn_before_l2=a**.5,bn_delta_l2=b**.5,bn_relative_l2=(b/a)**.5 if a else None)
        return {'version':VERSION,'scope':'same epoch start-to-end; sampled parameters, all BN running moments; not accuracy or full parameter norms',
                'sample_budget':self.budget,'sample_names':names,'components':rows,'batchnorm':bn_rows,
                'observation_seconds':self.capture_seconds+time.perf_counter()-start}
