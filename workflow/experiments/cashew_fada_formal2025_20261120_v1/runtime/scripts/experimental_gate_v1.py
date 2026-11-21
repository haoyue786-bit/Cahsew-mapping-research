"""Opt-in candidate only. Never reads point validation files."""
import hashlib
import json
import math
from pathlib import Path
import torch
from statistics import median

VERSION = 'experimental_gate_v1'
FIELDS = ('cls', 'adv', 'local_D_acc', 'global_D_acc', 'effective_gradient_ratio')

def reconcile_committed_skip(root, load_checkpoint):
    """Clear only a fully committed local skip; all other uncertain states block.

    Full/Agent reviews and incomplete transactions require separate exact
    reconciliation; no model call or parameter action occurs here.
    """
    root=Path(root).resolve(); pending=root/'experimental_gate_pending.json'
    if not pending.exists():return 'no_pending'
    try:
        item=json.loads(pending.read_text());epoch=item['epoch']
        if type(epoch) is not int or epoch<1 or item.get('version')!=VERSION or item.get('state')!='epoch_uncommitted':raise ValueError('invalid pending marker')
        light=root/'light_checkpoints';gate=root/'experimental_gate'
        receipt=json.loads((light/('epoch_%03d_committed.json'%epoch)).read_text())
        if receipt.get('epoch')!=epoch or receipt.get('review_mode')!='experimental_machine_gate' or receipt.get('normal_stop_allowed') is not False:raise ValueError('not a machine-skip receipt')
        checkpoint=(light/('epoch_%03d_post.pt'%epoch)).resolve()
        decision=(gate/('epoch_%03d_skip.json'%epoch)).resolve()
        if checkpoint.parent!=light.resolve() or decision.parent!=gate.resolve():raise ValueError('unsafe artifact location')
        if Path(receipt['checkpoint']).resolve()!=checkpoint or Path(receipt['decision']).resolve()!=decision:raise ValueError('receipt paths mismatch')
        def digest(path):
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            return h.hexdigest()
        if digest(checkpoint)!=receipt['checkpoint_sha256'] or digest(decision)!=receipt['decision_sha256']:raise ValueError('artifact hash mismatch')
        d=json.loads(decision.read_text())
        if d.get('epoch')!=epoch or d.get('decision_kind')!='experimental_machine_gate' or d.get('action')!='KEEP' or d.get('continue_training') is not True:raise ValueError('invalid skip decision')
        payload=load_checkpoint(checkpoint)
        if not isinstance(payload,dict) or payload.get('epoch')!=epoch:raise ValueError('checkpoint epoch mismatch')
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError('experimental transaction requires exact manual reconciliation') from exc
    pending.unlink()
    return 'committed_skip_verified'

def assert_finite_gradients(*optimizers):
    """Check gradients and AdamW group state before any paired optimizers step."""
    flags = []
    for optimizer in optimizers:
        connected = False
        for group in optimizer.param_groups:
            for key in ('lr','weight_decay','eps'):
                value=group.get(key)
                if type(value) not in (int,float) or not math.isfinite(value) or value<0 or (key=='eps' and value==0):
                    raise RuntimeError('invalid optimizer '+key+': update prevented')
            betas=group.get('betas')
            if not isinstance(betas,(tuple,list)) or len(betas)!=2 or any(type(b) not in (int,float) or not math.isfinite(b) or not 0<=b<1 for b in betas):
                raise RuntimeError('invalid optimizer betas: update prevented')
            for p in group['params']:
                if p.grad is not None:
                    connected = True
                    grad = p.grad.coalesce().values() if p.grad.is_sparse else p.grad
                    flags.append(torch.isfinite(grad).all())
        if not connected:
            raise RuntimeError('optimizer has no connected gradients: update prevented')
    if flags and not bool(torch.stack(flags).all().item()):
        raise RuntimeError('nonfinite gradient: optimizer update prevented')

def supplementary_context(metrics):
    """Stable actual exposure is not by itself a reason for an Agent call."""
    exposure = metrics.get('supplementary_actual_exposure')
    if not isinstance(exposure,dict):
        return {'supp_exposure_missing':False, 'supp_exposure_unknown':True}, True
    keys=('attempted_occurrences','loaded_occurrences','completed_occurrences')
    values=[exposure.get(k) for k in keys]
    if any(type(v) is not int or v<0 for v in values):
        return {'supp_exposure_missing':False, 'supp_exposure_unknown':True}, True
    missing = not (values[0]==values[1]==values[2])
    return {'supp_exposure_missing':missing, 'supp_exposure_unknown':False}, missing

def supplementary_integrity(exposure):
    """Check completed worker counters; zero intended exposure is legitimate."""
    keys=('planned_steps','planned_occurrences','planned_unique_patches','attempted_steps','attempted_occurrences','attempted_unique_patches',
          'loaded_steps','loaded_occurrences','loaded_unique_patches','forward_steps','loss_steps',
          'completed_optimizer_steps','completed_occurrences','completed_unique_patches','human_negative_effective_steps',
          'human_positive_effective_steps','human_negative_effective_pixel_occurrences','human_positive_effective_pixel_occurrences',
          'human_negative_effective_unique_patches','human_positive_effective_unique_patches','unknown_sign_completed_steps')
    if not isinstance(exposure,dict) or exposure.get('version')!='supplementary_actual_exposure_v1' or any(k not in exposure for k in keys):
        return {'hard':[],'missing':['supplementary_actual_exposure']}
    if any(type(exposure[k]) is not int or exposure[k]<0 for k in keys):
        return {'hard':['invalid_supplementary_exposure'],'missing':[]}
    x=exposure;bad=[]
    if not x['planned_occurrences']>=x['attempted_occurrences']==x['loaded_occurrences']==x['completed_occurrences']:
        bad.append('inconsistent_supplementary_occurrences')
    if not x['planned_steps']>=x['attempted_steps']==x['loaded_steps']==x['forward_steps']==x['loss_steps']==x['completed_optimizer_steps']:
        bad.append('inconsistent_supplementary_steps')
    for stage in ('planned','attempted','loaded','completed'):
        if x[stage+'_unique_patches']>x[stage+'_occurrences']:bad.append('inconsistent_supplementary_unique')
    if not x['planned_unique_patches']>=x['attempted_unique_patches']==x['loaded_unique_patches']==x['completed_unique_patches']:
        bad.append('inconsistent_supplementary_unique')
    for sign in ('negative','positive'):
        if x['human_'+sign+'_effective_unique_patches']>x['completed_unique_patches'] or x['human_'+sign+'_effective_steps']>x['completed_optimizer_steps']:
            bad.append('inconsistent_human_sign_exposure')
    if x['unknown_sign_completed_steps']>x['completed_optimizer_steps']:bad.append('inconsistent_human_sign_exposure')
    return {'hard':sorted(set(bad)),'missing':[]}

def epoch_integrity(metrics, config, scheduled):
    """Numerical and executed-vs-planned checks, respecting paused D state.

    A missing observation triggers review (unknown), while corruption or an
    impossible execution contract pauses training. No target evidence used.
    """
    hard=[];missing=[]
    if not isinstance(metrics,dict) or not isinstance(config,dict) or not isinstance(scheduled,dict):
        return {'route':'LIGHT_REVIEW','hard':[],'missing':['epoch_state']}
    supp=supplementary_integrity(metrics.get('supplementary_actual_exposure'));hard.extend(supp['hard']);missing.extend(supp['missing'])
    state=config.get('adversarial_state')
    if state not in ('active','paused'):missing.append('adversarial_state')
    steps=metrics.get('steps')
    if steps is None:missing.append('steps')
    elif type(steps) is not int or steps<1:hard.append('invalid_steps')
    # Adversarial zeros in a paused branch are placeholders, not measurements.
    keys=['cls']+(['cls_grad','adv','adv_grad','D_loss','local_D_acc','global_D_acc','effective_gradient_ratio'] if state=='active' else [])
    for key in keys:
        value=metrics.get(key)
        if value is None:missing.append(key);continue
        if type(value) not in (int,float) or not math.isfinite(value):hard.append('nonfinite_or_invalid_'+key);continue
        if value<0 or (key in ('local_D_acc','global_D_acc') and value>1):hard.append('invalid_range_'+key)
    updates=metrics.get('experimental_actual_updates')
    if not isinstance(updates,dict):missing.append('actual_updates');updates={}
    if updates.get('finite') is False:hard.append('nonfinite_sampled_update')
    elif updates.get('finite') is not True:missing.append('sampled_update_finiteness')
    actual=updates.get('steps');attempted=updates.get('attempted_steps');with_grad=updates.get('steps_with_gradient');groups=updates.get('optimizer_groups')
    if not isinstance(actual,dict):missing.append('actual_step_counts');actual={}
    if not isinstance(attempted,dict):missing.append('attempted_step_counts');attempted={}
    if not isinstance(with_grad,dict):missing.append('gradient_connected_step_counts');with_grad={}
    if not isinstance(groups,dict):missing.append('actual_optimizer_groups');groups={}
    for name in ('G','D_local','D_global'):
        count=actual.get(name)
        if count is None:missing.append(name+'_steps');continue
        if type(count) is not int or count<0:hard.append('invalid_'+name+'_steps');continue
        if name not in attempted or name not in with_grad:missing.append(name+'_gradient_step_evidence')
        elif any(type(v) is not int or v<0 for v in (attempted[name],with_grad[name])) or attempted[name]!=count or with_grad[name]!=count:hard.append('mismatched_'+name+'_gradient_steps')
        if type(steps) is int and steps>0:
            if name=='G':expected=steps
            elif state=='paused':expected=0
            elif state=='active' and type(config.get('D_steps_per_G_step')) is int and config['D_steps_per_G_step']>0:expected=steps*config['D_steps_per_G_step']
            else:expected=None
            if expected is not None and count!=expected:hard.append('mismatched_'+name+'_steps')
        if count==0:continue
        observed=groups.get(name)
        if not isinstance(observed,list) or not observed:missing.append(name+'_optimizer_groups');continue
        for i,group in enumerate(observed):
            if not isinstance(group,dict):hard.append('invalid_'+name+'_group');continue
            expected_lr=scheduled.get('main_lr' if name=='G' else 'D_lr')
            expected_decay=config.get('main_weight_decay' if name=='G' else 'D_weight_decay')
            mult=group.get('lr_multiplier')
            if expected_lr is None or expected_decay is None or mult is None:missing.append(name+'_expected_controls');continue
            vals=(group.get('lr'),group.get('weight_decay'),group.get('eps'),mult,expected_lr,expected_decay)
            if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in vals):hard.append('invalid_'+name+'_group_values');continue
            if not math.isclose(group['lr'],expected_lr*mult,rel_tol=1e-6,abs_tol=1e-15) or not math.isclose(group['weight_decay'],expected_decay,rel_tol=1e-6,abs_tol=1e-15):hard.append('mismatched_'+name+'_optimizer_group_'+str(i))
            betas=group.get('betas')
            if not isinstance(betas,(list,tuple)) or len(betas)!=2 or any(type(b) not in (int,float) or not math.isfinite(b) or not 0<=b<1 for b in betas) or group['eps']<=0:hard.append('invalid_'+name+'_optimizer_hyperparameters')
    return {'route':'SAFETY_STOP' if hard else 'LIGHT_REVIEW' if missing else 'OK','hard':sorted(set(hard)),'missing':sorted(set(missing))}


def validate(job):
    p = job.get('experimental_gate')
    if p is None:
        return None
    if p.get('version') != VERSION or p.get('experimental_only') is not True:
        raise ValueError('experimental gate must be explicitly declared')
    if job.get('review_interval') != 1 or job.get('deep_review_interval') != 5 or job.get('two_tier_review_policy') != 'light_per_epoch_deep_every_five_v1':
        raise ValueError('experimental gate requires per-epoch checks and five-epoch full review')
    if p.get('mode') not in ('shadow', 'enforce'):
        raise ValueError('mode must be shadow or enforce')
    if type(p.get('lookback')) is not int or p['lookback'] < 4:
        raise ValueError('lookback must be >= 4')
    sampling=p.get('recorder_sampling')
    if sampling is not None:
        if not isinstance(sampling,dict) or set(sampling)!={'version','max_tensors_per_group','elements_per_tensor'} or sampling['version']!='bounded_v1':
            raise ValueError('invalid bounded recorder policy')
        if type(sampling['max_tensors_per_group']) is not int or not 1<=sampling['max_tensors_per_group']<=1024 or type(sampling['elements_per_tensor']) is not int or not 1<=sampling['elements_per_tensor']<=1024:
            raise ValueError('invalid bounded recorder budget')
    for name in ('relative_limits', 'absolute_scales'):
        limits = p.get(name)
        if not isinstance(limits, dict) or set(limits) != set(FIELDS):
            raise ValueError('explicit thresholds required for all training fields')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in limits.values()):
            raise ValueError('invalid threshold')
    return p


def fingerprint(config, scheduled):
    """Hash full config + scheduled controls. Unknown/unserializable means review."""
    try:
        payload = json.dumps([config, scheduled], sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()
    except (TypeError, ValueError):
        return None


def automatic_context(controller_present, pool_present, config, metrics):
    """Separate stable exposure from generation/weight/expiry transitions.

    Expiration and unknown metadata fail closed. Cursor advancement itself is
    expected and does not make an otherwise identical control look like a new one.
    """
    if not controller_present:
        summary=metrics.get('automatic_negative')
        if summary is not None:
            if not isinstance(summary,dict) or any(type(summary.get(k)) not in (int,float) or not math.isfinite(summary[k]) for k in ('updates','effective_weight')):return None,True
            if summary['updates']!=0 or summary['effective_weight']!=0:return None,True
        return {'mode':'not_configured'}, False
    state = config.get('_automatic_negative_state')
    if state is None and not pool_present:
        summary=metrics.get('automatic_negative')
        if summary is not None:
            if not isinstance(summary,dict) or any(type(summary.get(k)) not in (int,float) or not math.isfinite(summary[k]) for k in ('updates','effective_weight')):return None,True
            if summary['updates']!=0 or summary['effective_weight']!=0:return None,True
        return {'mode':'no_pool'}, False
    summary = metrics.get('automatic_negative')
    required_state = ('manifest','manifest_sha256','weight','generation_epoch','cursor')
    required_summary = ('configured_weight','effective_weight','updates','generation_start_cursor','generation_end_cursor','generation_horizon_updates')
    if not isinstance(state, dict) or not pool_present or not isinstance(summary, dict) or any(k not in state for k in required_state) or any(k not in summary for k in required_summary):
        return None, True
    if any(type(summary[k]) not in (int,float) or not math.isfinite(summary[k]) for k in required_summary):
        return None, True
    if type(state['weight']) not in (int,float) or not math.isfinite(state['weight']) or state['weight'] < 0:
        return None, True
    if any(type(state[k]) is not int or state[k] < 0 for k in ('generation_epoch','cursor')):
        return None, True
    if any(type(summary[k]) is not int or summary[k] < 0 for k in ('updates','generation_start_cursor','generation_end_cursor','generation_horizon_updates')):
        return None, True
    if summary['generation_horizon_updates'] == 0 or summary['effective_weight'] < 0 or summary['updates'] > summary['generation_end_cursor']-summary['generation_start_cursor']:
        return None, True
    try:
        identity = state['manifest_sha256']
        if not isinstance(identity,str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
            return None, True
        if hashlib.sha256(Path(state['manifest']).read_bytes()).hexdigest() != identity:
            return None, True
    except (OSError, TypeError, ValueError):
        return None, True
    end, horizon = summary['generation_end_cursor'], summary['generation_horizon_updates']
    if not (0 <= summary['generation_start_cursor'] <= end <= horizon) or state['cursor'] != end or state['weight'] != summary['configured_weight']:
        return None, True
    # Natural pool exhaustion is a planned exposure event, not an error;
    # retain it in evidence for the next scheduled full review.
    transition = False
    # Exposure count and effective per-epoch dose remain in metrics, but do not
    # identify the pool configuration: ordinary workload fluctuations must not
    # turn an unchanged pool into a different fingerprint on every epoch.
    return {'mode':'pool', 'manifest_sha256':state['manifest_sha256'], 'weight':state['weight'],
            'generation_epoch':state['generation_epoch'], 'expired':end >= horizon}, transition


def stable_config(config):
    """Exclude only monotonically changing bookkeeping, preserve pool identity."""
    result = dict(config)
    state = result.get('_automatic_negative_state')
    if isinstance(state, dict):
        result['_automatic_negative_state'] = {k:v for k,v in state.items() if k not in ('cursor','last_epoch_exposure')}
    return result


def classify(epoch, metrics, config_hash, history, policy, *, active_aux, pending_action, adversarial_active=True):
    if type(adversarial_active) is not bool:return {'route':'LIGHT_REVIEW','reason':'adversarial branch state unknown'}
    fields=FIELDS if adversarial_active else ('cls',)
    facts = {k: metrics.get(k) for k in fields}
    if any(type(v) in (int, float) and not math.isfinite(v) for v in facts.values()):
        return {'route': 'SAFETY_STOP', 'reason': 'nonfinite training metric'}
    if epoch % 5 == 0:
        return {'route': 'FULL_REVIEW', 'reason': 'five epoch schedule'}
    if any(type(v) not in (int, float) for v in facts.values()):
        return {'route': 'LIGHT_REVIEW', 'reason': 'missing training metric'}
    if (any(facts[k] < 0 for k in fields if k in ('cls','adv','effective_gradient_ratio')) or
        any(not 0 <= facts[k] <= 1 for k in fields if k in ('local_D_acc','global_D_acc'))):
        return {'route':'LIGHT_REVIEW','reason':'training metric outside physical range'}
    if active_aux is not False or pending_action is not False or config_hash is None:
        return {'route': 'LIGHT_REVIEW', 'reason': 'auxiliary supervision, action response, or config state unknown/active'}
    past = history[-policy['lookback']:]
    if len(past) != policy['lookback'] or [x.get('epoch') for x in past] != list(range(epoch-len(past), epoch)):
        if epoch <= policy['lookback'] and [x.get('epoch') for x in past] == list(range(1,epoch)):
            return {'route': 'SKIP', 'reason': 'initial baseline not yet available; record unknown'}
        return {'route': 'LIGHT_REVIEW', 'reason': 'unexpected missing or nonconsecutive history'}
    changed=any(x.get('config_hash') != config_hash or x.get('stable') is not True for x in past)
    for k in fields:
        values = [x['facts'].get(k) for x in past]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            return {'route': 'LIGHT_REVIEW', 'reason': 'invalid prior facts'}
        center = median(values)
        if abs(facts[k]-center) / max(abs(center), policy['absolute_scales'][k]) > policy['relative_limits'][k]:
            return {'route': 'LIGHT_REVIEW', 'reason': ('deviation across changed controls in ' if changed else 'deviation in ')+k}
    if changed:
        return {'route': 'SKIP', 'reason': 'changed controls or phase; comparison is approximate, not proof of health'}
    return {'route': 'SKIP', 'reason': 'no configured alarm; not evidence of target accuracy'}


class StepRecorder:
    """Fixed deterministic element samples, one device-to-host transfer per epoch."""
    def __init__(self, count=8):
        self.count = count
        self.counts = {'G': 0, 'D_local': 0, 'D_global': 0}
        self.attempted_counts = {'G': 0, 'D_local': 0, 'D_global': 0}
        self.with_grad_counts = {'G': 0, 'D_local': 0, 'D_global': 0}
        self.groups = {}
        self.samples = []
        self.logit_samples = []

    def observe_logits(self, source_logits, target_logits, config, step):
        """Recompute class knowledge only on fixed existing logits; no labels/gradients."""
        if config.get('local_mode') != 'soft':
            return
        temperature, cap = float(config['temperature']), float(config['confidence_cap'])
        if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(cap) or not 0 < cap <= 1:
            raise ValueError('invalid knowledge temperature or cap')
        with torch.no_grad():
            for domain, logits in (('source',source_logits),('target_unlabeled',target_logits)):
                flat=logits.detach().permute(0,2,3,1).reshape(-1,logits.shape[1]).float()
                n=min(256,flat.shape[0])
                if n == 0: continue
                idx=torch.linspace(0,flat.shape[0]-1,n,device=flat.device).round().long()
                x=flat[idx]; base=torch.softmax(x/temperature,dim=1); clipped=base.clamp(max=cap)
                # These are hypothetical local probes, not proposed decisions.
                # Keep them in this runtime's advertised base parameter bounds.
                tlo=torch.softmax(x/max(1.2,temperature*.9),dim=1).clamp(max=cap)
                thi=torch.softmax(x/min(3.0,temperature*1.1),dim=1).clamp(max=cap)
                clo=base.clamp(max=max(.8,cap*.9)); chi=base.clamp(max=min(.95,cap*1.1))
                row=torch.stack(((base>cap).any(dim=1).float().mean(), (base>cap).float().mean(),
                    (tlo-clipped).abs().mean(), (thi-clipped).abs().mean(),
                    ((tlo-clipped).abs()>1e-7).any(dim=1).float().mean(),
                    ((thi-clipped).abs()>1e-7).any(dim=1).float().mean(),
                    (clo-clipped).abs().mean(),(chi-clipped).abs().mean(),
                    ((clo-clipped).abs()>1e-7).any(dim=1).float().mean(),
                    ((chi-clipped).abs()>1e-7).any(dim=1).float().mean()))
                self.logit_samples.append((domain,int(step),n,temperature,cap,row))

    def step(self, name, optimizer, sample):
        self.attempted_counts[name] += 1
        connected=any(p.grad is not None for group in optimizer.param_groups for p in group['params'])
        if not connected:raise RuntimeError('optimizer has no connected gradients: update prevented')
        self.groups[name] = [{'lr':float(g['lr']), 'lr_multiplier':float(g.get('lr_multiplier', 1.0)), 'weight_decay':float(g.get('weight_decay', 0)),
                              'betas':list(g.get('betas', ())), 'eps':g.get('eps')} for g in optimizer.param_groups]
        before = []
        if sample:
            with torch.no_grad():
                for gi, group in enumerate(optimizer.param_groups):
                    for p in group['params']:
                        if p.grad is None or not p.numel():
                            continue
                        n = min(self.count, p.numel())
                        idx = torch.linspace(0, p.numel()-1, n, device=p.device).round().long()
                        before.append((gi, p, idx, p.detach().reshape(-1)[idx].clone()))
        optimizer.step()
        self.counts[name] += 1
        self.with_grad_counts[name] += 1
        if sample:
            with torch.no_grad():
                for gi, p, idx, b in before:
                    a = p.detach().reshape(-1)[idx]
                    d = (a-b).float()
                    finite = torch.isfinite(a).all() & torch.isfinite(b).all() & torch.isfinite(d).all()
                    self.samples.append((name, gi, torch.stack((d.square().sum(), b.float().square().sum(), finite.float(), d.new_tensor(float(idx.numel()))))))

    def finish(self):
        # Reduce all tiny tensors on-device; exactly one transfer for all samples.
        if not self.samples:
            return {'steps':self.counts, 'attempted_steps':self.attempted_counts, 'steps_with_gradient':self.with_grad_counts, 'optimizer_groups':self.groups, 'sample_groups':[],
                    'finite':None, 'logit_response':self.finish_logits(),
                    'meaning':'no sampled optimizer updates; missing is not a zero update'}
        devices = {x[2].device for x in self.samples}
        if len(devices) != 1:
            raise RuntimeError('unexpected multi-device optimizers')
        keys = sorted({(name, gi) for name, gi, _ in self.samples})
        rows = []
        for key in keys:
            x = torch.stack([v for name, gi, v in self.samples if (name, gi) == key])
            rows.append(torch.stack((x[:,0].sum(), x[:,1].sum(), x[:,2].min(), x[:,3].sum())))
        values = torch.stack(rows).cpu().tolist()
        groups = [{'optimizer':k[0], 'group':k[1], 'sample_delta_l2':math.sqrt(max(v[0],0)),
                   'sample_before_l2':math.sqrt(max(v[1],0)), 'finite':bool(v[2]), 'sample_count':int(v[3])}
                  for k, v in zip(keys, values)]
        logit_response = self.finish_logits()
        return {'steps':self.counts, 'attempted_steps':self.attempted_counts, 'steps_with_gradient':self.with_grad_counts, 'optimizer_groups':self.groups, 'sample_groups':groups,
                'finite':all(v['finite'] for v in groups),
                'logit_response':logit_response,
                'meaning':'deterministic sampled elements, not a full update norm or component attribution'}

    def finish_logits(self):
        if not self.logit_samples:
            return {'available':False,'reason':'no sampled discriminator soft-mode logits'}
        keys=('current_cap_pixel_fraction','current_cap_element_fraction',
              'temperature_down_mean_abs_change','temperature_up_mean_abs_change',
              'temperature_down_pixel_fraction_changed','temperature_up_pixel_fraction_changed',
              'cap_down_mean_abs_change','cap_up_mean_abs_change',
              'cap_down_pixel_fraction_changed','cap_up_pixel_fraction_changed')
        bydomain={}
        for domain in ('source','target_unlabeled'):
            selected=[x for x in self.logit_samples if x[0] == domain]
            if not selected: continue
            n=sum(x[2] for x in selected)
            weighted=torch.stack([x[5]*x[2] for x in selected]).sum(dim=0)/n
            values=weighted.cpu().tolist()
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError('nonfinite sampled logit response: safe pause')
            bydomain[domain]={'sampled_steps':[x[1] for x in selected],'sampled_pixels':n,
                              **dict(zip(keys,values))}
        first=self.logit_samples[0]
        return {'available':True,'temperature':first[3],'confidence_cap':first[4],
                'temperature_alternatives':[max(1.2,first[3]*.9),min(3.0,first[3]*1.1)],
                'confidence_cap_alternatives':[max(.8,first[4]*.9),min(.95,first[4]*1.1)],
                'domains':bydomain,'interpretation':'same observed logits only; hypothetical local knowledge response, not future training or accuracy'}
