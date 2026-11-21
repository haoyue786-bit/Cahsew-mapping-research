"""Matched, discardable initialization trials; retain the formal starting state."""
import copy
import csv
import random
import time
from pathlib import Path
import numpy as np
import torch

VERSION='matched_initialization_trials_v1'


def run(t,gan,common,model,local_d,global_d,dataset,source_names,source_valid_names,
        target_names,config,job,device,formal_optimizers):
    policy=job.get('initialization_trial_policy')
    if not policy:return None
    if policy.get('version')!=VERSION:raise ValueError('unsupported initialization trial policy')
    steps=int(policy.get('steps_per_arm',32))
    if steps not in {16,32,64}:raise ValueError('initialization trial budget outside supported bounded scope')
    if any(opt.state for opt in formal_optimizers):raise ValueError('initialization trials require unstepped formal optimizers')
    networks=(model,local_d,global_d);modes=[m.training for m in networks]
    states=[{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for m in networks]
    digests=[t.model_state_digest(m) for m in networks]
    py=random.getstate();np_state=np.random.get_state();cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else None
    panel_path=Path(job['human_samples_evaluation']['relevance_report']).with_name('feature_probe_locations.csv')
    with panel_path.open(encoding='utf-8-sig') as f:panel=list(csv.DictReader(f))
    configs=[('reference',copy.deepcopy(config),None),
             ('weaker_adversarial',copy.deepcopy(config),'lambda_adv'),
             ('smaller_model_steps',copy.deepcopy(config),'main_lr')]
    configs[1][1]['lambda_adv']=max(t.PARAM_BOUNDS['lambda_adv'][0],config['lambda_adv']*.5)
    configs[2][1]['main_lr']=max(t.PARAM_BOUNDS['main_lr'][0],config['main_lr']*.5)
    results=[]
    def restore():
        for m,state in zip(networks,states):
            m.load_state_dict(state,strict=True)
            for p in m.parameters():p.grad=None
        random.setstate(py);np.random.set_state(np_state);torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
    try:
        for name,c,changed in configs:
            restore();started=time.monotonic()
            opts=[torch.optim.AdamW(t.model_parameter_groups(model,c['main_lr'],c['main_weight_decay'],(.9,.999),1e-8)),
                  torch.optim.AdamW(local_d.parameters(),lr=c['D_lr'],weight_decay=c['D_weight_decay']),
                  torch.optim.AdamW(global_d.parameters(),lr=c['D_lr'],weight_decay=c['D_weight_decay'])]
            metrics=t.train_epoch(gan,common,model,local_d,global_d,*opts,dataset,len(source_names),len(target_names),[],None,None,c,device,
                                  np.random.RandomState(271828),steps_override=steps)
            model.eval();source=common.old_diag.evaluate_source(model,list(source_valid_names))
            hist=np.zeros(4,dtype=np.int64);soft=np.zeros(4,dtype=np.float64);valid_pixels=0
            with torch.no_grad():
                for start in range(0,len(panel),16):
                    x=np.stack([common.read_raster(Path(r['tif_path']),dtype=np.float32) for r in panel[start:start+16]])
                    tensor=torch.from_numpy(x).to(device);logits,_=gan.forward_fuse(model,tensor.float())
                    valid=torch.isfinite(tensor).all(1)&(tensor!=0).any(1);prediction=logits.argmax(1)
                    probs=torch.softmax(logits,1).permute(0,2,3,1)[valid]
                    hist+=torch.bincount(prediction[valid],minlength=4).cpu().numpy()
                    soft+=probs.sum(0).cpu().numpy();valid_pixels+=int(valid.sum())
            human=t.human.evaluation(gan,common,model,job,device,t.supplementary_batch)
            results.append(dict(name=name,changed_parameter=changed,
                actual_controls={k:c[k] for k in t.parameter_bounds(c)},completed_updates=metrics['steps'],
                source_validation=source,target_panel=dict(patches=len(panel),tiles=len({r['tile_id'] for r in panel}),
                    valid_pixels=valid_pixels,hard_fraction=(hist/max(1,valid_pixels)).tolist(),mean_probability=(soft/max(1,valid_pixels)).tolist(),
                    scope='small fixed spatial feature panel; unweighted diagnostic, not whole-group area estimate or target accuracy'),
                human_supervision_evaluation=human,training_metrics=metrics,elapsed_seconds=time.monotonic()-started))
            del opts
    finally:
        restore()
        for m,mode in zip(networks,modes):m.train(mode)
    if any(t.model_state_digest(m)!=digest for m,digest in zip(networks,digests)) or any(opt.state for opt in formal_optimizers):
        raise RuntimeError('discardable initialization trials contaminated formal state')
    return dict(version=VERSION,completed_arms=results,steps_per_arm=steps,total_updates=sum(r['completed_updates'] for r in results),
        formal_model_discriminators_optimizers_rng_restored=True,formal_weights_adopted=False,
        selection_owner='Agent chooses parameters from evidence; formal training begins from original source initialization',
        limitations='Short candidates probe immediate response, not long-term accuracy or an optimal starting configuration. CIV point validation not used. Current polygons are training-only; no independent heldout polygon feedback exists.')
