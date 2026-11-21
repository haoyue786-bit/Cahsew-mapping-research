"""Explicit human-label adapter for this user-authorized group training cohort."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import torch
import numpy as np
from masked_cashew_polygon_loss import loss, VERSION

MODE='human_polygon_binary'
SCALE='supplementary_positive_scale'

@contextmanager
def frozen_auxiliary_bn(model):
    modules=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    states=[m.training for m in modules]
    try:
        for m in modules:m.eval()
        yield
    finally:
        for m,state in zip(modules,states):m.train(state)

def forward(gan,model,images,mode):
    if mode!=MODE:return gan.forward_fuse(model,images.float())
    with frozen_auxiliary_bn(model):return gan.forward_fuse(model,images.float())

def supervised_loss(logits,labels,images,config):
    return loss(logits,labels,images,positives=True,positive_scale=config[SCALE])

def validate(job,template,report,names):
    assert job['polygon_supervision_policy']==template['polygon_supervision_policy']
    assert job['supplementary']['mode']==MODE
    assert report['selection_rule']==template['supplementary_selection']=='human_polygon_geographic500km_v1'
    assert names==[r['name'] for r in report['selected']]
    assert len(names)==len(set(names))==job['supplementary']['count']
    assert report['old_supplementary_labels_allowed'] is False
    assert report['validation_patch_exclusion'] is True
    assert report['geographic_filter']['radius_km']==500
    assert report['fixed_top_k_used'] is False
    assert report['feature_gate_used'] is False
    for r in report['selected']:
        assert 0<=r['geo_distance_km']<=500
        assert r['negative_pixels']+r['positive_pixels']>0
        for key,hashkey in [('mask_path','mask_sha256')]:
            assert hashlib.sha256(Path(r[key]).read_bytes()).hexdigest()==r[hashkey]
    return True

def packet(job,report):
    result = dict(version=MODE,origin='Current user-drawn replacement Cashew/non-Cashew polygons',
        prior_weak_labels_used=False,feature_gate_used=False,validation_patch_exclusion=True,
        unmarked_pixels='ignored; zero supplementary logit gradient',negative_label='not Cashew, not a specific four-class alternative',
        positive_label='Cashew index2',loss=VERSION,auxiliary_batchnorm='frozen running moments only during supplementary forward; full-model gradients',
        coefficients='w_neg=supplementary_loss_weight; w_pos=w_neg*supplementary_positive_scale; scale0 is negative-only',
        activation='both signs off initially; may activate at review E50 or later; zero/one-sign pools legal',
        exposure='1..4 occurrences per annotated patch per epoch; cadence is a separate control',
        uncertainty='Human polygon labels may contain errors; geographic transfer benefit and local Cashew retention need assessment.',
        counts=dict(unique_patches=len(report['selected']),negative_patches=sum(r['negative_pixels']>0 for r in report['selected']),positive_patches=sum(r['positive_pixels']>0 for r in report['selected']),sign_counts_overlap='a patch can contain both signs; their counts need not sum to unique patches'))
    if job.get('human_samples_policy',{}).get('version')=='human_samples_agent_timing_v1':
        result['activation']='initially off; Agent can activate at any trained review; no fixed50-epoch boundary; source-only collapse recovery disables both signs'
        result['uncertainty'] += ' Feature similarity is soft relevance evidence, not a label or a gate; preserve novel labelled types.'
        evidence=job['human_samples_evaluation']
        p=Path(evidence['relevance_report'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=evidence['relevance_sha256']:raise ValueError('Human relevance report identity changed')
        result['relevance']=json.loads(p.read_text())
        result['location_holdout']=report['location_holdout']
    return result


def binary_counts(labels, prediction, probability, valid):
    negative=(labels==0)&valid;positive=(labels==2)&valid
    n=int(negative.sum());p=int(positive.sum())
    fp=int((negative&prediction).sum());tp=int((positive&prediction).sum())
    return dict(negative_pixels=n,positive_pixels=p,false_positive_pixels=fp,false_negative_pixels=p-tp,
                true_positive_pixels=tp,negative_p2_sum=float(probability[negative].sum()),positive_p2_sum=float(probability[positive].sum()))


def summarize_counts(counts):
    c=dict(counts);n=c['negative_pixels'];p=c['positive_pixels'];tp=c['true_positive_pixels'];fp=c['false_positive_pixels'];fn=c['false_negative_pixels']
    c.update(false_positive_rate=fp/n if n else None,cashew_recall=tp/p if p else None,
             cashew_precision=tp/(tp+fp) if p and n and tp+fp else None,
             cashew_f1=2*tp/(2*tp+fp+fn) if p and n and 2*tp+fp+fn else None,
             negative_mean_p2=c['negative_p2_sum']/n if n else None,
             positive_mean_p2=c['positive_p2_sum']/p if p else None)
    return c


def evaluation(gan,common,model,job,device,load_batch):
    """Current polygons only; held-out locations never enter auxiliary loss."""
    spec=job.get('human_samples_evaluation')
    if not spec:return None
    report_path=Path(spec['heldout_report'])
    if hashlib.sha256(report_path.read_bytes()).hexdigest()!=spec['heldout_sha256']:raise ValueError('Held-out polygon report changed')
    held=json.loads(report_path.read_text())['selected']
    training=json.loads(Path(job['supplementary']['selection']['report']).read_text())['selected']
    relevance=json.loads(Path(spec['relevance_report']).read_text())
    order=[];seen=set()
    lookup={r['name']:r for r in training}
    # A bounded difficulty panel; explicitly trained labels, not validation.
    for row in relevance['top_relevance_examples']:
        if row['name'] in lookup and row['name'] not in seen:
            order.append(lookup[row['name']]);seen.add(row['name'])
    result={};was_training=model.training;model.eval()
    try:
        with torch.no_grad():
            for role,rows in [('heldout_locations',held),('training_difficulty_only',order)]:
                names=[r['name'] for r in rows];locations={}
                for start in range(0,len(rows),16):
                    indices=list(range(start,min(start+16,len(rows))))
                    images,labels=load_batch(common,names,indices,Path(job['supplementary']['image_root']),Path(job['supplementary']['label_root']),device,augment=False)
                    logits,_=gan.forward_fuse(model,images.float())
                    if logits.shape[-2:]!=labels.shape[-2:]:
                        logits=torch.nn.functional.interpolate(logits,size=labels.shape[-2:],mode='bilinear',align_corners=False)
                    prob=torch.softmax(logits,dim=1)[:,2];prediction=logits.argmax(1)==2
                    valid=torch.isfinite(images).all(1)&(images!=0).any(1)
                    for k,index in enumerate(indices):
                        c=binary_counts(labels[k],prediction[k],prob[k],valid[k]);key=rows[index]['location_id']
                        if key not in locations:locations[key]={name:0 for name in c}
                        for name,value in c.items():locations[key][name]+=value
                zero=dict(negative_pixels=0,positive_pixels=0,false_positive_pixels=0,false_negative_pixels=0,true_positive_pixels=0,negative_p2_sum=0.,positive_p2_sum=0.)
                for c in locations.values():
                    for name,value in c.items():zero[name]+=value
                result[role]=dict(patch_count=len(rows),location_count=len(locations),
                    aggregate=summarize_counts(zero),by_location={k:summarize_counts(v) for k,v in locations.items()})
    finally:model.train(was_training)
    result.update(version='human_samples_agent_timing_v1',decoding='four_class_argmax_equals2',
        ignored_pixels='unmarked10, nonfinite/all-zero imagery',CIV_point_labels_used=False,
        role='supplementary TRAINING fit only; no independent holdout exists; cannot demonstrate generalization',
        limitation='Correlated annotated pixels and nearby-region support do not establish group-wide accuracy; inspect locations and sign support.')
    return result
