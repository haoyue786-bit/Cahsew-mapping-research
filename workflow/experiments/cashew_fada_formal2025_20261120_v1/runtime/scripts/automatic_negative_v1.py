"""Independent experimental pseudo-negatives; never a human-label adapter.

NumPy candidate logic is CPU-only. Torch is imported only by the auxiliary loss.
No geography-only labels, absent-prior zeros, source-weight changes or projection.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np

VERSION = 'priorv3_source_feature_auto_negative_experiment_v1'

@dataclass(frozen=True)
class CandidatePolicy:
    maximum_prior_support: float = .35
    minimum_evidence_confidence: float = .35
    maximum_conflict_fraction: float = .25
    minimum_model_p2: float = .80
    patch_cashew_alert_fraction: float = .15
    maximum_valid_pixel_fraction: float = .02
    maximum_pixels_per_patch: int = 256
    maximum_pool_patches: int = 64
    maximum_age_epochs: int = 5
    initial_weight: float = 0.
    maximum_weight: float = .05
    pilot_weight: float = .02
    occurrence_cap: int = 1
    interval_steps: int = 4

def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

def feature_evidence(features, prototypes, chunk_size=2048):
    """Compare raw normalized 64-band source-labelled feature prototypes.

    Absolute non-Cashew similarity prevents novelty alone from assigning labels.
    The margin compares strongest non-Cashew and strongest Cashew support.
    """
    x = normalize(features)
    proto = np.asarray(prototypes, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 64 or proto.ndim != 3 or proto.shape[0] != 4 or proto.shape[2] != 64:
        raise ValueError('Expected N x64 features and4 xK x64 prototypes')
    proto = normalize(proto)
    scores=[]
    for i in range(0,len(x),chunk_size):
        scores.append(np.max((x[i:i+chunk_size] @ proto.reshape(-1,64).T).reshape(-1,4,proto.shape[1]),axis=2))
    classes=np.concatenate(scores) if scores else np.empty((0,4),np.float32)
    negative=classes[:,[0,1,3]].max(1)
    return dict(noncashew_cosine=negative,cashew_cosine=classes[:,2],margin=negative-classes[:,2])

def candidate_mask(images, probabilities, prior, prototypes, feature_thresholds,
                   forbidden_pixels=None, patch_forbidden=False, policy=CandidatePolicy()):
    """Return0 only for individually supported negative candidates, else ignore10.

    Probabilities must be the frozen current-state four-class softmax. Prior
    summaries and anomaly fractions rank a test; they impose no crop quota.
    """
    x=np.asarray(images,dtype=np.float32); p=np.asarray(probabilities,dtype=np.float32)
    if x.ndim!=3 or x.shape[0]!=64 or p.shape!=(4,)+x.shape[1:]:
        raise ValueError('Unaligned64-band image/four-class probabilities')
    labels=np.full(x.shape[1:],10,np.uint8)
    report=dict(version=VERSION,policy=asdict(policy),selected_pixels=0,reason=None)
    if patch_forbidden: report['reason']='heldout_or_validation_patch'; return labels,report
    values=[prior.get(k) for k in ['cashew_prior_support_score','prior_evidence_confidence','evidence_conflict_fraction']]
    if not prior.get('prior_data_available') or any(v is None or not math.isfinite(float(v)) for v in values):
        report['reason']='prior_unknown'; return labels,report
    support,confidence,conflict=map(float,values)
    if not (0<=support<=1 and 0<=confidence<=1 and 0<=conflict<=1): raise ValueError('Prior outside unit range')
    if support>policy.maximum_prior_support or confidence<policy.minimum_evidence_confidence or conflict>policy.maximum_conflict_fraction:
        report['reason']='regional_context_insufficient'; return labels,report
    finite=np.isfinite(x).all(0)&(x!=0).any(0)&np.isfinite(p).all(0)
    if np.any(finite & ((p.min(0)<0)|(p.max(0)>1)|(np.abs(p.sum(0)-1)>1e-4))):
        raise ValueError('Frozen prediction must contain normalized probabilities')
    if not finite.any(): report['reason']='invalid_image'; return labels,report
    argmax=p.argmax(0)==2
    report['valid_pixels']=int(finite.sum())
    report['predicted_cashew_fraction']=float((argmax&finite).sum()/finite.sum())
    if report['predicted_cashew_fraction']<policy.patch_cashew_alert_fraction:
        report['reason']='no_model_alert'; return labels,report
    eligible=finite&argmax&(p[2]>=policy.minimum_model_p2)
    if forbidden_pixels is not None:
        forbidden=np.asarray(forbidden_pixels,dtype=bool)
        if forbidden.shape!=labels.shape: raise ValueError('Unaligned protection mask')
        eligible &= ~forbidden
    flat=np.flatnonzero(eligible)
    evidence=feature_evidence(x.reshape(64,-1).T[flat],prototypes)
    keep=(evidence['margin']>=feature_thresholds['minimum_noncashew_margin']) & (evidence['noncashew_cosine']>=feature_thresholds['minimum_noncashew_cosine'])
    eligible_flat=flat[keep]
    report['supported_before_cap']=int(keep.sum())
    limit=min(policy.maximum_pixels_per_patch,int(policy.maximum_valid_pixel_fraction*finite.sum()))
    # Stable ranked cap is a maximum experimental exposure, never an output quota.
    rank=evidence['margin'][keep]*p[2].ravel()[eligible_flat]*confidence
    order=np.argsort(-rank,kind='stable')[:limit]
    labels.ravel()[eligible_flat[order]]=0
    report.update(selected_pixels=int((labels==0).sum()),reason='candidates' if (labels==0).any() else 'no_supported_pixel',
                  source_thresholds=dict(feature_thresholds),negative_label='experimental_noncashew',unmarked_label=10)
    return labels,report

def loss(logits, labels, images):
    """Binary non-Cashew softmax loss with zero gradient on ignore10.

    This loss is separate from confirmed-human loss and four-class source loss.
    It neither chooses a particular alternate class nor conditions FADA.
    """
    import torch
    if not set(torch.unique(labels).detach().cpu().tolist()) <= {0,10}:
        raise ValueError('Automatic masks may contain only0/ignore10')
    if logits.shape[-2:]!=labels.shape[-2:]:
        logits=torch.nn.functional.interpolate(logits,size=labels.shape[-2:],mode='bilinear',align_corners=False)
    valid=(labels==0)&torch.isfinite(images).all(1)&(images!=0).any(1)
    p2=torch.softmax(logits,dim=1)[:,2]
    count=int(valid.sum())
    value=-torch.log1p(-p2[valid].clamp(max=1-1e-6)).mean() if count else logits.sum()*0
    return value,dict(selected_pixels=count,mean_p2=float(p2[valid].detach().mean()) if count else None)

def conflict_review(records, baseline_source_f1, current_source_f1,
                    heldout_recall_before=None, heldout_recall_after=None):
    """A bounded warning/pause signal; gradient opposition is not label truth.

    The pilot must inspect per-location support; these are diagnostic thresholds.
    No gradient projection of pseudo-label conflict is allowed.
    """
    severe=[r for r in records if r.get('source_negative_cosine') is not None and r['source_negative_cosine']<-.5 and (r.get('negative_to_source_norm_ratio') or 0)>.25]
    degradation=current_source_f1 < baseline_source_f1-.03
    recall_loss=(heldout_recall_before is not None and heldout_recall_after is not None and heldout_recall_after<heldout_recall_before-.03)
    pause=(len(severe)>=3 and degradation) or recall_loss
    return dict(action='pause_quarantine_review' if pause else 'continue_diagnostic_review',
                persistent_severe_probes=len(severe),source_degradation=degradation,heldout_cashew_loss=recall_loss,
                conflict_alone_proves_wrong_label=False,gradient_projection_allowed=False)
