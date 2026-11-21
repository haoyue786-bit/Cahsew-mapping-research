"""Candidate prior-review comparison; no point labels or model invocation.

This is not wired into the worker or full-review request. Both numeric artifacts
must be separately registered from the live source/imagery-only producer, and
binding membership must be authenticated by the same-run worker outside this
module. A plain hash set, snapshots of file metadata, or a caller's claim of a
run identity does not prove producer authenticity, content equality, a blind
training trajectory, or independence from held-out target points.
"""
import hashlib
import json
import math
import re

from sealed_packet import PANEL, SOURCE, evidence

HEX = re.compile(r'^[a-f0-9]{64}$')
ID = re.compile(r'^[A-Za-z0-9_.-]{1,100}$')
DELTA_FIELDS = frozenset(('previous_epoch', 'current_epoch', 'last_action_epoch',
    'same_input_snapshot', 'source_support_equal', 'panel_support_equal',
    'source_class2_f1_change', 'source_loss_change', 'panel_class2_fraction_change',
    'panel_mean_entropy_change'))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def bind(job_id, run_id, source, panel, audit):
    """Build an identity record to be separately authenticated and registered.

    Called only immediately after the live producer and source/panel journal
    registration. This itself does not prove either origin or lineage.
    """
    valid_run = isinstance(run_id, str) and len(run_id) <= 160 and all(ID.fullmatch(part) and part not in ('.', '..') for part in run_id.split('/'))
    if not isinstance(job_id, str) or not ID.fullmatch(job_id) or not valid_run:
        raise ValueError('invalid run identity')
    if not isinstance(audit,dict) or audit.get('version') != 'live_review_evidence_v1':
        raise ValueError('unsupported review producer')
    epoch=audit.get('epoch')
    if type(epoch) is not int or epoch < 0:
        raise ValueError('invalid review epoch')
    for artifact,kind in ((source,'source'),(panel,'panel')):
        if not isinstance(artifact,dict) or artifact.get('kind')!=kind or artifact.get('epoch')!=epoch:
            raise ValueError('review identity mismatch')
        if not isinstance(artifact.get('sha256'),str) or not HEX.fullmatch(artifact['sha256']):
            raise ValueError('invalid artifact hash')
    cp=audit.get('checkpoint_sha256'); snapshot=audit.get('input_snapshot_sha256')
    if not all(isinstance(x,str) and HEX.fullmatch(x) for x in (cp,snapshot)):
        raise ValueError('missing checkpoint or input snapshot identity')
    binding={'version':'prior_review_binding_v1','job_id':job_id,'run_id':run_id,'epoch':epoch,
             'checkpoint_sha256':cp,'input_snapshot_sha256':snapshot,
             'source_sha256':source['sha256'],'panel_sha256':panel['sha256']}
    return binding,hashlib.sha256(_canonical(binding)).hexdigest()


def _read(record,kind,cutoff,trusted):
    # Epoch zero is allowed for an actual initialization measurement.
    return evidence(record,kind,SOURCE if kind=='source' else PANEL,cutoff,trusted,min_epoch=0)


def _checked_record(record, job_id, run_id, cutoff, trusted_values, trusted_bindings):
    if not isinstance(record,dict) or set(record)!={'source','panel','binding'}:
        raise ValueError('review record must include source, panel and binding')
    s=_read(record['source'],'source',cutoff,trusted_values)
    p=_read(record['panel'],'panel',cutoff,trusted_values)
    b=record['binding']
    fields={'version','job_id','run_id','epoch','checkpoint_sha256','input_snapshot_sha256','source_sha256','panel_sha256'}
    if not isinstance(b,dict) or set(b)!=fields or b['version']!='prior_review_binding_v1':
        raise ValueError('invalid binding')
    if b['job_id']!=job_id or b['run_id']!=run_id or type(b['epoch']) is not int or b['epoch']!=s['epoch'] or s['epoch']!=p['epoch']:
        raise ValueError('cross-run or epoch mismatch')
    if b['source_sha256']!=s['sha256'] or b['panel_sha256']!=p['sha256']:
        raise ValueError('values not bound to review')
    if any(not isinstance(b[k],str) or not HEX.fullmatch(b[k]) for k in ('checkpoint_sha256','input_snapshot_sha256')):
        raise ValueError('invalid input or checkpoint digest')
    sha=hashlib.sha256(_canonical(b)).hexdigest()
    if (job_id,run_id,b['epoch'],sha) not in trusted_bindings:
        raise ValueError('unregistered review binding')
    return s['values'],p['values'],b


def _prob(x):
    return x is not None and type(x) in (int,float) and math.isfinite(x) and 0<=x<=1


def _positive_count(x):
    return type(x) is int and x>0


def compare(previous,current,*,job_id,run_id,trusted_values,trusted_bindings,last_action_epoch=None):
    """Compare consecutive full reviews on the same registered input snapshot.

    A missing previous review returns None. Input snapshots hash file *metadata*,
    not contents or individual validity masks. A successful comparison is only
    conditional evidence that the frame is stable, and must say so to the Agent.
    Difference in class proportions is not a pixelwise map-change measurement.
    Any response after an intervention is temporal and potentially confounded.
    """
    if previous is None:
        return None
    valid_run = isinstance(run_id, str) and len(run_id) <= 160 and all(ID.fullmatch(part) and part not in ('.', '..') for part in run_id.split('/'))
    if not isinstance(job_id, str) or not ID.fullmatch(job_id) or not valid_run:
        raise ValueError('invalid run identity')
    if not isinstance(current,dict) or not isinstance(current.get('binding'),dict):
        raise ValueError('missing current binding')
    epoch=current['binding'].get('epoch')
    if type(epoch) is not int or epoch<1:
        raise ValueError('invalid full-review epoch')
    ps,pp,pb=_checked_record(previous,job_id,run_id,epoch,trusted_values,trusted_bindings)
    cs,cp,cb=_checked_record(current,job_id,run_id,epoch,trusted_values,trusted_bindings)
    if cb['epoch']!=epoch or not 1<=epoch-pb['epoch']<=5:
        raise ValueError('previous observation outside previous five-epoch window')
    if last_action_epoch is not None and (type(last_action_epoch) is not int or not pb['epoch']<=last_action_epoch<epoch):
        raise ValueError('action outside comparison interval')
    same=pb['input_snapshot_sha256']==cb['input_snapshot_sha256']
    source_same=same and _positive_count(ps.get('validation_pixels')) and ps.get('validation_pixels')==cs.get('validation_pixels')
    panel_same=same and _positive_count(pp.get('sample_valid_pixel_count')) and pp.get('sample_valid_pixel_count')==cp.get('sample_valid_pixel_count') and _positive_count(pp.get('sample_patch_count')) and pp.get('sample_patch_count')==cp.get('sample_patch_count')
    out={'previous_epoch':pb['epoch'],'current_epoch':epoch,'last_action_epoch':last_action_epoch,
         'same_input_snapshot':int(same),'source_support_equal':int(source_same),'panel_support_equal':int(panel_same),
         'source_class2_f1_change':None,'source_loss_change':None,
         'panel_class2_fraction_change':None,'panel_mean_entropy_change':None}
    if source_same and _prob(ps.get('class2_f1')) and _prob(cs.get('class2_f1')):
        out['source_class2_f1_change']=cs['class2_f1']-ps['class2_f1']
    if source_same and all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in (ps.get('source_loss'),cs.get('source_loss'))):
        out['source_loss_change']=cs['source_loss']-ps['source_loss']
    if panel_same and _prob(pp.get('predicted_class2_fraction')) and _prob(cp.get('predicted_class2_fraction')):
        out['panel_class2_fraction_change']=cp['predicted_class2_fraction']-pp['predicted_class2_fraction']
    if panel_same and all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=math.log(4)+1e-6 for x in (pp.get('mean_entropy'),cp.get('mean_entropy'))):
        out['panel_mean_entropy_change']=cp['mean_entropy']-pp['mean_entropy']
    return out
