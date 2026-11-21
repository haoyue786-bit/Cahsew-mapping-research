"""Publish verified inference-only predictors while a full checkpoint is in RAM.

Full resumable states remain authoritative. Selection reuses immutable receipts
and checks file identities, and hashes/loads the selected compact predictor.
"""
from pathlib import Path
import hashlib,io,json,os
import torch

VERSION='verified_predictor_artifacts_v3'

def stat_identity(path):
    st=Path(path).stat()
    return dict(dev=st.st_dev,ino=st.st_ino,size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)

def enabled(job):return job.get('predictor_artifact_policy')==VERSION

def publish(root,job,payload,full_sha,relative):
    if not enabled(job):return None
    from complete_history_store import _write_immutable,_canonical
    from transfer_record_contract_v1 import model_digest
    root=Path(root);epoch=payload['epoch'];state=payload['state_dict']
    folder=root/'predictors';folder.mkdir(exist_ok=True)
    dest=folder/('epoch_%06d.pt'%epoch);receipt_path=folder/('epoch_%06d.json'%epoch)
    if receipt_path.exists():return verify(root,job,epoch,full_sha)
    model_sha=model_digest(state)
    if any((v.is_floating_point() or v.is_complex()) and not torch.isfinite(v).all().item() for v in state.values()):
        raise ValueError('nonfinite inference state')
    bn={k:v for k,v in state.items() if any(x in k.rsplit('.',1)[-1] for x in ('running_mean','running_var','num_batches_tracked','domain_mode'))}
    buffer=io.BytesIO();torch.save({'format':VERSION,'epoch':epoch,'state_dict':state,
        'architecture':job['architecture'],'bn_update_mode':payload['config'].get('bn_update_mode',0),
        'decoder':'four_class_argmax_equals_2'},buffer)
    raw=buffer.getvalue();_write_immutable(dest,raw)
    receipt=dict(version=VERSION,job_id=job['job_id'],run_id=job['run_id'],epoch=epoch,
        full_checkpoint=relative,full_checkpoint_sha256=full_sha,full_checkpoint_stat=stat_identity(root/relative),
        predictor=str(dest.relative_to(root)),predictor_sha256=hashlib.sha256(raw).hexdigest(),
        predictor_stat=stat_identity(dest),model_state_sha256=model_sha,batchnorm_buffers_sha256=model_digest(bn) if bn else hashlib.sha256(b'no_batchnorm_buffers').hexdigest(),
        bn_update_mode=payload['config'].get('bn_update_mode',0),finite_state_checked=True,
        identity_scope='Full bytes checked at publication; later stat checks assume the owned immutable journal and receipt store remain trusted.')
    _write_immutable(receipt_path,_canonical(receipt));return receipt

def verify(root,job,epoch,full_sha,*,load_predictor=False):
    root=Path(root);path=root/'predictors'/('epoch_%06d.json'%epoch)
    if path.is_symlink():raise ValueError('unsafe predictor receipt')
    r=json.loads(path.read_text())
    if (r['version'],r['job_id'],r['run_id'],r['epoch'],r['full_checkpoint_sha256'])!=(VERSION,job['job_id'],job['run_id'],epoch,full_sha):
        raise ValueError('predictor receipt identity mismatch')
    from checkpoint_selection_evidence_v2 import checkpoint_relative
    if r['full_checkpoint']!=checkpoint_relative(epoch):raise ValueError('unowned full checkpoint')
    for key,sk in [('full_checkpoint','full_checkpoint_stat'),('predictor','predictor_stat')]:
        p=root/r[key]
        if p.is_symlink() or p.resolve().parent not in ((root/'predictors').resolve(),(root/'journal').resolve(),(root/'journal'/('epoch_%06d'%epoch)).resolve()):
            raise ValueError('unsafe predictor/checkpoint path')
        if stat_identity(p)!=r[sk]:raise ValueError('published state changed; explicit reconciliation required')
    if load_predictor:
        raw=(root/r['predictor']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['predictor_sha256']:raise ValueError('predictor bytes changed')
        value=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        from transfer_record_contract_v1 import model_digest
        if value['format']!=VERSION or value['epoch']!=epoch or model_digest(value['state_dict'])!=r['model_state_sha256']:
            raise ValueError('predictor model differs from publication')
        if any((v.is_floating_point() or v.is_complex()) and not torch.isfinite(v).all().item() for v in value['state_dict'].values()):raise ValueError('predictor is nonfinite')
    return r

def predictor_identity(receipt):
    from checkpoint_selection_contract_v2 import DECODER
    return dict(DECODER,model_state_sha256=receipt['model_state_sha256'],
        batchnorm_buffers_sha256=receipt['batchnorm_buffers_sha256'],
        checkpoint_sha256=receipt['full_checkpoint_sha256'],compact_predictor=receipt['predictor'],
        compact_predictor_sha256=receipt['predictor_sha256'],bn_update_mode=receipt['bn_update_mode'])

def freeze_candidates(root,job,candidates,stop_epoch):
    from selection_freeze import publish as publish_json,hash_file
    from checkpoint_selection_contract_v2 import VERSION as SELECTION
    from checkpoint_selection_evidence_v2 import collect
    root=Path(root);out=[]
    for item in candidates:
        e=item['epoch'];detail=collect(root,job['job_id'],job['run_id'],e,include_annotation_fit=False)
        r=verify(root,job,e,detail['checkpoint_sha256'])
        supplied=Path(item['path']).resolve()
        if supplied!=(root/r['full_checkpoint']).resolve():raise ValueError('candidate path differs')
        out.append({'epoch':e,'path':r['full_checkpoint'],'sha256':r['full_checkpoint_sha256']})
    if [x['epoch'] for x in out]!=sorted(set(x['epoch'] for x in out)) or out[0]['epoch']!=0 or out[-1]['epoch']!=stop_epoch:
        raise ValueError('invalid frozen candidate sequence')
    manifest=dict(version='candidate_freeze_v1',job_id=job['job_id'],run_id=job['run_id'],stop_epoch=stop_epoch,
        stop_receipt='stop_receipt.json',stop_receipt_sha256=hash_file(root/'stop_receipt.json'),candidates=out,selection_policy=SELECTION)
    publish_json(root/'frozen_candidates.json',manifest);return manifest
