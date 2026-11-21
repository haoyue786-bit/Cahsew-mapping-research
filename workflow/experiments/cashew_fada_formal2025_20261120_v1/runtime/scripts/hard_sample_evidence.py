"""Checkpoint-bound pre/post training evidence and frozen-library comparisons."""

from collections import Counter
from pathlib import Path

from hard_sample_training import (PER_REVIEW, comparison, identity, manifest, mode,
                                  read, round_dir, save, summary, work_root)


def exposure_between(job, start, end):
    root = Path(job['output_root'])/'evidence_store/snapshots'
    used = Counter(); steps = 0; pixels = [0]*4
    for e in range(max(1,start),end+1):
        path = root/('epoch_%06d.json'%e)
        snap = read(path)
        if (snap['epoch'],snap['run_id'],snap['job_id']) != (e,job['run_id'],job['job_id']):
            raise ValueError('Annotation exposure belongs to another run')
        result = snap['training']['annotated_training_updates']
        used.update(result['sampled_keys']); steps += result['completed_steps']
        pixels = [a+b for a,b in zip(pixels,result['pixels_by_class'])]
    return used, dict(optimizer_steps=steps, class_pixel_occurrences=pixels, unique_patches=len(used))


def review(job, gan, common, model, device, epoch, checkpoint_sha):
    if epoch % PER_REVIEW:
        return None
    library = manifest(job,epoch)
    if library['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Accepted batch belongs to a different checkpoint')
    scores = comparison(gan,common,model,device,library['entries'])
    root = round_dir(job,epoch); prior_epoch = epoch-PER_REVIEW
    prior = read(round_dir(job,prior_epoch)/'training_fit.json')['rows'] if prior_epoch else []
    prev = {s['key']:s for s in prior}
    paired = [s for s in scores if s['key'] in prev]
    baseline = [prev[s['key']] for s in paired]
    used, batch_exposure = exposure_between(job,prior_epoch+1,epoch)
    if any((a['valid'],a['pixels_by_class']) != (b['valid'],b['pixels_by_class'])
           for a,b in zip(baseline,paired)):
        raise ValueError('Noncomparable accepted sample coverage')
    fresh = [s for s in scores if s['key'] not in prev]
    document = dict(epoch=epoch, checkpoint_sha256=checkpoint_sha, backend=mode(job),
        library_revision=library['revision'], role='training_fit', independent=False,
        inventory={'requested':library['report']['requested'],'resolved':library['report']['resolved'],
                   'statuses':dict(Counter(r['status'] for r in library['report']['attempts'])),
                   'attempted':library['report']['attempted'],'added':library['report']['added']},
        all_training_fit=summary(scores), new_before_first_use=summary(fresh),
        previous_library_comparison={'from_epoch':prior_epoch,'baseline':summary(baseline),
            'current':summary(paired),'actually_exposed_patches':sum(s['key'] in used for s in paired)},
        interval_exposure=batch_exposure)
    save(root/'training_fit.json',dict(summary=document,rows=scores))
    from complete_history_store import _csv_view
    _csv_view(Path(job['output_root'])/'evidence_store/annotated_training'/('epoch_%06d.csv'%epoch),document)
    return document


def freeze_library(job, stop):
    boundary = stop//PER_REVIEW*PER_REVIEW
    library = manifest(job,boundary)
    document = dict(epoch=stop, backend=mode(job), boundary=boundary, revision=library['revision'],
                    entries=library['entries'])
    save(work_root(job)/'selection_library.json',document)
    return document


def fit_candidates(job, gan, common, model, device, candidates, library):
    """Measure eligible model-only artifacts on one frozen set with target BN."""
    import torch
    from io import BytesIO
    from predictor_artifacts_v3 import verify
    root = Path(job['output_root']); selection_root = work_root(job)/'selection_fit'
    for slot in candidates:
        epoch = slot['epoch']; record = verify(root,job,epoch,slot['sha256'])
        path = selection_root/('epoch_%06d.json'%epoch)
        binding = dict(epoch=epoch, checkpoint_sha256=slot['sha256'],
                       model_state_sha256=record['model_state_sha256'],
                       library_revision=library['revision'], backend=library['backend'])
        if path.is_file():
            if read(path)['binding'] != binding:
                raise ValueError('Existing SELECT comparison does not match frozen inputs')
            continue
        if library['entries']:
            payload = (root/record['predictor']).read_bytes()
            import hashlib
            if hashlib.sha256(payload).hexdigest() != record['predictor_sha256']:
                raise ValueError('Predictor bytes changed before training-fit comparison')
            tensor_state = torch.load(BytesIO(payload),weights_only=True,map_location='cpu')
            from transfer_record_contract_v1 import model_digest
            if tensor_state['epoch'] != epoch or model_digest(tensor_state['state_dict']) != record['model_state_sha256']:
                raise ValueError('Predictor identity differs from candidate')
            model.load_state_dict(tensor_state['state_dict'],strict=True)
        rows = comparison(gan,common,model,device,library['entries'])
        exposures,_ = exposure_between(job,1,epoch)
        result = dict(role='training_fit',independent=False,all=summary(rows),
                      exposed=summary([s for s in rows if exposures[s['key']]]),
                      not_yet_exposed=summary([s for s in rows if not exposures[s['key']]]),
                      exposures_total=sum(exposures.values()))
        save(path,dict(binding=binding,summary=result,rows=rows))
        from complete_history_store import _csv_view
        _csv_view(root/'evidence_store/annotation_selection_fit'/('epoch_%06d.csv'%epoch),result)


def selection_fit(job, epoch, checkpoint_sha, model_sha):
    root = work_root(job); path = root/'selection_fit'/('epoch_%06d.json'%epoch)
    data = read(path)
    revision = read(root/'selection_library.json')['revision']
    if data['binding'] != dict(epoch=epoch,checkpoint_sha256=checkpoint_sha,
                               model_state_sha256=model_sha,library_revision=revision,backend=mode(job)):
        raise ValueError('Selection measurement is not from this frozen sample library and model')
    return dict(library_revision=revision,backend=mode(job),**data['summary'])
