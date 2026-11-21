"""Regression checks for the new worker's annotation boundary and loss."""

from pathlib import Path
import sys
from threading import Barrier

import numpy as np
import pytest
import rasterio
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime/scripts'))
from hard_sample_acquisition import acquire, checked_label, process_round
from hard_sample_scan import CATEGORIES, eligible, encode_mask, masks
from hard_sample_training import NoTrainableAnnotations, manifest, round_dir, save, training_loss


@pytest.fixture
def job(tmp_path):
    run = tmp_path/'verification/jobs/group1'; run.mkdir(parents=True)
    return dict(job_id='group1',run_id='test/group1',output_root=str(run),target_year=2019,
                hard_sample_policy=dict(version='hard_samples_v1',annotation_backend='simulated'))


def chip(number, tmp_path):
    path = tmp_path/(f'aef-{number}.tif')
    transform = rasterio.transform.from_origin(300000+number*2000,1000000,10,10)
    with rasterio.open(path,'w',driver='GTiff',dtype='float32',width=128,height=128,
                      count=1,crs='EPSG:32631',transform=transform) as output:
        output.write(np.ones((128,128),dtype='float32'),1)
    return dict(aef=str(path),path=path.name,country='NGA')


def test_all_four_classes_have_gradient_and_unknown_is_ignored():
    logits=torch.zeros(1,4,1,5,requires_grad=True)
    targets=torch.tensor([[[0,1,2,3,10]]])
    training_loss(logits,targets).backward()
    assert (logits.grad[0,:,0,:4].argmin(dim=0) == torch.arange(4)).all()
    assert (logits.grad[0,:,0,4] == 0).all()
    zeros=torch.ones(1,4,1,1,requires_grad=True)
    training_loss(zeros,torch.full((1,1,1),10)).backward()
    assert not zeros.grad.any()


def test_detects_two_kinds_of_indecision_only_with_spatial_support():
    probs=np.broadcast_to(np.array([.12,.48,.38,.02])[:,None,None],(4,128,128)).copy()
    probs[:,64:] = np.array([.27,.28,.25,.20])[:,None,None]
    support=np.ones((128,128),dtype=bool)
    two,many=masks(probs,support)
    assert two[:64].all() and not two[64:].any()
    assert many[64:].all() and not many[:64].any()
    isolated=np.zeros_like(support); isolated[::3,::3]=True
    assert eligible(two,support) and eligible(many,support) and not eligible(isolated,support)


def test_skips_retries_partially_annotates_and_reuses_without_another_call(job,tmp_path):
    candidates=[chip(i,tmp_path) for i in range(4)]
    first=round_dir(job,5)
    region=np.zeros((128,128),dtype=bool); region[:64]=True
    screen=dict(epoch=5,checkpoint_sha256='f'*64,scanned=4,valid_patches=4,valid_pixels=4*128**2,
                existing_patches=0,candidates={k:candidates if k=='competing' else [] for k in CATEGORIES},
                masks={r['path']:{'competing':encode_mask(region)} for r in candidates})
    save(first/'screen.json',screen)
    plan={k:3 if k=='competing' else 0 for k in CATEGORIES}
    plan['rationale']='Complete enough of one difficult region to test integration.'
    batch=process_round(job,5,{},planner=lambda *_:plan)
    assert [v['status'] for v in batch['report']['attempts']] == ['no_imagery','accepted','accepted']
    assert batch['report']['attempts'][1]['issue_coverage'] == 0
    assert batch['report']['added']==2 and batch['report']['resolved']['competing']==1
    assert len(list((first/'attempts').glob('*.error.json')))==1
    assert process_round(job,5,{},planner=lambda *_:pytest.fail('Plan should be pinned'))==batch
    second=dict(job,run_id='test/group2',output_root=str(Path(job['output_root']).with_name('group2')))
    status, (entry, _) = acquire(second,5,candidates[2])
    assert status=='reused' and entry['receipt']==batch['entries'][-1]['receipt']
    with pytest.raises(ValueError,match='grid or period'):
        checked_label(candidates[2],2025,Path(entry['receipt']),'simulated')
    with pytest.raises(ValueError,match='Formal visual review'):
        checked_label(candidates[2],2019,Path(entry['receipt']),'sample_assist')
    assert manifest(job,5)['entries']==batch['entries']


def test_entire_plan_starts_together_without_replacing_unusable_samples(job, monkeypatch):
    import hard_sample_acquisition as acquisition

    root = round_dir(job, 5)
    quotas = {'competing': 4, 'diffuse': 3, 'random_probe': 1}
    counts = dict(quotas, competing=5)
    candidates = {category: [{'path': f'{category}-{i}'} for i in range(counts.get(category, 0))]
                  for category in CATEGORIES}
    plan = {category: quotas.get(category, 0) for category in CATEGORIES}
    plan['rationale'] = 'Use the full selected plan.'
    count = sum(counts.values())
    save(root / 'screen.json', dict(epoch=5, checkpoint_sha256='f'*64, scanned=count,
                                    valid_patches=count, valid_pixels=count*128**2,
                                    existing_patches=0, candidates=candidates, masks={}))
    initial = Barrier(8, timeout=8)

    def acquire_candidate(job, epoch, candidate, *, ordinal):
        assert ordinal < 8
        initial.wait()
        if ordinal == 0:
            return 'no_imagery', None
        return 'accepted', (dict(candidate, key=candidate['path']), np.ones((128, 128), dtype=bool))

    monkeypatch.setattr(acquisition, 'acquire', acquire_candidate)
    batch = process_round(job, 5, {}, planner=lambda *_: plan)
    assert len(acquisition.read(root / 'dispatch' / '000.json')) == 8
    assert len(list((root / 'dispatch').glob('*.json'))) == 1
    assert batch['report']['attempted'] == 8 and batch['report']['added'] == 7
    assert batch['report']['attempts'][0]['status'] == 'no_imagery'
    assert batch['report']['resolved'] == {category: plan[category] - int(category == 'competing')
                                           for category in CATEGORIES}


@pytest.mark.parametrize('failure', ['no_imagery', 'transport_error'])
def test_formal_empty_intake_cannot_publish_ready(job, monkeypatch, failure):
    import hard_sample_acquisition as acquisition

    job['target_year'] = 2025
    job['hard_sample_policy']['annotation_backend'] = 'sample_assist'
    root = round_dir(job, 5)
    candidates = {category: [{'path': 'selected.tif'}] if category == 'competing' else []
                  for category in CATEGORIES}
    save(root / 'screen.json', dict(epoch=5, checkpoint_sha256='a' * 64, scanned=1, valid_patches=1,
                                    valid_pixels=128 ** 2, existing_patches=0,
                                    candidates=candidates, masks={}))
    plan = {category: int(category == 'competing') for category in CATEGORIES}
    plan['rationale'] = 'Check the available competing example.'
    monkeypatch.setattr(acquisition, 'acquire', lambda *args, **kwargs: (failure, None))
    with pytest.raises(NoTrainableAnnotations, match='No verified training labels'):
        process_round(job, 5, {}, planner=lambda *_: plan)
    blocked = acquisition.read(root / 'label_intake_failed.json')
    assert blocked['usable_labels'] == 0
    assert blocked['report']['attempts'][0]['status'] == failure
    assert not (root / 'ready.json').exists()


def test_recovery_preserves_legacy_assignments_and_dispatches_open_slots(job, tmp_path, monkeypatch):
    import hard_sample_acquisition as acquisition

    root = round_dir(job, 5)
    chips = [chip(i, tmp_path) for i in range(4)]
    plan = {category: 0 for category in CATEGORIES}
    plan.update(competing=2, diffuse=1, random_probe=1, rationale='Resume the recorded allocation.')
    candidates = {category: [] for category in CATEGORIES}
    candidates.update(competing=chips[:2], diffuse=chips[2:3], random_probe=chips[3:])
    save(root / 'screen.json', dict(epoch=5, checkpoint_sha256='a'*64, scanned=4,
                                    valid_patches=4, valid_pixels=4*128**2, existing_patches=0,
                                    candidates=candidates, masks={}))
    for i, candidate in enumerate(chips[:2]):
        save(root / 'results' / f'{i:03d}.json',
             dict(path=candidate['path'], category='competing', status='no_imagery' if i == 0 else 'accepted',
                  resolved=i != 0, entry=dict(candidate, key=candidate['path']) if i != 0 else None))
    key = acquisition.grid_key(acquisition.grid(chips[2], 2019))
    save(root / 'attempts' / 'legacy.json', dict(key=key, ordinal=2, attempt=1, backend='simulated'))
    together = Barrier(2, timeout=8)

    def resume_candidate(job, epoch, candidate, *, ordinal):
        assert ordinal in (2, 3) and candidate == chips[ordinal]
        together.wait()
        return 'accepted', (dict(candidate, key=candidate['path']), np.ones((128, 128), dtype=bool))

    monkeypatch.setattr(acquisition, 'acquire', resume_candidate)
    batch = process_round(job, 5, {}, planner=lambda *_: plan)
    assert [entry['path'] for entry in batch['entries']] == [item['path'] for item in chips[1:]]
    assert len(acquisition.read(root / 'dispatch' / '000.json')) == 3
    assert acquisition.read(root / 'dispatch' / '003.json') == [dict(category='random_probe', path=chips[3]['path'])]


@pytest.mark.parametrize('current', [False, True])
def test_formal_adapter_accepts_the_skill_receipt_schema_with_mocked_service(job,tmp_path,monkeypatch,current):
    import json
    import hard_sample_acquisition as a
    from hard_sample_training import sha

    job['hard_sample_policy']['annotation_backend']='sample_assist'
    monkeypatch.setattr(a,'LIBRARY',tmp_path/'mock_formal_library')
    candidate=chip(8,tmp_path); calls=[]
    def mock(command, **options):
        calls.append(command)
        assert Path(command[1])==a.ANNOTATOR/'scripts/run_sample.py'
        args=dict(zip(command[2::2],command[3::2]))
        assert set(args)=={'--aef','--case','--country','--period','--run-dir'}
        assert (args['--aef'],args['--country'],args['--period'])==(candidate['aef'],'NGA','2019')
        folder=Path(args['--run-dir'])/args['--case']
        folder.mkdir(parents=True)
        y,x=np.indices((128,128)); labels=(x//64+2*(y//64)).astype('uint8')
        label=folder/'labels_native128.tif'
        with rasterio.open(candidate['aef']) as aef:
            profile=dict(aef.profile,dtype='uint8')
        with rasterio.open(label,'w',**profile) as output:
            output.write(labels,1)
        receipt=folder/'RECEIPT.json'
        value=dict(status='completed',grid=a.grid(candidate,2019),acquisition_dates=['2019-02-01'],
                   label_geotiff=str(label),label_sha256=sha(label))
        audit=folder/'mock_audit.json'
        audit_fields=dict(decision_turn_id='mock_turn',verified_image_ids=['rgb'],missing_image_ids=[]) if current else {}
        save(audit,dict(status='verified',**audit_fields))
        selection=folder/'mock_selection.json'
        provider='Google' if current else 'Esri'
        prescreen=dict(selection_method='code_preflight') if current else {}
        save(selection,dict(status='selected',grid=value['grid'],frame=dict(
            provider=provider,acquisition_dates=value['acquisition_dates'],acquisition_range_verified=True,
            frame_georeferencing_verified=True,native128_valid_fraction=1.),
            review=dict(usable=True,needs_dry_season=False),**prescreen))
        decision=(dict(annotation_id='mock_plan',decision_turn_id='mock_turn',service_tier='default')
                  if current else dict(reviewed_preview_id='mock_preview'))
        value.update(model='gpt-6.2-sol',effort='max',imagery_provider=provider,**decision,
                     selection=str(selection),selection_sha256=sha(selection),vision_audit=str(audit))
        receipt.write_text(json.dumps(value))
    monkeypatch.setattr(a.subprocess,'run',mock)
    status,(entry,valid)=a.acquire(job,5,candidate)
    assert status=='accepted' and len(calls)==1 and all(entry['counts']) and valid.any()
    assert a.acquire(job,10,candidate)[0]=='reused' and len(calls)==1
    payload=json.loads(Path(entry['receipt']).read_text())
    if current:
        wrong_turn=Path(entry['receipt']).with_name('wrong_turn.json')
        save(wrong_turn,dict(payload,decision_turn_id='other_turn'))
        with pytest.raises(ValueError,match='annotation decision turn'):
            a.checked_label(candidate,2019,wrong_turn,'sample_assist')
    with rasterio.open(entry['label'],'r+') as label:
        label.update_tags(simulated='true')
    alternate=Path(entry['receipt']).with_name('mock_tagged_receipt.json')
    tagged=dict(payload,label_sha256=sha(entry['label']))
    save(alternate,tagged)
    with pytest.raises(ValueError,match='Simulated GeoTIFF'):
        a.checked_label(candidate,2019,alternate,'sample_assist')


def test_selection_requires_one_common_library_and_exposure_aware_evidence():
    from checkpoint_selection_contract_v2 import VERSION, artifact, summary
    from review_contract import build_checkpoint_selection, validate_checkpoint_selection
    from sealed_packet import values_digest

    fit = dict(library_revision='a'*64,backend='simulated',role='training_fit',independent=False,
               all=dict(patches=1,valid_pixels=9,class_pixels=[2,2,3,2]))
    def records(other):
        details=[]; coarse=[]
        for epoch,value in ((0,fit),(5,other)):
            record=dict(epoch=epoch,checkpoint_sha256='d'*64,model_state_sha256='e'*64,
                producer_request_sha256='b'*64,source_validation=dict(class2_f1=.4),
                unlabelled_panel=dict(predicted_class2_fraction=.1),human_training=None,
                training=dict(cls=.6) if epoch else {},controls=dict(hard_sample_weight=.5),
                controls_timing='used_during_completed_epoch' if epoch else 'after_initialization_before_training',
                review_action='STOP' if epoch else 'KEEP',review_action_applies_after_epoch=epoch,
                annotated_training=value)
            details.append(artifact(record))
            brief=summary(record)
            coarse.append(dict(kind='candidate',epoch=epoch,values=brief,sha256=values_digest('candidate',epoch,brief)))
        trusted={(r['kind'],r['epoch'],r['sha256']) for r in coarse+details}
        return build_checkpoint_selection(5,coarse,trusted,history_required=True,selection_policy=VERSION,
                                          candidate_details=details)
    with pytest.raises(ValueError,match='populations differ'):
        records(dict(fit,library_revision='c'*64))
    request=records(fit)
    answer=dict(selected_epoch=5,reason='Check training response.',uncertainty='Simulated labels.',
                baseline_comparison=dict(benefits='Fit',tradeoffs='Domain shift',unresolved='Generalization'),
                strongest_alternative=dict(epoch=0,why_competitive='Source retention',
                    benefits_of_selected='Exposure exercised',tradeoffs='Source behavior',unresolved='Target accuracy'),
                evidence_used=[dict(epoch=e,section='unlabelled_panel') for e in (0,5)])
    with pytest.raises(ValueError,match='frozen-library fit'):
        validate_checkpoint_selection(request,answer)
    answer['evidence_used']=[dict(epoch=e,section='annotated_training') for e in (0,5)]
    validate_checkpoint_selection(request,answer)


def test_trial_delivery_projects_actual_annotation_dose_and_validates_old_transcript():
    from training_tool_output_v1 import delivery, verify_delivery
    def arm(exposure):
        return dict(end={},observations=[dict(annotated_training_updates=dict(
            epoch=6,completed_steps=exposure,pixels_by_class=[4*exposure]*4,raw_loss=.8,
            gradient_probes=[dict(annotation_norm=.4)]))])
    result=dict(status='completed',baseline={},arms=dict(control=arm(3),treatment=arm(6)))
    tool='run_matched_training_trial'
    compact=delivery(result,tool,1)
    assert compact['arms']['treatment']['annotation_exposure']['optimizer_steps']==6
    assert compact['arms']['control']['annotation_exposure']['class_pixel_occurrences']==[12]*4
    for view in (compact,delivery(result,tool,1,annotation_summary=False)):
        verify_delivery(dict(result=result,delivered_result=view),tool,1)


def test_trial_capability_obeys_run_budget(job):
    from paired_trial_v1 import POLICY, capability, spec
    job.update(paired_trial_policy=POLICY,point_validation_online=False,max_epoch=10)
    request=dict(contract_version='safe_full_review_candidate_v1',epoch=5,
                 allowed_parameter_bounds={'hard_sample_weight':[0.,1.]},
                 matched_trial_capability=capability(job,epoch=5))
    assert spec(request)[0]['inputSchema']['properties']['steps']['enum']==[63,126,315]
    assert capability(job,epoch=10) is None
