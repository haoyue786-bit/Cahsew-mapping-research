"""User-authorized, separate Agent-selected weak negatives for CIV groups.

Finite 128-update exposure per generated pool, then off until a trained review.
No patch-count ceiling, geography quota, high-p2 gate or non-Cashew template.
"""
import copy, csv, fcntl, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape, box
from pyproj import Geod
import torch
from automatic_negative_v1 import normalize
from automatic_negative_pool_uncapped_v3 import AutomaticNegativePool as UncappedPool
from supplementary_control_v4 import rng_scope
from persistent_luna import PersistentLunaClient
from agent_account_route import apply_worker_route
from automatic_negative_selection_validation_v1 import validated_turn
from expert_policy_v4 import MODEL, EFFORT

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'agent_selected_rising_AEF_automatic_negative_production_v4'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def dump(p, value):
    p = Path(p);p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name+'.tmp.'+str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n');temp.replace(p)

def validate_job(job):
    policy = job.get('automatic_negative_policy')
    if not policy:return
    assert policy['version'] == VERSION and policy['user_authorized']
    assert policy['patch_count_cap'] is None and not policy['geographic_distribution_quota']
    assert policy['exposure_horizon_updates'] == 128 and policy['source_class_weights_changed'] is False
    assert job['max_epoch'] == 200 and job['source_profile'] == 'source2019_benin_e173'
    inputs=job['automatic_negative_inputs']
    assert job['country'] in ['Ivoire','Nigeria','Senegal'] and job['target_year'] in [2019,2025]
    for key in ['prior_index','prior_raster','human_polygons','validation_geometry','source_feature_manifest']:
        assert sha(inputs[key])==inputs[key+'_sha256'],key
    if job['country']=='Nigeria':
        assert inputs['manual_pool_sha256']=='5f44d1d959ff9cdbfe06f5b04463927d026a46aedf5afc0385ff44a017856832'
    assert job['point_validation_online'] is False

def evidence(job):
    if not job.get('automatic_negative_policy'):return None
    p = Path(job['output_root'])/'automatic_negative/state.json'
    state = read(p) if p.exists() else {}
    return {'policy':job['automatic_negative_policy'], 'state':state,
            'interpretation':'Independent user-authorized pseudo-negative hypothesis, not confirmed human labels. Rising p2 and original source Cashew AEF distance are soft evidence; approved country regional prior is soft context. No calibrated crop quota. Source weights unchanged.',
            'response_required':'Assess actual exposure, source validation, training-polygon fit (not independent evaluation) and other permitted proxies together. A smaller map, feature novelty or gradient opposition alone is not label correctness.',
            'information_boundary':'Independent target-country point labels and posthoc threshold/Planet area audits are withheld.'}

class Pool(UncappedPool):
    """Slice one finite generation-wide schedule across ordinary epochs."""
    def __init__(self, manifest, expected, weight, cursor):
        super().__init__(manifest, expected, weight)
        self.cursor = int(cursor)

    def begin_epoch(self, steps, config, seed):
        generation = self.manifest['generation_epoch']
        self.actual_epoch_steps = steps
        super().begin_epoch(128, config, self.manifest['schedule_seed'])
        entire = self.schedule
        self.schedule = {k-self.cursor:v for k,v in entire.items() if self.cursor <= k < min(128,self.cursor+steps)}
        if self.cursor >= 128:self.active_weight=0.
        self.next_cursor = min(128, self.cursor+steps)
        self.cumulative_expected_patches = len({i for k,v in entire.items() if k < self.next_cursor for i in v})

    def summary(self):
        value = super().summary()
        value.update(deployment_policy=VERSION, configured_weight=self.weight,
                     generation_start_cursor=self.cursor,generation_end_cursor=self.next_cursor,
                     generation_horizon_updates=128, generation_expected_cumulative_exposed_patches=self.cumulative_expected_patches,
                     all_selected_patches_scheduled=self.next_cursor>=128, occurrence_cap_per_generation=1)
        return value

class Controller:
    def __init__(self, job, gan, common, model, device):
        validate_job(job)
        self.job,self.gan,self.common,self.model,self.device=job,gan,common,model,device
        self.out=Path(job['output_root'])/'automatic_negative';self.out.mkdir(exist_ok=True)
        self.active_pool=None

    def previous(self, epoch):
        roots=[Path(self.job['output_root'])]
        if self.job.get('resume'):roots.append(Path(self.job['resume']['source_run_root']))
        found=[]
        for root in roots:
            for p in (root/'checkpoints').glob('epoch_*.pt'):
                e=int(p.stem.split('_')[1])
                if e<epoch:found.append((e,p))
        assert found, 'Rising-p2 assessment requires a real earlier checkpoint.'
        return max(found,key=lambda a:a[0])

    def catalog(self, epoch, checkpoint, destination):
        from train_expert_v4 import load_resumable_checkpoint, make_architecture_model, read_lines
        previous_epoch, previous=self.previous(epoch)
        target=read_lines(Path(self.job['target_train_list']))
        pid=lambda name:'_'.join(Path(name).stem.split('_')[-3:])
        ids={pid(name) for name in target}
        evaluation={pid(name) for name in read_lines(Path(self.job['target_evaluation']['names_file']))}
        assert not ids&evaluation
        index=Path(self.job['automatic_negative_inputs']['prior_index'])
        with index.open() as stream:
            priors={r['patch_id']:r for r in csv.DictReader(stream) if r['patch_id'] in ids}
        inputs=self.job['automatic_negative_inputs']
        polygons=[shape(f['geometry']) for f in read(inputs['human_polygons'])['features']]
        held=[];held_ids=set()
        # Prepared geometry-only protection over both years/all countries.
        points=[shape(f['geometry']) for f in read(inputs['validation_geometry'])['features']]
        source=read(inputs['source_feature_manifest'])
        assert sha(source['prototypes'])==source['prototypes_sha256']
        prototypes=normalize(np.load(source['prototypes'])['prototypes'][2])
        geod=Geod(ellps='WGS84');audit=[];rows=[];was_training=self.model.training
        (destination/'cache').mkdir(parents=True,exist_ok=True)
        with rng_scope(np.random.RandomState(16),700003+epoch,self.device):
            reference,_=make_architecture_model(self.gan,self.job,self.device)
            reference.load_state_dict(load_resumable_checkpoint(previous)['state_dict'],strict=True)
            reference.eval();self.model.eval()
            def p2(module,x):
                logits,_=self.gan.forward_fuse(module,torch.from_numpy(x[None]).to(self.device))
                if logits.shape[-2:]!=(128,128):logits=torch.nn.functional.interpolate(logits,size=(128,128),mode='bilinear',align_corners=False)
                return torch.softmax(logits,1)[0,2].cpu().numpy()
            try:
                with torch.no_grad():
                    for i,name in enumerate(target):
                        patch=pid(name);path=Path(self.job['target_root'])/name
                        with rasterio.open(path) as ds:
                            x=ds.read().astype(np.float32);bounds=box(*ds.bounds);transform=ds.transform
                            assert x.shape==(64,128,128) and ds.crs.to_epsg()==4326
                        lon,lat=bounds.centroid.x,bounds.centroid.y
                        forbidden=patch in held_ids or any(bounds.covers(pt) for pt in points)
                        if not forbidden:forbidden=any(geod.inv(lon,lat,float(r['longitude']),float(r['latitude']))[2]<=2500 for r in held)
                        if forbidden:
                            audit.append({'patch_id':patch,'reason':'heldout_or_validation_geometry_protection'});continue
                        intersect=[g for g in polygons if g.intersects(bounds)]
                        marked=rasterize([(g,1) for g in intersect],out_shape=(128,128),transform=transform,fill=0,dtype='uint8') if intersect else np.zeros((128,128),np.uint8)
                        before,after=p2(reference,x),p2(self.model,x)
                        valid=np.isfinite(x).all(0)&(x!=0).any(0)&np.isfinite(before)&np.isfinite(after)
                        candidate=valid&~marked.astype(bool)&(after>before)
                        selected=np.flatnonzero(candidate)
                        if not len(selected):
                            audit.append({'patch_id':patch,'reason':'no_unmarked_rising_p2'});continue
                        features=normalize(x.reshape(64,-1).T[selected]);distance=1-(features@prototypes.T).max(1)
                        increase=after.ravel()[selected]-before.ravel()[selected]
                        cache=destination/'cache'/(patch+'.npz')
                        np.savez_compressed(cache,indices=selected,distance=distance,delta=increase)
                        prior=priors.get(patch);known=prior is not None and prior['prior_data_available']=='True'
                        support=float(prior['cashew_prior_support_score']) if known else None
                        confidence=float(prior['prior_evidence_confidence']) if known else None
                        conflict=float(prior['evidence_conflict_fraction']) if known else None
                        regional=.5+(confidence*(.5-support)*(1-conflict) if known else 0.)
                        rows.append(dict(patch_id=patch,image_path=str(path),cache=str(cache),cache_sha256=sha(cache),valid_pixels=int(valid.sum()),
                                         rising_pixels=len(selected),source_cashew_distance_mean=float(distance.mean()),p2_increase_mean=float(increase.mean()),
                                         p2_before_mean=float(before.ravel()[selected].mean()),p2_after_mean=float(after.ravel()[selected].mean()),
                                         regional_context_rank=regional,prior_available=known,prior_support=support,prior_confidence=confidence))
                        audit.append({'patch_id':patch,'reason':'rising_candidate','candidate_pixels':len(selected)})
                        if (i+1)%100==0:print(json.dumps({'automatic_stage':'catalog','group':self.job['target_domain']['gid'],'epoch':epoch,'processed':i+1,'candidate_patches':len(rows)}),flush=True)
            finally:
                del reference;self.model.train(was_training)
        dump(destination/'candidate_catalog.json',rows);dump(destination/'protection_audit.json',audit)
        identity=dict(previous_checkpoint=str(previous),previous_epoch=previous_epoch,previous_checkpoint_sha256=sha(previous),
                      source_checkpoint=str(checkpoint),source_checkpoint_sha256=sha(checkpoint),
                      prior_index=str(index),prior_index_sha256=sha(index),source_prototypes_sha256=source['prototypes_sha256'],
                      target_train_list_sha256=sha(self.job['target_train_list']),CIV_validation_attributes_read=False,
                      human_marked_pixels_protected=True,heldout_locations_excluded=True,target_evaluation_ids_excluded=True)
        dump(destination/'catalog_identity.json',identity)
        return rows,identity

    def select(self, epoch, destination, rows, identity, review, config):
        raise RuntimeError('Legacy separate negative-selection model call is not permitted; use the explicitly bound joint Cashew review')

    def refresh(self, epoch, checkpoint, review, config):
        from legacy_agent_guard_v1 import block_legacy
        block_legacy(self.job, 'direct_automatic_negative_refresh')
        destination=self.out/('epoch_%03d'%epoch);destination.mkdir(exist_ok=True)
        manifest_path=destination/'manifest.json'
        if manifest_path.exists():
            manifest=read(manifest_path)
            assert manifest['source_checkpoint_sha256']==sha(checkpoint)
        else:
            if (destination/'candidate_catalog.json').exists() and (destination/'catalog_identity.json').exists():
                identity=read(destination/'catalog_identity.json')
                assert identity['source_checkpoint_sha256']==sha(checkpoint)
                assert identity['previous_checkpoint_sha256']==sha(identity['previous_checkpoint'])
                assert identity['target_train_list_sha256']==sha(self.job['target_train_list'])
                assert identity['prior_index_sha256']==sha(identity['prior_index'])
                # Reuse the manifest declared by this job.  The formal cohort
                # intentionally keeps automatic-negative inputs in the pinned
                # legacy source pool; resolving relative to the formal run
                # root incorrectly made every job fail at its first review.
                source=read(self.job['automatic_negative_inputs']['source_feature_manifest'])
                assert identity['source_prototypes_sha256']==sha(source['prototypes'])
                rows=read(destination/'candidate_catalog.json')
                for row in rows:assert row['cache_sha256']==sha(row['cache'])
                print(json.dumps({'automatic_stage':'verified_cached_catalog_reused','epoch':epoch,'candidate_patches':len(rows),'model_rescans':0}),flush=True)
            else:
                rows,identity=self.catalog(epoch,checkpoint,destination)
            if (destination/'selection_decision.json').exists() and (destination/'selection_provenance.json').exists():decision=read(destination/'selection_decision.json')
            else:decision=self.select(epoch,destination,rows,identity,review,config)
            w=decision['ranking_weights']
            rows=sorted(rows,key=lambda r:(-(w['source_cashew_distance']*r['source_cashew_distance_mean']/2+w['p2_increase']*r['p2_increase_mean']+w['regional_prior']*r['regional_context_rank']),r['patch_id']))[:decision['selected_count']]
            selected=[];(destination/'masks').mkdir(exist_ok=True)
            for row in rows:
                assert sha(row['cache'])==row['cache_sha256']
                cache=np.load(row['cache']);limit=min(256,int(.02*row['valid_pixels']))
                if not limit:continue
                score=w['source_cashew_distance']*cache['distance']/2+w['p2_increase']*cache['delta']+w['regional_prior']*row['regional_context_rank']
                indices=cache['indices'][np.argsort(-score,kind='stable')[:limit]]
                mask=np.full((128,128),10,np.uint8);mask.ravel()[indices]=0
                with rasterio.open(row['image_path']) as ds:profile=ds.profile.copy()
                profile.update(count=1,dtype='uint8',nodata=10,compress='deflate')
                p=destination/'masks'/('automatic_noncashew_'+row['patch_id']+'.tif')
                with rasterio.open(p,'w',**profile) as ds:ds.write(mask,1)
                selected.append(dict(patch_id=row['patch_id'],image_path=row['image_path'],image_sha256=sha(row['image_path']),mask_path=str(p),mask_sha256=sha(p),selected_pixels=len(indices)))
            manifest=dict(identity,version='agent_selected_training_pool_rising_p2_uncapped_v3',deployment_policy=VERSION,origin='experimental_automatic_negative',
                          generation_epoch=epoch,schedule_seed=16+epoch*100000,selected=selected,selected_patch_count=len(selected),selected_unique_pixels=sum(r['selected_pixels'] for r in selected),
                          requested_patch_count=decision['selected_count'],agent_selected_weight=decision['automatic_weight'],ranking_weights=w,
                          exclusion_audit_passed=True,source_class_weights_changed=False,patch_count_cap=None,geographic_distribution_quota=False,
                          selection_decision=str(destination/'selection_decision.json'),selection_decision_sha256=sha(destination/'selection_decision.json'),
                          occurrence_cap_per_generation=1,exposure_horizon_updates=128,policy={'maximum_age_epochs':5})
            dump(manifest_path,manifest)
        previous=config.get('_automatic_negative_state',{})
        if previous.get('manifest')==str(manifest_path):state=previous
        else:state={'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),'weight':manifest['agent_selected_weight'],
                    'generation_epoch':epoch,'cursor':0,'selected_patch_count':manifest['selected_patch_count'],'selected_pixels':manifest['selected_unique_pixels']}
        config['_automatic_negative_state']=state;dump(self.out/'state.json',state)
        print(json.dumps({'automatic_stage':'Agent_pool_committed','group':self.job['target_domain']['gid'],'epoch':epoch,'patches':state['selected_patch_count'],'pixels':state['selected_pixels'],'weight':state['weight']}),flush=True)

    def for_epoch(self, config):
        state=config.get('_automatic_negative_state')
        if not state:return None
        return Pool(state['manifest'],state['manifest_sha256'],state['weight'],state['cursor'])

    def after_epoch(self, config, metrics):
        state=config.get('_automatic_negative_state')
        if state and metrics.get('automatic_negative'):
            summary=metrics['automatic_negative'];state['cursor']=summary['generation_end_cursor']
            state['last_epoch_exposure']={k:v for k,v in summary.items() if k not in ['unique_exposed_patch_ids']}
            dump(self.out/'state.json',state)
            print(json.dumps({'automatic_stage':'actual_training_exposure','group':self.job['target_domain']['gid'],'weight':summary['effective_weight'],'updates':summary['updates'],'unique_patches':summary['unique_patches'],'pixels':summary['pixel_occurrences'],'pool_cursor':state['cursor']}),flush=True)


# This isolated user-authorized adapter reuses the miner; no second selection call.
from joint_negative import make_controller
Controller = make_controller(Controller)
