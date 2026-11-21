"""Fixed geographically registered target samples and tile-weighted distribution estimates."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

VERSION='fixed_tile_stratified_evaluation_v1'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def allocate(groups,budget,minimum=8):
    if budget<len(groups):raise ValueError('Evaluation budget cannot cover every tile')
    budget=min(budget,sum(len(v['eligible']) for v in groups.values()))
    quotas={t:min(minimum,len(v['eligible'])) for t,v in groups.items()}
    if sum(quotas.values())>budget:quotas={t:1 for t in groups}
    population=sum(v['population'] for v in groups.values())
    desired={t:budget*v['population']/population for t,v in groups.items()}
    while sum(quotas.values())<budget:
        available=[t for t,v in groups.items() if quotas[t]<len(v['eligible'])]
        if not available:raise ValueError('Cannot fill sample budget')
        t=max(sorted(available),key=lambda t:desired[t]-quotas[t]);quotas[t]+=1
    return quotas


def build(job,destination,ratio=3,seed=916):
    """Draw once from the declared ADM1 CSV and fixed full/training lists."""
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
    full=[x.strip() for x in Path(job['target_full_list']).read_text().splitlines() if x.strip()]
    training={x.strip() for x in Path(job['target_train_list']).read_text().splitlines() if x.strip()}
    if len(full)!=len(set(full)) or not training.issubset(full):raise ValueError('Invalid sampling frame')
    root=Path(job['target_root']);csv_path=Path(job['target_domain']['patch_csv'])
    with csv_path.open(encoding='utf-8-sig') as f:spatial={r['patch_id']:r for r in csv.DictReader(f)}
    groups=defaultdict(lambda:dict(population=0,eligible=[]))
    for name in sorted(full):
        identity='_'.join(Path(name).stem.rsplit('_',3)[1:]);row=spatial[identity];tile=row['tile_id']
        groups[tile]['population']+=1
        if name not in training and identity not in set(job.get('protected_patch_ids',[])):groups[tile]['eligible'].append((identity,name,row))
    # Tiny domains may have a tile fully consumed by training. Register that
    # overlap explicitly rather than silently deleting the tile from estimates.
    for tile,g in groups.items():
        if not g['eligible']:
            for name in sorted(training):
                identity='_'.join(Path(name).stem.rsplit('_',3)[1:]);row=spatial[identity]
                if row['tile_id']==tile:g['eligible'].append((identity,name,row))
    budget=min(max(512,int(np.ceil(ratio*len(training)))),sum(len(g['eligible']) for g in groups.values()))
    quotas=allocate(groups,budget);rng=np.random.RandomState(seed);rows=[];tiles=[]
    for tile,g in sorted(groups.items()):
        n=quotas[tile];weight=g['population']/n
        for ix in rng.choice(len(g['eligible']),n,replace=False):
            identity,name,location=g['eligible'][int(ix)]
            west,south,east,north=[float(location[x]) for x in ['west','south','east','north']]
            rows.append(dict(adm1_id=job['target_domain']['gid'],adm1_name=job['target_domain']['name'],year=job['target_year'],
                tile_id=tile,patch_id=identity,relative_tif=name,tif_path=str(root/name),center_lon=(west+east)/2,center_lat=(south+north)/2,
                west=west,south=south,east=east,north=north,population_tile_patches=g['population'],sampled_tile_patches=n,
                expansion_weight=weight,overlaps_training=name in training,source_csv_path=str(csv_path),
                prior_card_path=job['geographic_prior']['card']))
        tiles.append(dict(tile_id=tile,population_count=g['population'],eligible_count=len(g['eligible']),sample_count=n,expansion_weight=weight))
    rows.sort(key=lambda r:(r['tile_id'],r['patch_id']))
    listing=destination/'evaluation_samples_absolute.csv'
    with listing.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,[k for k in rows[0] if k!='relative_tif'],extrasaction='ignore');w.writeheader();w.writerows(rows)
    names_path=destination/'target_evaluation_fixed.txt';names_path.write_text('\n'.join(r['relative_tif'] for r in rows)+'\n')
    report=dict(version=VERSION,seed=seed,requested_ratio=ratio,training_count=len(training),population_count=len(full),sample_count=len(rows),
        tile_count=len(tiles),overlap_training_count=sum(r['overlaps_training'] for r in rows),tile_allocation=tiles,
        allocation='Proportional tile quotas with minimum 8 where available; uniform random without replacement within tile, no prior-based weighting',
        estimate='sum_h (N_h/n_h)*Class2_valid_pixels_h / sum_h (N_h/n_h)*valid_pixels_h',
        estimate_limits='Ratio estimate of prediction distribution, not target accuracy or a full census; training exclusion and spatial dependence limit representativeness.',
        boundary_policy='Same complete boundary-crossing patches as the declared target footprint',
        population_list_sha256=file_hash(job['target_full_list']),training_list_sha256=file_hash(job['target_train_list']),
        names_file=str(names_path),names_sha256=file_hash(names_path),locations_csv=str(listing),locations_csv_sha256=file_hash(listing))
    report_path=destination/'evaluation_sampling_report.json';report_path.write_text(json.dumps(report,indent=2)+'\n')
    return dict(version=VERSION,names_file=str(names_path),names_sha256=file_hash(names_path),locations_csv=str(listing),
        locations_csv_sha256=file_hash(listing),report=str(report_path),report_sha256=file_hash(report_path),sample_count=len(rows),
        population_count=len(full),tile_count=len(tiles),seed=seed,scope='fixed_tile_stratified_sample_valid_pixels',estimated_population_distribution=True)


class EvaluationNames(list):
    def __init__(self,names,policy,locations,report):
        super().__init__(names);self.policy=policy;self.locations=locations;self.report=report


def names_for_job(job):
    policy=job.get('target_evaluation')
    if not policy:return [x.strip() for x in Path(job['target_full_list']).read_text().splitlines() if x.strip()]
    if policy.get('version')!=VERSION:raise ValueError('Unsupported evaluation sample version')
    for path_key,hash_key in [('names_file','names_sha256'),('locations_csv','locations_csv_sha256'),('report','report_sha256')]:
        if file_hash(policy[path_key])!=policy[hash_key]:raise ValueError('Evaluation sample artifact changed: '+path_key)
    names=[x.strip() for x in Path(policy['names_file']).read_text().splitlines() if x.strip()]
    with Path(policy['locations_csv']).open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for row in rows:row['relative_tif']=str(Path(row['tif_path']).relative_to(job['target_root']))
    report=json.loads(Path(policy['report']).read_text())
    if len(names)!=policy['sample_count'] or len(set(names))!=len(names) or names!=[r['relative_tif'] for r in rows]:raise ValueError('Evaluation list/locations mismatch')
    if file_hash(job['target_full_list'])!=report['population_list_sha256'] or file_hash(job['target_train_list'])!=report['training_list_sha256']:raise ValueError('Evaluation frame changed')
    full=set(Path(job['target_full_list']).read_text().splitlines())
    if not set(names).issubset(full):raise ValueError('Evaluation outside target footprint')
    populations=defaultdict(int)
    for name in full:populations[Path(name).stem.rsplit('_',3)[1]]+=1
    strata=defaultdict(list)
    for row in rows:
        if row['tile_id']!=Path(row['relative_tif']).stem.rsplit('_',3)[1]:raise ValueError('Tile identity mismatch')
        if not Path(row['tif_path']).is_absolute() or Path(row['tif_path'])!=Path(job['target_root'])/row['relative_tif']:raise ValueError('Invalid absolute sample image path')
        strata[row['tile_id']].append(row)
    if set(strata)!=set(populations):raise ValueError('Evaluation must cover every population tile')
    for tile,items in strata.items():
        for row in items:
            if int(row['population_tile_patches'])!=populations[tile] or int(row['sampled_tile_patches'])!=len(items):raise ValueError('Invalid sample/population count')
            if not np.isclose(float(row['expansion_weight']),populations[tile]/len(items)):raise ValueError('Invalid expansion weight')
    return EvaluationNames(names,policy,rows,report)


def measure(base,gan,common,model,target_root,names,device,batch_size=16,return_argmax=False):
    if not isinstance(names,EvaluationNames):return base(gan,common,model,target_root,names,device,batch_size=batch_size,return_argmax=return_argmax)
    strata=defaultdict(list)
    for row in names.locations:strata[row['tile_id']].append(row)
    counts=np.zeros(4,np.float64);raw_counts=np.zeros(4,np.int64);prob=np.zeros(4,np.float64);pixels=0.;raw_pixels=0;entropy=0.;codes=[];tile_results=[]
    for tile,rows in sorted(strata.items()):
        try:
            metric=base(gan,common,model,target_root,[r['relative_tif'] for r in rows],device,batch_size=batch_size,return_argmax=return_argmax)
        except RuntimeError as exc:
            if str(exc)!='target prediction metric found no valid support pixels':raise
            tile_results.append(dict(tile_id=tile,sample_patch_count=len(rows),population_patch_count=int(rows[0]['population_tile_patches']),sampled_valid_pixels=0,class2_fraction=None,reason='all_sampled_input_pixels_invalid'))
            continue
        weight=float(rows[0]['expansion_weight']);n=metric['pixel_count'];weighted=weight*n
        counts+=weight*np.asarray(metric['predicted_class_counts']);raw_counts+=np.asarray(metric['predicted_class_counts'])
        pixels+=weighted;raw_pixels+=n;prob+=weighted*np.asarray(metric['mean_softmax_distribution']);entropy+=weighted*metric['entropy']
        tile_results.append(dict(tile_id=tile,sample_patch_count=len(rows),population_patch_count=int(rows[0]['population_tile_patches']),
            expansion_weight=weight,sampled_valid_pixels=n,class2_fraction=metric['class_distribution'][2],
            sample_bounds_wgs84=[min(float(r['west']) for r in rows),min(float(r['south']) for r in rows),max(float(r['east']) for r in rows),max(float(r['north']) for r in rows)]))
        if return_argmax:codes.append(metric['_argmax_codes'])
    if pixels<=0:raise RuntimeError('Evaluation sample found no valid support pixels')
    result=dict(class_distribution=(counts/pixels).tolist(),class_distribution_kind='tile_weighted_hard_argmax_population_ratio_estimate',
        predicted_class_counts=raw_counts.tolist(),pixel_count=raw_pixels,patch_count=len(names),inference_batch_size=batch_size,
        scope='fixed_tile_stratified_sample_valid_pixels',model_mode='eval',support_definition='pixels where not all input bands are zero',
        mean_softmax_distribution=(prob/pixels).tolist(),entropy=entropy/pixels,normalized_entropy=entropy/pixels/float(np.log(4)),
        estimated_population_class_counts=counts.tolist(),estimated_population_valid_pixels=pixels,
        sample_unweighted_class_distribution=(raw_counts/raw_pixels).tolist(),estimated_population_distribution=True,
        evaluation_sample=names.policy,sampling_limits=names.report['estimate_limits'],tile_estimates=tile_results,
        spatial_stability_scope='Fixed sampled pixels only; temporal agreement and IoU are descriptive sample consistency, not whole-region verification')
    if return_argmax:result['_argmax_codes']=np.concatenate(codes)
    return result
