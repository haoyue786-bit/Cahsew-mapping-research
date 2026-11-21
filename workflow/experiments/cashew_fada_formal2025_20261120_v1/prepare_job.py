"""Register a fresh job from an existing imagery grouping and source profile."""

import argparse
from collections import defaultdict
import csv
import fcntl
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT/'runtime/scripts'))
from hard_sample_training import VERSION, save, sha

COUNTRY_CODES = dict(Benin='BEN', Togo='TGO', Ghana='GHA', Ivoire='CIV', Cote_d_Ivoire='CIV',
                     Nigeria='NGA', Senegal='SEN', Guinea='GIN', Mali='MLI',
                     Burkina_Faso='BFA', Gambia='GMB', Guinea_Bissau='GNB')
COUNTRY_CODES.update({name.replace('_','-'):code for name,code in tuple(COUNTRY_CODES.items())})


def rebuild_registration(old, workspace, job, replace, local):
    registration = replace(json.loads((old/'panel_registration_candidate.json').read_text()))
    registration.update(run_id=job['run_id'],job_id=job['job_id'])
    for rec in registration['artifacts'].values():
        target = workspace/rec['path']; target.parent.mkdir(exist_ok=True,parents=True)
        target.write_text(replace((old/rec['path']).read_text()))
    panel = job['target_evaluation']
    for key in ('names','locations_csv'):
        field = 'names_file' if key == 'names' else key
        panel[key+'_sha256'] = sha(local(panel[field]))
    panel_report = local(panel['report']); report = json.loads(panel_report.read_text())
    report.update(names_sha256=panel['names_sha256'], locations_csv_sha256=panel['locations_csv_sha256'])
    panel_report.write_text(json.dumps(report)+'\n')
    panel['report_sha256'] = sha(panel_report)
    for rec in registration['artifacts'].values():
        target = workspace/rec['path']; st = target.stat()
        rec.update(sha256=sha(target),dev=st.st_dev,ino=st.st_ino,size=st.st_size,max_bytes=max(1,st.st_size))
    path = workspace/'panel_registration_candidate.json'; save(path,registration)
    job['panel_registration_sha256'] = sha(path)


def protect_geometry(countries, years):
    """Read shape geometry outside the worker; never open the point attribute table."""
    from pyproj import CRS, Transformer
    import shapefile
    by_degree = defaultdict(list); sources = []
    for code in sorted(countries):
        for year in years:
            root = Path('@ASSETS@/country_shp/data')/str(year)/code
            paths = sorted(root.glob(f'*_{year}_4class.shp'))
            if len(paths) != 1:
                raise ValueError('Expected one authoritative point geometry layer at ' + str(root))
            shape = paths[0]; projection = shape.with_suffix('.prj')
            transform = Transformer.from_crs(CRS.from_wkt(projection.read_text()), 'EPSG:4326', always_xy=True)
            with shapefile.Reader(shp=str(shape),shx=str(shape.with_suffix('.shx'))) as geometry:
                for item in geometry.iterShapes():
                    if item.shapeType != shapefile.POINT or len(item.points) != 1:
                        raise ValueError('Unexpected protected layer geometry type')
                    x,y = transform.transform(*item.points[0][:2]); by_degree[(int(x//1),int(y//1))].append((x,y))
            sources.append(dict(country=code,year=year,geometry_sha256=sha(shape),projection_sha256=sha(projection)))
    return by_degree,sources


def acquisition_frame(job, size, local, protection=None):
    import math
    root = Path(job['target_root']).resolve(); domain = job['target_domain']
    protected_ids = set(job.get('protected_patch_ids') or [])
    allowed = set(local(job['target_full_list']).read_text().splitlines())
    path = local(domain['patch_csv'])
    if domain.get('patch_csv_sha256') and sha(path) != domain['patch_csv_sha256']:
        raise ValueError('Imagery group footprint table changed')
    with path.open(encoding='utf-8-sig') as stream:
        rows = []
        for row in csv.DictReader(stream):
            tif = Path(row['tif_path'])
            if not tif.is_absolute():
                tif = root/tif
            name = tif.relative_to(root).as_posix()
            if name not in allowed or row['patch_id'] in protected_ids:
                continue
            if '..' in Path(name).parts or len(Path(name).parts) != 4:
                raise ValueError('Unexpected target imagery path')
            country = COUNTRY_CODES[Path(name).parts[1]]
            rows.append((row,name,tif,country))
    countries = {country for _,_,_,country in rows}
    blocks, sources = protection or protect_geometry(countries,(job['target_year'],))
    tiles = defaultdict(list); excluded = 0
    for row,name,tif,country in rows:
        w,s,e,n = (float(row[k]) for k in ('west','south','east','north'))
        if not all(math.isfinite(v) for v in (w,s,e,n)) or not (w<e and s<n):
            raise ValueError('Invalid chip footprint')
        # Include the edge and a small coordinate-tolerance margin. Tile membership is
        # only a frame; the underlying geographic footprint controls the exclusion.
        delta = .00012
        if any(w-delta <= x <= e+delta and s-delta <= y <= n+delta
               for i in range(math.floor(w-delta),math.floor(e+delta)+1)
               for j in range(math.floor(s-delta),math.floor(n+delta)+1) for x,y in blocks.get((i,j),())):
            excluded += 1; continue
        tiles[row['tile_id']].append(dict(tile=row['tile_id'],path=name,aef=str(tif),country=country,
                                         lon=(w+e)/2,lat=(s+n)/2,bounds=[w,s,e,n]))
    if not tiles:
        raise ValueError('No acquisition imagery remains after protection mask')
    rng = random.Random(int(job['seed'])); result=[]
    for tile, records in sorted(tiles.items()):
        # Take evenly spaced quantiles across the tile and randomly sample within each.
        records.sort(key=lambda r:(round(r['lat'],2),r['lon'],r['path']))
        requested = min(len(records), max(8,size//len(tiles)))
        for i in range(requested):
            result.append(records[rng.randrange(i*len(records)//requested,(i+1)*len(records)//requested)])
    return result,dict(excluded_overlap=excluded,excluded_patch_ids=len(protected_ids),
                       geometry_only=True,protection_sources=sources)


def prepare(old, name, backend, max_epoch, size):
    old = Path(old).resolve(strict=True)
    original = json.loads((old/'job_candidate.json').read_text())
    if (original.get('point_validation_online') is not False or original['target_year'] not in (2019,2025)
            or original.get('deep_review_interval') != 5):
        raise ValueError('Expected a registered five-epoch point-isolated four-class template')
    if backend == 'simulated' and max_epoch < 10:
        raise ValueError('Simulation exercises two distinct five-epoch checkpoints')
    parent = ROOT/('verification/jobs' if backend == 'simulated' else 'jobs')
    parent.mkdir(parents=True,exist_ok=True)
    work = parent/name
    with (parent/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if work.exists():
            raise FileExistsError('A registered path already exists: '+str(work))
        with TemporaryDirectory(prefix='.'+name+'-',dir=str(parent)) as temporary:
            workspace = Path(temporary)
            job,frame,protection = prepare_files(old,work,workspace,original,backend,max_epoch,size)
            save(workspace/'job_candidate.json',job)
            workspace.rename(work)
    print(json.dumps(dict(job=str(work),backend=backend,scanned_patches=len(frame),max_epoch=max_epoch,
                          geometry_exclusions=protection['excluded_overlap'])),flush=True)
    return work


def prepare_files(old, work, workspace, original, backend, max_epoch, size):
    def local(value):
        p = Path(value)
        try:
            return workspace/p.relative_to(work)
        except ValueError:
            return p
    def replace(v):
        if isinstance(v,dict):return {k:replace(x) for k,x in v.items()}
        if isinstance(v,list):return [replace(x) for x in v]
        if isinstance(v,str):return v.replace(str(old),str(work))
        return v
    job = replace(original)
    job.update(job_id=work.name,run_id=ROOT.name+'/'+backend+'/'+work.name, output_root=str(work),max_epoch=max_epoch)
    job['hard_sample_policy'] = dict(version=VERSION,annotation_backend=backend)
    job['supplementary'] = dict(enabled=False,count=0,mode='none',loss_weight=0.)
    # Reuse the registered imagery layout and independently registered pretraining,
    # with fresh optimizer/RNG/decision history and no old target-trained lineage.
    for key in ('initial_transfer_weights_v3','initialization_policy','transfer_source','transfer_regularization_policy',
                'supplementary_default_policy','period_library_manifest','negative_control_policy',
                'automatic_negative_policy',
                'polygon_supervision_policy','conditional_negative_policy','negative_response_policy',
                'human_training_evidence_policy','formal_release_identity','release_note'):
        job.pop(key,None)
    job['geographic_prior'] = {'card':str(work/'no_online_prior.md')}
    (workspace/'no_online_prior.md').write_text('No geographic prior supplied to online decisions.\n')
    for key in ('patch_csv','sampling_report'):
        value = original['target_domain'].get(key)
        if not value or value == job['target_domain'].get(key):
            continue
        source = Path(value)
        digest = original['target_domain'].get(key+'_sha256')
        if digest and sha(source) != digest:
            raise ValueError('Template group data changed: '+key)
        copy = local(job['target_domain'][key])
        copy.parent.mkdir(parents=True,exist_ok=True)
        copy.write_text(replace(source.read_text()))
        job['target_domain'][key+'_sha256'] = sha(copy)
    rebuild_registration(old,workspace,job,replace,local)
    frame, protection = acquisition_frame(job,size,local)
    path = workspace/'hard_sample_frame.json'; save(path,frame)
    job['hard_sample_frame'] = dict(path=str(work/path.name),sha256=sha(path))
    save(workspace/'hard_sample_protection_receipt.json',protection)
    return job,frame,protection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('template', type=Path)
    parser.add_argument('--name', required=True)
    parser.add_argument('--backend', default='simulated', choices=('simulated','sample_assist'))
    parser.add_argument('--max-epoch', type=int)
    parser.add_argument('--acquisition-patches',type=int,default=360)
    args = parser.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,90}',args.name):
        parser.error('Use a unique simple group name')
    maximum = args.max_epoch or (10 if args.backend == 'simulated' else 200)
    if not (5 <= maximum <= 200 and maximum % 5 == 0) or args.acquisition_patches < 8:
        parser.error('Use a positive five-epoch budget, up to 200, and at least eight candidate patches')
    prepare(args.template,args.name,args.backend,maximum,args.acquisition_patches)
