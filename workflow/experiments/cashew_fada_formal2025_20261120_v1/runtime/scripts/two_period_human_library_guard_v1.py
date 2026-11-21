"""Verify year-specific human training-library bindings before use.

A geometry explicitly valid in both periods can be reused with separate
period-matched images and a spatially aligned mask. This check does not infer
label correctness or decide whether a point is independent evaluation data.
The worker must authenticate the manifest and maintain evaluation isolation.
"""
import hashlib,json,re
from pathlib import Path
import rasterio
from human_negative_label_paths import parse_image_name,resolve_label_path

PERIOD={2018:'2018-2019',2019:'2018-2019',2024:'2024-2025',2025:'2024-2025'}

def validate(job,names,image_root,label_root):
    if not names:return {'version':'two_period_human_library_guard_v1','count':0}
    if not isinstance(job,dict) or job.get('target_year') not in PERIOD:raise ValueError('unsupported target image year')
    year=job['target_year'];period=PERIOD[year]
    spec=job.get('period_library_manifest')
    if not isinstance(spec,dict) or set(spec)!={'path','sha256'}:raise ValueError('explicit period library manifest required')
    path=Path(spec['path']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('period library manifest changed')
    manifest=json.loads(raw)
    if not isinstance(manifest,dict) or set(manifest)!={'version','target_year','samples'} or manifest['version']!='two_period_human_library_v1' or manifest['target_year']!=year:
        raise ValueError('library manifest target year mismatch')
    samples=manifest['samples'];required={'name','image_year','valid_periods','image_sha256','mask_sha256','geometry_id'}
    if not isinstance(samples,list) or any(not isinstance(s,dict) or set(s)!=required for s in samples):raise ValueError('invalid period library sample')
    if [s['name'] for s in samples]!=list(names) or len(names)!=len(set(names)):raise ValueError('library names do not match selected training samples')
    image_root=Path(image_root);label_root=Path(label_root)
    for s in samples:
        name=s['name'];parsed_year,_=parse_image_name(name)
        if type(s['image_year']) is not int or parsed_year!=s['image_year'] or PERIOD.get(s['image_year'])!=period:
            raise ValueError('human training image year mismatch')
        periods=s['valid_periods']
        if not isinstance(periods,list) or len(set(periods))!=len(periods) or set(periods)-set(PERIOD.values()) or period not in periods:
            raise ValueError('human label period is unknown or invalid')
        if not isinstance(s['geometry_id'],str) or not s['geometry_id'].strip():raise ValueError('missing label geometry identity')
        image=(image_root/name).resolve(strict=True);mask=resolve_label_path(label_root,name).resolve(strict=True)
        for p,key in ((image,'image_sha256'),(mask,'mask_sha256')):
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=s[key]:raise ValueError('library image or mask changed')
        # The year in the source path is independently checked after resolving
        # symlinks, to reject a 2025 raster renamed as a 2019 sample.
        # The actual filename, after resolving symlinks, identifies the AEF
        # acquisition year. A 2025-period folder may contain 2024 imagery;
        # folder names or surrounding run names cannot override that fact.
        actual_year=s['image_year']
        source_name=re.match(r'^(?:AlphaEarth_|[A-Za-z]+(?:-[A-Za-z]+)*)(20\d{2})_',image.name)
        if not source_name or int(source_name.group(1))!=actual_year:
            raise ValueError('image source path has no matching year')
        with rasterio.open(image) as a,rasterio.open(mask) as b:
            if a.count!=64 or b.count!=1 or a.crs!=b.crs or a.shape!=b.shape or not a.transform.almost_equals(b.transform,precision=1e-9):
                raise ValueError('human image and mask grids do not align')
    return {'version':'two_period_human_library_guard_v1','count':len(samples),'target_year':year,'period':period,'manifest_sha256':spec['sha256']}
