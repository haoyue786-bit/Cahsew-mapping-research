"""AEF tile descriptors and stable geographic ordering."""

import csv
import json
from pathlib import Path

import numpy as np
from pyproj import Geod
import rasterio


GEOD = Geod(ellps='WGS84')


def dump(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def write_csv(p, rows):
    with Path(p).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def descriptor(records):
    n=sum(x['pixels'] for x in records)
    assert n>0, 'No valid feature pixels in stratified sample'
    mean=sum(x['sum'] for x in records)/n
    sd=np.sqrt(np.maximum(sum(x['sum2'] for x in records)/n-mean*mean, 0))
    return dict(mean=mean.tolist(),sd=sd.tolist(),pixels=n,
                valid_patches=sum(x['pixels']>0 for x in records))

def extract(item, output):
    cache=output/'features'/(item['tile_id']+'.json')
    if cache.exists():
        result=json.loads(cache.read_text());assert result['directory']==item['path']
        return result
    samples=json.loads(item['header_samples_json'])
    corners=[Path(s['file']).stem.rsplit('_',2) for s in samples]
    prefix=corners[0][0]
    maxc=max(int(x[1]) for x in corners);maxr=max(int(x[2]) for x in corners)
    cols=np.unique(np.rint(np.linspace(0,maxc,6)).astype(int))
    rows=np.unique(np.rint(np.linspace(0,maxr,6)).astype(int))
    records=[];manifest=[]
    for ir,row in enumerate(rows):
        for ic,col in enumerate(cols):
            p=Path(item['path'])/(prefix+'_%d_%d.tif'%(col,row))
            with rasterio.open(p) as ds:
                assert ds.count==64 and ds.width==128 and ds.height==128
                values=ds.read().astype(np.float64)
            valid=np.isfinite(values).all(0)&np.any(values!=0,axis=0)
            values=values[:,valid]
            record=dict(pixels=int(valid.sum()),sum=values.sum(1),sum2=(values*values).sum(1),half=(ic+ir)%2)
            records.append(record)
            manifest.append(dict(path=str(p),col=int(col),row=int(row),half=record['half'],valid_pixels=record['pixels']))
    complete=descriptor(records)
    halves=[descriptor([x for x in records if x['half']==i]) for i in [0,1]]
    b=[x['bounds'] for x in samples]
    bounds=[min(x[0] for x in b),min(x[1] for x in b),max(x[2] for x in b),max(x[3] for x in b)]
    result=dict(tile_id=item['tile_id'],directory=item['path'],bounds=bounds,
        center=[(bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2],
        full=complete,halves=halves,samples=manifest,
        descriptor='Valid-pixel weighted64-band mean and standard deviation; full 128x128 sample patches.')
    dump(cache,result)
    return result

def feature_vectors(features, half=None):
    d=[x['full'] if half is None else x['halves'][half] for x in features]
    mean=np.asarray([x['mean'] for x in d]);sd=np.asarray([x['sd'] for x in d])
    mean/=np.maximum(np.linalg.norm(mean,axis=1,keepdims=True),1e-12)
    sd/=np.maximum(np.linalg.norm(sd,axis=1,keepdims=True),1e-12)
    return np.concatenate([np.sqrt(.8)*mean,np.sqrt(.2)*sd],axis=1)

def stable_ids(labels, centers):
    order=sorted(set(labels),key=lambda k:(-centers[labels==k,1].mean(),centers[labels==k,0].mean()))
    return np.asarray([order.index(v)+1 for v in labels])
