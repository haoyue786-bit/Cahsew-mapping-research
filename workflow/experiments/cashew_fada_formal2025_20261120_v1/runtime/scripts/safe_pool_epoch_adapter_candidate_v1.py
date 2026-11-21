"""Actual train_epoch adapter for a bound weights-only automatic pool.

The production source/catalog/point-exclusion certificate does not exist: the
private certificate hook unconditionally rejects. It has no job flag, boolean
callback or environment-variable bypass. Tests may patch it in a test scope.
This module does not authorize training or publish a checkpoint.
"""
import hashlib,json,os,stat
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rasterio.io import MemoryFile
from automatic_negative_production_v4 import Pool
from automatic_pool_epoch_adapter_sim_v1 import ActualEpochPool
from sealed_safe_final_selection_v1 import SafeRoot
from unified_epoch_pool_bridge_sim_v1 import UnifiedEpochPool

def _certified_pool_binding(job,epoch,checkpoint_sha256,pool):
    raise RuntimeError('automatic-pool source and exclusion certificate unavailable')

def _sha(raw):return hashlib.sha256(raw).hexdigest()

def _limited_image_read(root,path,remaining):
    """Check size BEFORE reading an image and reject all symlink components."""
    parent,name=root._parent(path)
    try:
        try:fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
        except OSError as exc:raise ValueError('unsafe automatic image') from exc
        try:
            before=os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1:raise ValueError('automatic image is not private regular file')
            if before.st_size>remaining:raise ValueError('automatic image read budget exhausted')
            with os.fdopen(os.dup(fd),'rb') as f:raw=f.read(before.st_size+1)
            after=os.fstat(fd);linked=os.stat(name,dir_fd=parent,follow_symlinks=False)
            fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
            if len(raw)!=before.st_size or any(getattr(before,k)!=getattr(after,k) or getattr(after,k)!=getattr(linked,k) for k in fields):
                raise ValueError('automatic image changed during read')
            return raw
        finally:os.close(fd)
    finally:os.close(parent)

class _PinnedPool(Pool):
    """Exact old schedule/loss methods; images/masks read from pinned bytes."""
    def __init__(self,manifest,weight,cursor,read):
        self.manifest=manifest;self.rows=manifest['selected'];self.weight=float(weight)
        self.cursor=cursor;self.schedule={};self._pinned_read=read
    def step(self,step,gan,common,model,device,seed):
        return super().step(step,gan,SimpleNamespace(read_raster=self._pinned_read),model,device,seed)

def run_epoch(bridge,epoch,previous,train_args,train_kwargs):
    """Execute one bound epoch, returning measured metrics/exposure only.

    The caller must hold the bridge lease and persist a complete epoch through
    bridge.finish_training; no cursor is committed here. On exceptions the
    worker must discard the partially mutated in-memory models/optimizers.
    """
    if type(bridge) is not UnifiedEpochPool or type(epoch) is not int or epoch<1 or not isinstance(previous,bytes):
        raise ValueError('invalid pool worker identity')
    budget=bridge.journal.job.get('experimental_catalog_snapshot_budget')
    if (not isinstance(budget,dict) or set(budget)!={'max_images','max_bytes'}
        or any(type(budget[k]) is not int or budget[k]<1 for k in ('max_images','max_bytes'))):
        raise ValueError('explicit automatic image read budget required')
    from safe_pool_checkpoint_codec_sim_v1 import decode
    checkpoint=decode(previous);checkpoint_pool=checkpoint.get('pool')
    if checkpoint['epoch']!=epoch-1 or not isinstance(checkpoint_pool,dict):raise ValueError('pool checkpoint is not preceding epoch')
    # This must remain the first potentially authorizing operation and must
    # precede reading any pool imagery/masks or invoking a model.
    spec=_certified_pool_binding(bridge.journal.job,epoch,_sha(previous),checkpoint_pool)
    if not isinstance(spec,dict) or set(spec)!={'manifest_path','manifest_sha256','image_sha256'}:
        raise ValueError('invalid certified pool binding')
    bridge._guard();status=bridge.recover()
    if status['state']!='TRAIN_REQUIRED' or status['epoch']!=epoch or status['checkpoint']!=previous:
        raise ValueError('pool execution is not bound to current epoch checkpoint')
    state=bridge._cp_state(previous);pool=state['pool']
    if pool!=checkpoint_pool:raise ValueError('certified pool differs from checkpoint')
    if not isinstance(train_args,tuple) or len(train_args)!=17 or not isinstance(train_kwargs,dict):raise ValueError('invalid train_epoch arguments')
    if 'automatic_negative_runtime' in train_kwargs:raise ValueError('caller may not replace pool adapter')
    if not isinstance(spec['image_sha256'],dict) or not spec['image_sha256']:
        raise ValueError('missing certified image identities')
    asset=target=None
    try:
        asset=SafeRoot(bridge.assets);target=SafeRoot(bridge.journal.job['target_root'])
        manifest_rel=asset.rel(spec['manifest_path'])
        if manifest_rel!=pool['generation']+'/legacy_manifest.json':
            raise ValueError('legacy manifest outside single-authority generation')
        manifest_raw=asset.read(manifest_rel)
        if _sha(manifest_raw)!=spec['manifest_sha256']:raise ValueError('legacy manifest changed')
        legacy=json.loads(manifest_raw);generation=pool['generation'];folder=generation+'/masks/'
        canonical=json.loads(asset.read(generation+'/manifest.json'))
        if (_sha(asset.read(generation+'/manifest.json'))!=pool['manifest_sha256'] or canonical.get('generation')!=generation
            or canonical.get('horizon')!=pool['horizon'] or canonical.get('weight')!=pool['weight']):
            raise ValueError('bridge generation manifest mismatch')
        if (legacy.get('version')!='agent_selected_training_pool_rising_p2_uncapped_v3'
            or legacy.get('origin')!='experimental_automatic_negative' or legacy.get('exclusion_audit_passed') is not True
            or type(legacy.get('schedule_seed')) is not int or type(legacy.get('generation_epoch')) is not int):
            raise ValueError('invalid distinct automatic generation')
        rows=legacy.get('selected');masks=canonical.get('masks')
        if not isinstance(rows,list) or not rows or not isinstance(masks,dict) or len(rows)!=len(masks):
            raise ValueError('selection and mask count differ')
        mappings={};seen=set();seen_images=set()
        for row in rows:
            if not isinstance(row,dict):raise ValueError('invalid pool row')
            mask_rel=asset.rel(row.get('mask_path',''));image_rel=target.rel(row.get('image_path',''))
            if not mask_rel.startswith(folder) or '/' in mask_rel[len(folder):]:raise ValueError('mask outside generation')
            name=mask_rel[len(folder):]
            if name in seen or masks.get(name)!=row.get('mask_sha256'):raise ValueError('mask manifest mismatch')
            if image_rel not in spec['image_sha256']:raise ValueError('uncertified automatic image')
            mask_key=str(Path(row['mask_path']));image_key=str(Path(row['image_path']))
            if image_rel in seen_images or mask_key==image_key or mask_key in mappings or image_key in mappings:
                raise ValueError('duplicate or ambiguous pool raster identity')
            seen.add(name);seen_images.add(image_rel);mappings[mask_key]=(asset,mask_rel,row['mask_sha256'],False)
            mappings[image_key]=(target,image_rel,spec['image_sha256'][image_rel],True)
        if seen!=set(masks):raise ValueError('pool contains unselected masks')
        if seen_images!=set(spec['image_sha256']) or len(seen_images)>budget['max_images']:
            raise ValueError('automatic image count or identity exceeds budget')
        usage={'bytes':0,'reads':0}
        def read_checked(key):
            if key not in mappings:raise ValueError('automatic raster not in certified selection')
            root,path,want,is_image=mappings[key]
            if is_image:
                raw=_limited_image_read(root,path,budget['max_bytes']-usage['bytes'])
                usage['bytes']+=len(raw);usage['reads']+=1
            else:raw=root.read(path)
            if _sha(raw)!=want:raise ValueError('automatic raster changed')
            return raw
        def check_assets():
            for key in mappings:read_checked(key)
            return True
        def read_raster(path,dtype=np.float32):
            raw=read_checked(str(Path(path)))
            with MemoryFile(raw) as memory:
                with memory.open() as ds:array=ds.read()
            if np.issubdtype(np.dtype(dtype),np.floating):array=np.nan_to_num(array,nan=0.,posinf=0.,neginf=0.)
            return array.astype(dtype,copy=False)
        check_assets()
        inner=_PinnedPool(legacy,pool['weight'],pool['cursor'],read_raster)
        adapter=ActualEpochPool(inner,cursor=pool['cursor'],asset_check=check_assets)
        import train_expert_v4 as worker
        with adapter.bind_generator(train_args[5]):
            metrics=worker.train_epoch(*train_args,automatic_negative_runtime=adapter,**train_kwargs)
        measured=metrics.get('automatic_negative')
        if not isinstance(measured,dict):raise ValueError('pool exposure missing')
        metrics['automatic_pool_resource_exposure']={'version':'automatic_pool_image_read_budget_v1',
          'unique_images':len(seen_images),'encoded_image_read_calls':usage['reads'],'encoded_image_bytes_read':usage['bytes'],
          'meaning':'includes repeated verification/epoch image reads; excludes catalog construction and decoded-array memory'}
        exposure={'mode':'paused' if not measured['effective_weight'] else 'active',
          'positions':measured['processed_contiguous_positions'],'completed_g':measured['completed_G_updates'],
          'auxiliary_steps':measured['completed_G_auxiliary_steps'],'patch_occurrences':measured['completed_G_patch_occurrences'],
          'pixel_occurrences':measured['completed_G_pixel_occurrences']}
        return metrics,exposure
    finally:
        if target is not None:target.close()
        if asset is not None:asset.close()
