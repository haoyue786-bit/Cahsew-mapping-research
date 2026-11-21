"""Expose every Agent-selected training patch; no patch-count truncation.

Microbatching controls memory only. The finite training-sample population is the
natural bound; no geographic quota or arbitrary pool-size ceiling is imposed.
"""
import json
from pathlib import Path
import numpy as np
from automatic_negative_v1 import loss
from human_polygon_v1 import frozen_auxiliary_bn
import supplementary_control_v4 as human_control
from automatic_negative_pool_agent_v2 import AutomaticNegativePool as BasePool
from automatic_negative_pool_agent_v2 import sha

VERSION='agent_selected_training_pool_rising_p2_uncapped_v3'

class AutomaticNegativePool(BasePool):
    def __init__(self,manifest_path,expected_sha256,weight=0.):
        if not 0<=weight<=.05:raise ValueError('Expected independent low automatic dose')
        p=Path(manifest_path)
        if sha(p)!=expected_sha256:raise ValueError('Pool identity changed')
        self.manifest=json.loads(p.read_text())
        if self.manifest['version']!=VERSION or self.manifest['origin']!='experimental_automatic_negative':raise ValueError('Distinct automatic origin required')
        assert self.manifest['exclusion_audit_passed']
        self.rows=self.manifest['selected'];self.weight=float(weight)
        for row in self.rows:assert sha(row['mask_path'])==row['mask_sha256']
        self.schedule={}

    def begin_epoch(self,steps,config,seed):
        age=float(config.get('_negative_epoch_start',0))-float(self.manifest['generation_epoch']);assert age>=0
        self.exposed=[];self.pixels=0;self.losses=[];self.probes=[];self.age=age
        self.active_weight=self.weight if age<5 and not config.get('_automatic_source_only_repair') else 0.
        rng=np.random.RandomState((seed+700001)%(2**32-1));order=rng.permutation(len(self.rows)) if self.active_weight else []
        # Preserve cadence4 for small pools. When more patches are selected,
        # distribute all of them over available optimizer updates; never slice
        # away patches because of a hidden slot/batch-size ceiling.
        slots=list(range(0,steps,4))
        if len(order)>len(slots)*4:slots=list(range(steps))
        if len(order):
            batches=[list(a) for a in np.array_split(order,min(len(slots),max(1,int(np.ceil(len(order)/4)))))]
            selected_slots=np.rint(np.linspace(0,len(slots)-1,len(batches))).astype(int)
            self.schedule={slots[int(slot)]:batch for slot,batch in zip(selected_slots,batches)}
        else:self.schedule={}
        flattened=[i for batch in self.schedule.values() for i in batch]
        assert len(flattened)==len(order) and len(set(flattened))==len(order)

    def step(self,step,gan,common,model,device,seed):
        import torch
        indices=self.schedule.get(step,[])
        if not indices:return None
        values=[];counts=[]
        for offset in range(0,len(indices),4):
            micro=indices[offset:offset+4];images=[];labels=[]
            for index in micro:
                row=self.rows[index];images.append(common.read_raster(Path(row['image_path']),dtype=np.float32))
                a=common.read_raster(Path(row['mask_path']),dtype=np.int64);labels.append(a[0] if a.ndim==3 else a)
            x=torch.from_numpy(np.stack(images)).to(device);y=torch.from_numpy(np.stack(labels)).to(device)
            with human_control.rng_scope(np.random.RandomState(seed),seed+700002+step*1001+offset,device):
                with frozen_auxiliary_bn(model):logits,_=gan.forward_fuse(model,x.float())
            raw,stats=loss(logits,y,x);values.append(raw);counts.append(stats['selected_pixels'])
        total=sum(counts)
        raw=sum(value*count for value,count in zip(values,counts))/total if total else sum(values)*0
        self.exposed.extend(indices);self.pixels+=total;self.losses.append(float(raw.detach()))
        return self.active_weight*raw

    def summary(self):
        summary=super().summary();summary.update(version=VERSION,all_selected_patches_scheduled=True,selected_patch_count=len(self.rows),
            patch_count_cap=None,geographic_distribution_quota=False,microbatch_size_is_memory_control_only=True)
        return summary
