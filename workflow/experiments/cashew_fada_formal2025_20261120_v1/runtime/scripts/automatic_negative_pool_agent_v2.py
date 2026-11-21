"""Hash-pinned, age-limited auxiliary pool used only by the controlled trial."""
import hashlib
import json
from pathlib import Path
import numpy as np
from automatic_negative_v1 import CandidatePolicy, loss
VERSION='agent_ranked_source_cashew_distance_rising_p2_v2'
from human_polygon_v1 import frozen_auxiliary_bn
import supplementary_control_v4 as human_control

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class AutomaticNegativePool:
    def __init__(self, manifest_path, expected_sha256, weight=0.):
        if not 0<=weight<=CandidatePolicy().maximum_weight: raise ValueError('Unbounded automatic dose')
        p=Path(manifest_path)
        if sha(p)!=expected_sha256: raise ValueError('Automatic pool identity changed')
        self.manifest=json.loads(p.read_text())
        if self.manifest['version']!=VERSION or self.manifest['origin']!='experimental_automatic_negative':
            raise ValueError('Human/automatic label routes must remain separate')
        if not self.manifest['exclusion_audit_passed']: raise ValueError('Pool leakage audit missing')
        self.rows=self.manifest['selected'];self.weight=float(weight)
        for r in self.rows:
            if sha(r['mask_path'])!=r['mask_sha256']: raise ValueError('Automatic mask identity changed')
        self.schedule={}

    def begin_epoch(self, steps, config, seed):
        from collections import Counter
        age=float(config.get('_negative_epoch_start',0))-float(self.manifest['generation_epoch'])
        if age<0: raise ValueError('Automatic labels generated from a future checkpoint')
        self.exposed=[];self.pixels=0;self.losses=[];self.probes=[]
        self.active_weight=self.weight if age<CandidatePolicy().maximum_age_epochs else 0.
        if config.get('_automatic_source_only_repair'): self.active_weight=0.
        rng=np.random.RandomState((seed+700001)%(2**32-1))
        slots=list(range(0,steps,CandidatePolicy().interval_steps))
        order=rng.permutation(len(self.rows)) if self.active_weight else []
        batches=[list(order[i:i+4]) for i in range(0,len(order),4)][:len(slots)]
        selected_slots=np.rint(np.linspace(0,len(slots)-1,len(batches))).astype(int) if batches else []
        self.schedule={slots[int(i)]:batch for i,batch in zip(selected_slots,batches)}
        assert max(Counter(v for batch in batches for v in batch).values(),default=0)<=1
        self.age=age

    def step(self, step, gan, common, model, device, seed):
        import torch
        indices=self.schedule.get(step,[])
        if not indices: return None
        images=[];labels=[]
        for index in indices:
            r=self.rows[index]
            images.append(common.read_raster(Path(r['image_path']),dtype=np.float32))
            a=common.read_raster(Path(r['mask_path']),dtype=np.int64)
            labels.append(a[0] if a.ndim==3 else a)
        x=torch.from_numpy(np.stack(images)).to(device)
        y=torch.from_numpy(np.stack(labels)).to(device)
        # Restore all random streams and frozen BN moments after the extra forward.
        with human_control.rng_scope(np.random.RandomState(seed),seed+700002+step,device):
            with frozen_auxiliary_bn(model): logits,_=gan.forward_fuse(model,x.float())
        raw,stats=loss(logits,y,x)
        self.exposed.extend(indices);self.pixels+=stats['selected_pixels'];self.losses.append(float(raw.detach()))
        return self.active_weight*raw

    def record_gradients(self, model, source, automatic, adversarial, coefficient, step):
        self.probes.append(dict(step=step,component='automatic_only',
            measurements=human_control.component_gradients(model,source,automatic,adversarial,coefficient)))

    def summary(self):
        from collections import Counter
        return dict(version=VERSION,origin='experimental_automatic_negative',effective_weight=self.active_weight,
            pool_age_epochs=self.age,updates=len(self.losses),unique_patches=len(set(self.exposed)),
            patch_occurrences=len(self.exposed),pixel_occurrences=self.pixels,
            unique_exposed_patch_ids=sorted(self.rows[i]['patch_id'] for i in set(self.exposed)),
            maximum_patch_occurrences=max(Counter(self.exposed).values(),default=0),
            raw_loss_mean=float(np.mean(self.losses)) if self.losses else None,
            component_gradient_probes=self.probes,random_stream_isolated=True,automatic_gradient_projection=False,
            confirmed_human_exposure_reported_separately=True)
