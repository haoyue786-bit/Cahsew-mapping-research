"""Divide a fixed ADM1 patch budget across tile IDs, then sample within tiles."""
from collections import defaultdict
import numpy as np


def sample_by_tile(chosen, total=1000, seed=16):
    if total < 0:
        raise ValueError('Sample budget must be nonnegative')
    identities=[identity for identity,path in chosen]
    names=[path for identity,path in chosen]
    if len(set(identities))!=len(identities) or len(set(names))!=len(names):
        raise ValueError('Duplicate candidate spatial identity or image path')
    groups=defaultdict(list)
    for identity,path in sorted(chosen):
        tile_id=identity.rsplit('_',2)[0]
        groups[tile_id].append((identity,path))
    rng=np.random.RandomState(seed)
    tile_ids=sorted(groups)
    order=list(tile_ids)
    rng.shuffle(order)
    budget=min(total,len(chosen))
    quotas={tile:0 for tile in tile_ids}
    remaining=budget
    while remaining:
        eligible=[tile for tile in order if quotas[tile]<len(groups[tile])]
        if not eligible:raise RuntimeError('Cannot allocate declared available budget')
        for tile in eligible:
            if not remaining:break
            quotas[tile]+=1
            remaining-=1
    selected=[]
    for tile in tile_ids:
        rows=groups[tile]
        indices=rng.choice(len(rows),quotas[tile],replace=False)
        selected.extend(rows[int(index)] for index in indices)
    sampled=sorted(path for identity,path in selected)
    if len(sampled)!=budget or len(set(sampled))!=budget:raise RuntimeError('Incorrect sample allocation')
    report=dict(policy='tile_equal_quota_random_without_replacement_v1',seed=seed,
        requested_count=total,sampled_count=budget,candidate_count=len(chosen),eligible_tile_count=len(tile_ids),
        sampled_tile_count=sum(q>0 for q in quotas.values()),
        allocation='Round-robin equal tile quotas; exhausted small tiles return remaining places to other tiles; seeded random within each tile',
        tile_counts=[dict(tile_id=tile,candidate_count=len(groups[tile]),sampled_count=quotas[tile]) for tile in tile_ids],
        selected_patch_ids=sorted(identity for identity,path in selected),
        user_scope='Use existing tile grid only; no feature clustering, crop-prior weighting or new geographic subdivisions')
    return sampled,report
