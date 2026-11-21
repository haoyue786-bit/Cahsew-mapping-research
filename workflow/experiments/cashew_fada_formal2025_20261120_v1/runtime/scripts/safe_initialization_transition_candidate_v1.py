"""Experimental, offline initialization transition for new safe checkpoints.

This does not call a model, authenticate evidence, commit a transaction or
authorize training. Its caller must reconstruct the sealed request and verify
the response and release provenance before applying the returned checkpoint.
"""
from decision_bridge import normalize_initialization
from native_safe_checkpoint_codec_v1 import decode
from sealed_safe_router_sim_v1 import _post


def apply(request, answer, checkpoint, job):
    cp = decode(checkpoint)
    if cp.get('epoch') != 0 or request.get('epoch') != 0:
        raise ValueError('initialization requires an epoch-zero checkpoint')
    config = cp.get('config')
    if not isinstance(config, dict):
        raise ValueError('initial configuration missing')
    decision = normalize_initialization(request, answer, config, job)
    if decision.get('action') not in ('KEEP', 'ADJUST'):
        raise ValueError('unsupported initialization action')
    post = _post(checkpoint, decision)
    return decision, post
