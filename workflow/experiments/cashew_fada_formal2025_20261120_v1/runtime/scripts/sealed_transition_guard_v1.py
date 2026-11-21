"""Compare a candidate transition against the native ordinary transition.

This checks all serialized fields, including tensor weights, optimizer state,
RNG and metrics, and binds optional pool changes to the existing bridge receipt.
It is a synthetic candidate, not provenance or permission to run production.
"""
import io
import torch
from unified_epoch_pool_bridge_sim_v1 import _equal
from sealed_joint_negative_decision_v1 import ordinary_transition

def verify(pre,decision,post,*,pool_journal=None):
    if not isinstance(post,bytes):raise ValueError('transition must return checkpoint bytes')
    expected=ordinary_transition(pre,decision)
    bridge=getattr(pool_journal,'_active_pool_bridge',None) if pool_journal is not None else None
    if bridge is None:
        unpack=lambda raw:torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        a=unpack(expected);b=unpack(post)
    else:
        a=bridge.decode(expected);b=bridge.decode(post)
        if (a.get('pool'),a.get('retired'))!=(b.get('pool'),b.get('retired')):
            action=(decision.get('automatic_negative') or {}).get('action')
            if action not in ('ACTIVATE','DISABLE'):
                raise ValueError('unapproved pool change in checkpoint')
            bridge._audit_pool_action(decision['epoch'],bridge._cp_state(pre),post)
            a={k:v for k,v in a.items() if k not in ('pool','retired')}
            b={k:v for k,v in b.items() if k not in ('pool','retired')}
    if not _equal(a,b):raise ValueError('checkpoint transition differs from authorized native transition')
    return post
