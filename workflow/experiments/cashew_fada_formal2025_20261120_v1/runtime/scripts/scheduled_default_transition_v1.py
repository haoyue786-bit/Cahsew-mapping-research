"""Pure, opt-in fixed supplementary schedule update on checkpoint bytes.

No model or optimizer update takes place: the supplementary loss coefficient is
read from config at the next epoch, not stored in an optimizer parameter group.
Caller validates trusted checkpoint bytes and durably publishes the result.
"""
import copy
import io
import torch
from fixed_supplementary_default_v1 import POLICY, advance


def transition(raw, epoch):
    if not isinstance(raw,bytes) or type(epoch) is not int or epoch<1:
        raise ValueError('invalid scheduled transition input')
    payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    if not isinstance(payload,dict) or payload.get('epoch')!=epoch or not isinstance(payload.get('config'),dict):
        raise ValueError('scheduled transition checkpoint mismatch')
    config=payload['config']; policy=config.get('supplementary_default')
    if policy is None:return raw,None
    if not isinstance(policy,dict) or policy.get('version')!=POLICY:
        raise ValueError('unknown supplementary default')
    if policy.get('state') not in {'pending','activated','no_samples','agent_override'}:
        raise ValueError('invalid supplementary default state')
    if policy['state']=='pending':
        if type(policy.get('activation_after_epoch')) is not int or policy['activation_after_epoch']<0:
            raise ValueError('invalid default activation epoch')
    before=copy.deepcopy(config)
    event=advance(config,epoch)
    if event is None:
        if config!=before:raise RuntimeError('unexpected silent scheduled mutation')
        return raw,None
    if not isinstance(event,dict) or event.get('state') not in {'activated','no_samples'} or event.get('epoch')!=epoch:
        raise ValueError('unexpected default transition')
    # The existing worker's advance modifies only this config. Model,
    # optimizer groups, source metrics and RNG stay bytewise semantically equal.
    out=io.BytesIO();torch.save(payload,out)
    return out.getvalue(),event
