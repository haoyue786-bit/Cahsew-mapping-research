"""Checkpointed BN bank policy; affine parameters and architecture stay shared.

The ordinary running_* buffers always define target inference. Mode changes
apply at the next training interval, not retroactively to a SELECT candidate.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import torch
from torch import nn
from torch.nn import functional as F

_DOMAIN = ContextVar('cashew_bn_domain', default='target')

@contextmanager
def domain(name):
    if name not in ('source', 'target'): raise ValueError('invalid BN domain')
    token = _DOMAIN.set(name)
    try: yield
    finally: _DOMAIN.reset(token)

class BankedBN2d(nn.BatchNorm2d):
    def forward(self, x):
        self._check_input_dim(x)
        mode = int(self.domain_mode)
        source = _DOMAIN.get() == 'source'
        split = mode == 3 and source
        mean = self.domain_source_running_mean if split else self.running_mean
        var = self.domain_source_running_var if split else self.running_var
        count = self.domain_source_num_batches_tracked if split else self.num_batches_tracked
        update = self.training and (mode == 0 or mode == 3 or (mode == 1 and source))
        factor = self.momentum if self.momentum is not None else 0.
        if update:
            count.add_(1)
            if self.momentum is None: factor = 1. / float(count)
        return F.batch_norm(x, mean, var, self.weight, self.bias, update, factor, self.eps)

def install(model):
    for m in model.modules():
        if type(m) is not nn.BatchNorm2d: continue
        if not m.track_running_stats: raise ValueError('BN policy needs tracked moments')
        m.__class__ = BankedBN2d
        m.register_buffer('domain_source_running_mean', m.running_mean.detach().clone())
        m.register_buffer('domain_source_running_var', m.running_var.detach().clone())
        m.register_buffer('domain_source_num_batches_tracked', m.num_batches_tracked.detach().clone())
        m.register_buffer('domain_mode', torch.zeros((), dtype=torch.int64, device=m.running_mean.device))

@torch.no_grad()
def load_inherited(model, state):
    """Allow only new policy buffers to be absent from an old transfer state."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    permitted = {name for name in model.state_dict() if '.domain_' in name}
    if unexpected or set(missing) - permitted: raise ValueError('transfer model state differs')
    for name, m in model.named_modules():
        if isinstance(m, BankedBN2d) and name+'.domain_mode' in missing:
            m.domain_source_running_mean.copy_(m.running_mean)
            m.domain_source_running_var.copy_(m.running_var)
            m.domain_source_num_batches_tracked.copy_(m.num_batches_tracked)
            m.domain_mode.zero_()

@torch.no_grad()
def apply(model, config):
    from module_controls_v2 import enabled, validate
    if not enabled(config): return
    validate(config); mode = config['bn_update_mode']; count = 0
    for m in model.modules():
        if not isinstance(m, BankedBN2d): continue
        count += 1
        if mode == 3 and int(m.domain_mode) != 3:
            # Fork both domains from the current inference state, never stale E0.
            m.domain_source_running_mean.copy_(m.running_mean)
            m.domain_source_running_var.copy_(m.running_var)
            m.domain_source_num_batches_tracked.copy_(m.num_batches_tracked)
        m.domain_mode.fill_(mode)
    if not count: raise ValueError('declared module BN policy was not installed')
