"""Full model/optimizer numerical check for an opt-in candidate runtime.

Call only while the worker owns the model, after restoring a complete checkpoint
and after a completed epoch. Not an authorization, provenance check, gradient
clipping mechanism, or replacement for checking gradients before an update.
All floating and complex model state and optimizer state tensors are inspected;
integer/bool tensors cannot encode NaN or infinity. Per-device reduction avoids
syncing once per tensor. The GPU cost is not yet benchmarked in full training.
"""
import math
import numbers
from collections import defaultdict

import torch

VERSION = 'full_state_finiteness_v1'
_FLOAT = frozenset((torch.float16, torch.bfloat16, torch.float32, torch.float64))
_INT = frozenset((torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64))


class NonFiniteState(RuntimeError):
    """A completed or restored state has non-finite numerical contents."""


def _leaves(value, path, active):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, bool):
        return
    elif isinstance(value, numbers.Real):
        if not math.isfinite(value):
            raise NonFiniteState('non-finite scalar: ' + path)
    elif isinstance(value, numbers.Complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise NonFiniteState('non-finite complex scalar: ' + path)
    elif isinstance(value, dict):
        if id(value) in active:
            raise ValueError('cyclic optimizer state: ' + path)
        active.add(id(value))
        try:
            for key, item in value.items():
                yield from _leaves(item, path + '/' + str(key), active)
        finally:
            active.remove(id(value))
    elif isinstance(value, (tuple, list)):
        if id(value) in active:
            raise ValueError('cyclic optimizer state: ' + path)
        active.add(id(value))
        try:
            for i, item in enumerate(value):
                yield from _leaves(item, path + '/' + str(i), active)
        finally:
            active.remove(id(value))
    else:
        raise TypeError('unsupported optimizer/model state type at ' + path)


def _dense_values(value, path):
    if value.layout == torch.strided:
        return value.detach()
    if value.layout == torch.sparse_coo:
        # Inspect stored values directly: coalesce would add duplicate entries,
        # potentially overflow and alter the meaning of a state-finiteness test.
        return value.detach()._values()
    if value.layout in (torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc):
        return value.detach().values()
    raise TypeError('unsupported tensor layout at ' + path)


def check(models, optimizers):
    """Raise on non-finite state; return coverage and synchronization metadata.

    ``models`` and ``optimizers`` are nonempty mappings of logical names to live
    ``nn.Module``/``Optimizer`` objects. Synchronize externally against mutation;
    this function cannot provide an atomic snapshot under concurrent updates.
    """
    if not isinstance(models, dict) or not models or not isinstance(optimizers, dict) or not optimizers:
        raise ValueError('nonempty model and optimizer mappings required')
    if any(not isinstance(k, str) or not k for k in (*models.keys(), *optimizers.keys())):
        raise ValueError('logical names must be nonempty strings')
    tensors = []
    for name, model in models.items():
        if not isinstance(model, torch.nn.Module):
            raise TypeError('invalid model: ' + name)
        tensors.extend(_leaves(model.state_dict(), 'model/' + name, set()))
    for name, opt in optimizers.items():
        if not isinstance(opt, torch.optim.Optimizer):
            raise TypeError('invalid optimizer: ' + name)
        # Use positional identifiers, never stringified Parameter objects.
        for i, state in enumerate(opt.state.values()):
            tensors.extend(_leaves(state, 'optimizer/' + name + '/' + str(i), set()))
    grouped = defaultdict(list)
    other = []
    skipped = 0
    elements = 0
    for path, tensor in tensors:
        tensor = _dense_values(tensor, path)
        if tensor.dtype in _INT:
            skipped += 1
            continue
        if not (tensor.is_floating_point() or tensor.is_complex()):
            raise TypeError('unsupported tensor dtype at ' + path + ': ' + str(tensor.dtype))
        elements += tensor.numel()
        if not tensor.numel():
            continue
        if tensor.dtype in _FLOAT and tensor.device.type in ('cpu', 'cuda'):
            grouped[(tensor.device, tensor.dtype)].append((path, tensor))
        else:
            other.append((path, tensor))
    # Each flag remains on its device until one stacked transfer per device.
    flags = defaultdict(list)
    with torch.no_grad():
        for (device, _dtype), rows in grouped.items():
            try:
                norms = torch._foreach_norm([t for _, t in rows], float('inf'))
                for (path, _), norm in zip(rows, norms):
                    flags[device].append((path, torch.isfinite(norm)))
            except (NotImplementedError, RuntimeError) as exc:
                # This compatibility fallback is safe for unsupported foreach
                # kernels. Do not mask memory exhaustion and retry larger work.
                if 'out of memory' in str(exc).lower():
                    raise
                for path, tensor in rows:
                    flags[device].append((path, torch.isfinite(tensor).all()))
        for path, tensor in other:
            flags[tensor.device].append((path, torch.isfinite(tensor).all()))
        bad = []
        for device, rows in flags.items():
            statuses = torch.stack([flag for _, flag in rows]).cpu().tolist()
            bad.extend(path for (path, _), ok in zip(rows, statuses) if not ok)
    if bad:
        raise NonFiniteState('non-finite model/optimizer state: ' + ', '.join(bad[:16]))
    return {'version': VERSION, 'floating_tensors': sum(len(rows) for rows in flags.values()),
            'floating_elements': elements, 'integer_or_bool_tensors': skipped,
            'devices_checked': len(flags), 'scope': 'all present model state and optimizer state; not future updates or target accuracy'}
