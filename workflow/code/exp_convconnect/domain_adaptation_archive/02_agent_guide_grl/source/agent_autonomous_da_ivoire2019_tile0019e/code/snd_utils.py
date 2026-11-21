"""Memory-bounded, deterministic Soft Neighborhood Density."""

import numpy as np
import torch
import torch.nn.functional as F


def sample_global_locations(valid_counts, max_points=10000, seed=42):
    counts = np.asarray(valid_counts, dtype=np.int64)
    cumulative = np.cumsum(counts)
    total = int(cumulative[-1]) if len(cumulative) else 0
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = min(total, int(max_points))
    global_ids = np.sort(rng.choice(total, size=n, replace=False))
    patch_ids = np.searchsorted(cumulative, global_ids, side="right")
    starts = np.concatenate(([0], cumulative[:-1]))
    pixel_ids = global_ids - starts[patch_ids]
    return patch_ids.astype(np.int64), pixel_ids.astype(np.int64)


def soft_neighborhood_density(features, temperature=0.05, row_chunk_size=512):
    """Return mean entropy of the soft neighborhood distribution."""
    x = torch.as_tensor(features, dtype=torch.float32)
    if x.ndim != 2:
        raise ValueError("features must be N x C")
    n = x.shape[0]
    if n <= 1:
        return 0.0
    x = F.normalize(x, p=2, dim=1)
    entropies = []
    for start in range(0, n, int(row_chunk_size)):
        rows = x[start:start + int(row_chunk_size)]
        logits = rows @ x.T
        row_ids = torch.arange(start, min(start + len(rows), n), device=x.device)
        logits[torch.arange(len(rows), device=x.device), row_ids] = -torch.inf
        probs = torch.softmax(logits / float(temperature), dim=1)
        entropies.append(-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1))
    return float(torch.cat(entropies).mean().item())
