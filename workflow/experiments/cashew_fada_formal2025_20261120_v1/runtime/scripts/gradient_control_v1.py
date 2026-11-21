"""Source/target-adversarial diagnostics and optional one-sided projection.

The Agent chooses a strength; a negative cosine alone never enables control.
All quantities refer to raw gradients, not the preconditioned AdamW update.
"""
import copy
import math
import torch

VERSION = 'source_guard_projection_v1'
CONTROL = 'adversarial_projection_strength'


def enabled(config):
    return (config.get('gradient_control_policy') or {}).get('version') == VERSION


def bounds(config):
    return {CONTROL: [0.0, 1.0]} if enabled(config) else {}


def initialize(config, job):
    policy = job.get('gradient_control_policy')
    if not policy:
        return
    if policy.get('version') != VERSION:
        raise ValueError('unsupported gradient control policy')
    config['gradient_control_policy'] = copy.deepcopy(policy)
    config[CONTROL] = 0.0


def strength(config):
    value = config.get(CONTROL, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('projection strength must be a real number')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('projection strength must be finite and in [0,1]')
    if value and not enabled(config):
        raise ValueError('projection was not advertised by the run policy')
    return value


def probe_steps(steps):
    return {0, steps // 2, steps - 1} if steps > 0 else set()


def _shared(a, b):
    return [(x.detach(), y.detach()) for x, y in zip(a, b)
            if x is not None and y is not None]


def _stats(pairs):
    # Aggregate on the device before transferring three scalars to the CPU.
    if not pairs:
        return 0.0, 0.0, 0.0
    values = torch.stack([
        torch.stack(((s.float() * a.float()).sum(), s.float().square().sum(),
                     a.float().square().sum())) for s, a in pairs
    ]).sum(0)
    dot, ss, aa = (float(x) for x in values.cpu())
    if not all(math.isfinite(x) for x in (dot, ss, aa)):
        raise FloatingPointError('nonfinite gradient diagnostics')
    return dot, ss, aa


def cosine(a, b):
    dot, ss, aa = _stats(_shared(a, b))
    return dot / math.sqrt(ss * aa) if ss > 0 and aa > 0 else None


def prepare(params, source, adv, local, glob, weight, amount):
    """Return a compact report and a pending correction, without mutation."""
    if not (len(params) == len(source) == len(adv) == len(local) == len(glob)):
        raise ValueError('gradient/parameter lengths differ')
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(amount) or not 0 <= amount <= 1:
        raise ValueError('invalid projection coefficient')
    indices = [i for i, (s, a) in enumerate(zip(source, adv))
               if s is not None and a is not None]
    pairs = [(source[i].detach(), adv[i].detach()) for i in indices]
    dot, ss, aa = _stats(pairs)
    coefficient = -amount * weight * dot / ss if dot < 0 and ss > 0 else 0.0
    report = {
        'source_adversarial_cosine': dot / math.sqrt(ss * aa) if ss > 0 and aa > 0 else None,
        'source_local_cosine': cosine(source, local),
        'source_global_cosine': cosine(source, glob),
        'local_global_cosine': cosine(local, glob),
        'source_norm_shared': math.sqrt(ss),
        'weighted_adversarial_norm_shared': weight * math.sqrt(aa),
        'weighted_opposition_relative_to_source': max(0.0, -weight * dot / ss) if ss > 0 else None,
        'shared_parameter_tensors': len(indices),
        'projection_strength': amount,
        'projection_applied': coefficient > 0,
        'correction_norm': coefficient * math.sqrt(ss),
    }
    return report, ([(params[i], source[i].detach()) for i in indices], coefficient)


def apply(correction):
    """After total-loss backward, remove only the chosen opposing component.

    Source and supplementary gradients are not independently rewritten. No
    correction is injected into parameters unused by the adversarial loss.
    """
    pairs, coefficient = correction
    if not coefficient:
        return
    with torch.no_grad():
        for param, source in pairs:
            if param.grad is None:
                raise RuntimeError('backward did not populate a shared gradient')
            param.grad.add_(source, alpha=coefficient)


def summarize(records, steps, amount):
    measured = [r for r in records if r['source_adversarial_cosine'] is not None]
    def median(key):
        values = sorted(r[key] for r in records if r.get(key) is not None)
        if not values:
            return None
        n = len(values)
        return (values[(n-1)//2] + values[n//2]) / 2
    return {
        'version': VERSION, 'projection_strength': amount,
        'completed_steps': steps, 'measured_steps': len(records),
        'nonzero_shared_gradient_steps': len(measured),
        'opposing_steps': sum(r['source_adversarial_cosine'] < 0 for r in measured),
        'projected_steps': sum(r['projection_applied'] for r in records),
        'source_adversarial_cosine_median': median('source_adversarial_cosine'),
        'weighted_opposition_relative_to_source_median': median('weighted_opposition_relative_to_source'),
        'source_local_cosine_median': median('source_local_cosine'),
        'source_global_cosine_median': median('source_global_cosine'),
        'local_global_cosine_median': median('local_global_cosine'),
        'records': records,
        'scope': 'shared feature parameters for each reported pair; pure source supervision, excludes supplementary loss',
        'sampling': 'every update while projection enabled; otherwise first/middle/last update',
        'actual_optimizer_update_measured': False,
        'limitation': 'Raw-gradient diagnostics are local evidence, not AdamW-update guarantees, validation preservation or target accuracy. Weak conflict does not require projection.',
    }


def enrich_packet(packet, job):
    if not enabled(job):
        return packet
    policy = packet.setdefault('source_protection_policy', {})
    policy.update({
        'role': 'catastrophic_safety_plus_agent_early_degradation_diagnosis',
        'early_intervention_owner': 'agent',
        'instruction': 'Allow ordinary temporary source-validation fluctuations. Diagnose sustained material source-validation degradation together with repeated, materially weighted source/adversarial conflict before collapse. Compare recent observations and alternative explanations, including source overfit; source training gradients are not infallible. Weak conflict: retain projection off. For material conflict the Agent may choose the advertised projection strength, weaker adversarial pressure or another justified reversible intervention, then reassess. Negative cosine alone is not a trigger. Source F1 alone does not rank target accuracy. Keep the catastrophic floor and source-only recovery.',
    })
    packet['gradient_control_capability'] = {
        'version': VERSION, 'parameter': CONTROL, 'bounds': [0.0, 1.0],
        'zero_means': 'off', 'owner': 'agent',
        'method': 'one-sided removal of source-opposing generator adversarial gradient on shared feature parameters',
        'adoption_evidence': 'Two previous mild-conflict short cases had no Cashew FP/FN/F1 improvement; efficacy for stronger conflict remains unproven.',
        'instruction': 'Use weighted opposition, repeated support and source-validation trends. Choose switch and strength explicitly; reassess benefit and possible overconstraint. This does not guarantee AdamW descent, prevent all source collapse or establish target-class alignment.',
    }
    return packet
