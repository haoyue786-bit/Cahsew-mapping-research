"""Explicit optimizer policy for newly declared trajectories only."""
import math

VERSION = 'inherited_parameter_groups_v1'
CONTROL = 'refiner_lr_multiplier'

def initialize(config, job):
    policy = job.get('parameter_group_policy')
    if policy is None:
        return False
    if policy != {'version': VERSION}:
        raise ValueError('unsupported parameter group policy')
    inherited = bool(job.get('transfer_source'))
    config['parameter_group_policy'] = {'version': VERSION, 'refiner_inherited': inherited}
    config[CONTROL] = 1.0 if inherited else 5.0
    return True

def bounds(config):
    policy = config.get('parameter_group_policy')
    if policy is None:
        return {}
    if (set(policy) != {'version', 'refiner_inherited'} or policy['version'] != VERSION
            or type(policy['refiner_inherited']) is not bool):
        raise ValueError('invalid parameter provenance policy')
    return {CONTROL: [.1, 5.]}

def value(config):
    if not bounds(config):
        raise ValueError('parameter group policy not declared')
    v = config.get(CONTROL)
    if type(v) not in (int, float) or not math.isfinite(v) or not .1 <= v <= 5.:
        raise ValueError('invalid refiner learning-rate multiplier')
    return float(v)

def apply(config, optimizer):
    from module_controls_v2 import apply as apply_modules
    if apply_modules(config, optimizer): return
    if not bounds(config):
        return
    multiplier = value(config)
    groups = [g for g in optimizer.param_groups if g.get('component') == 'spatial_refiner']
    if len(groups) != 1:
        raise ValueError('missing or ambiguous refiner optimizer group')
    expected = 'inherited' if config['parameter_group_policy']['refiner_inherited'] else 'new'
    if groups[0].get('parameter_origin') != expected:
        raise ValueError('optimizer parameter origin differs from declared transfer')
    groups[0]['lr_multiplier'] = multiplier
