"""Versioned module controls. Defaults preserve inherited effective rates."""
import math

VERSION = 'module_controls_v2'
MULTIPLIERS = ('encoder_lr_multiplier', 'pyramid_lr_multiplier', 'fusion_lr_multiplier',
               'refiner_lr_multiplier', 'decoder_lr_multiplier', 'domain_norm_lr_multiplier')
BOUNDS = {k: [.1, 5.] for k in MULTIPLIERS}
BOUNDS['bn_update_mode'] = [0, 3]
BN_MODES = {0: 'shared', 1: 'source_updates_only', 2: 'frozen_running_moments',
            3: 'separate_source_target_running_moments_shared_affine'}

def enabled(obj):
    return obj.get('module_control_policy') == VERSION

def initialize(config, job):
    if not enabled(job): return
    config['module_control_policy'] = VERSION
    for name in MULTIPLIERS:
        config[name] = config.get(name, 1.) if name == 'refiner_lr_multiplier' else 1.
    config['bn_update_mode'] = 0

def bounds(config):
    return dict(BOUNDS) if enabled(config) else {}

def validate(config):
    if not enabled(config): return
    for name, (lo, hi) in BOUNDS.items():
        value = config.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('invalid module control: ' + name)
    if type(config['bn_update_mode']) is not int: raise ValueError('BN mode must be integer')

def component(name):
    if name.startswith('Swin.patch_embed.'): return 'embedding'
    if name.startswith('Swin.layers.'):
        return 'encoder_stage' + str(int(name.split('.')[2]) + 1)
    if name.startswith('Swin.stage_convs.'): return 'pyramid_projection'
    if name.startswith('Swin.head.'): return 'pyramid_fusion'
    if name.startswith('a3_spatial_refiner.'):
        return 'refiner' + name.split('.')[1]
    if name.startswith('fuse_conv.'): return 'fusion'
    if name.startswith('decoder.'): return 'decoder'
    if name.startswith('norm.'): return 'domain_norm'
    return 'other'

COMPONENTS = ('embedding', 'encoder_stage1', 'encoder_stage2', 'encoder_stage3',
              'encoder_stage4', 'pyramid_projection', 'pyramid_fusion', 'fusion',
              'refiner0', 'refiner1', 'decoder', 'domain_norm', 'other', 'D_local', 'D_global')

def group_control(name):
    if name.startswith(('Swin.patch_embed.', 'Swin.layers.')): return 'encoder_lr_multiplier'
    if name.startswith(('Swin.stage_convs.', 'Swin.head.')): return 'pyramid_lr_multiplier'
    if name.startswith('a3_spatial_refiner.'): return 'refiner_lr_multiplier'
    if name.startswith('fuse_conv.'): return 'fusion_lr_multiplier'
    if name.startswith('decoder.'): return 'decoder_lr_multiplier'
    if name.startswith('norm.'): return 'domain_norm_lr_multiplier'
    return None

def groups(model, config, weight_decay, betas, eps):
    validate(config)
    buckets = {name: [] for name in (*MULTIPLIERS, None)}
    for name, parameter in model.named_parameters(): buckets[group_control(name)].append(parameter)
    result = []
    for control, params in buckets.items():
        if not params: continue
        multiplier = float(config[control]) if control else 1.
        result.append(dict(params=params, lr=float(config['main_lr'])*multiplier,
                           lr_multiplier=multiplier, weight_decay=weight_decay, betas=betas,
                           eps=eps, module_control=control or 'none'))
    return result

def apply(config, optimizer):
    if not enabled(config): return False
    validate(config)
    seen = set()
    for group in optimizer.param_groups:
        key = group.get('module_control')
        if key not in (*MULTIPLIERS, 'none') or key in seen:
            raise ValueError('invalid module optimizer topology')
        seen.add(key)
        group['lr_multiplier'] = float(config[key]) if key != 'none' else 1.
    return True
