"""Numeric-only module observation contract; no tensor library dependency."""
VERSION = 'module_dynamics_v1'
COMPONENTS = ('backbone', 'refiner', 'fusion', 'decoder', 'other')
FIELDS = ('sample_before_l2', 'sample_delta_l2', 'sample_relative_l2',
          'sample_elements', 'sample_tensors', 'eligible_tensors', 'lr_min', 'lr_max',
          'bn_before_l2', 'bn_delta_l2', 'bn_relative_l2', 'bn_elements',
          'bn_modules', 'bn_forward_count_delta')

def project(block):
    if not isinstance(block,dict) or block.get('version')!=VERSION:raise ValueError('invalid module dynamics')
    return {'module_'+c+'_'+f:block['components'][c][f] for c in COMPONENTS for f in FIELDS}
