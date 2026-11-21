"""Explicit Class2 evidence for existing four-class initialization measurements.

Derives named scalars from recorded arrays; never changes arrays, controls,
probes, weights or training. Used by the staged runtime correction.
"""
import copy
import math

CLASS2_INDEX = 2


def _four_values(values, name):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(name + ' must contain four zero-based class values')
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError(name + ' contains invalid class measurements')
    return values


def initialization_baseline_evidence(review):
    result = copy.deepcopy(review)
    source = _four_values(result['source']['f1_per_class'], 'source F1')
    target = result['target_unsupervised']
    fractions = _four_values(target['class_distribution'], 'target class distribution')
    fraction_kind = target.get('class_distribution_kind')
    if fraction_kind not in ('hard_argmax_valid_pixel_fraction', 'tile_weighted_hard_argmax_population_ratio_estimate'):
        raise ValueError('Cannot label a soft or unknown distribution as a hard Class2 fraction')
    named = {'source_class2_f1': float(source[CLASS2_INDEX]),
             'target_class2_fraction': float(fractions[CLASS2_INDEX])}
    for name, value in named.items():
        if name in result and not math.isclose(result[name], value, rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError(name + ' contradicts the recorded zero-based class array')
        result[name] = value
    result['class_index_contract'] = {
        'array_order': 'zero_based_class_indices_0_1_2_3', 'Class2_index': CLASS2_INDEX,
        'source_class2_f1_origin': 'source.f1_per_class[2]',
        'target_class2_fraction_origin': 'target_unsupervised.class_distribution[2]',
        'target_fraction_kind': fraction_kind,
        'target_fraction_scope': target.get('scope', 'full_target_valid_pixels'),
        'estimated_population_distribution': bool(target.get('estimated_population_distribution', False)),
        'instruction': 'Class2 uses array index2, the third entry. Index3 is Class3. Source F1 remains a collapse diagnostic; hard fractions are not target accuracy.'}
    return result


def initialization_probe_evidence(probe):
    result = copy.deepcopy(probe)
    result['gradient_probe']['lambda_candidates_semantics'] = {
        'values_are': 'lambda_base_amplitudes',
        'key_effective_ratio_means': 'desired_effective_gradient_norm_ratio',
        'formula': 'clip(desired_ratio * combined_segmentation_grad_median / max(combined_adversarial_grad_median,1e-12), lambda_bounds)',
        'reference_lambda_effective_ratio': 'actual probe estimate at the reference lambda, a separate field',
        'not_a_required_or_optimal_ratio': True,
        'instruction': 'A lambda_candidates value is not a measured effective ratio; consider bound clipping. These examples do not establish an optimal lambda or optimizer-update ratio.'}
    return result
