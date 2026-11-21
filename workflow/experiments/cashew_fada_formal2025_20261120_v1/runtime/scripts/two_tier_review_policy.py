"""Evidence and action contract for lightweight versus full training reviews."""
import math


def mode(epoch, interval=5):
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError('epoch must be a positive integer')
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError('interval must be a positive integer')
    return 'full' if epoch % interval == 0 else 'light'


def light_observation(epoch, metrics, controls):
    """Only measurements from this epoch; absence is never filled with old validation."""
    wanted = ('cls', 'seg_cls', 'adv', 'D_loss', 'local_D_acc', 'global_D_acc',
              'effective_gradient_ratio', 'steps', 'gradient_direction',
              'automatic_negative', 'polygon_supervision')
    current = {k: metrics[k] for k in wanted if k in metrics}
    for key, fields in {
        'gradient_direction': ('measured_steps','completed_steps','opposing_steps',
            'source_adversarial_cosine_median','weighted_opposition_relative_to_source_median','scope'),
        'automatic_negative': ('updates','patch_occurrences','pixel_occurrences','configured_weight','effective_weight'),
        'polygon_supervision': ('negative_pixel_occurrences','positive_pixel_occurrences',
            'effective_negative_weight_min','effective_negative_weight_max',
            'effective_positive_weight_min','effective_positive_weight_max')
    }.items():
        if isinstance(current.get(key), dict):
            current[key] = {field: current[key][field] for field in fields if field in current[key]}
    numeric_errors = [k for k in ('cls', 'seg_cls', 'adv', 'D_loss')
                      if k in current and (isinstance(current[k], bool) or
                         not isinstance(current[k], (int, float)) or not math.isfinite(current[k]))]
    return {'epoch': epoch, 'review_mode': 'light', 'metrics': current,
            'base_controls': controls, 'measured_scheduled_controls': metrics.get('scheduled_controls'),
            'nonfinite_or_invalid_fields': numeric_errors,
            'source_validation': None, 'target_unsupervised': None,
            'evidence_scope': 'Current epoch training measurements only. Source validation and target map diagnostics were not measured; older values are not current.',
            'normal_stop_allowed': False, 'normal_model_selection_allowed': False,
            'emergency_signal': bool(numeric_errors)}
