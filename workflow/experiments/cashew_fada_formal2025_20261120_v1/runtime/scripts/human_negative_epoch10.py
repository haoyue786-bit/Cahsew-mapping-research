"""Versioned 2019 human-supervision policy for newly initialized jobs.

v1 retains its historical automatic event after epoch 9.
v2 installs no automatic event: the worker and journal require a full Agent
review after epoch 9; retaining zero weight is valid.
"""

POLICY = 'user_2019_human_negative_from_epoch10_v1'
AGENT_POLICY = 'user_2019_human_decision_before_epoch10_v2'
E50_POLICY = 'user_2019_human_default_epoch50_v1'
EXISTING_JOURNAL_VERSION = 'fixed_supplementary_default_v1'


def initialize(config, job):
    declared = job.get('supplementary_default_policy')
    if declared not in {POLICY, AGENT_POLICY, E50_POLICY}:
        return False
    if job.get('target_year') != 2019:
        raise ValueError('the epoch-10 default applies to 2019 jobs only')
    if config.get('negative_control_policy', {}).get('version') != 'tile_negative_controls_v1':
        raise ValueError('negative control policy is required')
    supp = config.get('supplementary_training', {})
    if supp.get('mode') != 'human_polygon_binary' or not supp.get('enabled'):
        raise ValueError('year-matched human polygon training is required')
    if type(supp.get('count')) is not int or supp['count'] < (0 if declared == AGENT_POLICY else 1):
        raise ValueError('no supplementary samples are available')
    if config.get('supplementary_loss_weight') != 0.0:
        raise ValueError('default requires zero initial supplementary weight')
    if config.get('supplementary_default') is not None:
        raise ValueError('a supplementary default is already installed')
    config.update(supplementary_occurrence_cap=4, supplementary_interval_steps=1,
                  supplementary_horizon_epochs=100, supplementary_positive_scale=.25)
    if declared == AGENT_POLICY:
        config['human_epoch10_review'] = {
            'version': AGENT_POLICY, 'review_after_epoch': 9,
            'review_before_training_epoch': 10, 'automatic_activation': False,
            'zero_weight_allowed': True,
        }
        return True
    config['supplementary_default'] = {
        'version': EXISTING_JOURNAL_VERSION,
        'activation_after_epoch': 49 if declared == E50_POLICY else 9,
        'activation_weight': .5,
        'state': 'pending',
        'rule': 'Agent weight/schedule intervention cancels pending activation',
        'declared_policy': declared,
        'first_default_training_epoch': 50 if declared == E50_POLICY else 10,
    }
    return True


def describe(config):
    """Project only the verified, non-point-derived default into a brief."""
    if config.get('human_epoch10_review') is not None:
        return validate_brief(config['human_epoch10_review'])
    policy = config.get('supplementary_default')
    if not policy or policy.get('declared_policy') not in (POLICY, E50_POLICY):
        return None
    if (policy.get('version') != EXISTING_JOURNAL_VERSION
            or policy.get('activation_after_epoch') != (49 if policy.get('declared_policy') == E50_POLICY else 9)
            or policy.get('first_default_training_epoch') != (50 if policy.get('declared_policy') == E50_POLICY else 10)
            or policy.get('activation_weight') != .5
            or policy.get('state') not in {'pending', 'activated', 'no_samples', 'agent_override'}):
        raise ValueError('invalid epoch-10 human-negative default')
    return {'version': policy['declared_policy'], 'first_default_training_epoch': policy['first_default_training_epoch'],
            'activation_weight': .5, 'state': policy['state'],
            'prior_weight_or_schedule_agent_intervention_cancels_default': True}


def validate_brief(value):
    if value is None:
        return None
    if isinstance(value, dict) and value.get('version') == AGENT_POLICY:
        expected = {'version': AGENT_POLICY, 'review_after_epoch': 9,
                    'review_before_training_epoch': 10, 'automatic_activation': False,
                    'zero_weight_allowed': True}
        if value != expected or any(type(value[k]) is not type(v) for k,v in expected.items()):
            raise ValueError('invalid Agent-owned epoch-10 review')
        return dict(value)
    if (not isinstance(value, dict) or set(value) != {
            'version', 'first_default_training_epoch', 'activation_weight', 'state',
            'prior_weight_or_schedule_agent_intervention_cancels_default'}
            or value['version'] not in (POLICY, E50_POLICY) or value['first_default_training_epoch'] != (50 if value['version'] == E50_POLICY else 10)
            or value['activation_weight'] != .5
            or value['state'] not in {'pending', 'activated', 'no_samples', 'agent_override'}
            or value['prior_weight_or_schedule_agent_intervention_cancels_default'] is not True):
        raise ValueError('invalid human-negative default brief')
    return dict(value)

def review_due(job, epoch):
    return job.get('supplementary_default_policy') == AGENT_POLICY and epoch == 9
