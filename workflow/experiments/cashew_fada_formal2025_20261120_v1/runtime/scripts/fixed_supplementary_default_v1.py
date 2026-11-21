"""Fixed-cohort supplementary default for explicitly opted-in fresh pilots.

An Agent intervention on the supplementary weight or its schedule cancels the
pending default. State lives inside the saved training config for exact recovery.
"""
POLICY = 'fixed_supplementary_default_v1'
WEIGHT = 'supplementary_loss_weight'

def initialize(config, job):
    if job.get('supplementary_default_policy') != POLICY:
        return
    if config.get('negative_control_policy', {}).get('version') != 'tile_negative_controls_v1':
        raise ValueError('fixed supplementary default requires negative controls')
    if config[WEIGHT] != 0:
        raise ValueError('fixed supplementary default requires zero initial weight')
    config.update(supplementary_occurrence_cap=4, supplementary_interval_steps=1,
                  supplementary_horizon_epochs=100, supplementary_positive_scale=.25)
    config['supplementary_default'] = dict(version=POLICY, activation_after_epoch=50,
        activation_weight=.5, state='pending', rule='Agent weight/schedule intervention cancels pending activation')

def record_agent_intervention(config, decision):
    policy = config.get('supplementary_default')
    if not policy or policy['state'] != 'pending':
        return
    intervention=decision['intervention']
    updates=intervention.get('schedule_updates') or []
    if WEIGHT in intervention.get('changed_parameters', []) or any(
        isinstance(u,dict) and u.get('component') == WEIGHT for u in updates) or any(
        isinstance(u,dict) and u.get('component') == WEIGHT and u.get('curve') != 'constant'
        for u in (intervention.get('schedule_choices') or [])):
        policy['state']='agent_override'
        policy['override_epoch']=decision.get('epoch')

def advance(config, epoch):
    policy=config.get('supplementary_default')
    if not policy or policy['state'] != 'pending' or epoch < policy['activation_after_epoch']:
        return None
    if not config['supplementary_training'].get('count'):
        policy['state']='no_samples'; return dict(state='no_samples',epoch=epoch)
    config[WEIGHT]=policy['activation_weight']
    policy['state']='activated';policy['activated_after_epoch']=epoch
    return dict(state='activated',epoch=epoch,weight=config[WEIGHT],positive_scale=config['supplementary_positive_scale'])
