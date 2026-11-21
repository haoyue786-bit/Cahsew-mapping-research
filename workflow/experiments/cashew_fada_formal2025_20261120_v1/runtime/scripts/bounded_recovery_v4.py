"""Bounded recovery bookkeeping; never chooses controls or starts a worker.

The owning supervisor supplies a verified healthy full-state anchor, an explicit
one-control strategy or registered source-only repair and a reconciled outcome. Existing STOP/SELECT
decisions and checkpoints are immutable. This module has no Agent/GPU caller.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch


FULL_STATE_KEYS = {
    'epoch', 'state_dict', 'local_discriminator', 'global_discriminator',
    'model_optimizer', 'local_optimizer', 'global_optimizer', 'config', 'rng_state',
}


def _finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else False
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def verify_full_state(payload, *, expected_epoch, saved_cuda_ordinal):
    """Check the loaded trusted checkpoint, before a supervisor applies it.

    File identity, job/data/runtime provenance and strict module load compatibility
    are separate caller checks. No complete receipt may omit those checks.
    """
    if not isinstance(payload, dict) or FULL_STATE_KEYS - payload.keys():
        raise ValueError('Incomplete model/D/optimizer/config/RNG checkpoint')
    epoch = payload['epoch']
    if type(epoch) is not int or epoch != expected_epoch or epoch <= 0:
        raise ValueError('Restore requires the recorded trained epoch, not epoch0')
    for key in ['state_dict', 'local_discriminator', 'global_discriminator']:
        if not isinstance(payload[key], dict) or not payload[key]:
            raise ValueError('Empty model/discriminator state: ' + key)
    for key in ['model_optimizer', 'local_optimizer', 'global_optimizer']:
        optimizer = payload[key]
        if (not isinstance(optimizer, dict) or not isinstance(optimizer.get('state'), dict)
                or not isinstance(optimizer.get('param_groups'), list) or not optimizer['param_groups']
                or any(not isinstance(group, dict) or not group.get('params') for group in optimizer['param_groups'])):
            raise ValueError('Incomplete optimizer: ' + key)
    config = payload['config']
    if not isinstance(config, dict) or not isinstance(config.get('schedule_state'), dict):
        raise ValueError('Missing configuration/schedule state')
    import supplementary_control_v4 as negative
    components = {'main_lr', 'D_lr', 'lambda_adv'}
    if negative.enabled(config):
        components.add(negative.WEIGHT)
    if set(config['schedule_state']) != components:
        raise ValueError('Incomplete four-component schedule state' if negative.enabled(config)
                         else 'Incomplete three-component schedule state')
    if negative.enabled(config):
        state = config['schedule_state'][negative.WEIGHT]
        if (not isinstance(state, dict)
                or state.get('curve') not in negative.CURVES[negative.WEIGHT]
                or state.get('mode') not in {'hold', 'advance'}
                or type(state.get('phase_at_anchor')) not in {int, float}
                or not 0 <= state['phase_at_anchor'] <= 1
                or type(state.get('anchor_epoch')) not in {int, float}
                or not math.isfinite(state['anchor_epoch']) or state['anchor_epoch'] < 0):
            raise ValueError('Invalid supplementary schedule state')
    rng = payload['rng_state']
    if not isinstance(rng, dict) or {'python', 'numpy', 'torch', 'cuda'} - rng.keys():
        raise ValueError('Incomplete RNG state')
    if (not isinstance(rng['python'], tuple) or not isinstance(rng['numpy'], tuple)
            or not isinstance(rng['torch'], torch.Tensor) or rng['torch'].dtype != torch.uint8):
        raise ValueError('Invalid Python/NumPy/Torch RNG state')
    cuda = rng['cuda']
    if (type(saved_cuda_ordinal) is not int or saved_cuda_ordinal < 0
            or not isinstance(cuda, (tuple, list)) or saved_cuda_ordinal >= len(cuda)
            or not isinstance(cuda[saved_cuda_ordinal], torch.Tensor)
            or cuda[saved_cuda_ordinal].dtype != torch.uint8):
        raise ValueError('Original active saved CUDA generator is unknown or missing')
    if not _finite(payload):
        raise ValueError('Nonfinite saved model/D/optimizer/config/RNG state')
    return {'epoch': epoch, 'full_state_fields_verified': True, 'finite_state_verified': True,
            'saved_active_cuda_ordinal': saved_cuda_ordinal,
            'mapped_cuda_rng_state': cuda[saved_cuda_ordinal].clone(),
            'file_job_provenance_and_strict_load_still_required': True}


class BoundedRecovery:
    """An explicit finite state machine, to be persisted by the owning supervisor."""
    def __init__(self, *, source_collapsed, max_attempts, max_extra_epochs, original_remaining_epochs, anchor):
        if type(source_collapsed) is not bool:
            raise ValueError('Recovery requires an explicit measured collapse flag')
        for value in [max_attempts, max_extra_epochs, original_remaining_epochs]:
            if type(value) is not int or value < 0:
                raise ValueError('Recovery limits must be explicit nonnegative integers')
        self.state = {'status': 'NEEDS_STRATEGY' if source_collapsed else 'NO_RECOVERY_REQUIRED',
                      'collapse_confirmed': bool(source_collapsed), 'max_attempts': max_attempts,
                      'epoch_limit': min(max_extra_epochs, original_remaining_epochs),
                      'epochs_used': 0, 'epochs_reserved': 0, 'attempts': [],
                      'anchor': copy.deepcopy(anchor), 'delivery': None}

    def begin(self, *, additional_epochs, strategy, verified_anchor):
        state = self.state
        if state['status'] not in {'NEEDS_STRATEGY', 'ATTEMPT_FAILED_RECONCILED'}:
            raise ValueError('No blind restart, duplicate attempt or ordinary-drop rescue')
        if len(state['attempts']) >= state['max_attempts']:
            raise ValueError('Recovery attempt allowance exhausted')
        if (type(additional_epochs) is not int or additional_epochs <= 0
                or additional_epochs > state['epoch_limit'] - state['epochs_used']):
            raise ValueError('Recovery epoch allowance exhausted')
        anchor = state['anchor']
        if (verified_anchor != anchor or type(anchor.get('epoch')) is not int or anchor['epoch'] <= 0
                or anchor.get('source_collapsed') is not False
                or any(anchor.get(key) is not True for key in
                       ['finite_state_verified', 'full_state_verified', 'checkpoint_hash_verified',
                        'job_data_runtime_provenance_verified', 'strict_load_verified'])
                or not anchor.get('checkpoint_sha256')):
            raise ValueError('Healthy trained full-state anchor remains unverified')
        source_only = (isinstance(strategy, dict) and strategy.get('strategy_kind') == 'source_only_repair'
                       and strategy.get('source_only_controls_verified') is True
                       and strategy.get('repair_controls') == {'adversarial_state': 'paused',
                                                               'supplementary_loss_weight': 0}
                       and strategy.get('matched_causal_experiment') is False
                       and type(strategy.get('changed_control_count')) is int
                       and strategy['changed_control_count'] in {1, 2})
        if (not isinstance(strategy, dict) or strategy.get('validated_by_normalizer') is not True
                or type(strategy.get('changed_control_count')) is not int
                or (strategy['changed_control_count'] != 1 and not source_only)
                or (strategy.get('strategy_kind') == 'source_only_repair' and not source_only)
                or not strategy.get('decision_sha256') or not strategy.get('mechanism')
                or not strategy.get('reversal_condition')):
            raise ValueError('Require one explicit validated strategy, never a fixed rescue default')
        attempt = {'number': len(state['attempts']) + 1, 'status': 'RUNNING_UNRECONCILED',
                   'epoch_cap': additional_epochs, 'restore_epoch': anchor['epoch'],
                   'restore_checkpoint_sha256': anchor['checkpoint_sha256'],
                   'strategy': copy.deepcopy(strategy)}
        state['attempts'].append(attempt)
        state['epochs_reserved'] = additional_epochs
        state['status'] = 'RUNNING_UNRECONCILED'
        return copy.deepcopy(attempt)

    def finish(self, *, completed_epochs, worker_reaped, source_collapsed, finite_state, delivery,
               charged_epochs=None):
        state = self.state
        if state['status'] != 'RUNNING_UNRECONCILED':
            raise ValueError('No active recovery attempt')
        if worker_reaped is not True:
            raise ValueError('Reconcile and reap the worker before retry or delivery')
        attempt = state['attempts'][-1]
        if type(completed_epochs) is not int or not 0 <= completed_epochs <= attempt['epoch_cap']:
            raise ValueError('Missing or excessive actual recovery cost')
        charge = completed_epochs if charged_epochs is None else charged_epochs
        if type(charge) is not int or not completed_epochs <= charge <= attempt['epoch_cap']:
            raise ValueError('Recovery budget charge must cover the observed cost within the reservation')
        healthy = source_collapsed is False and finite_state is True
        if healthy:
            epoch = delivery.get('epoch') if isinstance(delivery, dict) else None
            outcome_epoch = delivery.get('outcome_epoch', epoch) if isinstance(delivery, dict) else None
            if (completed_epochs <= 0 or charge != completed_epochs
                    or type(epoch) is not int or type(outcome_epoch) is not int or not 0 < epoch <= outcome_epoch
                    or outcome_epoch != attempt['restore_epoch'] + completed_epochs
                    or (outcome_epoch != epoch and (not delivery.get('outcome_checkpoint_sha256')
                                                   or delivery.get('outcome_checkpoint_verified') is not True))
                    or not delivery.get('checkpoint_sha256')
                    or delivery.get('checkpoint_hash_verified') is not True):
                raise ValueError('Healthy outcome needs the exact verified resulting checkpoint')
        state['epochs_used'] += charge
        state['epochs_reserved'] = 0
        attempt.update({'completed_epochs': completed_epochs, 'worker_reaped': True,
                        'charged_epochs': charge,
                        'charge_kind': 'observed_completed_epochs' if charge == completed_epochs else 'reserved_upper_bound',
                        'source_collapsed': source_collapsed, 'finite_state': finite_state,
                        'status': 'HEALTHY_COMPLETED' if healthy else 'FAILED_RECONCILED'})
        state['status'] = 'HEALTHY_COMPLETED' if healthy else 'ATTEMPT_FAILED_RECONCILED'
        if healthy:
            kind = 'trained_recovery' if delivery['epoch'] > attempt['restore_epoch'] else 'trained_fallback'
            state['delivery'] = {**copy.deepcopy(delivery), 'kind': kind,
                                 'adapted': True, 'target_quality_proven': False}
        return copy.deepcopy(state)

    def fallback(self, *, reason, verified_candidate):
        state = self.state
        if state['status'] == 'RUNNING_UNRECONCILED':
            raise ValueError('Do not deliver or retry an unresolved live attempt')
        if not reason or not isinstance(verified_candidate, dict):
            raise ValueError('Fallback needs a concrete reason and checkpoint')
        epoch = verified_candidate.get('epoch')
        if (type(epoch) is not int or epoch < 0 or verified_candidate.get('source_collapsed') is not False
                or verified_candidate.get('finite_state_verified') is not True
                or verified_candidate.get('checkpoint_hash_verified') is not True
                or not verified_candidate.get('checkpoint_sha256')):
            raise ValueError('No usable finite noncollapsed fallback checkpoint')
        state['delivery'] = {**copy.deepcopy(verified_candidate), 'reason': reason,
                             'kind': 'trained_fallback' if epoch else 'unadapted_fallback',
                             'adapted': epoch > 0, 'target_quality_proven': False}
        state['status'] = 'FALLBACK_DELIVERED'
        return copy.deepcopy(state)
