"""Deterministic full-state restoration and explicit recovery delivery records."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from bounded_recovery_v4 import verify_full_state
from expert_policy_v4 import _check_config, flat


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def active_saved_cuda_ordinal(payload, resume, source_run):
    saved = payload['rng_state']
    candidates = []
    for ordinal in [saved.get('active_cuda_ordinal'), resume.get('saved_active_cuda_ordinal')]:
        if ordinal is not None:
            if type(ordinal) is not int or ordinal < 0:
                raise ValueError('Invalid original active CUDA ordinal')
            candidates.append(ordinal)
    # Older v4 checkpoints lack the ordinal; their own saved launch/status
    # provides the actual logical device, rather than a guessed current device.
    if not candidates:
        if resume.get('recovery_mode') == 'same_run_existing_checkpoint':
            raise ValueError('Same-run original CUDA identity missing; current startup status is not provenance')
        for name in ['launch.json', 'status.json']:
            path = Path(source_run) / name
            if not path.is_file():
                continue
            value = json.loads(path.read_text()).get('gpu')
            if isinstance(value, str) and value.startswith('cuda:'):
                candidates.append(int(value.split(':', 1)[1]))
                break
            if name == 'launch.json' and isinstance(value, (str, int)):
                candidates.append(int(value))
                break
    if not candidates or len(set(candidates)) != 1:
        raise ValueError('Original active CUDA ordinal missing or contradictory')
    return candidates[0]


def restore_full_state(payload, *, expected_epoch, saved_cuda_ordinal, device,
                       model, local_d, global_d, model_opt, local_opt, global_opt):
    """Restore an isolated worker only after complete finite-state checks.

    Caller verifies the file/job identities before loading. Strict model and
    optimizer compatibility must succeed before any optimizer step is possible.
    Only the originally active saved CUDA generator is restored on this worker.
    """
    check = verify_full_state(payload, expected_epoch=expected_epoch,
                              saved_cuda_ordinal=saved_cuda_ordinal)
    _check_config(flat(payload['config']))
    for module, name in [(model, 'state_dict'), (local_d, 'local_discriminator'),
                         (global_d, 'global_discriminator')]:
        module.load_state_dict(payload[name], strict=True)
    for optimizer, name in [(model_opt, 'model_optimizer'), (local_opt, 'local_optimizer'),
                            (global_opt, 'global_optimizer')]:
        source_groups = payload[name]['param_groups']
        if (len(source_groups) != len(optimizer.param_groups)
                or any(len(a['params']) != len(b['params'])
                       for a, b in zip(source_groups, optimizer.param_groups))):
            raise ValueError('Optimizer parameter-group topology changed')
        optimizer.load_state_dict(payload[name])
        for parameter, state in optimizer.state.items():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    if key in {'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'} and value.shape != parameter.shape:
                        raise ValueError('Optimizer moment shape changed')
                    state[key] = value.to(device)
    saved = payload['rng_state']
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    torch.set_rng_state(saved['torch'])
    cuda_applied = torch.device(device).type == 'cuda'
    if cuda_applied:
        torch.cuda.set_rng_state(check['mapped_cuda_rng_state'], device=device)
    return copy.deepcopy(payload['config']), {
        'epoch': expected_epoch, 'full_state_fields_verified': True,
        'finite_state_verified': True, 'strict_module_optimizer_load_verified': True,
        'python_numpy_torch_rng_restored': True, 'saved_active_cuda_ordinal': saved_cuda_ordinal,
        'active_cuda_generator_applied': cuda_applied,
        'target_device': str(device), 'new_training_steps': 0,
    }


def verify_resume_job_identity(job, source_job):
    """Compare method/data identities without scanning rasters or target labels."""
    for key in ['target_year', 'target_root', 'architecture', 'source_profile', 'source',
                'variant', 'seed', 'negative_control_policy', 'conditional_negative_policy',
                'area_alert_policy', 'geographic_prior_policy']:
        if job.get(key) != source_job.get(key):
            raise ValueError('Resume changes locked method/data identity: ' + key)
    if (job.get('geographic_prior', {}).get('sha256')
            != source_job.get('geographic_prior', {}).get('sha256')):
        raise ValueError('Resume changes locked geographic prior evidence')
    for key in ['target_full_list', 'target_train_list']:
        if not job.get(key) or not source_job.get(key) or digest(job[key]) != digest(source_job[key]):
            raise ValueError('Resume changes target-list identity: ' + key)
    current_supplementary = job.get('supplementary', {})
    source_supplementary = source_job.get('supplementary', {})
    # A new lineage may write the same selector/list into its own output root.
    # Compare actual list bytes and loss/data controls, rather than report paths.
    def controls(value):
        return {k: v for k, v in value.items() if k not in {'names_file', 'selection'}}
    if controls(current_supplementary) != controls(source_supplementary):
        raise ValueError('Resume changes supplementary data/loss controls')
    if current_supplementary.get('enabled'):
        if digest(current_supplementary['names_file']) != digest(source_supplementary['names_file']):
            raise ValueError('Resume changes supplementary-list identity')
        if (current_supplementary.get('selection', {}).get('method')
                != source_supplementary.get('selection', {}).get('method')):
            raise ValueError('Resume changes supplementary selector method')
    return {'job_method_data_identity_verified': True,
            'source_job_sha256': hashlib.sha256(json.dumps(source_job, sort_keys=True).encode()).hexdigest(),
            'target_full_list_sha256': digest(job['target_full_list']),
            'target_train_list_sha256': digest(job['target_train_list'])}


def resume_decisions_at_cutoff(decisions, epoch):
    return [copy.deepcopy(item) for item in decisions
            if type(item.get('epoch')) is int and 0 <= item['epoch'] <= epoch]


def prior_decisions_before_replay(decisions, epoch):
    """Checkpoint E>0 precedes its Agent action; replay that action once."""
    if type(epoch) is not int or epoch <= 0:
        raise ValueError('same-epoch action replay requires a trained checkpoint')
    return [copy.deepcopy(item) for item in decisions
            if type(item.get('epoch')) is int and 0 <= item['epoch'] < epoch]


def verify_decision_checkpoint_controls(decision, checkpoint_config, phase):
    """Check the controls on the correct side of a saved Agent action."""
    if phase not in ('before', 'after'):
        raise ValueError('phase must be before or after')
    values = (decision.get('intervention') or {}).get('parameters_' + phase)
    if not isinstance(values, dict) or not values:
        raise RuntimeError('decision has no verifiable ' + phase + ' controls')
    actual = dict(checkpoint_config)
    if 'negative_pool_count' not in actual:
        actual['negative_pool_count'] = checkpoint_config.get('supplementary_training', {}).get('count')
    mismatch = [key for key, value in values.items()
                if key not in actual or actual[key] != value]
    if mismatch:
        raise RuntimeError({'resume_decision_checkpoint_mismatch': mismatch, 'phase': phase})


def load_epoch_zero_resume(source_run, publication_check, checkpoint_config):
    """Recover the already-applied initialization and baseline, with no Agent call."""
    root = Path(source_run)
    init = root / 'agent' / 'initialization'
    if not publication_check(init / 'request.json'):
        raise RuntimeError('epoch-0 recovery requires a verified initialization publication')
    decision = json.loads((init / 'decision.json').read_text())
    if not decision.get('continue_training'):
        raise RuntimeError('epoch-0 initialization did not authorize training')
    verify_decision_checkpoint_controls(decision, checkpoint_config, 'after')
    baseline = json.loads((root / 'review_epoch_000.json').read_text())
    return decision, float(baseline['source_class2_f1']), float(baseline['target_class2_fraction'])


def resume_history_at_cutoff(history, epoch, recorded_review_epochs):
    """Use an explicit outcome epoch or the next actually recorded review.

    Old action rows have no outcome epoch. Their formal response was recorded
    at the next review, so a same-epoch action's later completed response must
    not appear when restoring that earlier checkpoint. No future value is used
    to infer an unobserved outcome or a quality claim.
    """
    reviews = sorted(set(recorded_review_epochs))
    result = []
    for item in history:
        decision_epoch = item.get('decision_epoch')
        if (type(decision_epoch) is not int or decision_epoch > epoch
                or item.get('model_fraction_after') is None):
            continue
        outcome_epoch = item.get('outcome_epoch')
        if outcome_epoch is None:
            outcome_epoch = next((value for value in reviews if value > decision_epoch), None)
        if type(outcome_epoch) is not int or not decision_epoch < outcome_epoch <= epoch:
            continue
        result.append(copy.deepcopy(item))
    return result


def delivery_record(*, selected_epoch, selected_checkpoint, source_collapsed,
                    stop_reason, stop_epoch, original_max_epoch, recovery_policy=None):
    """Always describe delivery; a safety stop must not silently lose a tile."""
    if type(selected_epoch) is not int or selected_epoch < 0:
        raise ValueError('Unknown selected checkpoint epoch')
    if not 0 <= selected_epoch <= stop_epoch <= original_max_epoch:
        raise ValueError('Selected/stop/budget epoch identities disagree')
    record = {'selected_epoch': selected_epoch, 'checkpoint': str(Path(selected_checkpoint).resolve()),
              'checkpoint_sha256': digest(selected_checkpoint), 'adapted': selected_epoch > 0,
              'delivery_kind': ('trained_fallback' if source_collapsed else 'trained_selected')
                               if selected_epoch > 0 else 'unadapted_fallback',
              'source_collapse_safety_exit': bool(source_collapsed), 'stop_reason': stop_reason,
              'target_quality_proven': False, 'prediction_complete': False}
    policy = recovery_policy or {}
    for key in ['max_attempts', 'max_extra_epochs']:
        if type(policy.get(key, 0)) is not int or policy.get(key, 0) < 0:
            raise ValueError('Recovery limits must be explicit nonnegative integers')
    remaining = max(0, int(original_max_epoch) - int(stop_epoch))
    record['recovery'] = {
        'state': 'NEEDS_BOUNDED_SPECIFICATION' if source_collapsed else 'NOT_REQUIRED',
        'original_remaining_epochs': remaining,
        'max_attempts': int(policy.get('max_attempts', 0)),
        'max_extra_epochs': min(remaining, int(policy.get('max_extra_epochs', 0))),
        'automatic_fixed_control_change': False,
        'next': ('Review a complete finite healthy trained checkpoint and record a separate bounded strategy/lineage; if unavailable, keep this explicit fallback.'
                 if source_collapsed else 'Deliver the recorded selected checkpoint.'),
    }
    return record
