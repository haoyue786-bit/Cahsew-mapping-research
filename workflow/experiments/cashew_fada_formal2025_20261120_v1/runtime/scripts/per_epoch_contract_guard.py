"""Reject inconsistent v2 pilot execution before CUDA or Agent calls."""
def validate(job):
    stop_policy = job.get('evidence_based_stop_policy')
    if stop_policy not in (None, 'agent_judgment_v1'):
        raise ValueError('Unknown evidence-based stop policy')
    if stop_policy and job.get('per_epoch_policy') != 'per_epoch_evidence_v2':
        raise ValueError('Evidence-based stop policy requires the per-epoch v2 contract')
    if job.get('per_epoch_policy') != 'per_epoch_evidence_v2':
        return
    if job.get('review_interval') != 1:
        raise ValueError('per_epoch_evidence_v2 requires review_interval=1')
    if job.get('planet_decision_reference') is not False:
        raise ValueError('per_epoch_evidence_v2 requires target evaluation isolation')
    if job.get('point_validation_online') is not False:
        raise ValueError('per_epoch_evidence_v2 requires online target-point evaluation disabled')
    if (job.get('automatic_negative_policy') or {}).get('validation_point_labels_for_training') is not False:
        raise ValueError('per_epoch_evidence_v2 requires target-point labels excluded from automatic negatives')
    if job.get('supplementary_default_policy') != 'fixed_supplementary_default_v1':
        raise ValueError('per_epoch_evidence_v2 requires its matched supplementary default')
    route=job.get('agent_account_route') or {}
    if route.get('codex_home') != '@WORKSPACE@/.codex_training_current_account' or route.get('model') != 'gpt-6.1-sol':
        raise ValueError('per_epoch_evidence_v2 pilot requires Sol via the isolated current account')

    if job.get('two_tier_review_policy') == 'light_per_epoch_deep_every_five_v1':
        if job.get('deep_review_interval') != 5:
            raise ValueError('Two-tier review requires a full review every five epochs')
        if (job.get('automatic_negative_policy') or {}).get('refresh_at_trained_review') != 5:
            raise ValueError('Two-tier review requires automatic-negative refresh every five epochs')
    elif job.get('two_tier_review_policy') is not None:
        raise ValueError('Unknown two-tier review policy')
