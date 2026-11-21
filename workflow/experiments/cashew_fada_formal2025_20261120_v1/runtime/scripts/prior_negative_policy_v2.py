"""Deterministic review priorities; no automatic parameter or prediction mutation."""
from __future__ import annotations
import copy

VERSION='spatial_prior_negative_guidance_v2'


def packet_evidence(packet):
    card=packet.get('geographic_prior_evidence') or {}
    if not card.get('version','').startswith('cashew_spatial_prior_v2_'):return None
    area=packet.get('area_alert_evidence') or {}
    current=area.get('current') or {}
    kind=packet.get('decision_kind')
    config=packet.get('current_config') or packet.get('historical_anchor_config') or {}
    negative=packet.get('conditional_negative_evidence') or {}
    pool=int(negative.get('candidate_pool_count',0))
    metrics=(packet.get('current_review') or {}).get('metrics') or {}
    exposure=metrics.get('negative_controls') or {}
    known_exposure=exposure.get('negative_updates',0)>0 and exposure.get('effective_weight_max',0)>0
    source=(packet.get('source_protection_policy') or {}).get('source_collapsed',False)
    source=source or (packet.get('safety_envelope') or {}).get('source_collapsed',False)
    alert=current.get('level') in ('warning','high','severe')
    low=bool(card.get('limited_context_covers_majority'))
    competitor=bool(card.get('competing_crop_context'))
    budget=packet.get('experiment_capability') or {}
    completed_pair=(packet.get('experiment_result') or {}).get('state')=='completed'
    action='diagnose_from_remote_sensing'
    blockers=[]
    if kind=='checkpoint_selection':action='compare_remote_sensing_and_intervention_evidence_not_minimum_area'
    elif source:action='disable_negative_pressure_and_follow_source_repair'
    elif kind=='initialization':action='initialize_negatives_off_and_plan_background_diagnostics'
    elif not area.get('late_activation_eligible',False):action='keep_negatives_off_until_late_boundary'
    elif current.get('level')=='unmeasured':action='measure_full_target_hard_area_before_area_prior_comparison'
    elif pool==0:
        action='retain_empty_negative_branch';blockers.append('empty_selected_pool; do not enlarge 500km radius silently')
    elif kind=='experiment_review':
        action=('evaluate_completed_pair_then_adopt_or_test_larger_single_factor_dose_if_weak_and_safe'
                if completed_pair else 'reconcile_incomplete_or_failed_pair_without_claiming_treatment_effect')
    elif alert:
        if config.get('supplementary_loss_weight',0)>0 and not known_exposure:
            action='check_effective_negative_exposure_before_judging_dose'
        elif config.get('supplementary_loss_weight',0)>0:
            action='compare_measured_response_then_prioritize_higher_single_factor_dose_if_weak_and_safe'
        else:action='prioritize_paired_negative_dose_trial'
        if not budget.get('enabled',False):blockers.append('paired experiment currently unavailable; record capability/budget blocker and evaluate existing completed evidence')
    return {'version':VERSION,'recommended_review_focus':action,
            'prior_vs_area_trigger':bool(alert and low),
            'competing_crop_vs_area_trigger':bool(alert and competitor),
            'trial_priority':'urgent' if alert and low else 'high' if alert and competitor else 'normal',
            'limited_context':card.get('cashew_context','unknown'),'area_alert':copy.deepcopy(current),
            'competing_crop_context_strength':card.get('competing_crop_context_strength','unknown'),
            'late_activation_eligible':area.get('late_activation_eligible',False),
            'selected_negative_pool':pool,'last_epoch_exposure_recorded':known_exposure,
            'completed_pair_available':completed_pair,
            'source_collapsed':bool(source),'blockers':blockers,
            'required_before_adoption':[
                'Identify suspect background from imagery/features and distinguish it from plausible cashew; text is not a pixel label.',
                'Check that selected weak negatives cover that background and exclude plausible cashew/unknown pixels; 500km distance and feature similarity do not validate a label.',
                'Use one-factor same-state/sample/schedule comparison within the advertised budget; record both arms and all endpoints.',
                'Read effective dose, curve phase, cap/cadence, unique samples and eligible pixels; last-epoch metrics are not whole-arm totals.',
                'Compare suspect-background suppression, preserved plausible cashew, source safety and contrary evidence. Lower area or lower loss alone is not effectiveness.',
                'Safe weak measured response motivates a larger controlled coefficient OR exposure test, one factor at a time. Weak response alone never turns the branch off.',
                'Disable/reduce on source collapse, nonfinite values, credible cashew damage or unreliable negatives; area targets and geographic prior cannot override these observations.'
            ],'automatic_parameter_change':False,'automatic_pixel_label':False,
            'statement':'This is a diagnostic/action priority, not proof of overprediction, target accuracy, or an optimal dose. A dose may be adopted as a reversible monitored trial while target accuracy is unknown, with a concrete hypothesis and reversal condition.'}
