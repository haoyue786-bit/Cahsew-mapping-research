"""Versioned formal guidance distilled from cutoff-bound historical procedures."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

POLICY = {
    'version': 'formal_distilled_v1',
    'stopping': 'marginal_review_value_v1',
    'selection': 'role_phase_coverage_v2',
    'tuning': 'paired_response_v1',
}
HELPERS = Path(__file__).resolve().parents[1] / '.codex/skills/cashew-fada-training-expert/scripts'


def enabled(job):
    value = job.get('agentguide_policy')
    if value is None:
        return False
    if value != POLICY:
        raise ValueError('unknown formal AgentGuide policy; preserve the pinned job version')
    if job.get('research_stopselect_policy'):
        raise ValueError('formal and research stopping overrides cannot be combined')
    return True


def helper(name):
    path = HELPERS / (name + '.py')
    spec = importlib.util.spec_from_file_location('formal_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def enrich(packet, candidates, decisions):
    """Add measured evidence without choosing actions or opening evaluation labels."""
    if not enabled(packet.get('job', {})):
        return packet
    cutoff = int(packet['epoch'])
    observed = [c for c in candidates if int(c['epoch']) <= cutoff]
    past_decisions = [d for d in decisions if isinstance(d.get('epoch'), int) and d['epoch'] <= cutoff]
    kind = packet['decision_kind']
    if kind == 'training_review':
        past_decisions = [d for d in past_decisions if d['epoch'] < cutoff]
    audit = helper('audit_decision_evidence')
    if kind in {'training_review', 'experiment_review'}:
        prior = [d for d in past_decisions if 0 < d['epoch'] < cutoff and (
            (d.get('intervention') or {}).get('changed_parameters') or (d.get('intervention') or {}).get('schedule_updates') or d.get('action') in {'PAUSE_ADV', 'RESUME_ADV'})]
        response_request = dict(packet)
        response_request['previous_agent_decision'] = prior[-1] if prior else packet.get('previous_agent_decision')
        response_request['recent_reviews'] = [copy.deepcopy(c['review']) for c in observed if c.get('review')]
        # The canonical reviews and current packet have different layouts;
        # avoid presenting the same epoch twice to the evidence-card helper.
        if response_request['recent_reviews']:
            response_request['current_review'] = None
        packet['intervention_evidence'] = audit.intervention_card(response_request)
        packet['intervention_evidence']['cutoff_epoch'] = cutoff
        packet['intervention_evidence']['changed_schedules'] = copy.deepcopy(((prior[-1] if prior else {}).get('intervention') or {}).get('schedule_updates', []))
        if not prior:
            packet['intervention_evidence']['state'] = 'no_noninitial_intervention'
            packet['intervention_evidence']['baseline_role'] = 'previous_keep_observation_only_not_a_treatment_response'
        rows = []
        for candidate in observed[-5:]:
            review = candidate.get('review') or {}
            metrics = review.get('metrics') or {}
            stability = review.get('target_region_stability') or {}
            adjacent = stability.get('adjacent_comparisons') or []
            transition = adjacent[-1] if adjacent else {}
            rows.append({'epoch': candidate['epoch'], 'target_class2_fraction': review.get('target_class2_fraction'),
                         'source_collapsed': bool(not finite(candidate.get('source_class2_f1')) or
                             candidate['source_class2_f1'] < packet['source_protection_policy']['effective_floor']),
                         'all_class_agreement': transition.get('all_class_pixel_agreement'),
                         'class2_iou': transition.get('class2_iou'),
                         'segmentation_loss': metrics.get('seg_cls'), 'supplementary_loss': metrics.get('supplementary_cls'),
                         'effective_gradient_ratio': metrics.get('effective_gradient_ratio'),
                         'scheduled_controls': copy.deepcopy(metrics.get('scheduled_controls')),
                         'scope': (review.get('target') or {}).get('scope')})
        packet['next_review_evidence'] = {'cutoff_epoch': cutoff, 'observations': rows,
            'recent_interventions': [{'epoch': d['epoch'], 'action': d.get('action'),
                'changed_parameters': (d.get('intervention') or {}).get('changed_parameters', []),
                'schedule_updates': (d.get('intervention') or {}).get('schedule_updates', [])}
                for d in past_decisions if d['epoch'] >= max(0, cutoff - 20)],
            'target_quality_known': False,
            'instruction': 'Compare repeated trends and the latest intervention response. State what another five epochs could resolve, the strongest contrary signal and the evidence that would reverse STOP. A temporary lull or recovery remains possible; do not add automatic fixed patience.'}
    if kind == 'checkpoint_selection':
        packet['candidate_identity_evidence'] = audit.selection_card(packet)
        if packet['candidate_identity_evidence']['missing_checkpoint_hashes'] or packet['candidate_identity_evidence']['conflicting_checkpoint_hashes']:
            packet['candidate_memory'] = {'state': 'unavailable', 'reason': 'candidate identity incomplete; retain runtime eligibility and explain uncertainty'}
            return packet
        reviews = {}
        for candidate in observed:
            review = candidate.get('review') or {}
            stability = review.get('target_region_stability') or {}
            adjacent = stability.get('adjacent_comparisons') or []
            last = adjacent[-1] if adjacent else {}
            reviews[str(candidate['epoch'])] = {
                'target_class2_fraction': review.get('target_class2_fraction'),
                'prediction_scope': (review.get('target') or {}).get('scope'),
                'spatial_stability_scope': stability.get('scope'),
                'full_region_stable': stability.get('full_region_stable'),
                'class2_iou': last.get('class2_iou'), 'all_class_agreement': last.get('all_class_pixel_agreement')}
        projection = {'job_id': packet['job']['job_id'], 'tile_id': packet['job'].get('target_tile', ''), 'arm': 'formal',
            'checkpoints': [{'epoch': c['epoch'], 'checkpoint_sha256': c.get('checkpoint_sha256')} for c in observed],
            'reviews': reviews, 'decisions': [{'epoch': d['epoch'], 'action': d.get('action'),
                'changed_parameters': list((d.get('intervention') or {}).get('changed_parameters', [])) + ['schedule:' + x['component'] for x in (d.get('intervention') or {}).get('schedule_updates', [])]}
                for d in past_decisions]}
        memory = helper('persistent_candidate_memory').build_memory(projection, cutoff, capacity=8)
        eligible = set(packet['eligible_candidates'])
        for record in memory['candidates']:
            record['eligible'] = record['epoch'] in eligible
        memory['state'] = 'available'
        memory['role_semantics'] = {'observed_recovery': 'observed overlap/agreement rebound only; target-quality recovery unknown'}
        memory['terminal_ranking'] = 'agent_joint_evidence_with_uncertainty_no_automatic_ranker'
        memory['instruction'] = 'Use roles and phase coverage to compare early, intervention, recovery and recent candidates. This is an eight-record attention aid; it never removes other runtime-eligible candidates. Epoch zero remains diagnostic or catastrophic-collapse fallback. Do not pick the newest, most stable or highest source-F1 candidate automatically.'
        packet['candidate_memory'] = memory
    return packet
