"""Supply checkpoint/map-bound facts for an Agent's qualitative prior check."""
from __future__ import annotations
import math

VERSION='prior_prediction_semantic_check_v1_20261008'


def packet_evidence(packet):
    prior=packet.get('geographic_prior_evidence') or {}
    background=prior.get('planting_background') or {}
    if not background:return None
    review=packet.get('current_review') or {}
    fraction=review.get('target_class2_fraction')
    evaluation=(packet.get('job') or {}).get('target_evaluation') or {}
    sampled=evaluation.get('version')=='fixed_tile_stratified_evaluation_v1'
    if fraction is not None and (not math.isfinite(float(fraction)) or not 0<=float(fraction)<=1):
        raise ValueError('Invalid measured Class2 fraction')
    return dict(version=VERSION,decision_kind=packet.get('decision_kind'),epoch=packet.get('epoch'),
        shared_planting_context=background.get('cashew_context'),
        where_documented=background.get('where_documented',[]),scale_summary=background.get('scale_summary'),
        support_evidence_ids=background.get('support_evidence_ids',[]),
        measured_whole_target_class2_fraction=None if sampled else fraction,
        measured_target_class2_fraction_estimate=fraction if sampled else None,
        prediction_scope=('fixed tile-stratified sampled valid pixels, tile-weighted raw-argmax population ratio estimate; sample consistency only' if sampled else 'whole target valid pixels, raw four-class argmax; not the separately requested >0.6 decoding'),
        evaluation_sample_locations_csv=evaluation.get('locations_csv'),
        evaluation_sample_locations_csv_sha256=evaluation.get('locations_csv_sha256'),
        named_place_spatial_matching='not_verified_by_regional_fraction; use registered sampled patch bounds for approximate location checks' if sampled else 'not_verified_by_whole_target_fraction',
        context_report=prior.get('context_report'),
        agent_assessment=['State broadly compatible, concern requiring investigation, or insufficient evidence, using actual measured scope and cited evidence IDs.',
            'Compare named planting locations and source-described scale when spatial outputs are available; use audit_prior_prediction_semantics.py for a frozen raster.',
            'A regional positive source does not prove every patch contains Cashew; near-zero regional predictions merit diagnosis but do not by themselves establish false negatives.',
            'Major producing-region descriptions do not impose a predicted-area minimum; tonnes and seedlings cannot be converted into pixel fraction.',
            'Retain competing crops, source disagreements, boundary editions and plausible novel Cashew. Agreement with text is not target accuracy.',
            'Without a named-place spatial measurement, report that limitation explicitly; do not claim an inferred location match.',
            'At SELECT compare the chosen checkpoint with an eligible alternative using model/intervention evidence, and summarize semantic concerns separately.'],
        automatic_parameter_change=False,automatic_checkpoint_gate=False,automatic_pixel_label=False)
