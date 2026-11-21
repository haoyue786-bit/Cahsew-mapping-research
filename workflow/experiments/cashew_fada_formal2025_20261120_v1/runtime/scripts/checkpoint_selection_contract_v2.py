"""Complete, point-isolated selection evidence and comparison contract."""
import re
from sealed_packet import SOURCE, PANEL, CONTROL, TRAIN, nums, values_digest
from human_training_evidence_contract import validate as validate_human

VERSION = 'complete_evidence_selection_v2'
SECTIONS = ('source_validation', 'unlabelled_panel', 'human_training', 'annotated_training', 'training', 'controls')
TRAIN_FIELDS = ({'cls', 'seg_cls', 'adv', 'D_loss', 'cls_grad', 'adv_grad', 'effective_gradient_ratio', 'steps', 'G_steps',
                 'D_local_steps', 'D_global_steps', 'local_D_acc', 'global_D_acc'} |
    {k for k in TRAIN if k.startswith(('supp_loaded_', 'supp_completed_', 'supp_human_',
        'human_polygon_', 'gradient_direction_'))} |
    {k for k in TRAIN if k.startswith(('module_', 'domain_', 'detail_', 'gradient_', 'regularization_', 'actual_update_'))})
DECODER = {'version': 'frozen_predictor_v1', 'selection_comparison': 'argmax_class_2',
           'probability_output': 'softmax_class_2', 'point_calibration': 'post_selection_only',
           'checkpoint_includes_normalization_buffers': True}


def enabled(job):
    version = job.get('checkpoint_selection_policy')
    if version is None:
        return False
    if version != VERSION:
        raise ValueError('unknown checkpoint selection policy')
    return True


def _fields(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError('invalid selection evidence fields')


def validate_detail(value):
    fields = {'epoch', 'checkpoint_sha256', 'model_state_sha256', 'producer_request_sha256',
        'source_validation', 'unlabelled_panel', 'human_training', 'training', 'controls',
        'controls_timing', 'review_action', 'review_action_applies_after_epoch'}
    if 'annotated_training' in value:
        fields.add('annotated_training')
    _fields(value, fields)
    epoch = value['epoch']
    if type(epoch) is not int or not 0 <= epoch <= 200:
        raise ValueError('invalid selection evidence epoch')
    for key in ('checkpoint_sha256', 'model_state_sha256', 'producer_request_sha256'):
        if not isinstance(value[key], str) or not re.fullmatch('[0-9a-f]{64}', value[key]):
            raise ValueError('invalid selection evidence identity')
    for key, allowed in (('source_validation', SOURCE), ('unlabelled_panel', PANEL),
                          ('training', TRAIN_FIELDS), ('controls', CONTROL)):
        nums(value[key], allowed)
    if value['source_validation'].get('class2_f1') is None or value['unlabelled_panel'].get('predicted_class2_fraction') is None:
        raise ValueError('selection candidate lacks source/panel observations')
    if 'annotated_training' in value:
        from online_boundary import verify_online
        verify_online(value['annotated_training'])
        if value['human_training'] is not None:
            raise ValueError('Old polygon evidence mixed with current image annotations')
    else:
        human = validate_human(value['human_training'])
        if human['epoch'] != epoch or human['model_state_sha256'] != value['model_state_sha256']:
            raise ValueError('human measurement and selected model differ')
    if value['controls_timing'] != ('after_initialization_before_training' if epoch == 0 else 'used_during_completed_epoch'):
        raise ValueError('ambiguous selection control timing')
    if (value['review_action'] not in ('KEEP', 'ADJUST', 'CONTINUE', 'STOP', 'PAUSE_ADV', 'RESUME_ADV')
            or type(value['review_action_applies_after_epoch']) is not int
            or value['review_action_applies_after_epoch'] != epoch):
        raise ValueError('ambiguous selection action timing')
    if epoch == 0 and value['training']:
        raise ValueError('initialization cannot claim training exposure')
    return value


def artifact(value):
    validate_detail(value)
    return {'kind': 'selection_candidate', 'epoch': value['epoch'], 'values': value,
            'sha256': values_digest('selection_candidate', value['epoch'], value)}


def checked(record, trusted):
    _fields(record, ('kind', 'epoch', 'values', 'sha256'))
    if record != artifact(record['values']) or ('selection_candidate', record['epoch'], record['sha256']) not in trusted:
        raise ValueError('untrusted detailed selection evidence')
    return {k: record[k] for k in ('epoch', 'sha256', 'values')}


def summary(value):
    validate_detail(value)
    result = {'source_class2_f1': value['source_validation']['class2_f1'],
              'target_unlabelled_class2_fraction': value['unlabelled_panel']['predicted_class2_fraction']}
    if value['training'].get('cls') is not None:
        result['training_cls_loss'] = value['training']['cls']
    return result


def comparison_schema(epochs):
    text = {'type': 'string', 'minLength': 1}
    tradeoff = {'type': 'object', 'additionalProperties': False,
        'required': ['benefits', 'tradeoffs', 'unresolved'],
        'properties': {k: dict(text) for k in ('benefits', 'tradeoffs', 'unresolved')}}
    alternative = {'type': 'object', 'additionalProperties': False,
        'required': ['epoch', 'why_competitive', 'benefits_of_selected', 'tradeoffs', 'unresolved'],
        'properties': {'epoch': {'type': 'integer', 'enum': epochs},
            **{k: dict(text) for k in ('why_competitive', 'benefits_of_selected', 'tradeoffs', 'unresolved')}}}
    return {'type': 'object', 'additionalProperties': False,
        'required': ['selected_epoch', 'reason', 'uncertainty', 'baseline_comparison', 'strongest_alternative', 'evidence_used'],
        'properties': {'selected_epoch': {'type': 'integer', 'enum': epochs}, 'reason': dict(text),
            'uncertainty': dict(text), 'baseline_comparison': tradeoff, 'strongest_alternative': alternative,
            'evidence_used': {'type': 'array', 'minItems': 2, 'maxItems': 30, 'uniqueItems': True,
                'items': {'type': 'object', 'additionalProperties': False, 'required': ['epoch', 'section'],
                    'properties': {'epoch': {'type': 'integer', 'enum': epochs},
                                   'section': {'type': 'string', 'enum': list(SECTIONS)}}}}}}


def validate_comparison(request, answer):
    selected = answer['selected_epoch']; alternative = answer['strongest_alternative']['epoch']
    if type(alternative) is not int or alternative == selected:
        raise ValueError('strongest alternative must be a different eligible candidate')
    def nonempty(value):
        if isinstance(value, str) and not value.strip():
            raise ValueError('selection comparison text cannot be blank')
        if isinstance(value, dict):
            for v in value.values(): nonempty(v)
    nonempty(answer)
    details = {r['epoch']: r['values'] for r in request['candidate_details']}
    cited = set()
    for item in answer['evidence_used']:
        value = details[item['epoch']].get(item['section'])
        if not value:
            raise ValueError('selection cites an unavailable measurement')
        cited.add((item['epoch'], item['section']))
    if not {0, selected, alternative}.issubset({e for e, _ in cited}):
        raise ValueError('selection must compare E0, selected and strongest alternative evidence')
    for epoch in {0, selected, alternative}:
        if not any((epoch, kind) in cited for kind in ('human_training', 'annotated_training', 'unlabelled_panel')):
            raise ValueError('source validation alone cannot justify the comparison')
        fit = details[epoch].get('annotated_training') or {}
        if fit.get('all', {}).get('patches',0) and (epoch,'annotated_training') not in cited:
            raise ValueError('SELECT must compare available frozen-library fit for each contrasted predictor')
    return {'baseline_comparison': answer['baseline_comparison'],
            'strongest_alternative': answer['strongest_alternative'], 'evidence_used': answer['evidence_used']}


INSTRUCTION = (
    ' Compare the frozen complete predictors, including E0 and normalization buffers. '
    'Use the fixed argmax rule for the supplied diagnostic comparisons; no threshold fitting or BN replacement is allowed in SELECT. '
    'Use source validation for source preservation, annotation diagnostics for training fit, and the unlabelled panel for prediction response. '
    'Do not maximize a universal source-F1, stability, entropy or area score, invent a weighted proxy score, or favor an epoch because it is later or earlier. '
    'Identify the strongest competing candidate; explain benefits, tradeoffs and unresolved evidence relative to both it and E0. '
    'If E0 is selected, its baseline comparison must explicitly describe retaining the identical starting predictor. '
    'Query initialization and candidate_details for E0, the selected predictor and the strongest alternative. '
    'The worker checks those actual queries and evidence references. Missing measurements remain unknown; E0 has no executed training interval. '
    'Controls describe the completed interval; review actions apply afterwards. Epoch drift is not cumulative drift and does not by itself prove harm. '
    'Candidate training includes full-module displacement, actual-update probes, component gradients and regularization/exposure controls when measured. '
    'Retrieve relevant interventions and module_detail/module_gradients to explain key pre/post-intervention alternatives; do not treat their absence from a compact view as missing collection. '
    'Each domain normally uses its own BN; a source score with target BN is not target accuracy or a reason to require shared BN. '
    'Later authorized calibration is a separate post-selection operation on this exact frozen model, never feedback to training or selection.')
