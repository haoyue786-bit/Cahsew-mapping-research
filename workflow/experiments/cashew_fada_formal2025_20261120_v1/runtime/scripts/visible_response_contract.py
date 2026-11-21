"""Expose the caller's existing JSON schema; do not change scientific policy."""
import json


def visible_contract(schema):
    return (
        'REQUIRED OUTPUT CONTRACT: return one JSON object with exactly training and automatic_negative. '
        'The input brief is evidence, not an output template. The training object must contain every '
        'required field below, including continue_training (boolean) and selected_epoch (integer or null). '
        'Never substitute a selection object for selected_epoch. Do not add keys. '
        'For a non-selection review selected_epoch must be null. Keep the scientific decision yours; '
        'this schema specifies representation only. Output JSON without markdown.\n'
        'AUTHORITATIVE_JSON_SCHEMA:\n' + json.dumps(schema, separators=(',', ':'))
    )


def validate_training_fields(schema, value):
    """Report actionable missing/extra keys before the unchanged native validator."""
    def check(node, specification, path):
        if not isinstance(node, dict):
            raise ValueError({'path': path, 'expected': 'object'})
        missing = sorted(set(specification.get('required', [])) - set(node))
        extra = sorted(set(node) - set(specification.get('properties', {})))
        if missing or (extra and specification.get('additionalProperties') is False):
            raise ValueError({'path': path, 'missing_required_fields': missing,
                              'unexpected_fields': extra,
                              'required_fields': specification.get('required', [])})
    check(value['training'], schema['properties']['training'], 'training')
    check(value['training'].get('intervention'), schema['$defs']['intervention'], 'training.intervention')
