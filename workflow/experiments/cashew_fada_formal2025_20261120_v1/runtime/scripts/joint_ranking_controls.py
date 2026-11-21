"""Bounded scientific controls, with host-owned unit-sum arithmetic.

No clipping, guessing, or automatic acceptance of invalid legacy responses.
The original strict numerical selection validator remains authoritative.
"""
import copy
import math

FIELDS = {'regional_prior_fraction', 'p2_share_of_nonprior'}
WEIGHTS = {'source_cashew_distance', 'p2_increase', 'regional_prior'}


def bounded(value, lower, upper, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f'{name} must be finite in [{lower}, {upper}]')
    return float(value)


def decode_selection(selection):
    """Decode the advertised new controls; valid legacy fractions pass unchanged."""
    if not isinstance(selection, dict):
        raise ValueError('automatic_negative must be an object')
    result = copy.deepcopy(selection)
    if 'ranking_controls' not in result:
        return result  # The caller must run the unchanged legacy validator.
    if 'ranking_weights' in result:
        raise ValueError('Choose one ranking representation, never both')
    controls = result.pop('ranking_controls')
    if not isinstance(controls, dict) or set(controls) != FIELDS:
        raise ValueError('Expected exactly regional_prior_fraction and p2_share_of_nonprior')
    prior = bounded(controls['regional_prior_fraction'], .25, 1, 'regional_prior_fraction')
    share = bounded(controls['p2_share_of_nonprior'], 0, 1, 'p2_share_of_nonprior')
    remaining = 1.0 - prior
    probability = remaining * share
    result['ranking_weights'] = dict(source_cashew_distance=remaining - probability,
                                     p2_increase=probability, regional_prior=prior)
    return result


def explicit_legacy_reexpression(selection):
    """One-time audited recovery, called explicitly, never by normal validation.

    Preserve the stated prior fraction and source:p2 ratio. Invalid doses,
    actions, missing values, NaN, negative or out-of-range components still fail.
    This function alone does not approve or publish a scientific decision.
    """
    result = copy.deepcopy(selection)
    if 'ranking_controls' in result:
        raise ValueError('Recovery expects legacy ranking_weights only')
    weights = result.pop('ranking_weights')
    if not isinstance(weights, dict) or set(weights) != WEIGHTS:
        raise ValueError('Expected the three original ranking components')
    source = bounded(weights['source_cashew_distance'], 0, 1, 'source_cashew_distance')
    probability = bounded(weights['p2_increase'], 0, 1, 'p2_increase')
    prior = bounded(weights['regional_prior'], .25, 1, 'regional_prior')
    if source + probability == 0:
        if prior != 1:
            raise ValueError('No source:p2 ratio specified for a nonzero remainder')
        share = 0.0
    else:
        share = probability / (source + probability)
    result['ranking_controls'] = dict(regional_prior_fraction=prior, p2_share_of_nonprior=share)
    return result
