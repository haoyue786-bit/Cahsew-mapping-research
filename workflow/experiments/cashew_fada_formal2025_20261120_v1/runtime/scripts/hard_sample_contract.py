"""Point-isolated Expert contracts for acquisition and visual training fit."""

from jsonschema import Draft202012Validator

from hard_sample_scan import CATEGORIES, MAX_NEW
from sealed_packet import values_digest

ACQUISITION = 'safe_hard_sample_acquisition_v1'
REVIEW_KIND = 'annotated_training'
INVENTORY = {'scanned', 'valid_patches', 'valid_pixels', 'existing_patches', 'previously_unusable'} | {
    'available_' + k for k in CATEGORIES}


def planning_request(epoch, inventory, backend):
    if (type(epoch) is not int or epoch < 5 or epoch % 5 or set(inventory) != INVENTORY
            or any(type(v) is not int or v < 0 for v in inventory.values())
            or backend not in ('sample_assist', 'simulated')):
        raise ValueError('Invalid acquisition inventory')
    budget = {key:{'type':'integer','minimum':0,'maximum':min(MAX_NEW, inventory['available_'+key])}
              for key in CATEGORIES}
    fields = dict(budget, rationale={'type':'string','minLength':1})
    return dict(contract_version=ACQUISITION, epoch=epoch, backend=backend, evidence=inventory,
                instruction=('Plan the number of newly resolved locations in each category (total at most %d). '
                             'Start around six to ten only while distinct uncertain patterns merit that budget. '
                             'Read E0 and earlier evidence yourself. Consider reusable coverage and marginal '
                             'benefit. Include diffuse uncertainty and occasional geographically distinct checks. '
                             'Use miss_probe/extra_probe for small mid/late audits. A zero budget is valid. '
                             'This action changes only acquisition, with no optimizer or STOP decision. '
                             'Geometry, model predictions and candidate hypotheses stay with the worker. '
                             'A simulated backend supplies fabricated labels for integration verification.' % MAX_NEW),
                schema={'type':'object','additionalProperties':False,'required':list(fields),'properties':fields},
                capabilities={'external_tools':False,'historical_retrieval':True,
                              'history_query_required':True,'target_point_evidence':False})


def validate_plan(request, answer):
    Draft202012Validator(request['schema']).validate(answer)
    if sum(answer[key] for key in CATEGORIES) > MAX_NEW:
        raise ValueError('Acquisition total exceeds worker limit')
    return answer


def artifact(epoch, values):
    from online_boundary import verify_online
    verify_online(values)
    return dict(kind=REVIEW_KIND, epoch=epoch, values=values,
                sha256=values_digest(REVIEW_KIND, epoch, values))


def checked(value, epoch, trusted):
    if (value != artifact(epoch, value['values'])
            or (REVIEW_KIND, epoch, value['sha256']) not in trusted):
        raise ValueError('Untrusted annotated-training measurement')
    return {k:v for k,v in value.items() if k != 'kind'}
