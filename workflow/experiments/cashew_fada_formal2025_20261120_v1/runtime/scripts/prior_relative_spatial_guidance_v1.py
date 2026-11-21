"""User-observed density pattern, separate from hard labels and loss mechanics."""
import copy,hashlib,json
from pathlib import Path
VERSION='user_prior_observation_v1'
EXPERT_IDENTITY=(
    'You are a remote-sensing scientist, agricultural land-use expert and domain-adaptation training diagnostician with broad scientific common sense. '
    'Integrate evidence and mechanisms like a careful human expert. Do not mechanically execute prior quantiles or probability cutoffs. '
    'Before pixel selection or anomaly judgments, reason about the plausible order of magnitude of Cashew planting across WestAfrica, '
    'other crops, savannah/forest, drylands, settlements and localized orchards. Separate verified data from estimates/assumptions; '
    'do not invent acreage statistics or convert a rough base-rate estimate into a forced group/country area quota. '
    'Distinguish plantation footprint from canopy-pixel area, map12country coverage from the entire political subregion, and global rarity from locally dense production. '
    'Use source Cashew AEF embeddings accurately (they are not64 optical spectral bands), prior reliability, temporal maps, human FP/retention and gradient/domain evidence. '
    'Question both the prior and the model; explain spatial mechanisms, design a bounded useful tuning test, assess actual response and revise your judgment.'
)

def read_pinned(path,expected):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Pinned prior evidence identity changed')
    return json.loads(path.read_text())

def target(review):
    review=review or {}
    if 'review' in review:return target(review['review'])
    return review.get('target_unsupervised') or review.get('target') or {}

def enrich(packet,job,current_review,candidates):
    policy=job.get('prior_relative_spatial_guidance')
    if not policy:return packet
    observation=read_pinned(policy['observation'],policy['observation_sha256'])
    context=read_pinned(policy['tile_context'],policy['tile_context_sha256'])
    current={x['tile_id']:x for x in target(current_review).get('tile_estimates',[])}
    previous={}
    for item in reversed(candidates or []):
        if int(item['epoch']) < int(packet.get('epoch',0)):
            previous={x['tile_id']:x for x in target(item.get('review')).get('tile_estimates',[])}
            if previous:break
    spatial=[]
    for row in context['tiles']:
        value=copy.deepcopy(row);now=current.get(row['tile_id'],{});before=previous.get(row['tile_id'],{})
        fraction=now.get('class2_fraction');earlier=before.get('class2_fraction')
        value.update(predicted_Class2_fraction=fraction,previous_Class2_fraction=earlier,
                     change_in_fraction=None if fraction is None or earlier is None else fraction-earlier,
                     model_sampled_valid_pixels=now.get('sampled_valid_pixels'))
        spatial.append(value)
    packet['prior_relative_spatial_guidance']=dict(observation=observation,group_spatial_diagnostics=spatial,
        scope='Existing fixed3000 evaluation patches, tile-level predicted fractions and prior summaries; no target validation labels.',
        required_expert_review='State where predicted Cashew is unusually dense or spreading relative to the prior hierarchy, or state why the supplied spatial evidence is inconclusive. Assess the source/domain/human exposure mechanisms, choose a useful bounded adjustment or observation, then review its response. Do not reduce a healthy model merely to fit an area quota.',
        limitations='Tile/1km contexts can miss fine-scale orchards. Lower relative prior is a density warning, not crop absence. Sample prediction fractions are not measured crop fractions.')
    packet['job']['prior_relative_spatial_guidance']=copy.deepcopy(policy)
    return packet

def prompt(request):
    evidence=request.get('prior_relative_spatial_guidance')
    if not evidence:return ''
    return ('\nEXPERT IDENTITY AND APPROACH: '+EXPERT_IDENTITY+'\nMANDATORY USER OBSERVATION user_prior_observation_v1: prior values are generally high. '
        'The user observed dense Cashew planting mainly in the very highest-prior areas. '
        'Treat widespread dense predictions outside those strongest areas as a possible overprediction/negative-transfer pattern requiring spatial investigation, '
        'while retaining local exceptions and uncertainty. WestAfrica_whole_map_reference is the primary reference: use the ENTIRE approved12country WestAfrica map COMMON RAW-SCORE scale, '
        'not a within-group or within-country rescaling. Its p90/p95/p99 are about0.844/0.867/0.917; CIV roughly0.92/0.94 are its own upper-tail references. '
        'If every tile in a group has low raw scores, that group remains low relative to WestAfrica even when one tile ranks first locally. '
        'A positive within-group correlation between prior and predicted fractions does NOT rule out regionwide overprediction. '
        'Do not excuse dense predictions in an overall low-prior region merely because its densest tile is the best of that weak group. These references are '
        'not calibrated crop probabilities or hard masks. Missing prior is unknown. '
        'At EACH decision include user_prior_observation_v1 and WestAfrica_whole_map_reference in current_state_assessment.evidence and explain where the supplied spatial table is anomalous '
        'or why evidence is insufficient; distinguish inference from labels. Diagnose and tune like a human transfer expert: mechanisms, '
        'bounded parameter/dose tests, heldout human FP/positive retention, original source Cashew AEF distance, gradients and subsequent spatial response. '
        'This context informs ordinary tuning/STOP/SELECT as well as independent weak negatives; it must not become a compulsory negative count, '
        'high-p2 gate, crop-area quota or assumption of zero Cashew outside high values. The supplied quantiles describe the map, not decision rules. '
        'Use your own agricultural/remote-sensing common sense and uncertainty-aware whole-region base-rate reasoning before spatial diagnosis and sample selection. '
        'Source initialization and class weights remain unchanged.\n'
        'USER-APPROVED woody_wetland_confusion_expert_v1: diagnose supported forest/mixed-woody/wetland confusion with the supplied evidence and four-class head; there is no wetland posterior. '
        'External tree-cover includes orchards and cannot be a blanket negative. Localize errors and distinguish conditioning/alignment, vegetation ambiguity, label exposure and prior mistakes. '
        'Use a useful permitted reversible adjustment or matched experiment; choose its direction from mechanisms and measured response, not a fixed recipe. '
        'Track supported FP and positive retention, source validation and real exposure; area reduction alone is not success. Read the full approved Training Expert confusion section.\n')

def validate_semantic(request,semantic):
    if request.get('prior_relative_spatial_guidance'):
        evidence=(semantic.get('current_state_assessment') or {}).get('evidence',[])
        if not any(VERSION in str(x) for x in evidence) or not any('WestAfrica_whole_map_reference' in str(x) for x in evidence):
            raise ValueError('Explicitly review user_prior_observation_v1 AND WestAfrica_whole_map_reference in current_state_assessment.evidence; a weak-group local maximum cannot replace the wholeWestAfrica raw-score reference.')
