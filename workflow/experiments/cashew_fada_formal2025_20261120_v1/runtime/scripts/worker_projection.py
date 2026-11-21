"""Project live worker measurements to fixed numeric fields for isolated review.

This function never proves that an input was produced by a live worker: the
caller must authenticate provenance and epoch before use. It deliberately does
not copy arbitrary nested values, histories, paths, labels or free text.
"""
import hashlib,json,math
from sealed_packet import TRAIN, CONTROL, SOURCE, PANEL, nums

OPTIMIZERS=('G','D_local','D_global')
BASE_FIELDS=('cls','seg_cls','adv','D_loss','local_D_acc','global_D_acc','cls_grad',
             'adv_grad','effective_gradient_ratio','steps','regularization_loss')
RESPONSE_METRICS=frozenset(('source_class2_f1','source_loss','training_cls_loss',
 'effective_gradient_ratio','local_D_acc','global_D_acc','target_unlabelled_class2_fraction',
 'target_unlabelled_pixel_agreement','sampled_temperature_response'))
RESPONSE_METRICS |= frozenset(('human_training_negative_cashew_pixels','human_training_positive_missed_pixels','regularization_loss'))
RESPONSE_METRICS |= frozenset(k for k in TRAIN if
 (k.startswith('detail_') and k.endswith(('_full_initial_relative_l2','_full_epoch_relative_l2'))) or
 (k.startswith('gradient_') and k.endswith(('_human_negative_norm','_source_negative_cosine','_regularization_norm'))) or
 (k.startswith('actual_update_') and k.endswith('_relative_l2_median')))
LOGIT_FIELDS=('sampled_pixels','current_cap_pixel_fraction','current_cap_element_fraction',
 'temperature_down_mean_abs_change','temperature_up_mean_abs_change',
 'temperature_down_pixel_fraction_changed','temperature_up_pixel_fraction_changed',
 'cap_down_mean_abs_change','cap_up_mean_abs_change',
 'cap_down_pixel_fraction_changed','cap_up_pixel_fraction_changed')

def project_controls(metrics):
 """Configured/effective epoch controls; actual optimizer groups are separate."""
 if not isinstance(metrics,dict):raise ValueError('missing worker metrics')
 updates=metrics.get('experimental_actual_updates') or {}
 if not isinstance(updates,dict):raise ValueError('invalid updates')
 controls=updates.get('training_controls') or {}
 if not isinstance(controls,dict):raise ValueError('invalid control record')
 return nums({k:controls[k] for k in CONTROL if k in controls},CONTROL)

def project_full_review(source_stats,unlabelled_stats):
 """Project the two *separate live producer outputs*; no raw labels/history.

 The caller must bind these arguments to the worker's source validation and
 unlabelled-panel functions. A mapping or hash alone cannot prove origin.
 """
 if not isinstance(source_stats,dict) or not isinstance(unlabelled_stats,dict):raise ValueError('missing independent producers')
 f1=source_stats.get('f1_per_class');distribution=unlabelled_stats.get('class_distribution')
 if not isinstance(f1,(list,tuple)) or len(f1)!=4 or not isinstance(distribution,(list,tuple)) or len(distribution)!=4:
  raise ValueError('unexpected class metric dimensions')
 if any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in distribution) or abs(sum(distribution)-1)>1e-3:
  raise ValueError('invalid unlabelled class distribution')
 source=nums({'class2_f1':f1[2]},SOURCE);panel=nums({'predicted_class2_fraction':distribution[2]},PANEL)
 for value in (source['class2_f1'],panel['predicted_class2_fraction']):
  if value is not None and not 0<=value<=1:raise ValueError('metric outside probability range')
 return source,panel

def project_verified_panel(source_stats,unlabelled_stats,*,current,expected_current,roster,expected_roster_sha256,transition=None,previous=None,transition_identity=None,probe_identity=None,expected_probe_identity=None):
 """Return detailed numeric evidence only with independently bound identities.

 Fixed ordered tile indices and a SHA-256 of the ordered roster let the caller
 bind panel/comparison provenance outside this numeric packet. The hashes are
 identity hints, not proof of clean lineage or unchanged file contents. The
 first-64 SND is omitted unless its own subset/mask identities are supplied and
 independently verified. No target labels, example texts or tile IDs are sent.
 """
 from panel_projection_v1 import project
 if not isinstance(roster,(list,tuple)) or len(roster)>64:raise ValueError('unsupported tile roster')
 roster_sha=hashlib.sha256(json.dumps(list(roster),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
 if roster_sha!=expected_roster_sha256:raise ValueError('tile roster identity mismatch')
 d=project(source_stats,unlabelled_stats,current,roster=roster,expected_current=expected_current,transition=transition,previous=previous,transition_identity=transition_identity,probe_identity=probe_identity,expected_probe_identity=expected_probe_identity)
 s=d['source_validation'];t=d['unlabelled_target_panel'];f=s['class_f1'];q=t['class_fraction']
 source={'source_loss':s['loss'],'validation_pixels':s['validation_pixels'],**{'class%d_f1'%i:f[i] for i in range(4)}}
 panel={'predicted_class2_fraction':q[2],'class0_fraction':q[0],'class1_fraction':q[1],'class3_fraction':q[3],
  'mean_entropy':t['entropy'],'normalized_entropy':t['normalized_entropy'],'sample_patch_count':t['sample_patch_count'],
  'sample_valid_pixel_count':t['sample_valid_pixel_count'],'missing_tile_count':t['missing_tile_count'],'probe_snd':t['probe_snd']}
 if t.get('probe_support'):
  panel['probe_target_patch_count']=t['probe_support']['target_patch_count'];panel['probe_source_patch_count']=t['probe_support']['source_patch_count']
 for tile in t['tiles']:
  i=tile['tile_index']
  for field in ('class2_fraction','sample_patch_count','sampled_valid_pixels'):panel['tile_%d_%s'%(i,field)]=tile[field]
 c=d['adjacent_prediction_comparison']
 if c is not None:
  panel.update(pixel_agreement=c['all_class_pixel_agreement'],class2_iou=c['class2_iou'],comparison_from_epoch=c['from_epoch'],comparison_to_epoch=c['to_epoch'],comparison_class2_pixels_previous=c['class2_pixels_previous'],comparison_class2_pixels_current=c['class2_pixels_current'])
 return nums(source,SOURCE),nums(panel,PANEL)

def project_candidate(source_values,panel_values,training_values):
 """Candidate metrics are derived from separately projected current-epoch facts.

 Caller must commit this at the actual review epoch and bind the corresponding
 checkpoint; values alone do not establish checkpoint identity.
 """
 source=nums(source_values,SOURCE);panel=nums(panel_values,PANEL);training=nums(training_values,TRAIN)
 if source.get('class2_f1') is None or panel.get('predicted_class2_fraction') is None:raise ValueError('candidate needs measured source and unlabelled panel')
 candidate={'source_class2_f1':source['class2_f1'],'target_unlabelled_class2_fraction':panel['predicted_class2_fraction']}
 for key in ('pixel_agreement','class2_iou'):
  if panel.get(key) is not None:candidate['target_unlabelled_'+key]=panel[key]
 if training.get('cls') is not None:candidate['training_cls_loss']=training['cls']
 return candidate

def project_negative_summary(summary):
 """Keep only unlabelled-candidate distribution statistics; never row IDs."""
 if not isinstance(summary,dict) or type(summary.get('candidate_count')) is not int or summary['candidate_count']<0:raise ValueError('invalid candidate count')
 out={'candidate_count':summary['candidate_count']};distribution=summary.get('distribution')
 if not isinstance(distribution,dict):raise ValueError('missing candidate distribution')
 for field,prefix in (('source_cashew_distance_mean','source_distance'),('p2_increase_mean','p2_increase'),('regional_context_rank','regional_context')):
  record=distribution.get(field)
  if not isinstance(record,dict) or record.get('count')!=summary['candidate_count'] or not isinstance(record.get('quantiles'),dict):raise ValueError('invalid candidate distribution')
  if summary['candidate_count']:
   for q,label in (('0.25','q25'),('0.5','q50'),('0.75','q75')):
    if q not in record['quantiles']:raise ValueError('missing candidate quantile')
    out[prefix+'_'+label]=record['quantiles'][q]
 allowed={'candidate_count'}|{p+'_'+q for p in ('source_distance','p2_increase','regional_context') for q in ('q25','q50','q75')}
 return nums(out,allowed)

def project_committed_action(decision_bytes, receipt, cutoff):
 """Read a previously committed action without passing the Agent's prose.

 The caller must separately authenticate the worker's receipt and checkpoint;
 a matching hash establishes byte identity, not safe origin or actual execution.
 """
 if not isinstance(decision_bytes,bytes) or not isinstance(receipt,dict):raise ValueError('missing committed action')
 if hashlib.sha256(decision_bytes).hexdigest()!=receipt.get('decision_sha256'):raise ValueError('decision hash mismatch')
 d=json.loads(decision_bytes)
 if type(cutoff) is not int or type(d.get('epoch')) is not int or d['epoch']>=cutoff or d['epoch']<1 or receipt.get('epoch')!=d['epoch']:raise ValueError('wrong action epoch')
 action=d.get('action')
 if action not in ('ADJUST','KEEP','PAUSE_ADV','RESUME_ADV','CONTINUE','REQUEST_EXPERIMENT'):raise ValueError('unrecognized action')
 intervention=d.get('intervention')
 if not isinstance(intervention,dict):raise ValueError('missing intervention')
 before,after=intervention.get('parameters_before'),intervention.get('parameters_after')
 if not isinstance(before,dict) or not isinstance(after,dict):raise ValueError('missing parameter values')
 before=nums({k:before[k] for k in CONTROL if k in before},CONTROL)
 after=nums({k:after[k] for k in CONTROL if k in after},CONTROL)
 changed={k:{'before':before[k],'after':after[k]} for k in before.keys() & after.keys() if before[k]!=after[k]}
 meaning=('monitoring_without_parameter_intervention' if action in ('KEEP','CONTINUE') else
          'committed_intervention_later_observations_are_not_causal_proof')
 return {'epoch':d['epoch'],'action':action,'parameter_changes':changed,'interpretation':meaning,
         'status':'decision committed; execution must be independently authenticated'}

def structured_response_hypotheses(epoch, proposals):
 """Validate prospective observations; no free text or held-out point metrics."""
 if type(epoch) is not int or epoch < 1 or not isinstance(proposals,list) or len(proposals)>4:
  raise ValueError('invalid response hypotheses')
 safe=[];seen=set()
 for p in proposals:
  if not isinstance(p,dict) or set(p)!={'metric','direction','observe_after_epoch'}:raise ValueError('invalid response hypothesis')
  if p['metric'] not in RESPONSE_METRICS or p['direction'] not in ('increase','decrease','stable','inspect'):
   raise ValueError('unapproved response hypothesis')
  if type(p['observe_after_epoch']) is not int or not epoch<p['observe_after_epoch']<=epoch+20:
   raise ValueError('invalid observation epoch')
  identity=(p['metric'],p['observe_after_epoch'])
  if identity in seen:raise ValueError('duplicate response hypothesis')
  seen.add(identity)
  safe.append(dict(p))
 return safe

def project_committed_hypotheses(response_bytes,commit,decision_bytes,decision_receipt,*,job_id,run_id,cutoff):
 """Extract only structured claims from the committed response bytes.

 The caller MUST separately authenticate the immutable same-run transaction,
 original request/response receipt, skill binding and applied checkpoint. An
 unkeyed commit hash alone is not proof of origin, execution or blind lineage.
 Missing/null/empty legacy fields are unknown and never inferred from prose.
 """
 if not isinstance(response_bytes,bytes) or not isinstance(decision_bytes,bytes) or not isinstance(commit,dict) or not isinstance(decision_receipt,dict):raise ValueError('missing committed response')
 identity=commit.get('identity')
 if not isinstance(identity,dict) or identity.get('job_id')!=job_id or identity.get('run_id')!=run_id:raise ValueError('cross-run response')
 epoch=identity.get('epoch')
 if type(epoch) is not int or type(cutoff) is not int or not 1<=epoch<cutoff or decision_receipt.get('epoch')!=epoch:raise ValueError('response epoch mismatch')
 if hashlib.sha256(response_bytes).hexdigest()!=commit.get('response_sha256') or hashlib.sha256(decision_bytes).hexdigest()!=commit.get('decision_sha256') or decision_receipt.get('decision_sha256')!=commit.get('decision_sha256'):raise ValueError('response/decision commit mismatch')
 for key in ('request_sha256','pre_checkpoint_sha256','post_checkpoint_sha256'):
  value=commit.get(key)
  if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('incomplete commit identity')
 if any(not isinstance(commit.get(k),str) or not commit[k] for k in ('thread_id','turn_id')):raise ValueError('missing turn receipt identity')
 answer=json.loads(response_bytes)
 if not isinstance(answer,dict):raise ValueError('invalid response')
 decision=json.loads(decision_bytes)
 if not isinstance(decision,dict) or answer.get('action')!=decision.get('action'):
  raise ValueError('response/decision action mismatch')
 claims=answer.get('response_hypotheses')
 if claims is None:return []
 return structured_response_hypotheses(epoch,claims)

def project_training(metrics):
 from module_dynamics_contract import project as project_modules
 if not isinstance(metrics,dict):raise ValueError('missing worker metrics')
 output={k:metrics[k] for k in BASE_FIELDS if k in metrics}
 if 'module_dynamics' in metrics:output.update(project_modules(metrics['module_dynamics']))
 from module_observer_v2 import project as project_detail
 output.update(project_detail(metrics))
 domain=metrics.get('source_target_actual_exposure')
 if domain is not None:
  if not isinstance(domain,dict) or domain.get('version')!='source_target_exposure_v1':raise ValueError('unknown source/target exposure format')
  for field in ('planned_steps','loaded_steps','completed_g_steps','target_loss_active_completed_steps',
   'source_planned_occurrences','source_planned_unique','target_planned_occurrences','target_planned_unique',
   'source_loaded_occurrences','source_loaded_unique','target_loaded_occurrences','target_loaded_unique',
   'source_completed_occurrences','source_completed_unique','target_completed_occurrences','target_completed_unique',
   'target_loss_active_occurrences','target_loss_active_unique','target_generator_nonzero_weight_steps',
   'target_generator_nonzero_weight_occurrences','target_generator_nonzero_weight_unique'):
   if field in domain:output['domain_'+field]=domain[field]
 exposure=metrics.get('supplementary_actual_exposure')
 if exposure is not None:
  if not isinstance(exposure,dict) or exposure.get('version')!='supplementary_actual_exposure_v1':raise ValueError('unknown supplementary exposure format')
  fields=('planned_steps','planned_occurrences','planned_unique_patches','attempted_steps','attempted_occurrences','attempted_unique_patches','loaded_steps','loaded_occurrences','loaded_unique_patches',
   'forward_steps','loss_steps','completed_optimizer_steps','completed_occurrences','completed_unique_patches','human_negative_effective_steps','human_positive_effective_steps',
   'human_negative_effective_pixel_occurrences','human_positive_effective_pixel_occurrences','human_negative_effective_unique_patches','human_positive_effective_unique_patches','unknown_sign_completed_steps')
  for field in fields:
   if field in exposure:output['supp_'+field]=exposure[field]
 for src,prefix,fields in (
  ('polygon_supervision','human_polygon',('negative_pixel_occurrences','positive_pixel_occurrences','positive_scale','effective_negative_weight_min','effective_negative_weight_max','effective_positive_weight_min','effective_positive_weight_max')),
  ('automatic_negative','automatic_negative',('effective_weight','configured_weight','updates','unique_patches','patch_occurrences','pixel_occurrences','selected_patch_count','pool_age_epochs')),
  ('gradient_direction','gradient_direction',('measured_steps','completed_steps','opposing_steps','source_adversarial_cosine_median','source_local_cosine_median','source_global_cosine_median','local_global_cosine_median','weighted_opposition_relative_to_source_median')),
 ):
  if src not in metrics:continue
  block=metrics[src]
  if not isinstance(block,dict):raise ValueError('invalid measurement block')
  for field in fields:
   if field in block:output[prefix+'_'+field]=block[field]
 updates=metrics.get('experimental_actual_updates')
 if updates is None:return nums(output,TRAIN)
 if not isinstance(updates,dict):raise ValueError('invalid recorded updates')
 steps=updates.get('steps') or {}
 if not isinstance(steps,dict):raise ValueError('invalid step counts')
 for opt in OPTIMIZERS:
  if opt in steps:output[opt+'_steps']=steps[opt]
 for source,name in (('attempted_steps','attempted_steps'),('steps_with_gradient','steps_with_gradient')):
  values=updates.get(source) or {}
  if not isinstance(values,dict):raise ValueError('invalid optimizer execution counts')
  for opt in OPTIMIZERS:
   if opt in values:output[opt+'_'+name]=values[opt]
 groups=updates.get('optimizer_groups') or {}
 if not isinstance(groups,dict):raise ValueError('invalid optimizer groups')
 for opt in OPTIMIZERS:
  group_list=groups.get(opt,[])
  if not isinstance(group_list,list) or len(group_list)>32:raise ValueError('invalid optimizer group count')
  for i,group in enumerate(group_list):
   if not isinstance(group,dict):raise ValueError('invalid optimizer group')
   for field in ('lr','lr_multiplier','weight_decay','eps'):
    if field in group:output[f'{opt}_group_{i}_{field}']=group[field]
   betas=group.get('betas')
   if betas is not None:
    if not isinstance(betas,(list,tuple)) or len(betas)!=2:raise ValueError('invalid optimizer beta')
    output[f'{opt}_group_{i}_beta1'],output[f'{opt}_group_{i}_beta2']=betas
 seen=set()
 for group in updates.get('sample_groups',[]):
  if not isinstance(group,dict) or group.get('optimizer') not in OPTIMIZERS or type(group.get('group')) is not int or not 0<=group['group']<32:raise ValueError('invalid sampled group')
  opt=group['optimizer'];i=group['group']
  if (opt,i) in seen:raise ValueError('duplicate sampled group')
  seen.add((opt,i))
  if group.get('finite') is not True:raise ValueError('unverified sampled update')
  for field in ('sample_delta_l2','sample_before_l2','sample_count'):
   if field in group:output[f'{opt}_group_{i}_{field}']=group[field]
 coverage=updates.get('sample_tensor_coverage')
 if coverage is not None:
  if not isinstance(coverage,list) or len(coverage)>10000:raise ValueError('invalid tensor sample coverage')
  for row in coverage:
   if not isinstance(row,dict) or row.get('optimizer') not in OPTIMIZERS or type(row.get('group')) is not int or not 0<=row['group']<32:raise ValueError('invalid coverage group')
   eligible=row.get('eligible_tensors');sampled=row.get('sampled_tensors')
   if type(eligible) is not int or type(sampled) is not int or not 0<=sampled<=eligible:raise ValueError('invalid tensor sample counts')
   prefix=f"{row['optimizer']}_group_{row['group']}_coverage_"
   for field,value in (('sampled_steps',1),('sampled_tensors',sampled),('eligible_tensors',eligible)):
    output[prefix+field]=output.get(prefix+field,0)+value
 response=updates.get('logit_response')
 if response is not None:
  if not isinstance(response,dict) or type(response.get('available')) is not bool:raise ValueError('invalid response availability')
  if response['available']:
   for key,field in (('temperature','sampled_temperature'),('confidence_cap','sampled_cap')):
    if key not in response:raise ValueError('missing response baseline')
    output[field]=response[key]
   for key,prefix in (('temperature_alternatives','temperature_probe'),('confidence_cap_alternatives','cap_probe')):
    values=response.get(key)
    if not isinstance(values,list) or len(values)!=2:raise ValueError('invalid response alternatives')
    output[prefix+'_lower'],output[prefix+'_upper']=values
   domains=response.get('domains')
   if not isinstance(domains,dict):raise ValueError('missing response domains')
   for domain in ('source','target_unlabeled'):
    if domain not in domains:continue
    values=domains[domain]
    if not isinstance(values,dict):raise ValueError('invalid domain')
    for field in LOGIT_FIELDS:
     if field in values:output[domain+'_'+field]=values[field]
 return nums(output,TRAIN)
