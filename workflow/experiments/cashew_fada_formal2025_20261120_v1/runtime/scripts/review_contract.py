"""Experimental, tool-free structured review contract, not a live dispatcher.

Input records and receipts must be authenticated by a trusted, blind worker.
This module does not launch or authorize training, infer provenance, or select a
checkpoint. No historical free text, arbitrary retrieval, or target point data.
"""
import hashlib,json,math
from pathlib import Path
from jsonschema import Draft202012Validator
from sealed_packet import packet,training_window,evidence as checked_evidence,CONTROL,SOURCE,PANEL,AUTOMATIC_SUMMARY,AUTOMATIC_POOL,BRANCH
from worker_projection import (project_committed_action,project_committed_hypotheses,
                               structured_response_hypotheses,RESPONSE_METRICS)

ALLOWED_ACTIONS=frozenset(('KEEP','ADJUST','PAUSE_ADV','RESUME_ADV','REQUEST_EXPERIMENT','STOP'))
INTEGER_CONTROLS=frozenset(('D_steps_per_G_step','supplementary_occurrence_cap','supplementary_interval_steps','supplementary_horizon_epochs','main_lr_finish_epochs'))
INTEGER_CONTROLS = frozenset(INTEGER_CONTROLS) | {'bn_update_mode', 'hard_sample_repeats'}
CONTRACT_INSTRUCTION=('Review only the supplied structured evidence. Missing is unknown. '
 'Source validation and unlabeled target predictions do not establish target accuracy. '
 'Name a plausible alternative; do not force an adjustment. Any actual action is subject '
 'to worker bounds and committed only after validation. No external tools or history.')
HISTORY_REQUIRED_INSTRUCTION=('Review the current-epoch structured evidence and independently query this group\'s '
 'committed history with query_training_history before returning any decision. Query epoch 0 with kind '
 'initialization to inspect the starting model and baseline; when earlier trained epochs exist, query at least '
 'one relevant prior epoch using the evidence kinds and ranges you judge useful. Query additional intervals '
 'as needed to assess training response, earlier changes, and the expected value of continuing versus stopping. '
 'The current request does not preload a historical training window. The only available historical tool is '
 'the audited same-run query; target-point labels, scores and fitted thresholds are excluded. Missing '
 'measurements are unknown. Source validation and unlabeled target predictions do not establish target '
 'accuracy. Name a plausible alternative, do not force an adjustment, and stay within the worker bounds.')
TREND_INSTRUCTION=(' Prior-review changes, when present, compare a registered earlier review with '
 'the current one. Snapshot identity is based on file metadata, not verified content or pixel '
 'masks; equal support means counts only, not identical masks. A change in predicted class '
 'fraction is not pixelwise map agreement, accuracy, or evidence that an intervention caused it. '
 'A missing comparison or missing value means unknown; never invent a baseline.')
HYPOTHESIS_INSTRUCTION=(' For response_hypotheses, use null when unknown or select up to four '
 'advertised metrics with an observation direction and a future epoch. These are predictions '
 'to check, not measurements or proof of causality. Do not include target point metrics. '
 'After KEEP or CONTINUE, a hypothesis is monitoring without a parameter intervention. '
 'After ADJUST, PAUSE_ADV or RESUME_ADV, later changes are temporal observations, not proof '
 'of causation or that the intervention actually took effect. '
 'Free-text expectations never substitute for the structured fields.')
REFERENCE_NAMES=('hard-sample-training.md',)

def load_approved_references(skill_root,expected_hashes):
 """Load only preaudited, hash-pinned references, never general file access.

 The caller must authenticate the manifest and the trusted skill release; a
 caller-supplied digest alone does not establish that the text is safe.
 """
 if not isinstance(expected_hashes,dict) or set(expected_hashes)!=set(REFERENCE_NAMES):raise ValueError('reference manifest incomplete')
 directory=(Path(skill_root).resolve()/'references').resolve();out={}
 for name in REFERENCE_NAMES:
  p=(directory/name).resolve()
  if p.parent!=directory or not p.is_file():raise ValueError('invalid reference path')
  data=p.read_bytes()
  if len(data)>100000 or hashlib.sha256(data).hexdigest()!=expected_hashes[name]:raise ValueError('reference hash mismatch or too large')
  out[name]=data.decode('utf-8')
 return out

def validated_bounds(bounds):
 if not isinstance(bounds,dict) or not bounds or set(bounds)-CONTROL:raise ValueError('unapproved control bounds')
 cleaned={}
 for key,lim in sorted(bounds.items()):
  if not isinstance(lim,(list,tuple)) or len(lim)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in lim) or lim[0]>lim[1]:raise ValueError('invalid bounds')
  cleaned[key]=list(lim)
 return cleaned

def response_schema(bounds,allowed_actions,experiment_steps,schedules=None,initial_families=None,review_epoch=None,strict_schedule_schema=False):
 schedules=schedules or {}
 initial_families=initial_families or {}
 if type(strict_schedule_schema) is not bool:raise ValueError('invalid schedule schema policy')
 schema={'type':'object','additionalProperties':False,'required':['action','updates','schedule','initial_schedules','experiment','observations','diagnosis','alternative','expected_response','reversal_condition'],
  'properties':{'action':{'type':'string','enum':list(allowed_actions)},'updates':{'type':'array','maxItems':4,'items':{'type':'object','additionalProperties':False,'required':['parameter','value'],'properties':{'parameter':{'type':'string','enum':sorted(bounds)},'value':{'type':'number'}}}},
  'experiment':{'type':'object','additionalProperties':False,'required':['parameter','value','steps'],'properties':{'parameter':{'type':'string','enum':['none']+sorted(bounds)},'value':{'type':'number'},'steps':{'type':'integer','enum':[0]+sorted(experiment_steps)}}},
  'schedule':{'type':'object','additionalProperties':False,'required':['component','operation','phase'],'properties':{'component':{'type':'string','enum':['none']+sorted(schedules)},'operation':{'type':'string','enum':['none','hold','resume','set_phase','set_phase_hold','start_cosine_finish']},'phase':{'type':'number'}}},
  'initial_schedules':{'type':'array','maxItems':len(initial_families),'items':{'type':'object','additionalProperties':False,'required':['component','curve'],'properties':{'component':{'type':'string','enum':['none']+sorted(initial_families)},'curve':{'type':'string','enum':['constant','cosine','poly','dann_sigmoid','cosine_zero']}}}},
  'observations':{'type':'string'},'diagnosis':{'type':'string'},'alternative':{'type':'string'},'expected_response':{'type':'string'},'reversal_condition':{'type':'string'}}}
 if strict_schedule_schema and not schedules:
  schedule=schema['properties']['schedule']['properties']
  schedule['operation']['enum']=['none']
  schedule['phase']['enum']=[0]
 if review_epoch is not None:
  if type(review_epoch) is not int or review_epoch<1:raise ValueError('invalid hypothesis epoch')
  schema['required'].append('response_hypotheses')
  schema['properties']['response_hypotheses']={'anyOf':[{'type':'null'},{'type':'array','maxItems':4,'items':{
   'type':'object','additionalProperties':False,'required':['metric','direction','observe_after_epoch'],
   'properties':{'metric':{'type':'string','enum':sorted(RESPONSE_METRICS)},'direction':{'type':'string','enum':['increase','decrease','stable','inspect']},
    'observe_after_epoch':{'type':'integer','minimum':review_epoch+1,'maximum':review_epoch+20}}}}]}
  if 'representation_anchor_weight' in bounds or 'hard_sample_weight' in bounds:
   schema['required'].append('intervention_review')
   schema['properties']['intervention_review']={'type':'array','maxItems':6,'items':{
    'type':'object','additionalProperties':False,'required':['origin_epoch','status','evidence_epochs','assessment'],
    'properties':{'origin_epoch':{'type':'integer','minimum':1,'maximum':max(1,review_epoch-1)},
     'status':{'type':'string','enum':['supported','contradicted','inconclusive','pending']},
     'evidence_epochs':{'type':'array','minItems':1,'uniqueItems':True,'items':{'type':'integer','minimum':0,'maximum':review_epoch}},
     'assessment':{'type':'string','minLength':1}}}}
 return schema

def build_full_review(epoch,rows,source,panel,trusted,bounds,allowed_actions,previous=None,hypotheses=(),experiment_steps=(),context=None,schedules=None,automatic_summary=None,automatic_max_weight=.005,trend_context=None,prior_response=None,human_default=None,history_required=False,strict_schedule_schema=False,human_training=None,matched_trial_capability=None,annotated_training=None):
 if type(epoch) is not int or epoch<1:raise ValueError('invalid epoch')
 if not isinstance(allowed_actions,(list,tuple)) or not allowed_actions or len(set(allowed_actions))!=len(allowed_actions) or set(allowed_actions)-ALLOWED_ACTIONS:raise ValueError('invalid actions')
 if not isinstance(experiment_steps,(tuple,list)) or len(set(experiment_steps))!=len(experiment_steps) or any(type(s) is not int or s<1 or s>128 for s in experiment_steps):raise ValueError('invalid experiment budget')
 if 'REQUEST_EXPERIMENT' in allowed_actions and not experiment_steps:raise ValueError('experiments require an explicit budget')
 schedules=schedules or {}
 valid_operations={'hold','resume','set_phase','set_phase_hold','start_cosine_finish'}
 if not isinstance(schedules,dict) or set(schedules)-{'main_lr','D_lr','lambda_adv','supplementary_loss_weight'} or any(not isinstance(v,list) or not v or len(set(v))!=len(v) or set(v)-valid_operations for v in schedules.values()):raise ValueError('invalid schedule capabilities')
 evidence=packet(epoch,rows,source,panel,trusted)
 if human_default is not None:
  from human_negative_epoch10 import validate_brief
  evidence['human_supplementary_default']=validate_brief(human_default)
 if human_training is not None:
  from human_training_evidence_contract import checked
  evidence['human_training']=checked(human_training,epoch,trusted)
 if annotated_training is not None:
  from hard_sample_contract import checked
  evidence['annotated_training'] = checked(annotated_training,epoch,trusted)
 # This comparison is reconstructed from independently registered reviews,
 # never accepted as caller-supplied numeric deltas. The caller must attest
 # lineage, both producer identities and ownership of the run out-of-band.
 evidence['prior_review_trend']=None
 evidence['current_review_binding_sha256']=None
 if trend_context is not None:
  from review_trends_v1 import compare,_checked_record
  fields={'job_id','run_id','previous_review','current_binding','trusted_bindings','current_checkpoint_sha256'}
  if not isinstance(trend_context,dict) or set(trend_context)!=fields:raise ValueError('invalid trend worker context')
  current_binding=trend_context['current_binding']
  if not isinstance(current_binding,dict) or current_binding.get('epoch')!=epoch or current_binding.get('checkpoint_sha256')!=trend_context['current_checkpoint_sha256']:
   raise ValueError('trend checkpoint differs from live worker')
  registered=trend_context['trusted_bindings']
  if not isinstance(registered,(list,tuple,set,frozenset)) or any(not isinstance(x,(list,tuple)) or len(x)!=4 for x in registered):raise ValueError('invalid trusted review bindings')
  records={tuple(x) for x in registered}
  current={'source':source,'panel':panel,'binding':current_binding}
  # Even without a previous review, a supplied trend context must bind the
  # current measurements to the registered same-run checkpoint identity.
  _checked_record(current,trend_context['job_id'],trend_context['run_id'],epoch,trusted,records)
  evidence['current_review_binding_sha256']=hashlib.sha256(json.dumps(current_binding,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
  evidence['prior_review_trend']=compare(trend_context['previous_review'],current,
   job_id=trend_context['job_id'],run_id=trend_context['run_id'],trusted_values=trusted,trusted_bindings=records)
 # Facts about batch size and data exposure are numeric context, not evidence
 # of target accuracy and not automatically adjustable parameters.
 if context is not None:
  evidence['training_context']=checked_evidence(context,'context',{'batch_size','source_train_count','target_train_count','max_epoch','source_year','target_year','source_class2_weight','new_module_lr_multiplier'},epoch,trusted)
 if type(history_required) is not bool:raise ValueError('invalid history requirement')
 if history_required:
  if len(evidence['training_window'])!=1 or previous is not None or hypotheses or prior_response is not None or (trend_context is not None and trend_context.get('previous_review') is not None):
   raise ValueError('history-required review must not preload past evidence')
 elif len(evidence['training_window'])!=min(5,epoch):raise ValueError('incomplete review window')
 prior=None
 if previous is not None:
  if not isinstance(previous,tuple) or len(previous)!=2:raise ValueError('invalid prior action')
  prior=project_committed_action(previous[0],previous[1],epoch)
 expected=[]
 if prior_response is not None:
  if not prior or not isinstance(prior_response,dict) or set(prior_response)!={'response_bytes','commit','job_id','run_id'}:raise ValueError('hypotheses require a committed prior response')
  if trend_context is not None and (prior_response['job_id'],prior_response['run_id'])!=(trend_context['job_id'],trend_context['run_id']):raise ValueError('hypothesis/trend run mismatch')
  expected=project_committed_hypotheses(prior_response['response_bytes'],prior_response['commit'],previous[0],previous[1],job_id=prior_response['job_id'],run_id=prior_response['run_id'],cutoff=epoch)
 if hypotheses and list(hypotheses)!=expected:raise ValueError('hypotheses differ from committed response')
 bounds=validated_bounds(bounds)
 schema=response_schema(bounds,allowed_actions,experiment_steps,schedules,review_epoch=epoch,strict_schedule_schema=strict_schedule_schema)
 auto_limit=0
 if automatic_summary is not None:
  if type(automatic_max_weight) not in (int,float) or not math.isfinite(automatic_max_weight) or not 0<automatic_max_weight<=.005:raise ValueError('invalid automatic-negative dose cap')
  auto=checked_evidence(automatic_summary,'automatic_summary',AUTOMATIC_SUMMARY,epoch,trusted)
  if auto['epoch']!=epoch or type(auto['values'].get('candidate_count')) is not int or auto['values']['candidate_count']<0:raise ValueError('invalid automatic-negative summary')
  auto_limit=auto['values']['candidate_count'];evidence['automatic_negative_candidates']=auto
  if set(auto['values']) != (AUTOMATIC_SUMMARY if auto_limit else {'candidate_count'}):raise ValueError('automatic-negative summary is incomplete')
  if auto_limit:
   for prefix in ('source_distance','p2_increase','regional_context'):
    values=[auto['values'][prefix+'_'+q] for q in ('q25','q50','q75')]
    if any(x is None for x in values) or values!=sorted(values):raise ValueError('invalid candidate quantiles')
   if any(auto['values']['source_distance_'+q]<0 or not -1<=auto['values']['p2_increase_'+q]<=1 or not 0<=auto['values']['regional_context_'+q]<=1 for q in ('q25','q50','q75')):raise ValueError('candidate quantile outside range')
  schema['required'].append('automatic_negative')
  schema['properties']['automatic_negative']={'type':'object','additionalProperties':False,'required':['action','selected_count','weight','regional_prior_fraction','p2_share_of_nonprior','rationale','expected_response','reversal_condition'],
   'properties':{'action':{'type':'string','enum':['ABSTAIN','ACTIVATE'] if auto_limit else ['ABSTAIN']},
    'selected_count':{'type':'integer','minimum':0,'maximum':auto_limit},'weight':{'type':'number','minimum':0,'maximum':automatic_max_weight},
    'regional_prior_fraction':{'type':'number','minimum':.25,'maximum':1},'p2_share_of_nonprior':{'type':'number','minimum':0,'maximum':1},
    'rationale':{'type':'string'},'expected_response':{'type':'string'},'reversal_condition':{'type':'string'}}}
 instruction=(HISTORY_REQUIRED_INSTRUCTION if history_required else CONTRACT_INSTRUCTION)+TREND_INSTRUCTION+HYPOTHESIS_INSTRUCTION
 if 'representation_anchor_weight' in bounds or 'hard_sample_weight' in bounds:
  instruction+=(' Complete recall is available: query interventions for the latest actual ADJUST and any relevant earlier intervention; '
   'read its expected_response, reversal_condition, actual exposure and subsequent observations. '
   'Return intervention_review with origin_epoch, supported/contradicted/inconclusive/pending, evidence_epochs and assessment; '
   'use [] when no earlier ADJUST exists. Pending is valid before a meaningful response horizon. '
   'This run retains original source/training reasoning without keyword redaction. '
   'Distinguish raw component gradients, executed Adam updates and functional response. '
   'Use an advertised multi-epoch trial horizon that can answer the stated mechanism; do not choose the shortest budget by habit.')
 if 'representation_anchor_weight' in bounds:
  instruction+=' Optional inherited-parameter anchors start at zero and can be tested, increased, reduced or disabled.'
 if human_default is not None and human_default.get('version')=='user_2019_human_decision_before_epoch10_v2' and epoch==9:
  instruction+=(' This review decides human supervision for epoch 10. Query initialization and relevant earlier execution. '
   'Explicitly explain whether to enable, retain or defer it, and justify its dose and actual exposure. '
   'There is no automatic 0.5 activation. KEEP explicitly retains the current weight, including zero; '
   'ADJUST may set supplementary_loss_weight within the declared bounds. Zero remains a valid decision, '
   'and later reviews may reconsider it. Do not infer benefit from sample availability or mere activation.')
 if human_training is not None:
  instruction+=' Human training polygon evidence is a fixed-panel training-fit diagnostic, never independent accuracy. Pairing checks establish imagery year, geometry and content identity, not label correctness or spatial representativeness. Zero loss weight does not make probe measurements unavailable. Use negative and positive support separately; missing signs are unknown. Do not infer a causal effect from an unmatched before/after change or require a positive weight merely because samples exist. Query human_training together with training, controls and action at initialization and relevant earlier epochs to assess actual exposure, component gradients and response; current-only evidence does not replace history.'
 if annotated_training is not None:
  instruction+=(' Query annotated_training for the prior fit and check executed exposure. Judge STOP primarily from '
                'marginal reductions in comparable losses across intervals. New label intake changes the training '
                'objective and in-pool difficulty. Keep the source and paired previous-library fit separate. '
                'Consider the first-use baseline and actual optimizer dose before claiming the model learned from labels. '
                'The labeled subset is training data, and simulated data are for engineering tests only.')
 if strict_schedule_schema and not schedules:
  instruction+=(' No schedule operation is currently available. Set schedule component and operation to none, and phase to 0. '
                'A duration parameter by itself does not activate a schedule.')
 if automatic_summary is not None:
  instruction+=(' Automatic-negative candidates are unverified pseudo-labels, not confirmed negatives. '
   'Source distance, rising probabilities and regional context alone do not establish correctness. '
   'Consider alternative explanations, actual prior exposure, expected response and reversal; '
   'abstain when a candidate-and-dose direction is unsupported. Ranking fractions sum to one after host decoding.')
 result = {'contract_version':'safe_full_review_candidate_v1','instruction':instruction,'epoch':epoch,
  'evidence':evidence,'prior_committed_action':prior,'prior_structured_hypotheses':expected,'prior_hypotheses_status':'declared' if expected else 'unknown','allowed_parameter_bounds':bounds,'allowed_experiment_steps':list(experiment_steps),'allowed_schedules':schedules,'automatic_candidate_limit':auto_limit,'automatic_max_weight':automatic_max_weight if automatic_summary is not None else 0,'automatic_selection_enabled':automatic_summary is not None,'schema':schema,
  'capabilities':({'external_tools':False,'historical_retrieval':True,'history_query_required':True,'target_point_evidence':False} if history_required else {'external_tools':False,'historical_retrieval':False,'target_point_evidence':False})}
 if annotated_training is not None:
  if any(v is not None for v in (automatic_summary,human_default,human_training)):
   raise ValueError('Visual annotation and legacy supplementary routes cannot coexist')
  for key in ('automatic_candidate_limit','automatic_max_weight','automatic_selection_enabled'):
   result.pop(key)

 if matched_trial_capability is not None:
  from paired_trial_v1 import trial_max_epoch
  if trial_max_epoch(matched_trial_capability)<=epoch or not history_required:raise ValueError('invalid matched trial policy')
  result['matched_trial_capability'] = dict(matched_trial_capability)
  result['capabilities']['matched_training_trial'] = True
  result['instruction'] += ' The audited run_matched_training_trial tool is available for optional discriminating experiments. Read history first; state one mechanism, expected response and falsification before a trial. Compare control and treatment, including adverse or null outcomes. Small-panel response is not target accuracy; do not automatically select the largest source/proxy score. Trials never alter formal state or become SELECT candidates. Final ADJUST changes only advertised controls on the formal trajectory; KEEP retains them.'
 return result

def build_anomaly_review(epoch,rows,trusted,bounds,trigger,previous=None,automatic_pool=None,history_required=False,strict_schedule_schema=False):
 """A bounded daily review; no reused source/target measurements or STOP."""
 if trigger not in ('missing_training_metric','metric_out_of_range','unexpected_missing_history','unexpected_auxiliary_state','training_metric_deviation'):
  raise ValueError('unrecognized or non-review trigger')
 bounds=validated_bounds(bounds)
 if set(bounds)-{'main_lr','D_lr','lambda_adv'}:raise ValueError('light review has restricted controls')
 evidence=training_window(epoch,rows,trusted)
 if type(history_required) is not bool:raise ValueError('invalid history requirement')
 if history_required:
  if len(evidence)!=1 or previous is not None:raise ValueError('history-required review must not preload past evidence')
 elif len(evidence)!=min(5,epoch):raise ValueError('incomplete worker records')
 prior=None
 if previous is not None:
  if not isinstance(previous,tuple) or len(previous)!=2:raise ValueError('invalid prior action')
  prior=project_committed_action(previous[0],previous[1],epoch)
 actions=['KEEP','ADJUST']
 pool=None
 if automatic_pool is not None:
  pool=checked_evidence(automatic_pool,'automatic_pool',AUTOMATIC_POOL,epoch,trusted)
  weight=pool['values'].get('weight')
  if pool['epoch']!=epoch or type(weight) not in (int,float) or not 0<weight<=.005:raise ValueError('automatic pool is not active')
  actions.append('DISABLE_AUTOMATIC_NEGATIVE')
 instruction=(HISTORY_REQUIRED_INSTRUCTION if history_required else CONTRACT_INSTRUCTION)+HYPOTHESIS_INSTRUCTION
 if strict_schedule_schema:
  instruction+=' No schedule operation is currently available. Set schedule component and operation to none, and phase to 0.'
 return {'contract_version':'safe_anomaly_review_candidate_v1','instruction':instruction,
  'epoch':epoch,'trigger':trigger,'evidence':{'training_window':evidence,'active_automatic_pool':pool},'prior_committed_action':prior,
  'allowed_parameter_bounds':bounds,'allowed_experiment_steps':[],'allowed_schedules':{},'schema':response_schema(bounds,actions,[],review_epoch=epoch,strict_schedule_schema=strict_schedule_schema),
  'capabilities':({'external_tools':False,'historical_retrieval':True,'history_query_required':True,'target_point_evidence':False} if history_required else {'external_tools':False,'historical_retrieval':False,'target_point_evidence':False})}

def build_branch_review(epoch,rows,source,panel,trusted,bounds,allowed_actions,*,previous,branch,branch_parameter,schedules=None,context=None,automatic_summary=None,automatic_max_weight=.005):
 """Review a real branch at the same training epoch, never its raw narrative.

 Caller must authenticate branch executor's matched-state and fixed-panel
 metadata, compare against the committed prior decision, and reject untrusted
 historical results. A failed branch supplies no fabricated response values.
 """
 if not isinstance(previous,tuple) or len(previous)!=2 or branch_parameter not in CONTROL:raise ValueError('missing prior experiment identity')
 prior=project_committed_action(previous[0],previous[1],epoch+1)
 if prior['epoch']!=epoch or prior['action']!='REQUEST_EXPERIMENT':raise ValueError('prior decision is not this epoch experiment')
 original=json.loads(previous[0]);experiment=original.get('experiment') or {};updates=experiment.get('parameter_updates')
 if not isinstance(updates,list) or len(updates)!=1 or updates[0].get('parameter')!=branch_parameter:raise ValueError('prior experiment parameter mismatch')
 if 'REQUEST_EXPERIMENT' in allowed_actions:raise ValueError('nested experiment review is not authorized')
 b=checked_evidence(branch,'branch',BRANCH,epoch,trusted)
 if b['epoch']!=epoch or type(b['values'].get('status_code')) is not int or b['values']['status_code'] not in (0,1,2,3,4):raise ValueError('invalid branch result status')
 values=b['values'];status=values['status_code'];completed=values.get('completed_steps')
 if completed is not None and (type(completed) is not int or not 0<=completed<=128):raise ValueError('invalid branch completed steps')
 if status==1:
  before=(original.get('intervention') or {}).get('parameters_before') or {}
  if completed!=experiment.get('steps') or values.get('control_value')!=before.get(branch_parameter) or values.get('treatment_value')!=updates[0].get('value'):
   raise ValueError('branch execution does not match committed prior')
  if set(values)!=BRANCH or any(values.get(role+'_'+field) is None for role in ('control','treatment') for field in ('start_source_class2_f1','end_source_class2_f1','start_target_class2_fraction','end_target_class2_fraction')):
   raise ValueError('completed branch lacks observation support')
  for role in ('control','treatment'):
   for field in ('start_source_class2_f1','end_source_class2_f1','start_target_class2_fraction','end_target_class2_fraction'):
    if not 0<=values[role+'_'+field]<=1:raise ValueError('branch observation outside range')
   gradient=values[role+'_end_effective_gradient_ratio']
   if gradient is not None and gradient<0:raise ValueError('invalid branch gradient ratio')
  if values['control_start_source_class2_f1']!=values['treatment_start_source_class2_f1'] or values['control_start_target_class2_fraction']!=values['treatment_start_target_class2_fraction']:
   raise ValueError('branches have different baseline observations')
 else:
  if set(values)-{'status_code','completed_steps'}:raise ValueError('unfinished experiment may not claim response')
 request=build_full_review(epoch,rows,source,panel,trusted,bounds,allowed_actions,schedules=schedules,context=context,automatic_summary=automatic_summary,automatic_max_weight=automatic_max_weight)
 request['contract_version']='safe_branch_review_candidate_v1';request['prior_committed_action']=prior
 request['evidence']['matched_branch_result']=b;request['branch_parameter']=branch_parameter
 request['instruction']+=(' The reported paired branch is limited to the supplied start and step budget. '
  'Source and unlabelled target changes are not target accuracy; an unfinished branch has unknown effects. '
  'Do not infer causal attribution beyond the authenticated matching or automatically adopt a branch.')
 return request

def build_initialization_review(initial_controls,source,panel,trusted,bounds,initial_families=None,initial_schedule_baseline=None,transfer_record=None,human_default=None,human_training=None):
 """Build a zero-epoch, point-sealed initializer with no historical memories.

 Origin and baseline-checkpoint identity remain the worker's responsibility.
 """
 bounds=validated_bounds(bounds)
 c=checked_evidence(initial_controls,'initial_controls',CONTROL,0,trusted,min_epoch=0)
 s=checked_evidence(source,'source',SOURCE,0,trusted,min_epoch=0)
 p=checked_evidence(panel,'panel',PANEL,0,trusted,min_epoch=0)
 if set(c['values'])!=set(bounds):raise ValueError('initial control/bounds mismatch')
 for key,(lo,hi) in bounds.items():
  value=c['values'][key]
  if value is None or not lo<=value<=hi or (key in INTEGER_CONTROLS and type(value) is not int):raise ValueError('invalid initial control')
 initial_families=initial_families or {}
 allowed={'main_lr':{'constant','cosine','poly'},'D_lr':{'constant','cosine','poly'},'lambda_adv':{'constant','dann_sigmoid'},'supplementary_loss_weight':{'constant','cosine_zero'}}
 if not isinstance(initial_families,dict) or set(initial_families)-set(allowed) or any(not isinstance(v,list) or not v or len(set(v))!=len(v) or set(v)-allowed[k] for k,v in initial_families.items()):raise ValueError('invalid initial schedule families')
 initial_schedule_baseline=initial_schedule_baseline or {}
 if not isinstance(initial_schedule_baseline,dict) or set(initial_schedule_baseline)!=set(initial_families) or any(v not in initial_families[k] for k,v in initial_schedule_baseline.items()):raise ValueError('missing or invalid initial schedule baseline')
 schema=response_schema(bounds,['KEEP','ADJUST'],[],initial_families=initial_families)
 schema['properties']['updates']['maxItems']=len(bounds)
 from transfer_record_contract_v1 import validate as validate_transfer
 evidence={'initial_controls':c,'source_validation':s,'unlabelled_panel':p}
 if transfer_record is not None:evidence['transfer_identity']=validate_transfer(transfer_record)
 if human_default is not None:
  from human_negative_epoch10 import validate_brief
  evidence['human_supplementary_default']=validate_brief(human_default)
 instruction=CONTRACT_INSTRUCTION
 if human_training is not None:
  from human_training_evidence_contract import checked
  evidence['human_training']=checked(human_training,0,trusted)
  instruction+=' Human training polygon evidence is a fixed-panel training-fit diagnostic, never independent accuracy. Pairing checks establish imagery year, geometry and content identity, not label correctness or spatial representativeness. Zero loss weight does not make probe measurements unavailable. Use negative and positive support separately; missing signs are unknown. Do not infer a causal effect from an unmatched before/after change or require a positive weight merely because samples exist.'
 return {'contract_version':'safe_initialization_candidate_v1','instruction':instruction,
  'epoch':0,'evidence':evidence,
  'allowed_parameter_bounds':bounds,'allowed_experiment_steps':[],'allowed_schedules':{},'allowed_initial_families':initial_families,'initial_schedule_baseline':initial_schedule_baseline,'schema':schema,
  'capabilities':{'external_tools':False,'historical_retrieval':False,'target_point_evidence':False}}

def build_checkpoint_selection(stop_epoch,candidates,trusted,history_required=False,selection_policy=None,candidate_details=None):
 """Create a point-sealed candidate list after the stopping epoch is frozen.

 Caller authenticates stop state, artifact producer and checkpoint mapping.
 The returned selected epoch is not a frozen model and opens no evaluation.
 """
 from checkpoint_selection_contract_v2 import VERSION,checked as checked_detail,summary,comparison_schema,INSTRUCTION,DECODER
 complete=selection_policy==VERSION
 if selection_policy is not None and not complete:raise ValueError('unknown selection policy')
 if type(stop_epoch) is not int or stop_epoch<1 or not isinstance(candidates,list) or not 1<=len(candidates)<=(201 if complete else 200):raise ValueError('invalid frozen stop or candidates')
 allowed={'source_class2_f1','target_unlabelled_class2_fraction','target_unlabelled_pixel_agreement','target_unlabelled_class2_iou','training_cls_loss'}
 clean=[checked_evidence(c,'candidate',allowed,stop_epoch,trusted,min_epoch=0 if complete else 1) for c in candidates]
 epochs=[x['epoch'] for x in clean]
 if len(set(epochs))!=len(epochs) or epochs!=sorted(epochs):raise ValueError('duplicate or unordered candidate epoch')
 schema={'type':'object','additionalProperties':False,'required':['selected_epoch','reason','uncertainty'],
         'properties':{'selected_epoch':{'type':'integer','enum':epochs},'reason':{'type':'string'},'uncertainty':{'type':'string'}}}
 if type(history_required) is not bool:raise ValueError('invalid history requirement')
 request={'contract_version':'safe_checkpoint_selection_candidate_v1','instruction':HISTORY_REQUIRED_INSTRUCTION if history_required else CONTRACT_INSTRUCTION,
         'stop_epoch':stop_epoch,'candidate_observations':clean,'schema':schema,
         'capabilities':({'external_tools':False,'historical_retrieval':True,'history_query_required':True,'target_point_evidence':False} if history_required else {'external_tools':False,'historical_retrieval':False,'target_point_evidence':False})}
 if complete:
  if not history_required or len(epochs)<2 or epochs[0]!=0 or epochs[-1]!=stop_epoch:raise ValueError('complete selector requires E0, STOP and mandatory history')
  if not isinstance(candidate_details,list) or len(candidate_details)!=len(clean):raise ValueError('missing candidate details')
  details=[checked_detail(x,trusted) for x in candidate_details]
  if [x['epoch'] for x in details]!=epochs or any(summary(d['values'])!=c['values'] for d,c in zip(details,clean)):raise ValueError('candidate summary and details differ')
  fits=[x['values'].get('annotated_training') for x in details]
  sections = comparison_schema(epochs)
  section_enum = sections['properties']['evidence_used']['items']['properties']['section']['enum']
  if any(fit is not None for fit in fits):
   if not all(isinstance(fit,dict) for fit in fits):raise ValueError('Candidate annotation coverage is incomplete')
   signature=lambda f:(f['backend'],f['library_revision'],f['all']['patches'],f['all']['valid_pixels'],tuple(f['all']['class_pixels']))
   if len({signature(f) for f in fits})!=1:raise ValueError('Candidate annotation populations differ')
   section_enum.remove('human_training')
  else:
   humans=[x['values']['human_training'] for x in details]
   if not all(isinstance(h,dict) for h in humans) or len({(h['job_id'],h['run_id']) for h in humans})!=1 or any(h['inventory']!=humans[0]['inventory'] for h in humans):raise ValueError('candidate human populations differ')
   section_enum.remove('annotated_training')
  request.update(selection_policy=VERSION,candidate_details=details,prediction_contract=dict(DECODER),schema=sections)
  request['instruction']+=INSTRUCTION
 elif candidate_details is not None:raise ValueError('unadvertised candidate details')
 return request

def validate_checkpoint_selection(request,answer):
 error=next(Draft202012Validator(request['schema']).iter_errors(answer),None)
 if error is not None:raise ValueError('invalid selected checkpoint: '+error.message)
 if type(answer['selected_epoch']) is not int:raise ValueError('selected epoch must be an exact integer')
 if request.get('selection_policy') is not None:
  from checkpoint_selection_contract_v2 import VERSION,validate_comparison
  if request['selection_policy']!=VERSION:raise ValueError('unknown selection policy')
  validate_comparison(request,answer)
 return {'selected_epoch':answer['selected_epoch']}

def validate_review_response(request,answer):
 from hard_sample_contract import ACQUISITION, validate_plan
 if request.get('contract_version') == ACQUISITION:
  return validate_plan(request,answer)
 """Validate the response before any optimizer or schedule can change."""
 if not isinstance(answer,dict):raise ValueError('invalid review response')
 # Preserve reading of old already-committed answers; new strict schemas ask
 # explicitly for null or a structured array. Missing is unknown, not prose.
 answer=dict(answer)
 if 'response_hypotheses' in request['schema'].get('properties',{}) and 'response_hypotheses' not in answer:answer['response_hypotheses']=None
 error=next(Draft202012Validator(request['schema']).iter_errors(answer),None)
 if error is not None:raise ValueError('invalid review response: '+error.message)
 if any(not answer[k].strip() for k in ('observations','diagnosis','alternative','expected_response','reversal_condition')):raise ValueError('scientific justification fields must be non-empty')
 updates=answer['updates'];names=[u['parameter'] for u in updates]
 if any(type(u['value']) not in (int,float) or not math.isfinite(u['value']) for u in updates):raise ValueError('nonfinite parameter update')
 bounds=validated_bounds(request['allowed_parameter_bounds'])
 for u in updates:
  key,value=u['parameter'],u['value'];lo,hi=bounds[key]
  if not lo<=value<=hi or (key in INTEGER_CONTROLS and (type(value) is not int or isinstance(value,bool))):raise ValueError('parameter outside allowed bounds or integer contract')
 if len(names)!=len(set(names)):raise ValueError('duplicate parameter updates')
 schedule=answer['schedule'];component=schedule['component'];operation=schedule['operation'];phase=schedule['phase']
 if type(phase) not in (int,float) or not math.isfinite(phase):raise ValueError('invalid schedule phase')
 if component=='none':
  if operation!='none' or phase!=0:raise ValueError('invalid empty schedule')
  schedule_update=None
 else:
  if operation not in request['allowed_schedules'].get(component,[]):raise ValueError('schedule operation not advertised')
  if operation in ('set_phase','set_phase_hold'):
   if not 0<=phase<=1:raise ValueError('schedule phase outside bounds')
   schedule_update={'component':component,'operation':operation,'phase':phase}
  else:
   if phase!=0:raise ValueError('unused schedule phase must be zero')
   schedule_update={'component':component,'operation':operation,'phase':None}
 families=request.get('allowed_initial_families',{});initial_choices=answer['initial_schedules'];seen=set()
 for item in initial_choices:
  component=item['component']
  if component in seen or item['curve'] not in families.get(component,[]):raise ValueError('initial schedule choice not advertised')
  seen.add(component)
 if request.get('contract_version')=='safe_initialization_candidate_v1':
  baseline=request['evidence']['initial_controls']['values'];schedule_baseline=request['initial_schedule_baseline']
  changed=any(baseline[x['parameter']]!=x['value'] for x in updates) or any(schedule_baseline[x['component']]!=x['curve'] for x in initial_choices)
 else:changed=bool(updates or schedule_update or initial_choices)
 if (answer['action']=='ADJUST') != changed:raise ValueError('action and updates disagree')
 experiment=answer['experiment'];param=experiment['parameter'];value=experiment['value'];steps=experiment['steps']
 if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('nonfinite experiment value')
 if answer['action']=='REQUEST_EXPERIMENT':
  if param=='none' or steps not in request['allowed_experiment_steps']:raise ValueError('experiment missing bounded parameter or budget')
  lo,hi=bounds[param]
  if not lo<=value<=hi or (param in INTEGER_CONTROLS and type(value) is not int):raise ValueError('invalid experiment parameter')
 elif param!='none' or steps!=0 or value!=0:raise ValueError('unexpected experiment proposal')
 automatic=None
 if request.get('automatic_selection_enabled'):
  a=answer['automatic_negative'];count=a['selected_count'];weight=a['weight'];prior=a['regional_prior_fraction'];share=a['p2_share_of_nonprior']
  if type(count) is not int or not 0<=count<=request['automatic_candidate_limit'] or any(type(x) not in (int,float) or not math.isfinite(x) for x in (weight,prior,share)) or not 0<=weight<=request['automatic_max_weight'] or not .25<=prior<=1 or not 0<=share<=1:raise ValueError('invalid automatic-negative controls')
  if (a['action']=='ACTIVATE' and (count==0 or weight==0 or answer['action'] in ('STOP','REQUEST_EXPERIMENT'))) or (a['action']=='ABSTAIN' and (count!=0 or weight!=0)):raise ValueError('invalid automatic-negative action')
  if any(not a[k].strip() for k in ('rationale','expected_response','reversal_condition')):raise ValueError('automatic-negative rationale required')
  automatic=dict(a)
 claims=answer.get('response_hypotheses')
 claims=structured_response_hypotheses(request['epoch'],claims) if claims is not None else []
 return {'action':answer['action'],'updates':[dict(u) for u in updates], 'schedule_update':schedule_update,'initial_schedule_choices':[dict(x) for x in initial_choices],'experiment':dict(experiment) if answer['action']=='REQUEST_EXPERIMENT' else None,'automatic_negative':automatic,'response_hypotheses':claims}
