"""Experimental bridge into the actual Cashew worker's native normalizer.

No training, execution, file I/O or model calls. The worker must authenticate the
safe request and job lineage; unsupported stages and missing safety metadata
remain blocked. Agent prose is not used as an online history in later calls.
"""
import copy,hashlib,json
from review_contract import validate_review_response,validate_checkpoint_selection
from expert_policy_v4 import normalize,parameter_bounds,schedule_curves
from light_review_dispatch import normalize_light
from joint_ranking_controls import decode_selection
from automatic_negative_selection_validation_v1 import validate_selection

def normalize_review(safe_request,answer,config,job,*,minimum_stop_epoch,stop_allowed,source_collapsed,experiment_attempts=0,experiment_limit=0):
 kind=safe_request.get('contract_version')
 if kind not in ('safe_full_review_candidate_v1','safe_branch_review_candidate_v1'):raise ValueError('only full and branch reviews are implemented')
 if type(minimum_stop_epoch) is not int or minimum_stop_epoch<0 or type(stop_allowed) is not bool or type(source_collapsed) is not bool:raise ValueError('missing stopping safety state')
 result=validate_review_response(safe_request,answer);epoch=safe_request['epoch']
 actual=parameter_bounds(config);claimed=safe_request['allowed_parameter_bounds']
 if any(k not in actual or list(actual[k])!=list(v) for k,v in claimed.items()):raise ValueError('request bounds differ from current worker')
 action=result['action'];before_state=config.get('adversarial_state','active')
 if before_state not in ('active','paused'):raise ValueError('unknown adversarial state')
 if action=='PAUSE_ADV' and before_state!='active':raise ValueError('already paused')
 if action=='RESUME_ADV' and before_state!='paused':raise ValueError('not paused')
 after_state='paused' if action=='PAUSE_ADV' else 'active' if action=='RESUME_ADV' else before_state
 updates=result['updates']
 if action=='ADJUST' and any(config[u['parameter']]==u['value'] for u in updates):raise ValueError('requested update has no effect')
 experiment=None
 if result['experiment']:
  x=result['experiment'];experiment={'kind':'paired_training','hypothesis':answer['diagnosis'],
    'parameter_updates':[{'parameter':x['parameter'],'value':x['value']}],
    'adversarial_state_after':before_state,'steps':x['steps'],
    'expected_observation':answer['expected_response'],'falsification':answer['reversal_condition']}
 semantic={'action':action,'continue_training':action!='STOP','selected_epoch':None,'reason_codes':[],
  'analysis_summary':answer['diagnosis'],'previous_hypothesis_review':{'interventions':answer.get('intervention_review',[])},
  'current_state_assessment':{'state':'UNKNOWN','summary':answer['diagnosis'],'evidence':[answer['observations'],answer['alternative']]},
  'intervention':{'adversarial_state_after':after_state,'parameter_updates':updates,'schedule_choices':[],
    'schedule_updates':[result['schedule_update']] if result['schedule_update'] else [],'emergency':False,'rationale':answer['diagnosis']},
  'experiment':experiment,
  'next_stage_expectation':{'hypothesis':answer['diagnosis'],'expected_source':answer['expected_response'],
   'expected_target':'Unlabelled target response is not target accuracy.',
   'expected_discriminator':answer['expected_response'],'expected_gradient_ratio':answer['expected_response'],
   'failure_triggers':[answer['reversal_condition']]}}
 worker_request={'decision_kind':'experiment_review' if kind=='safe_branch_review_candidate_v1' else 'training_review','epoch':epoch,'current_config':config,
  'allowed_actions':safe_request['schema']['properties']['action']['enum'],'job':job,
  'safety_envelope':{'minimum_stop_epoch':minimum_stop_epoch,'stop_allowed':stop_allowed,'source_collapsed':source_collapsed},
  'source_protection_policy':{'source_collapsed':source_collapsed},
  'experiment_capability':{'enabled':bool(safe_request['allowed_experiment_steps']),'kinds':['paired_training'],
   'steps':safe_request['allowed_experiment_steps'],'attempts_used':experiment_attempts,'max_per_tile':experiment_limit}}
 request_hash=hashlib.sha256(json.dumps(safe_request,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 return normalize(worker_request,semantic,request_hash,planet_used=False)

def normalize_initialization(safe_request,answer,config,job):
 """Map point-sealed initialization to the worker's full native contract.

 Unmentioned parameters retain the recorded baseline; they are expanded for
 the native initializer, not presented as individual Agent changes.
 """
 if safe_request.get('contract_version')!='safe_initialization_candidate_v1' or safe_request.get('epoch')!=0:raise ValueError('invalid initialization contract')
 result=validate_review_response(safe_request,answer);bounds=parameter_bounds(config)
 if set(bounds)!=set(safe_request['allowed_parameter_bounds']) or any(list(bounds[k])!=list(safe_request['allowed_parameter_bounds'][k]) for k in bounds):raise ValueError('initial bounds differ from current worker')
 initial=safe_request['evidence']['initial_controls']['values']
 if any(initial[k]!=config[k] for k in bounds):raise ValueError('initial control baseline differs from worker')
 updates={x['parameter']:x['value'] for x in result['updates']}
 if result['action']=='ADJUST' and any(updates[k]==config[k] for k in updates):raise ValueError('requested update has no effect')
 actual_curves=schedule_curves(config);advertised=safe_request.get('allowed_initial_families',{})
 if any(k not in actual_curves or set(v)!=set(actual_curves[k]) for k,v in advertised.items()):raise ValueError('initial schedule capabilities differ from worker')
 if any(config.get('schedule_state',{}).get(k,{}).get('curve')!=v for k,v in safe_request.get('initial_schedule_baseline',{}).items()):raise ValueError('initial schedule baseline differs from worker')
 proposed={x['component']:x['curve'] for x in result['initial_schedule_choices']};choices=[]
 for component in actual_curves:
  state=config.get('schedule_state',{}).get(component)
  if not isinstance(state,dict) or state.get('curve') not in schedule_curves(config)[component]:raise ValueError('missing initial schedule')
  choices.append({'component':component,'curve':proposed.get(component,state['curve'])})
 if result['action']=='ADJUST' and not updates and not any(config['schedule_state'][k]['curve']!=v for k,v in proposed.items()):raise ValueError('initial adjustment has no effect')
 semantic={'action':result['action'],'continue_training':True,'selected_epoch':None,'reason_codes':[],
  'analysis_summary':answer['diagnosis'],'previous_hypothesis_review':{},
  'current_state_assessment':{'state':'UNKNOWN','summary':answer['diagnosis'],'evidence':[answer['observations'],answer['alternative']]},
  'intervention':{'adversarial_state_after':config.get('adversarial_state','active'),
    'parameter_updates':[{'parameter':k,'value':updates.get(k,config[k])} for k in sorted(bounds)],
    'schedule_choices':choices,'schedule_updates':[],'emergency':False,'rationale':answer['diagnosis']},
  'experiment':None,'next_stage_expectation':{'hypothesis':answer['diagnosis'],'expected_source':answer['expected_response'],
   'expected_target':'Unlabelled target response is not target accuracy.','expected_discriminator':answer['expected_response'],
   'expected_gradient_ratio':answer['expected_response'],'failure_triggers':[answer['reversal_condition']]}}
 worker_request={'decision_kind':'initialization','epoch':0,'historical_anchor_config':config,
   'allowed_actions':safe_request['schema']['properties']['action']['enum'],'job':job}
 request_hash=hashlib.sha256(json.dumps(safe_request,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 return normalize(worker_request,semantic,request_hash,planet_used=False)

def normalize_anomaly_review(safe_request,answer,config):
 """Apply the existing restricted light-review validator, without old history."""
 if safe_request.get('contract_version')!='safe_anomaly_review_candidate_v1':raise ValueError('invalid anomaly review contract')
 result=validate_review_response(safe_request,answer);actual=parameter_bounds(config)
 if any(k not in actual or list(actual[k])!=list(v) for k,v in safe_request['allowed_parameter_bounds'].items()):raise ValueError('light bounds differ from worker')
 if result['action']=='DISABLE_AUTOMATIC_NEGATIVE':
  evidence=safe_request['evidence'].get('active_automatic_pool') or {};recorded=(evidence.get('values') or {}).get('weight')
  if recorded is None or (config.get('_automatic_negative_state') or {}).get('weight')!=recorded:raise ValueError('active pool evidence is stale or missing')
 native={'action':result['action'],'parameter_updates':result['updates'],
  'observations':answer['observations'],'mechanism':answer['diagnosis']+'; alternative: '+answer['alternative'],
  'expected_response':answer['expected_response'],'reversal_condition':answer['reversal_condition']}
 request_hash=hashlib.sha256(json.dumps(safe_request,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 return normalize_light(native,config,safe_request['epoch'],request_hash)

def normalize_selection(safe_request,answer,config,job,*,frozen_stop_epoch):
 """Validate selection; the caller must still freeze a checkpoint hash."""
 if safe_request.get('contract_version')!='safe_checkpoint_selection_candidate_v1' or type(frozen_stop_epoch) is not int or safe_request.get('stop_epoch')!=frozen_stop_epoch:raise ValueError('selection stop identity mismatch')
 result=validate_checkpoint_selection(safe_request,answer);epoch=result['selected_epoch']
 semantic={'action':'SELECT_CHECKPOINT','continue_training':False,'selected_epoch':epoch,'reason_codes':[],
  'analysis_summary':answer['reason'],'previous_hypothesis_review':{},
  'current_state_assessment':{'state':'UNKNOWN','summary':answer['reason'],'evidence':[]},
  'intervention':{'adversarial_state_after':config.get('adversarial_state','active'),'parameter_updates':[],
   'schedule_choices':[],'schedule_updates':[],'emergency':False,'rationale':answer['reason']},
  'experiment':None,'next_stage_expectation':{'hypothesis':answer['reason'],'expected_source':'Unknown',
   'expected_target':'Unknown','expected_discriminator':'Unknown','expected_gradient_ratio':'Unknown',
   'failure_triggers':[answer['uncertainty']]}}
 worker_request={'decision_kind':'checkpoint_selection','epoch':frozen_stop_epoch,'current_config':config,
  'allowed_actions':['SELECT_CHECKPOINT'],'eligible_candidates':[x['epoch'] for x in safe_request['candidate_observations']],'job':job}
 request_hash=hashlib.sha256(json.dumps(safe_request,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 decision=normalize(worker_request,semantic,request_hash,planet_used=False)
 if safe_request.get('selection_policy') is not None:
  from checkpoint_selection_contract_v2 import validate_comparison
  decision['selection_comparison']=validate_comparison(safe_request,answer)
 return decision

def normalize_automatic_selection(safe_request,answer,*,actual_candidate_count,actual_max_weight):
 """Validate joint auto-negative choice; never execute or refresh a pool here."""
 result=validate_review_response(safe_request,answer);auto=result['automatic_negative']
 if auto is None:raise ValueError('automatic-negative selection was not advertised')
 if type(actual_candidate_count) is not int or actual_candidate_count!=safe_request['automatic_candidate_limit'] or actual_max_weight!=safe_request['automatic_max_weight']:raise ValueError('candidate catalog or job dose cap changed')
 native={'action':auto['action'],'selected_count':auto['selected_count'],'automatic_weight':auto['weight'],
  'ranking_controls':{'regional_prior_fraction':auto['regional_prior_fraction'],'p2_share_of_nonprior':auto['p2_share_of_nonprior']},
  'rationale':auto['rationale'],'expected_response':auto['expected_response'],'reversal_condition':auto['reversal_condition']}
 return validate_selection(decode_selection(native),actual_candidate_count)
