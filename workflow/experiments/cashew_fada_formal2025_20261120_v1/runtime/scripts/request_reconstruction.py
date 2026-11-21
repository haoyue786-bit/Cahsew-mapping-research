"""Reconstruct permitted request shapes before a future model call.

 The trusted set must come from an authenticated, blind worker producer and
 same-run journal. Hash membership alone does not prove semantic provenance.
"""
from review_contract import build_full_review,build_branch_review,build_anomaly_review,build_initialization_review,build_checkpoint_selection

def as_art(kind,d):
 if not isinstance(d,dict) or set(d)!={'epoch','sha256','values'}:raise ValueError('invalid projected artifact')
 return {'kind':kind,**d}
def rows_from(evidence):
 rows=[]
 for row in evidence['training_window']:
  if not isinstance(row,dict) or set(row)!={'training','controls'}:raise ValueError('invalid training row')
  rows.append({'training':as_art('training',row['training']),'controls':as_art('controls',row['controls'])})
 return rows

def verify_request(request,trusted,*,expected,previous=None):
 if not isinstance(request,dict):raise ValueError('invalid request')
 if not isinstance(expected,dict):raise ValueError('missing independent worker capabilities')
 history_required=expected.get('history_required',False)
 if type(history_required) is not bool:raise ValueError('invalid worker history requirement')
 strict_schedule_schema=expected.get('strict_schedule_schema',False)
 if type(strict_schedule_schema) is not bool:raise ValueError('invalid worker schedule schema policy')
 kind=request.get('contract_version');epoch=request.get('epoch');e=request.get('evidence',{})
 if kind!='safe_checkpoint_selection_candidate_v1' and (type(expected.get('epoch')) is not int or epoch!=expected['epoch']):raise ValueError('request epoch differs from live worker')
 if kind=='safe_full_review_candidate_v1':
  context=as_art('context',e['training_context']) if 'training_context' in e else None
  automatic=as_art('automatic_summary',e['automatic_negative_candidates']) if 'automatic_negative_candidates' in e else None
  rebuilt=build_full_review(epoch,rows_from(e),as_art('source',e['source_validation']),as_art('panel',e['unlabelled_panel']),trusted,
   expected['bounds'],expected['actions'],previous=previous,
   hypotheses=expected.get('hypotheses',()),experiment_steps=expected.get('experiment_steps',()),context=context,
   schedules=expected.get('schedules',{}),automatic_summary=automatic,automatic_max_weight=expected.get('automatic_max_weight',.005),
   trend_context=expected.get('trend_context'),prior_response=expected.get('prior_response'),human_default=expected.get('human_default'),history_required=history_required,strict_schedule_schema=strict_schedule_schema,human_training=expected.get('human_training'),matched_trial_capability=expected.get('matched_trial_capability'),annotated_training=expected.get('annotated_training'))
  if (automatic is not None)!=expected.get('automatic_enabled',False):raise ValueError('automatic-negative capability mismatch')
 elif kind=='safe_branch_review_candidate_v1':
  context=as_art('context',e['training_context']) if 'training_context' in e else None
  automatic=as_art('automatic_summary',e['automatic_negative_candidates']) if 'automatic_negative_candidates' in e else None
  rebuilt=build_branch_review(epoch,rows_from(e),as_art('source',e['source_validation']),as_art('panel',e['unlabelled_panel']),trusted,
   expected['bounds'],expected['actions'],previous=previous,branch=as_art('branch',e['matched_branch_result']),branch_parameter=expected['branch_parameter'],
   schedules=expected.get('schedules',{}),context=context,automatic_summary=automatic,automatic_max_weight=expected.get('automatic_max_weight',.005))
  if (automatic is not None)!=expected.get('automatic_enabled',False):raise ValueError('automatic-negative capability mismatch')
 elif kind=='safe_anomaly_review_candidate_v1':
  pool=e.get('active_automatic_pool');pool=as_art('automatic_pool',pool) if pool is not None else None
  rebuilt=build_anomaly_review(epoch,rows_from(e),trusted,expected['bounds'],expected['trigger'],previous=previous,automatic_pool=pool,history_required=history_required,strict_schedule_schema=strict_schedule_schema)
  if (pool is not None)!=expected.get('automatic_pool_enabled',False):raise ValueError('automatic pool capability mismatch')
 elif kind=='safe_initialization_candidate_v1':
  rebuilt=build_initialization_review(as_art('initial_controls',e['initial_controls']),as_art('source',e['source_validation']),as_art('panel',e['unlabelled_panel']),trusted,
   expected['bounds'],initial_families=expected.get('initial_families',{}),initial_schedule_baseline=expected.get('initial_schedule_baseline',{}),transfer_record=expected.get('transfer_record'),human_default=expected.get('human_default'),human_training=expected.get('human_training'))
 elif kind=='safe_checkpoint_selection_candidate_v1':
  rebuilt=build_checkpoint_selection(expected['stop_epoch'],[as_art('candidate',x) for x in request['candidate_observations']],trusted,history_required=history_required,selection_policy=expected.get('selection_policy'),candidate_details=expected.get('candidate_details'))
 elif kind=='safe_hard_sample_acquisition_v1':
  from hard_sample_contract import planning_request
  rebuilt=planning_request(epoch,expected['hard_sample_inventory'],expected['annotation_backend'])
 else:raise ValueError('unsupported request contract')
 if rebuilt!=request:raise ValueError('request contains unrecognized, modified or unverified content')
 return rebuilt
