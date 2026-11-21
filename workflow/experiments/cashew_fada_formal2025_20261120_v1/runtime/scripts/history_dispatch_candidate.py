"""Cashew dispatch with audited same-run history; not authorized to launch the worker.

Only a verified worker may invoke this with current-run trusted records and
independently derived capabilities. This module verifies content and routing,
not the provenance, source identity or blindness of the caller's records.
Unknown model submissions are never re-issued automatically.
"""
import fcntl,hashlib,json,os,tempfile,types
from pathlib import Path
from history_tool_client import HistoryClient, SPEC
from history_audit import verify as verify_history, HistoryQueryRequired
PersistentLunaClient=HistoryClient
from request_reconstruction import verify_request
from review_contract import load_approved_references,validate_review_response,validate_checkpoint_selection
from formal_training_account import TRAINING_HOME, verify_training_account_environment

SKILL=Path('@WORKSPACE@/.codex_training_current_account/skills/cashew-fada-training-expert/SKILL.md')
MODEL='gpt-6.2-sol';EFFORT='max'
_REAL_CLIENT=PersistentLunaClient
def sha(data):return hashlib.sha256(data).hexdigest()
def encode(value):return (json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')
def wire_schema(request):
 # The API strict output schema rejects uniqueItems. Keep the original schema
 # for local validation; only its unsupported transport keyword is removed.
 import copy
 schema=copy.deepcopy(request['schema'])
 def supported(value):
  if isinstance(value,dict):
   value.pop('uniqueItems',None)
   for child in value.values():supported(child)
  elif isinstance(value,list):
   for child in value:supported(child)
 supported(schema)
 return schema
def history_cutoff(request):
 if request.get('contract_version')=='safe_checkpoint_selection_candidate_v1':
  stop=request.get('stop_epoch')
  if type(stop) is not int or stop<1:raise ValueError('selection stop epoch missing')
  return stop+1
 epoch=request.get('epoch')
 if type(epoch) is not int or epoch<1:raise ValueError('review epoch missing')
 return epoch

def developer_instructions(request):
 if request.get('contract_version') == 'safe_hard_sample_acquisition_v1':
  return ('Plan only the annotation batch. Query E0 and a relevant prior committed record using query_training_history. '
          'Return the given counts and rationale. Treat simulation as an engineering exercise.')
 if request.get('selection_policy'):
  return ('Before SELECT use query_training_history yourself: retrieve epoch 0 initialization and candidate_details for epoch 0, your selected epoch, and your strongest alternative. You may query more candidates as needed. These records contain source validation, fixed unlabelled panel, annotated-training fit/exposure, training/BN dynamics and controls. Explain benefits, tradeoffs and unknowns versus E0 and the strongest alternative. Visual training fit is measured on the frozen annotation set; read the same cases and contrast exposure; source/proxy changes do not prove target accuracy. Never use target-point evidence. The worker rejects missing queries or comparisons. Return the supplied schema.')
 if request.get('capabilities',{}).get('history_query_required') is True:
  return ('Use the supplied current evidence and approved references. Before answering, use the audited '
          'query_training_history tool yourself to inspect this same run: include epoch 0 initialization '
          'and at least one relevant earlier committed training record when one exists. Select additional '
          'epochs and kinds as needed. Only the advertised audited history and matched-trial tools are permitted; target-point evidence is forbidden. '
          'For complete recall, query interventions for the latest actual ADJUST and assess it in the advertised intervention_review. '
          'Use the available experiment budget to answer the mechanism, not merely to finish quickly. '
          'An answer without the required queries will not be committed; if asked again, query before answering.')
 return ('Use the supplied request and approved references. The frozen packet marks historical_retrieval false; '
         'the separately audited query_training_history tool is the only exception for this same run. '
         'Never use target-point evidence. Return the supplied schema.')

def verify_history_attempts(stage,thread_id,turn_id):
 """Bind the accepted turn and retain any completed no-history attempt."""
 stage=Path(stage);marker=json.loads((stage/'accepted_attempt.json').read_text());number=marker.get('attempt')
 if type(number) is not int or number not in (1,2) or marker.get('turn_id')!=turn_id:
  raise ValueError('invalid accepted history attempt')
 current=stage/('history_attempt_%02d'%number)
 for name in ('actual_turn.json','binding.json','raw_result.json'):
  if (current/name).is_symlink() or (current/name).read_bytes()!=(stage/name).read_bytes():
   raise ValueError('accepted history attempt differs from published turn')
 accepted_raw=json.loads((current/'raw_result.json').read_text())
 if accepted_raw.get('turn_id')!=turn_id or (accepted_raw.get('event_tracking') or {}).get('turn_status')!='completed':
  raise ValueError('accepted history attempt is not a completed turn')
 turn_ids=[]
 for index in range(1,number):
  old=stage/('history_attempt_%02d'%index);rejected=json.loads((old/'rejected.json').read_text());raw=json.loads((old/'raw_result.json').read_text());turn=json.loads((old/'actual_turn.json').read_text());binding=json.loads((old/'binding.json').read_text())
  if (rejected.get('reason')!='required_history_not_retrieved' or rejected.get('turn_id')!=raw.get('turn_id')
      or not raw.get('turn_id') or raw.get('turn_id')==turn_id
      or turn.get('params',{}).get('threadId')!=thread_id or binding.get('valid') is not True
      or (raw.get('event_tracking') or {}).get('turn_status')!='completed'):
   raise ValueError('invalid rejected history attempt')
  turn_ids.append(raw['turn_id'])
 return turn_ids+[turn_id]
def publish(path,data):
 """Create a durable immutable artifact, or accept identical bytes only."""
 if not isinstance(data,bytes):data=encode(data)
 fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=str(path.parent))
 try:
  with os.fdopen(fd,'wb',closefd=False) as f:f.write(data);f.flush();os.fsync(fd)
  try:os.link(tmp,path,follow_symlinks=False)
  except FileExistsError:
   if path.is_symlink() or path.read_bytes()!=data:raise RuntimeError('conflicting transaction artifact: '+path.name)
  parent=os.open(str(path.parent),os.O_RDONLY)
  try:os.fsync(parent)
  finally:os.close(parent)
 finally:
  os.close(fd)
  try:os.unlink(tmp)
  except FileNotFoundError:pass

def dispatch(request,trusted,expected,stage,*,codex_binary,skill_sha256,verifier_sha256,reference_hashes,previous=None,before_turn=None,client_class=PersistentLunaClient,timeout=1800,history_root=None,history_job_id=None,history_run_id=None):
 """One fresh context, exact structured skill binding; unresolved calls block."""
 if client_class is _REAL_CLIENT:verify_training_account_environment(os.environ)
 if not callable(before_turn):raise ValueError('transaction claim callback is mandatory')
 if history_root is None or not history_job_id or not history_run_id:raise ValueError('authenticated same-run history context required')
 clean=verify_request(request,trusted,expected=expected,previous=previous)
 history_required=expected.get('history_required',False)
 skill=SKILL.resolve(strict=True);skill_bytes=skill.read_bytes()
 if sha(skill_bytes)!=skill_sha256:raise ValueError('installed Cashew skill differs from pinned release')
 refs=load_approved_references(skill.parent,reference_hashes)
 verifier=skill.parent/'scripts/verify_explicit_skill_binding.py'
 verifier_bytes=verifier.read_bytes()
 if sha(verifier_bytes)!=verifier_sha256:raise ValueError('Cashew skill verifier differs from pinned release')
 module=types.ModuleType('sealed_cashew_binding');module.__file__=str(verifier)
 exec(compile(verifier_bytes,str(verifier),'exec'),module.__dict__)
 base=Path(stage);base.mkdir(parents=True,exist_ok=True)
 if base.is_symlink():raise ValueError('stage must not be a symlink')
 with (base/'lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  request_bytes=encode(clean);reference_bytes=encode(refs)
  publish(base/'request.json',request_bytes);publish(base/'approved_references.json',reference_bytes)
  request_hash=sha(request_bytes)
  thread={'baseInstructions':skill_bytes.decode('utf-8'),'cwd':str(base.resolve()),'model':MODEL,'sandbox':'read-only',
   'approvalPolicy':'never','ephemeral':False,
   'developerInstructions':developer_instructions(clean),
   'dynamicTools':[SPEC,__import__('training_tool_output_v1').SPEC]+__import__('paired_trial_v1').spec(clean),
   'allowProviderModelFallback':False, 'runtimeWorkspaceRoots':[], 'environments':[], 'config':{'model_reasoning_effort':'max','project_doc_max_bytes':0,'features.skip_host_skill_discovery':True,'features.multi_agent':False,
    'features.shell_tool':False,'features.unified_exec':False,'features.apps':False,'web_search':'disabled'}}
  if (base/'receipt.json').exists():
   receipt=json.loads((base/'receipt.json').read_text());answer_path=base/'answer.json'
   if receipt.get('request_sha256')!=request_hash or receipt.get('skill_sha256')!=skill_sha256 or receipt.get('verifier_sha256')!=verifier_sha256 or receipt.get('approved_references_sha256')!=sha(reference_bytes) or not answer_path.is_file() or answer_path.is_symlink() or sha(answer_path.read_bytes())!=receipt.get('answer_sha256'):raise RuntimeError('invalid previous receipt')
   recorded_thread=json.loads((base/'thread.json').read_text());recorded_turn=json.loads((base/'actual_turn.json').read_text());audit=module.validate_binding(recorded_thread,recorded_turn,expected_model=MODEL,expected_effort=EFFORT)
   allowed_turn_ids=verify_history_attempts(base,receipt['thread_id'],receipt['turn_id']) if history_required else None
   if receipt.get('history_audit_sha256')!=verify_history(base,root=history_root,job_id=history_job_id,run_id=history_run_id,cutoff_epoch=history_cutoff(clean),thread_id=receipt.get('thread_id'),require_history=history_required,allowed_turn_ids=allowed_turn_ids,selection_request=clean,selection_answer=json.loads(answer_path.read_text())):raise RuntimeError('history transcript mismatch')
   if not __import__('training_tool_output_v1').thread_matches(recorded_thread,thread) or audit.get('skill_sha256')!=skill_sha256 or recorded_turn['params'].get('threadId')!=receipt.get('thread_id') or not receipt.get('turn_id'):raise RuntimeError('previous turn identity mismatch')
   expected_prompt=('$cashew-fada-training-expert\nApproved reference material:\n'+reference_bytes.decode('utf-8')+'\nPoint-sealed request:\n'+request_bytes.decode('utf-8'))
   inputs=recorded_turn['params'].get('input',[])
   if len(inputs)!=2 or [x.get('type') for x in inputs]!=['skill','text'] or inputs[1].get('text')!=expected_prompt or recorded_turn['params'].get('outputSchema') not in (clean['schema'],wire_schema(clean)):raise RuntimeError('previous turn content mismatch')
   answer=json.loads(answer_path.read_text())
   validated=(validate_checkpoint_selection(clean,answer) if clean['contract_version']=='safe_checkpoint_selection_candidate_v1' else validate_review_response(clean,answer))
   return {'answer':answer,'validated':validated,'replayed':True,'receipt':receipt}
  if (base/'submission.json').exists():raise RuntimeError('unresolved submission: reconcile exact thread and turn; never resend')
  publish(base/'thread.json',thread)
  prompt=('$cashew-fada-training-expert\nApproved reference material:\n'+reference_bytes.decode('utf-8')+
          '\nPoint-sealed request:\n'+request_bytes.decode('utf-8'))
  attempt={'number':1}
  class BoundClient(client_class):
   def _send(self,payload):
    if payload.get('method')=='turn/start':
     if SKILL.resolve(strict=True)!=skill or sha(SKILL.read_bytes())!=skill_sha256 or sha(verifier.read_bytes())!=verifier_sha256:raise RuntimeError('skill or verifier changed during transaction')
     params=payload.get('params',{});inputs=params.get('input',[])
     if len(inputs)!=2 or [x.get('type') for x in inputs]!=['skill','text'] or inputs[1].get('text')!=prompt or params.get('outputSchema')!=wire_schema(clean):
      raise ValueError('actual turn differs from sealed request')
     audit=module.validate_binding(thread,payload,expected_model=MODEL,expected_effort=EFFORT)
     target=base
     if history_required:
      target=base/('history_attempt_%02d'%attempt['number']);target.mkdir(exist_ok=False)
     publish(target/'actual_turn.json',payload);publish(target/'binding.json',audit)
    return super()._send(payload)
  client=BoundClient(codex_binary,timeout,root=history_root,job_id=history_job_id,run_id=history_run_id,cutoff_epoch=history_cutoff(clean),audit_dir=base)
  client.history_required=history_required
  client.trial_request=clean
  try:
   client.start_server();reply,_=client._request('thread/start',thread);tid=reply['thread']['id']
   if client_class is _REAL_CLIENT:
    if reply.get('model')!=MODEL or reply.get('reasoningEffort')!=EFFORT:raise ValueError('App-server model or effort differs from pinned Expert')
    publish(base/'thread_start_response.json',reply)
   if not isinstance(tid,str) or not tid:raise ValueError('invalid thread identity')
   client.history_thread=tid
   # The outer transaction can durably claim this logical decision before any
   # turn is sent; it also prevents bypassing a pending call via a new stage.
   before_turn(tid)
   # Marker precedes the attempted turn. A crash after this line needs explicit
   # reconciliation even when no model call was ultimately sent.
   publish(base/'submission.json',{'thread_id':tid,'request_sha256':request_hash,'skill_sha256':skill_sha256})
   previous_turn_id=None;completed_turn_ids=[]
   for number in range(1,3 if history_required else 2):
    attempt['number']=number
    resume_options={'previous_turn_id':previous_turn_id} if previous_turn_id is not None else {}
    result=client.run_structured_turn(thread_id=tid,text=prompt,skill_name='cashew-fada-training-expert',
     skill_path=SKILL,output_schema=wire_schema(clean),model=MODEL,reasoning_effort=EFFORT,timeout_seconds=timeout,**resume_options)
    if not isinstance(result.get('turn_id'),str) or not result['turn_id']:raise ValueError('missing turn identity')
    if result['turn_id'] in completed_turn_ids:raise ValueError('duplicate completed turn identity')
    completed_turn_ids.append(result['turn_id'])
    if any(event.get('method')=='model/rerouted' for event in result.get('events',[])):
     raise ValueError('Expert model was rerouted during the turn')
    answer=json.loads(result['final_text'])
    validated=(validate_checkpoint_selection(clean,answer) if clean['contract_version']=='safe_checkpoint_selection_candidate_v1' else validate_review_response(clean,answer))
    attempt_dir=base/('history_attempt_%02d'%number)
    if history_required:publish(attempt_dir/'raw_result.json',result)
    try:
     history_hash=verify_history(base,root=history_root,job_id=history_job_id,run_id=history_run_id,cutoff_epoch=history_cutoff(clean),thread_id=tid,require_history=history_required,allowed_turn_ids=completed_turn_ids if history_required else None,selection_request=clean,selection_answer=answer)
    except HistoryQueryRequired:
     publish(attempt_dir/'rejected.json',{'turn_id':result['turn_id'],'reason':'required_history_not_retrieved'})
     if number>=2:raise
     previous_turn_id=result['turn_id']
     continue
    if history_required:
     publish(base/'actual_turn.json',(attempt_dir/'actual_turn.json').read_bytes())
     publish(base/'binding.json',(attempt_dir/'binding.json').read_bytes())
     publish(base/'accepted_attempt.json',{'attempt':number,'turn_id':result['turn_id']})
    break
   publish(base/'raw_result.json',result)
   publish(base/'answer.json',answer)
   receipt={'thread_id':tid,'turn_id':result['turn_id'],'request_sha256':request_hash,'skill_sha256':skill_sha256,'verifier_sha256':verifier_sha256,
    'answer_sha256':sha((base/'answer.json').read_bytes()),'approved_references_sha256':sha(reference_bytes),
    'history_audit_sha256':history_hash}
   publish(base/'receipt.json',receipt)
   return {'answer':answer,'validated':validated,'replayed':False,'receipt':receipt}
  finally:client.close()
