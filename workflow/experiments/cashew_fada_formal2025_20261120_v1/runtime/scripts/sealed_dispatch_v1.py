"""Candidate tool-free Cashew dispatch; not authorized to launch the worker.

Only a verified worker may invoke this with current-run trusted records and
independently derived capabilities. This module verifies content and routing,
not the provenance, source identity or blindness of the caller's records.
Unknown model submissions are never re-issued automatically.
"""
import fcntl,hashlib,json,os,tempfile,types
from pathlib import Path
from persistent_luna import PersistentLunaClient
from request_reconstruction import verify_request
from review_contract import load_approved_references,validate_review_response,validate_checkpoint_selection
from formal_training_account import TRAINING_HOME, verify_training_account_environment

SKILL=Path('@WORKSPACE@/.codex_training_current_account/skills/cashew-fada-training-expert/SKILL.md')
MODEL='gpt-6.2-sol';EFFORT='max'
_REAL_CLIENT=PersistentLunaClient
def sha(data):return hashlib.sha256(data).hexdigest()
def encode(value):return (json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')
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

def dispatch(request,trusted,expected,stage,*,codex_binary,skill_sha256,verifier_sha256,reference_hashes,previous=None,before_turn=None,client_class=PersistentLunaClient,timeout=1800):
 """One fresh context, exact structured skill binding; unresolved calls block."""
 if client_class is _REAL_CLIENT:verify_training_account_environment(os.environ)
 if not callable(before_turn):raise ValueError('transaction claim callback is mandatory')
 clean=verify_request(request,trusted,expected=expected,previous=previous)
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
   'developerInstructions':'Use only this request and the supplied approved references. Do not use external tools, files, history or target-point evaluation. Return the supplied schema.',
   'allowProviderModelFallback':False, 'runtimeWorkspaceRoots':[], 'environments':[], 'config':{'model_reasoning_effort':'max','project_doc_max_bytes':0,'features.skip_host_skill_discovery':True,'features.multi_agent':False,
    'features.shell_tool':False,'features.unified_exec':False,'features.apps':False,'web_search':'disabled'}}
  if (base/'receipt.json').exists():
   receipt=json.loads((base/'receipt.json').read_text());answer_path=base/'answer.json'
   if receipt.get('request_sha256')!=request_hash or receipt.get('skill_sha256')!=skill_sha256 or receipt.get('verifier_sha256')!=verifier_sha256 or receipt.get('approved_references_sha256')!=sha(reference_bytes) or not answer_path.is_file() or answer_path.is_symlink() or sha(answer_path.read_bytes())!=receipt.get('answer_sha256'):raise RuntimeError('invalid previous receipt')
   recorded_thread=json.loads((base/'thread.json').read_text());recorded_turn=json.loads((base/'actual_turn.json').read_text());audit=module.validate_binding(recorded_thread,recorded_turn,expected_model=MODEL,expected_effort=EFFORT)
   if recorded_thread!=thread or audit.get('skill_sha256')!=skill_sha256 or recorded_turn['params'].get('threadId')!=receipt.get('thread_id') or not receipt.get('turn_id'):raise RuntimeError('previous turn identity mismatch')
   expected_prompt=('$cashew-fada-training-expert\nApproved reference material:\n'+reference_bytes.decode('utf-8')+'\nPoint-sealed request:\n'+request_bytes.decode('utf-8'))
   inputs=recorded_turn['params'].get('input',[])
   if len(inputs)!=2 or [x.get('type') for x in inputs]!=['skill','text'] or inputs[1].get('text')!=expected_prompt or recorded_turn['params'].get('outputSchema')!=clean['schema']:raise RuntimeError('previous turn content mismatch')
   answer=json.loads(answer_path.read_text())
   validated=(validate_checkpoint_selection(clean,answer) if clean['contract_version']=='safe_checkpoint_selection_candidate_v1' else validate_review_response(clean,answer))
   return {'answer':answer,'validated':validated,'replayed':True,'receipt':receipt}
  if (base/'submission.json').exists():raise RuntimeError('unresolved submission: reconcile exact thread and turn; never resend')
  publish(base/'thread.json',thread)
  prompt=('$cashew-fada-training-expert\nApproved reference material:\n'+reference_bytes.decode('utf-8')+
          '\nPoint-sealed request:\n'+request_bytes.decode('utf-8'))
  class BoundClient(client_class):
   def _send(self,payload):
    if payload.get('method')=='turn/start':
     if SKILL.resolve(strict=True)!=skill or sha(SKILL.read_bytes())!=skill_sha256 or sha(verifier.read_bytes())!=verifier_sha256:raise RuntimeError('skill or verifier changed during transaction')
     params=payload.get('params',{});inputs=params.get('input',[])
     if len(inputs)!=2 or [x.get('type') for x in inputs]!=['skill','text'] or inputs[1].get('text')!=prompt or params.get('outputSchema')!=clean['schema']:
      raise ValueError('actual turn differs from sealed request')
     audit=module.validate_binding(thread,payload,expected_model=MODEL,expected_effort=EFFORT)
     publish(base/'actual_turn.json',payload);publish(base/'binding.json',audit)
    return super()._send(payload)
  client=BoundClient(codex_binary,timeout)
  try:
   client.start_server();reply,_=client._request('thread/start',thread);tid=reply['thread']['id']
   if client_class is _REAL_CLIENT:
    if reply.get('model')!=MODEL or reply.get('reasoningEffort')!=EFFORT:raise ValueError('App-server model or effort differs from pinned Expert')
    publish(base/'thread_start_response.json',reply)
   if not isinstance(tid,str) or not tid:raise ValueError('invalid thread identity')
   # The outer transaction can durably claim this logical decision before any
   # turn is sent; it also prevents bypassing a pending call via a new stage.
   before_turn(tid)
   # Marker precedes the attempted turn. A crash after this line needs explicit
   # reconciliation even when no model call was ultimately sent.
   publish(base/'submission.json',{'thread_id':tid,'request_sha256':request_hash,'skill_sha256':skill_sha256})
   result=client.run_structured_turn(thread_id=tid,text=prompt,skill_name='cashew-fada-training-expert',
    skill_path=SKILL,output_schema=clean['schema'],model=MODEL,reasoning_effort=EFFORT,timeout_seconds=timeout)
   publish(base/'raw_result.json',result)
   if not isinstance(result.get('turn_id'),str) or not result['turn_id']:raise ValueError('missing turn identity')
   if any(event.get('method')=='model/rerouted' for event in result.get('events',[])):
    raise ValueError('Expert model was rerouted during the turn')
   answer=json.loads(result['final_text'])
   validated=(validate_checkpoint_selection(clean,answer) if clean['contract_version']=='safe_checkpoint_selection_candidate_v1' else validate_review_response(clean,answer))
   publish(base/'answer.json',answer)
   receipt={'thread_id':tid,'turn_id':result['turn_id'],'request_sha256':request_hash,'skill_sha256':skill_sha256,'verifier_sha256':verifier_sha256,
    'answer_sha256':sha((base/'answer.json').read_bytes()),'approved_references_sha256':sha(reference_bytes)}
   publish(base/'receipt.json',receipt)
   return {'answer':answer,'validated':validated,'replayed':False,'receipt':receipt}
  finally:client.close()
