"""Joint scientific review with short context and host-owned read-only retrieval.

The old normalizer, bounds, minimum epoch and selection validation remain the
executable authority. This adapter never launches training or queries a model
unless explicitly invoked by the future authorized numerical worker.
"""
from pathlib import Path
import argparse,copy,json,os,time,hashlib,datetime,shutil,uuid,math
from jsonschema import Draft202012Validator
from on_demand_memory import read,write,sha,compact,evidence_snapshot,brief,query
from expert_policy_v4 import MODEL,EFFORT,normalize,semantic_schema,apply_agent_early_stopping_policy
from persistent_luna import PersistentLunaClient,PersistentLunaRecoveryRequired,PersistentLunaTransportError,PersistentLunaProtocolError
from automatic_negative_selection_validation_v1 import validate_selection
from joint_ranking_controls import decode_selection
from prior_relative_spatial_guidance_v1 import validate_semantic
from visible_response_contract import visible_contract, validate_training_fields
import group_session_lifecycle as sessions
ROOT=Path(__file__).resolve().parents[1]
SKILL=Path('@WORKSPACE@/.codex_training_current_account/skills/cashew-fada-training-expert/SKILL.md')
import importlib.util
_spec=importlib.util.spec_from_file_location('skill_binding',SKILL.parent/'scripts/verify_explicit_skill_binding.py');_binding=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_binding)
SCHEMA=ROOT/'references/agent_decision_v4.schema.json'
AUTOMATIC_EVIDENCE_POLICY = (
 'ACTIVATE applies pseudo-negative loss to the main training trajectory: it is an intervention, not an isolated experiment merely because its dose is low. '
 'Activate only when observed evidence supports both the need and this particular candidate/dose direction, including plausible alternatives and a checkable reversal. '
 'One unlabelled prediction change, prior disagreement, or training-polygon error alone does not establish that unmarked candidates are true negatives. '
 'When the direction is not sufficiently supported, ABSTAIN or request a permitted matched branch experiment; do not force ACTIVATE. '
 'Conversely, act when the combined evidence genuinely supports a useful bounded change; perfect target accuracy evidence is neither available nor required. '
)

def negative_schema(count):
 return dict(type='object',additionalProperties=False,required=['action','selected_count','automatic_weight','ranking_controls','rationale','expected_response','reversal_condition'],properties={
 'action':dict(type='string',enum=['ACTIVATE','ABSTAIN']),'selected_count':dict(type='integer',minimum=0,maximum=count),'automatic_weight':dict(type='number',minimum=0,maximum=.005),
 'ranking_controls':dict(type='object',description='The three ranking weights share a TOTAL BUDGET OF 1. Choose prior fraction p and the p2 share q of the remaining 1-p, not a second absolute weight. Host derives prior=p, p2=(1-p)*q, source=(1-p)*(1-q). Example p=0.25,q=0.60 gives prior=0.25,p2=0.45,source=0.30, totaling 1. This is ranking emphasis, separate from loss dose.',additionalProperties=False,required=['regional_prior_fraction','p2_share_of_nonprior'],properties={k:dict(type='number',description='Fraction of the total budget of 1 reserved for the regional prior.' if k=='regional_prior_fraction' else 'Fraction of the remaining nonprior budget assigned to p2; this is not the absolute p2 weight.',minimum=.25 if k=='regional_prior_fraction' else 0,maximum=1) for k in ['regional_prior_fraction','p2_share_of_nonprior']}),
 **{k:dict(type='string') for k in ['rationale','expected_response','reversal_condition']}})

def schema_for(packet,negative):
 schema=semantic_schema(SCHEMA,packet);defs=schema.pop('$defs',{});schema.pop('$schema',None)
 return dict(type='object',additionalProperties=False,required=['training','automatic_negative'],properties=dict(training=schema,automatic_negative=dict(anyOf=[dict(type='null'),negative_schema(len(negative['rows']))]) if negative else dict(type='null')),**{'$defs':defs})

def validate_answer(packet,value,request_hash,negative):
 if not isinstance(value,dict) or set(value)!={'training','automatic_negative'}:
  raise ValueError('Return exactly training and automatic_negative; extra or missing fields require correction')
 schema=schema_for(packet,negative)
 validate_training_fields(schema,value)
 error=next(Draft202012Validator(schema).iter_errors(value),None)
 if error is not None:
  raise ValueError('Invalid structured response at '+'.'.join(map(str,error.absolute_path))+': '+error.message)
 semantic=value['training'];validate_semantic(packet,semantic);decision=normalize(packet,semantic,request_hash,planet_used=False)
 actual=value['automatic_negative'];needs=negative is not None and decision['continue_training'] and decision['action']!='REQUEST_EXPERIMENT'
 if needs:
  actual=decode_selection(actual);validate_selection(actual,len(negative['rows']))
 elif actual is not None:raise ValueError('No negative pool commitment for initialization, selection, STOP or pending experiment')
 return decision,actual

def parse_final_object(text):
 """Parse one joint object, tolerating markdown fences or trailing chatter.

 A repair turn remains mandatory for malformed or ambiguous answers. Harmless
 transport wrappers are tolerated only when they contain one joint decision.
 """
 if not isinstance(text,str):raise ValueError('Agent final answer is not text')
 raw=text.strip()
 if raw.startswith('```'):
  lines=raw.splitlines()
  if lines and lines[0].lstrip().startswith('```'):lines=lines[1:]
  if lines and lines[-1].strip().startswith('```'):lines=lines[:-1]
  raw='\n'.join(lines).strip()
 def unique_pairs(pairs):
  result={}
  for key,value in pairs:
   if key in result:raise ValueError('Duplicate JSON field: '+key)
   result[key]=value
  return result
 def reject_nonfinite(value):
  raise ValueError('Non-finite JSON number: '+value)
 def finite_float(value):
  parsed=float(value)
  if not math.isfinite(parsed):raise ValueError('Non-finite JSON number: '+value)
  return parsed
 try:return json.loads(raw,object_pairs_hook=unique_pairs,parse_constant=reject_nonfinite,parse_float=finite_float)
 except json.JSONDecodeError as first:
  decoder=json.JSONDecoder(object_pairs_hook=unique_pairs,parse_constant=reject_nonfinite,parse_float=finite_float);found=[];start=0
  while start<len(raw):
   if raw[start]!='{':start+=1;continue
   try:
    value,end=decoder.raw_decode(raw[start:])
    if isinstance(value,dict) and {'training','automatic_negative'}.issubset(value):found.append(value)
    start+=end
   except json.JSONDecodeError:start+=1
  if len(found)>1:raise ValueError('Ambiguous response contains multiple joint decisions')
  if found:return found[0]
  raise first

def archive_recoverable_session(stage):
 """Clear only a terminal failed short-session marker before retrying.

 The complete failed attempt remains under failed_attempts; an in-progress or
 otherwise uncertain turn is never cleared here.
 """
 failure=stage/'session_failure.json'
 if not failure.is_file():return False
 try:
  payload=read(failure);kind=payload.get('error_type','')
  receipt=read(stage/'session_receipt.json')
  submissions=sorted(stage.glob('submission_*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
  if not submissions:return False
  submitted=read(submissions[-1])
  request_hash=sha(stage/'request.json')
 except Exception:return False
 # Text of an error or an unrelated notification is not proof of a terminal
 # turn. Only a successful durable read of the exact submitted thread and its
 # last turn permits a fresh session. Anything ambiguous remains blocked for
 # explicit reconciliation; never infer terminal state from a substring.
 rec=payload.get('reconciliation')
 if not isinstance(rec,dict) or rec.get('reconciled') is not True:return False
 thread=rec.get('thread');turn=rec.get('turn');turns=rec.get('turns')
 if not isinstance(thread,dict) or not isinstance(turn,dict) or not isinstance(turns,list) or not turns:return False
 if not receipt.get('thread_id') or thread.get('id')!=receipt['thread_id']:return False
 if receipt.get('request_sha256')!=request_hash:return False
 if submitted.get('thread_id')!=receipt['thread_id']:return False
 if submitted.get('request_sha256',request_hash)!=request_hash:return False
 if not turn.get('id') or not isinstance(turns[-1],dict) or turns[-1].get('id')!=turn['id']:return False
 if turn['id']==submitted.get('previous_turn_id'):return False
 if turn.get('status') not in {'failed','interrupted'} or turns[-1].get('status')!=turn.get('status'):return False
 archive=stage.parents[1]/'failed_attempts'/('dispatch_retry_%d_%s'%(int(time.time()),uuid.uuid4().hex))/stage.name
 archive.mkdir(parents=True,exist_ok=True)
 for item in list(stage.iterdir()):
  if item.name in {'request.json','compact_brief.json','joint_output_schema.json'}:continue
  shutil.move(str(item),str(archive/item.name))
 write(stage/'retry_reconciliation.json',dict(state='terminal_failed_session_archived',archive=str(archive),error_type=kind))
 return True

TOOLS=[dict(type='function',name='get_training_evidence',description='Read factual evidence for this group and decision cutoff. Topics include current,controls,history,trajectory,accepted_decisions,prior,auxiliary,experiments,initialization,selection,negative_candidates,negative_generations,automatic_negative_usage,knowledge,complete_packet. automatic_negative_usage /since_previous_review gives measured interval totals; /by_epoch gives actual uses. Pool manifests are selection, not execution. For objects omit char_limit and use pointer; for arrays use offset/limit; char_offset/char_limit are ONLY for string values and char_limit is at most 2500. Use pointer/range/page to get only what changes your decision. Historical evidence is not absent because it was not sent.',inputSchema=dict(type='object',additionalProperties=False,properties=dict(topic=dict(type='string'),pointer=dict(type='string'),start_epoch=dict(type='integer',minimum=0),end_epoch=dict(type='integer',minimum=0),offset=dict(type='integer',minimum=0),limit=dict(type='integer',minimum=1,maximum=100),char_offset=dict(type='integer',minimum=0),char_limit=dict(type='integer',minimum=1,maximum=2500)),required=['topic']))]

def start_server_with_retry(client):
 """Bounded retries before a thread or model turn has been submitted."""
 for attempt in range(3):
  try:
   client.start_server();return
  except (PersistentLunaTransportError,TimeoutError,OSError):
   client.close()
   if attempt==2:raise
   time.sleep(attempt+1)

class RetrievalClient(PersistentLunaClient):
 def __init__(self,*args,store,tool_log,packet,**kwargs):
  super().__init__(*args,**kwargs);self.store=store;self.tool_log=tool_log;self.packet=packet;self.calls=[]
 def _send(self,payload):
  if payload.get('method')=='turn/start':
   thread=read(self.store.parent/'actual_thread_request.json')
   frozen=ROOT/'.codex/skills/cashew-fada-training-expert/SKILL.md'
   if sha(SKILL)!=sha(frozen):raise ValueError('Installed skill changed after cohort preparation')
   audit=_binding.validate_binding(thread,payload,expected_model=MODEL)
   n=len(list(self.store.parent.glob('actual_turn_request_*.json')))
   write(self.store.parent/('actual_turn_request_%d.json'%n),payload)
   write(self.store.parent/('explicit_binding_%d.json'%n),audit)
  return super()._send(payload)
 def initialize(self):
  r,_=self._request('initialize',dict(clientInfo=dict(name='cashew-on-demand-evidence',version='1'),capabilities=dict(experimentalApi=True)));self._send(dict(jsonrpc='2.0',method='initialized',params={}));return r
 def _read_line(self,deadline):
  while True:
   m=super()._read_line(deadline)
   if m.get('method') in ['thread/tokenUsage/updated','turn/started','turn/completed','turn/failed']:
    with self.tool_log.with_name('usage_events.jsonl').open('a') as stream:stream.write(compact(m)+'\n')
   if m.get('method')=='item/tool/call' and 'id' in m:
    p=m.get('params',{});args=p.get('arguments',{});name=p.get('tool');success=True
    try:
     if name=='get_training_evidence':result=query(self.store,args)
     elif name=='prepare_experiment_script':
      capability=self.packet.get('experiment_capability',{});path=Path(capability['custom_code']['script_path'])
      if not capability.get('enabled') or 'REQUEST_EXPERIMENT' not in self.packet.get('allowed_actions',[]):raise ValueError('No custom experiment authorized for this turn')
      if set(args)!={'python_source'} or not isinstance(args['python_source'],str):raise ValueError('Expected Python source only')
      compile(args['python_source'],str(path),'exec');path.parent.mkdir(parents=True,exist_ok=True)
      if path.exists() and path.read_text()!=args['python_source']:raise ValueError('Do not overwrite an existing experimental source')
      path.write_text(args['python_source']);result=dict(script=str(path),sha256=sha(path),executed=False,branch_not_adopted=True)
     else:raise ValueError('Unadvertised tool')
    except Exception as e:success=False;result=dict(error=str(e),no_state_change=True)
    self.calls.append(dict(call_id=p.get('callId'),tool=name,arguments=args,success=success,result=result,time=datetime.datetime.now().astimezone().isoformat()));write(self.tool_log,self.calls)
    self._send(dict(jsonrpc='2.0',id=m['id'],result=dict(contentItems=[dict(type='inputText',text=compact(result))],success=success)));continue
   return m

 def start_owner_session(self,packet,tools):
  self._requested_sandbox='read-only'
  instructions=('Use the supplied cashew-fada-training-expert Skill as scientific owner of this one group. '+
  ('This is the durable conversation for this group. Reuse relevant prior reasoning, but current measured facts and epoch cutoffs supersede old values. Do not poll training status. ' if sessions.policy(packet)==sessions.LONG else 'The factual store preserves the full trajectory; this new session has no hidden conversational history. ')+
  'Use get_training_evidence to inspect evidence you need, not shell/browser/global filesystem. '+
  'Output schema is authoritative. Never invent absent measurements or use future evidence. '+
  'No fixed minimum STOP epoch; cap200. No preference for early checkpoints. Target point validation is post-training only. Current human polygons are training-only, no independent heldout. Preserve source weights and existing controls. '+
  'When the host advertises candidates, return next automatic-negative controls in the same answer; no second negative-selection Agent. '+
  'An experiment request is not a formal treatment adoption. The original executor owns numerical execution.')
  params=dict(baseInstructions=SKILL.read_text(),cwd=str(self.store),model=MODEL,sandbox='read-only',approvalPolicy='never',ephemeral=False,developerInstructions=instructions,dynamicTools=tools,config={'project_doc_max_bytes':0,'features.skip_host_skill_discovery':True,'features.multi_agent':False,'features.shell_tool':False,'features.unified_exec':False,'features.apps':False,'web_search':'disabled'})
  write(self.store.parent/'actual_thread_request.json',params);r,_=self._request('thread/start',params);return r['thread']['id']

def prepare(request_path):
 from online_boundary import verify_online
 packet=read(request_path)
 trials=(packet.get('pre_run') or {}).get('matched_candidate_trials')
 if isinstance(trials,dict) and 'Locally held-out polygons are development labels.' in str(trials.get('limitations','')):
  trials['limitations']=trials['limitations'].replace('Locally held-out polygons are development labels.','Current polygons are training-only; no independent heldout polygon feedback exists.')
  write(request_path,packet)
  write(request_path.parent/'evidence_description_correction.json',dict(reason='Legacy descriptive sentence inconsistent with current verified zero-holdout contract',numeric_measurements_changed=False,training_changed=False))
 verify_online(packet);normalized=apply_agent_early_stopping_policy(packet)
 if normalized!=packet:write(request_path,normalized);packet=normalized
 job_dir=request_path.parents[2] if request_path.parent.name.startswith('epoch_') or request_path.parent.name in ['initialization','selection'] else Path(packet.get('runtime_artifacts',{}).get('output_root',''))
 # For experiment subreviews, the explicit job root remains authoritative.
 if packet.get('runtime_artifacts',{}).get('output_root'):job_dir=Path(packet['runtime_artifacts']['output_root'])
 if not (job_dir/'job.json').is_file():raise ValueError('Missing current-group job root')
 negative=None;ref=packet.get('joint_negative_candidates')
 if ref:
  if sha(ref['path'])!=ref['sha256']:raise ValueError('Negative candidate identity changed')
  candidate=read(ref['path'])
  if candidate['epoch']==packet['epoch']:negative=candidate
 store=request_path.parent/'memory_store';manifest=evidence_snapshot(packet,request_path,job_dir,store,negative)
 usage={k:query(store,dict(topic='automatic_negative_usage',pointer='/'+k))['value'] for k in ['version','cutoff_epoch','current_epoch_only','since_previous_review','whole_run','interpretation']} if 'automatic_negative_usage' in manifest['documents'] else None
 small=brief(packet,manifest,negative['summary'] if negative else None,usage=usage)
 if sessions.policy(packet)==sessions.LONG:
  small['memory_instruction']='Same persistent group conversation. Reuse already read unchanged knowledge. Retrieve original actions and comparable measurements when needed; current facts carry the current cutoff. All own-run external evidence remains available. Never poll between scheduled reviews.'
 write(request_path.parent/'compact_brief.json',small)
 return packet,negative,store,small

def verify_saved_response_identity(stage, request_hash, raw_path):
 """Do not publish a previous response against a changed evidence packet."""
 receipt_path=stage/'session_receipt.json'
 if not receipt_path.is_file():
  raise PersistentLunaRecoveryRequired('Saved response has no session receipt; reconcile before publication')
 receipt=read(receipt_path)
 if receipt.get('request_sha256')!=request_hash or not receipt.get('thread_id'):
  raise PersistentLunaRecoveryRequired('Saved response belongs to an unverified or changed request; reconcile before publication')
 response=read(raw_path)
 if response.get('thread_id')!=receipt['thread_id']:
  raise PersistentLunaRecoveryRequired('Saved response thread differs from session receipt')
 suffix=raw_path.stem.rsplit('_',1)[-1]
 submission=stage/('submission_'+suffix+'.json')
 if not submission.is_file():
  raise PersistentLunaRecoveryRequired('Saved response has no matching submission record')
 submitted=read(submission)
 if submitted.get('thread_id')!=receipt['thread_id'] or submitted.get('request_sha256',request_hash)!=request_hash:
  raise PersistentLunaRecoveryRequired('Saved response submission identity differs from current request')
 if not response.get('turn_id') or response['turn_id']==submitted.get('previous_turn_id'):
  raise PersistentLunaRecoveryRequired('Saved response has no new turn identity')
 return response

def dispatch(path,codex,timeout=900):
 decision_path=path.with_name('decision.json');provenance=path.with_name('provenance.json')
 if publication_complete(path):return dict(idempotent=True,decision=read(decision_path))
 packet,negative,store,small=prepare(path);reqsha=sha(path);schema=schema_for(packet,negative);write(path.parent/'joint_output_schema.json',schema)
 # A pause marker is an execution boundary, even if somebody accidentally runs
 # a prepared dispatcher. Preparation/CPU tests remain usable while paused.
 trial=ROOT.parent
 if (trial/'PAUSE_REQUESTED.json').exists() or not (trial/'TRAINING_START_AUTHORIZED.json').exists():raise RuntimeError('Trial launch not authorized; no model request made')
 # A completed answer can be published again after an interrupted file write.
 # An uncertain submitted turn must instead be reconciled, never resubmitted.
 saved=list(path.parent.glob('joint_raw_response_*.json'))
 for raw in sorted(saved,reverse=True):
  result=verify_saved_response_identity(path.parent,reqsha,raw)
  try:decision,automatic=validate_answer(packet,parse_final_object(result['final_text']),reqsha,negative)
  except (ValueError,KeyError,TypeError):continue
  return publish(path,packet,negative,small,result,decision,automatic)
 if (path.parent/'session_receipt.json').exists():
  if not archive_recoverable_session(path.parent):
   raise PersistentLunaRecoveryRequired('Existing session has no accepted durable answer; reconcile its thread before retrying. No new model request sent.')
 owner_dir=Path(packet['job'].get('run_id','owner').split('/')[0]);tools=copy.deepcopy(TOOLS)
 if packet.get('experiment_capability',{}).get('enabled'):
  tools.append(dict(type='function',name='prepare_experiment_script',description='Write exactly the custom experiment source path advertised for this turn, without executing or adopting it. CPU Python syntax checked.',inputSchema=dict(type='object',additionalProperties=False,required=['python_source'],properties=dict(python_source=dict(type='string')))))
 # Resolve the account at the last boundary before any new model call. Do not
 # trust a stale shell environment or a caller-provided alternate executable.
 from agent_account_route import apply_worker_route
 apply_worker_route(os.environ)
 codex=os.environ['CASHEW_FADA_CODEX_BINARY']
 client=RetrievalClient(codex,timeout,store=store,tool_log=path.parent/'retrieval_calls.json',packet=packet)
 try:
  start_server_with_retry(client);tid,previous,resumed=sessions.begin(client,packet,tools,path,MODEL,EFFORT,SKILL)
  receipt=dict(owner=packet['job']['thread_key'],thread_id=tid,state='started',request_sha256=reqsha,model=MODEL,reasoning_effort=EFFORT);write(path.parent/'session_receipt.json',receipt)
  instruction=('$cashew-fada-training-expert\nUse the current brief and retrieve only needed evidence with get_training_evidence. '+
   'Keep every advertised scalar/schedule explicit at initialization; get initialization probe and controls if needed. '+
   'Before deciding intervention response query accepted_decisions and comparable trajectory. Before SELECT retrieve selection candidate identities and relevant comparisons. '+
   'For automatic-negative execution use CURRENT_BRIEF.automatic_negative_measured_usage.since_previous_review and whole_run_observed, or query automatic_negative_usage /since_previous_review and /by_epoch. Current-epoch zero after a finite pool expires does not mean the interval had no training use. A pool manifest or empty pool query is not an execution counter. Correct earlier accepted narrative claims when contradicted by measured counts, including claims of no use before an earlier review: check the historical epoch range instead of repeating those claims. Actual use does not by itself prove benefit. '+
   'Before spatial claims inspect actual prior evidence, cite user_prior_observation_v1 AND WestAfrica_whole_map_reference. '+
   'training follows the exact legacy semantic schema. automatic_negative is a compact selection object only when continuing a trained review with advertised candidates; otherwise null. '+
   AUTOMATIC_EVIDENCE_POLICY+
   'Negative count remains your decision over all candidates; representative examples are not a cap. No hard prior/distance/probability gate. '+
   'Ranking has one shared TOTAL WEIGHT BUDGET OF 1. For ranking_controls choose regional_prior_fraction in [0.25,1] and p2_share_of_nonprior in [0,1]. Reserve prior first, then divide only the remaining 1-prior between p2 and source. The host computes p2=(1-prior)*share and source=(1-prior)*(1-share). Example prior=0.25,share=0.60 yields source=0.30,p2=0.45,prior=0.25,total=1. Do not output three independent weights; these are ranking weights, not the separate loss dose. '+
   'A REQUEST_EXPERIMENT leaves formal controls unchanged and automatic_negative=null until its result review. '+
   'Schedule family is immutable after initialization; later use only advertised operations. STOP changes no controls/schedules; SELECT is separate. '+
   'Return the joint object only, no extra metadata.\n'+visible_contract(schema)+'\nCURRENT_BRIEF:\n'+compact(small))
  if resumed:
   instruction += '\nPERSISTENT_GROUP_REVIEW: Continue the same group conversation. This brief is the new measured interval, not a new training run. Reuse already read unchanged knowledge; retrieve exact earlier facts when needed. Old observations retain their original epoch and are not current measurements.\n'
  write(path.parent/'initial_prompt.json',dict(text=instruction,characters=len(instruction)))
  error=None
  # jhyapi turns can legitimately take longer under heavy shared-account
  # load. Give the same bounded request more repair opportunities without
  # resubmitting an uncertain in-progress turn.
  for attempt in range(8):
   prompt=instruction if error is None else 'Correct only the previous completed invalid joint response. Original evidence cutoff and scientific intent unchanged. Validator: '+error+'\n'+visible_contract(schema)
   write(path.parent/('submission_%d.json'%attempt),dict(thread_id=tid,previous_turn_id=previous,request_sha256=reqsha,attempt=attempt,state='submission_intended',time=datetime.datetime.now().astimezone().isoformat()))
   # Only a completed model response can be repaired as a response-format
   # error.  Routing, skill-binding and transport errors happen before or
   # during submission and must propagate without issuing repair turns.
   result=client.run_structured_turn(thread_id=tid,text=prompt if '$cashew-fada-training-expert' in prompt else '$cashew-fada-training-expert\n'+prompt,skill_name='cashew-fada-training-expert',skill_path=SKILL,output_schema=schema,model=MODEL,reasoning_effort=EFFORT,timeout_seconds=timeout,previous_turn_id=previous,include_skill=True)
   write(path.parent/('joint_raw_response_%d.json'%attempt),result);previous=result['turn_id']
   try:
    value=parse_final_object(result['final_text']);decision,automatic=validate_answer(packet,value,reqsha,negative);break
   except (ValueError,KeyError,TypeError) as e:
    error=str(e);write(path.parent/('joint_validation_error_%d.json'%attempt),dict(error=error,turn_id=previous))
    if attempt==7:raise
  return publish(path,packet,negative,small,result,decision,automatic)
 except Exception as exc:
  write(path.parent/'session_failure.json',dict(error_type=type(exc).__name__,error=str(exc),events=client._events,reconciliation=client.last_reconciliation,no_automatic_resubmission=True))
  raise
 finally:client.close()

def publication_complete(path):
 receipt=path.with_name('publication_receipt.json')
 if not receipt.exists():return False
 data=read(receipt)
 return data.get('request_sha256')==sha(path) and all(Path(p).is_file() and sha(p)==h for p,h in data['files'].items())

def dispatch_with_terminal_retry(path,codex,timeout=900,max_retries=2):
 """Retry fresh short sessions only after the exact prior turn is terminal.

 Never retry an ambiguous, running, completed or different request/turn. The
 existing recovery gate verifies identity and archives complete evidence.
 """
 if type(max_retries) is not int or not 0<=max_retries<=2:
  raise ValueError('terminal retry limit must be between zero and two')
 for attempt in range(max_retries+1):
  try:
   return dispatch(path,codex,timeout)
  except PersistentLunaRecoveryRequired:
   if attempt==max_retries:raise
   packet=read(path)
   if sessions.policy(packet)!=sessions.SHORT or not archive_recoverable_session(path.parent):raise
   time.sleep(attempt+1)

def publish(path,packet,negative,small,result,decision,automatic):
 decision_path=path.with_name('decision.json');provenance=path.with_name('provenance.json');tid=result['thread_id']
 write(decision_path,decision)
 prov=dict(request=str(path.resolve()),request_sha256=sha(path),decision=str(decision_path),decision_sha256=sha(decision_path),thread_id=tid,turn_id=result['turn_id'],skill_name='cashew-fada-training-expert',skill_sha256=sha(SKILL),model=MODEL,reasoning_effort=EFFORT,token_usage=result.get('token_usage'),logical_group_owner=packet['job']['thread_key'],context='fresh per decision, complete own-run external memory',joint_negative_selection=negative is not None,second_negative_model_call=False,retrieval_log=str(path.parent/'retrieval_calls.json'),original_packet_characters=len(compact(packet)),initial_brief_characters=len(compact(small)),target_labels_exposed=False,created_at=datetime.datetime.now().astimezone().isoformat())
 prov['context']=sessions.policy(packet)
 prov['evidence_adapter_sha256']={n:sha(Path(__file__).with_name(n)) for n in ['memory_dispatch_v1.py','on_demand_memory.py','joint_negative.py','joint_ranking_controls.py','negative_usage_facts.py','visible_response_contract.py','group_session_lifecycle.py','persistent_luna.py']}
 if small.get('automatic_negative_measured_usage') is not None:prov['measured_automatic_negative_usage']=small['automatic_negative_measured_usage']
 raw_value=parse_final_object(result['final_text']).get('automatic_negative')
 if automatic is not None and isinstance(raw_value,dict) and 'ranking_controls' in raw_value:
  prov['ranking_representation']=dict(version='prior_fraction_and_nonprior_share_v1',agent_controls=raw_value['ranking_controls'],derived_weights=automatic['ranking_weights'])
 if result.get('host_recovery_receipt'):
  recovery=Path(result['host_recovery_receipt']);prov['host_recovery_receipt']=dict(path=str(recovery),sha256=sha(recovery))
 files=[decision_path,provenance]
 if automatic is not None:
  dest=Path(packet['joint_negative_candidates']['path']).parent;p=dest/'joint_selection.json'
  write(p,dict(epoch=packet['epoch'],checkpoint_sha256=negative['identity']['source_checkpoint_sha256'],automatic_negative=automatic,review_decision=str(decision_path),review_decision_sha256=sha(decision_path),thread_id=tid,turn_id=result['turn_id']));files.append(p)
 write(provenance,prov)
 write(path.parent/'session_receipt.json',dict(owner=packet['job']['thread_key'],request_sha256=sha(path),thread_id=tid,state='completed',turn_id=result['turn_id'],token_usage=result.get('token_usage')))
 write(path.parent/'publication_receipt.json',dict(request_sha256=sha(path),files={str(p):sha(p) for p in files}))
 sessions.complete(packet,path,result,MODEL,EFFORT,SKILL)
 return dict(decision=decision,provenance=prov)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--request',required=True,type=Path);p.add_argument('--timeout-seconds',type=int,default=900);p.add_argument('--codex',default=os.environ.get('CASHEW_FADA_CODEX_BINARY','codex'));p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 try:
  if a.prepare_only:
   packet,neg,store,small=prepare(a.request);print(compact(dict(state='prepared_only',characters=len(compact(small)),no_model_call=True)))
  else:print(compact(dispatch_with_terminal_retry(a.request,a.codex,a.timeout_seconds)))
 except Exception as e:
  write(a.request.parent/'dispatch_failure.json',dict(error_type=type(e).__name__,error=str(e),failure_kind='other_agent_failure'));raise
