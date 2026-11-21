"""Decision-gate tests for real same-run history tool semantics."""
import copy,hashlib,json,shutil,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch

import history_audit
import history_dispatch_candidate as dispatch
from fd_safe_decision_transaction_candidate_v1 import SafeTransaction
from history_bridge_candidate import attach_dispatch_receipt
from request_reconstruction import verify_request
from review_contract import build_anomaly_review,build_checkpoint_selection,build_full_review,REFERENCE_NAMES
from sealed_packet import values_digest

def art(kind,epoch,values):
 return {'kind':kind,'epoch':epoch,'values':values,'sha256':values_digest(kind,epoch,values)}

def fetch(root,*,job_id,run_id,cutoff_epoch,start_epoch,end_epoch,kinds,**kwargs):
 records=[]
 if start_epoch==0 and 'initialization' in kinds:
  records.append({'epoch':0,'initialization':{'action':'KEEP','request_evidence':{'transfer_identity':{'source_period':'2024-2025'}}}})
 if start_epoch<=1<=end_epoch and 'training' in kinds:
  records.append({'epoch':1,'training':{'cls':0.5}})
 return {'version':'same_run_complete_history_v2','job_id':job_id,'run_id':run_id,'cutoff_epoch':cutoff_epoch,
         'start_epoch':start_epoch,'end_epoch':end_epoch,'records':records}

class FakeHistoryClient:
 mode='first';action='KEEP';instances=[]
 def __init__(self,binary,timeout,*,root,job_id,run_id,cutoff_epoch,audit_dir):
  self.root=root;self.job_id=job_id;self.run_id=run_id;self.cutoff=cutoff_epoch;self.audit=Path(audit_dir);self.calls=0;self.round=0
  self.history_thread=None;FakeHistoryClient.instances.append(self)
 def start_server(self):pass
 def close(self):pass
 def _request(self,name,payload):return {'thread':{'id':'synthetic-thread'}},[]
 def _send(self,payload):pass
 def _query(self,thread,args):
  self.calls+=1;request={'method':'item/tool/call','params':{'tool':'query_training_history','threadId':thread,'turnId':'turn-%d'%self.round,'arguments':args}}
  result=fetch(self.root,job_id=self.job_id,run_id=self.run_id,cutoff_epoch=self.cutoff,**args)
  (self.audit/('tool_call_%02d.json'%self.calls)).write_text(json.dumps({'request':request,'result':result}))
 def run_structured_turn(self,*,thread_id,text,skill_name,skill_path,output_schema,model,reasoning_effort,timeout_seconds,previous_turn_id=None):
  self.round+=1
  if self.round>1 and previous_turn_id!='turn-1':raise AssertionError('retry must bind prior terminal turn')
  self._send({'method':'turn/start','params':{'threadId':thread_id,'model':model,'effort':reasoning_effort,'outputSchema':output_schema,
             'input':[{'type':'skill','name':skill_name,'path':str(skill_path)},{'type':'text','text':text}]}})
  if self.mode=='first' or (self.mode=='second' and self.round==2):
   self._query(thread_id,{'start_epoch':0,'end_epoch':0,'kinds':['initialization']})
   self._query(thread_id,{'start_epoch':1,'end_epoch':1,'kinds':['training']})
  answer={'action':'KEEP','updates':[],'schedule':{'component':'none','operation':'none','phase':0},'initial_schedules':[],
          'experiment':{'parameter':'none','value':0,'steps':0},'observations':'observed','diagnosis':'uncertain',
          'alternative':'other cause','expected_response':'review later','reversal_condition':'not applicable','response_hypotheses':None}
  answer['action']=self.action
  if self.action=='ADJUST':answer['updates']=[{'parameter':'temperature','value':2.0}]
  return {'thread_id':thread_id,'turn_id':'turn-%d'%self.round,'final_text':json.dumps(answer),
          'event_tracking':{'turn_completed':True,'turn_status':'completed'}}

class RequiredHistory(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.skill=self.root/'cashew-fada-training-expert'/'SKILL.md';self.skill.parent.mkdir()
  self.skill.write_text('---\nname: cashew-fada-training-expert\ndescription: fixture\n---\nSafe synthetic skill.\n')
  (self.skill.parent/'scripts').mkdir();source=Path('@WORKSPACE@/.codex_training_current_account/skills/cashew-fada-training-expert/scripts/verify_explicit_skill_binding.py')
  shutil.copyfile(source,self.skill.parent/'scripts/verify_explicit_skill_binding.py')
  self.verifier=hashlib.sha256((self.skill.parent/'scripts/verify_explicit_skill_binding.py').read_bytes()).hexdigest()
  (self.skill.parent/'references').mkdir();self.refs={}
  for name in REFERENCE_NAMES:
   blob=('synthetic permitted '+name).encode();(self.skill.parent/'references'/name).write_bytes(blob);self.refs[name]=hashlib.sha256(blob).hexdigest()
  self.shash=hashlib.sha256(self.skill.read_bytes()).hexdigest();self.stage=self.root/'job'/'review'/'transport';self.stage.parent.mkdir(parents=True)
  self.row={'training':art('training',5,{'cls':.5,'steps':63}),'controls':art('controls',5,{'temperature':1.8})}
  self.source=art('source',5,{'class2_f1':.7});self.panel=art('panel',5,{'predicted_class2_fraction':.2})
  self.trusted={(x['kind'],x['epoch'],x['sha256']) for x in list(self.row.values())+[self.source,self.panel]}
  self.bounds={'temperature':[1.2,3.]};self.actions=['KEEP','ADJUST','STOP']
  self.req=build_full_review(5,[self.row],self.source,self.panel,self.trusted,self.bounds,self.actions,history_required=True)
  self.expected={'epoch':5,'bounds':self.bounds,'actions':self.actions,'history_required':True}
  FakeHistoryClient.instances=[];FakeHistoryClient.mode='first';FakeHistoryClient.action='KEEP'

 def test_current_only_and_no_implicit_history(self):
  self.assertEqual(len(self.req['evidence']['training_window']),1)
  self.assertEqual(self.req['evidence']['training_window'][0]['training']['epoch'],5)
  self.assertIsNone(self.req['evidence']['prior_review_trend']);self.assertIsNone(self.req['prior_committed_action'])
  self.assertTrue(self.req['capabilities']['history_query_required']);self.assertFalse(self.req['capabilities']['target_point_evidence'])
  self.assertNotIn('No external tools or history.',self.req['instruction'])
  self.assertEqual(verify_request(self.req,self.trusted,expected=self.expected),self.req)
  five=[{'training':art('training',e,{'cls':.5}),'controls':art('controls',e,{'temperature':1.8})} for e in range(1,6)]
  trust={(v['kind'],v['epoch'],v['sha256']) for row in five for v in row.values()}|{(self.source['kind'],5,self.source['sha256']),(self.panel['kind'],5,self.panel['sha256'])}
  with self.assertRaisesRegex(ValueError,'must not preload'):
   build_full_review(5,five,self.source,self.panel,trust,self.bounds,self.actions,history_required=True)
  legacy=build_full_review(5,five,self.source,self.panel,trust,self.bounds,self.actions)
  self.assertEqual(len(legacy['evidence']['training_window']),5)
  self.assertFalse(legacy['capabilities']['historical_retrieval'])
  with self.assertRaises(ValueError):verify_request(legacy,trust,expected=self.expected)

 def test_anomaly_and_selector_advertise_mandatory_history(self):
  b={'main_lr':[1e-7,1e-5]};r=build_anomaly_review(5,[self.row],self.trusted,b,'training_metric_deviation',history_required=True)
  self.assertEqual(len(r['evidence']['training_window']),1);self.assertTrue(r['capabilities']['history_query_required'])
  self.assertEqual(verify_request(r,self.trusted,expected={'epoch':5,'bounds':b,'trigger':'training_metric_deviation','history_required':True}),r)
  c=art('candidate',5,{'source_class2_f1':.7});t={('candidate',5,c['sha256'])}
  s=build_checkpoint_selection(5,[c],t,history_required=True)
  self.assertEqual(verify_request(s,t,expected={'stop_epoch':5,'history_required':True}),s)

 def test_unavailable_schedule_cannot_be_advertised_in_new_reviews(self):
  from jsonschema import Draft202012Validator
  strict=build_full_review(5,[self.row],self.source,self.panel,self.trusted,self.bounds,self.actions,
                           history_required=True,strict_schedule_schema=True)
  expected={**self.expected,'strict_schedule_schema':True}
  self.assertEqual(verify_request(strict,self.trusted,expected=expected),strict)
  self.assertEqual(verify_request(self.req,self.trusted,expected=self.expected),self.req)
  with self.assertRaises(ValueError):verify_request(strict,self.trusted,expected=self.expected)
  schedule=strict['schema']['properties']['schedule']
  good={'component':'none','operation':'none','phase':0}
  invalid={'component':'none','operation':'start_cosine_finish','phase':0}
  self.assertTrue(Draft202012Validator(schedule).is_valid(good))
  self.assertFalse(Draft202012Validator(schedule).is_valid(invalid))
  self.assertFalse(Draft202012Validator(schedule).is_valid({**good,'phase':.5}))
  self.assertTrue(Draft202012Validator(self.req['schema']['properties']['schedule']).is_valid(invalid))
  self.assertIn('does not activate a schedule',strict['instruction'])
  b={'main_lr':[1e-7,1e-5]}
  light=build_anomaly_review(5,[self.row],self.trusted,b,'training_metric_deviation',history_required=True,strict_schedule_schema=True)
  ex={'epoch':5,'bounds':b,'trigger':'training_metric_deviation','history_required':True,'strict_schedule_schema':True}
  self.assertEqual(verify_request(light,self.trusted,expected=ex),light)
  self.assertFalse(Draft202012Validator(light['schema']['properties']['schedule']).is_valid(invalid))

 def _dispatch(self,mode):
  FakeHistoryClient.mode=mode;txn=SafeTransaction(self.stage.parent/'transaction','J1','R1',5,'full_review')
  txn.prepare(self.req,b'synthetic checkpoint',lambda raw,epoch:None)
  with patch.object(dispatch,'SKILL',self.skill),patch.object(history_audit,'fetch',side_effect=fetch):
   result=dispatch.dispatch(self.req,self.trusted,self.expected,self.stage,codex_binary='fake',skill_sha256=self.shash,
      verifier_sha256=self.verifier,reference_hashes=self.refs,before_turn=txn.claim_submission,client_class=FakeHistoryClient,
      history_root=self.root,history_job_id='J1',history_run_id='R1')
   receipt=attach_dispatch_receipt(txn,self.stage,trusted=self.trusted,expected=self.expected,skill_path=self.skill,
      skill_sha256=self.shash,verifier_sha256=self.verifier,reference_hashes=self.refs,history_root=self.root,history_job_id='J1',history_run_id='R1')
  return result,receipt,txn

 def test_valid_queries_are_accepted_and_bound_to_transaction(self):
  result,receipt,txn=self._dispatch('first')
  self.assertEqual(result['decision'] if 'decision' in result else result['validated']['action'],'KEEP')
  self.assertTrue((txn.root/'completed.json').is_file());self.assertEqual(json.loads((self.stage/'accepted_attempt.json').read_text())['attempt'],1)

 def test_completed_no_history_answer_retries_without_applying_it(self):
  result,receipt,txn=self._dispatch('second')
  self.assertEqual(FakeHistoryClient.instances[-1].round,2)
  self.assertEqual(json.loads((self.stage/'accepted_attempt.json').read_text())['attempt'],2)
  self.assertTrue((self.stage/'history_attempt_01/rejected.json').is_file())
  self.assertEqual(receipt['turn_id'],'turn-2');self.assertTrue((txn.root/'completed.json').is_file())

 def test_no_history_never_publishes_a_decision(self):
  with self.assertRaises(history_audit.HistoryQueryRequired):self._dispatch('never')
  self.assertEqual(FakeHistoryClient.instances[-1].round,2)
  self.assertFalse((self.stage/'receipt.json').exists());self.assertFalse((self.stage.parent/'transaction/completed.json').exists())

 def test_only_initialization_is_insufficient_when_prior_epochs_exist(self):
  self.stage.mkdir(parents=True);args={'start_epoch':0,'end_epoch':0,'kinds':['initialization']}
  rec={'request':{'method':'item/tool/call','params':{'tool':'query_training_history','threadId':'T','turnId':'turn-1','arguments':args}},
       'result':fetch(self.root,job_id='J1',run_id='R1',cutoff_epoch=5,**args)}
  (self.stage/'tool_call_01.json').write_text(json.dumps(rec))
  with patch.object(history_audit,'fetch',side_effect=fetch),self.assertRaises(history_audit.HistoryQueryRequired):
   history_audit.verify(self.stage,root=self.root,job_id='J1',run_id='R1',cutoff_epoch=5,thread_id='T',require_history=True,allowed_turn_ids=['turn-1'])

 def test_query_from_another_turn_cannot_satisfy_requirement(self):
  self.stage.mkdir(parents=True);args={'start_epoch':0,'end_epoch':0,'kinds':['initialization']}
  rec={'request':{'method':'item/tool/call','params':{'tool':'query_training_history','threadId':'T','turnId':'other-turn','arguments':args}},
       'result':fetch(self.root,job_id='J1',run_id='R1',cutoff_epoch=5,**args)}
  (self.stage/'tool_call_01.json').write_text(json.dumps(rec))
  with patch.object(history_audit,'fetch',side_effect=fetch),self.assertRaisesRegex(ValueError,'audited completed turn'):
   history_audit.verify(self.stage,root=self.root,job_id='J1',run_id='R1',cutoff_epoch=5,thread_id='T',require_history=True,allowed_turn_ids=['turn-1'])

 def test_missing_snapshot_does_not_trigger_old_checkpoint_reload(self):
  from complete_history_store import load_epoch
  root=self.root/'live';directory=root/'journal/epoch_000001';directory.mkdir(parents=True)
  (root/'journal/identity.json').write_text(json.dumps({'job_id':'J1','run_id':'R1','max_epoch':5}))
  (directory/'trained.json').write_text(json.dumps({'epoch':1,'checkpoint_sha256':'a'*64,'route':'SKIP'}))
  (directory/'commit.json').write_text(json.dumps({'epoch':1,'trained_checkpoint_sha256':'a'*64,'route':'SKIP'}))
  (directory/'trained_checkpoint.bin').write_bytes(b'old model must not be read')
  original=Path.read_bytes
  def guarded(path):
   if path.name=='trained_checkpoint.bin':raise AssertionError('old checkpoint was read')
   return original(path)
  with patch.dict('os.environ',{'CASHEW_FAST_RESUME':'1'}),patch.object(Path,'read_bytes',guarded):
   with self.assertRaisesRegex(ValueError,'snapshot is missing'):load_epoch(root,'J1','R1',1)

 def test_real_dynamic_tool_handler_records_model_requested_query(self):
  from history_tool_client import HistoryClient
  from persistent_luna import PersistentLunaClient
  self.stage.mkdir(parents=True);client=HistoryClient('fake',5,root=self.root,job_id='J1',run_id='R1',cutoff_epoch=5,audit_dir=self.stage)
  client.history_thread='T';args={'start_epoch':0,'end_epoch':0,'kinds':['initialization']}
  call={'method':'item/tool/call','id':7,'params':{'tool':'query_training_history','threadId':'T','arguments':args}}
  finished={'method':'turn/completed'}
  with patch.object(PersistentLunaClient,'_read_line',side_effect=[call,finished]),patch('history_tool_client.fetch',side_effect=fetch),patch.object(client,'_send') as send:
   self.assertEqual(client._read_line(time.monotonic()+5),finished)
  record=json.loads((self.stage/'tool_call_01.json').read_text());self.assertEqual(record['request'],call)
  self.assertEqual(record['result']['records'][0]['epoch'],0);self.assertTrue(send.call_args.args[0]['result']['success'])

 def test_point_evaluation_aliases_never_leave_the_history_tool(self):
  from history_tool_client import HistoryClient
  from persistent_luna import PersistentLunaClient
  self.stage.mkdir(parents=True);client=HistoryClient('fake',5,root=self.root,job_id='J1',run_id='R1',cutoff_epoch=5,audit_dir=self.stage)
  client.history_thread='T';args={'start_epoch':1,'end_epoch':1,'kinds':['training']}
  call={'method':'item/tool/call','id':7,'params':{'tool':'query_training_history','threadId':'T','arguments':args}}
  for unsafe in ({'point_fp':2},{'target_checkpoint_ranking':[1,2]},{'target_accuracy':.9},{'target':{'labels':[0,1]}}):
   with self.subTest(unsafe=unsafe),patch.object(PersistentLunaClient,'_read_line',return_value=call), \
        patch('history_tool_client.fetch',return_value={'records':[{'epoch':1,**unsafe}]}),patch.object(client,'_send') as send:
    with self.assertRaisesRegex(ValueError,'Sealed point evidence'):client._read_line(time.monotonic()+5)
    send.assert_not_called();self.assertFalse((self.stage/'tool_call_01.json').exists())

 def test_evaluation_producer_fields_rejected_and_legal_training_fields_kept(self):
  from complete_history_store import _project
  from online_boundary import verify_online
  raw={'source_validation':{'class2_f1':.7},'unlabelled_panel':{'predicted_class2_fraction':.2},
       'target_checkpoint_ranking':[1,2],'point_fp':3,'target_accuracy':.9,
       'target':{'labels':[0,1],'accuracy':.8},'controls':{'hard_threshold':.4}}
  with self.assertRaises(ValueError):_project(raw)
  clean=_project({'source_validation':{'class2_f1':.7,'precision':.8},'controls':{'hard_threshold':.4}})
  self.assertEqual(clean['source_validation']['class2_f1'],.7)
  self.assertEqual(clean['controls']['hard_threshold'],.4);verify_online(clean)

 def test_legal_training_prose_and_digest_are_not_rewritten(self):
  from online_boundary import isolate_history_text
  source={'version':'same_run_complete_history_v2','records':[{'epoch':3,'source':{'class2_f1':.7},
      'reasoning':{'a':'Source F1 was 0.95; this does not establish target accuracy','b':'At E42 inspect negative training fit and threshold control','c':'Observed source loss is 2.3'}}]}
  value=dict(source,sha256=hashlib.sha256(json.dumps(source,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest())
  safe=isolate_history_text(value);reason=safe['records'][0]['reasoning']
  self.assertEqual(reason,source['records'][0]['reasoning']);self.assertEqual(safe,value)
  self.assertEqual(safe['records'][0]['source']['class2_f1'],.7);h=safe.pop('sha256')
  self.assertEqual(h,hashlib.sha256(json.dumps(safe,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest())

 def test_worker_full_and_light_packets_only_send_current_epoch(self):
  import torch
  import train_expert_v4 as worker
  from native_safe_checkpoint_codec_v1 import encode
  from sealed_safe_epoch_journal_v1 import SafeTrainingJournal
  from experimental_production_orchestration_v1 import ExperimentalProductionWorker
  nets=[torch.nn.Linear(1,1) for _ in range(3)];opts=[torch.optim.AdamW(n.parameters()) for n in nets]
  cfg=worker.config_anchor({});metrics={'experimental_gate_decision':{'route':'LIGHT_REVIEW','reason':'missing training metric','integrity':{}}}
  cp=encode(worker.checkpoint_payload(5,*nets,*opts,cfg,metrics));signature=hashlib.sha256(cp).hexdigest()
  row={'epoch':5,'training':{'cls':.5,'steps':63},'controls':{'temperature':1.8}}
  source={'class2_f1':.7};panel={'predicted_class2_fraction':.2}
  class FS:
   def json(self,path):return {'epoch':5,'checkpoint_sha256':signature,'source':source,'panel':panel}
  class Journal:
   _verify_review_request=SafeTrainingJournal._verify_review_request
   _fs=FS();max_epoch=200;job={}
   def _held(self):pass
   def recover(self):return {'state':self.route,'epoch':5,'checkpoint':cp}
   def training_window_evidence(self,epoch,*,current_only=False):
    if not current_only:raise AssertionError('worker attempted to preload earlier rounds')
    return [row],[]
   def prepare_full_review(self,epoch):return {'checkpoint':cp}
   def current_review_context(self,epoch):return None,set()
   def prior_action_context(self,epoch):raise AssertionError('worker attempted to preload previous decisions')
   def active_pool_evidence(self,epoch):return None
  journal=Journal();w=ExperimentalProductionWorker({},journal,None)
  with patch.dict('os.environ',{'CASHEW_HISTORY_REQUIRED':'1'}),patch.object(w,'_authorize'),patch.object(w,'_authorize_review_sources'),patch.object(w,'_authorize_stop'):
   for route in ('FULL_REVIEW','LIGHT_REVIEW'):
    journal.route=route
    q,trusted,expected,previous=(w.prepare_full_review_request(5,allow_stop=True) if route=='FULL_REVIEW' else w.prepare_light_review_request(5))
    self.assertEqual(len(q['evidence']['training_window']),1);self.assertEqual(q['evidence']['training_window'][0]['training']['epoch'],5)
    self.assertTrue(expected['history_required']);self.assertIsNone(previous)
    self.assertEqual(verify_request(q,trusted,expected=expected,previous=previous),q)

 def test_actual_router_applies_only_audited_keep_adjust_or_stop(self):
  import torch
  import train_expert_v4 as worker
  from native_safe_checkpoint_codec_v1 import encode,decode
  from history_router_candidate import route
  for action in ('KEEP','ADJUST','STOP'):
   with self.subTest(action=action):
    output=self.root/('job_'+action);output.mkdir()
    job={'job_id':'J1','run_id':'R1','output_root':str(output),'point_validation_online':False,
         'experimental_gate':{'version':'experimental_gate_v1','experimental_only':True,'mode':'enforce'}}
    nets=[torch.nn.Linear(1,1) for _ in range(3)]
    cfg=worker.config_anchor({});cfg['class_knowledge']={'temperature':cfg['temperature'],'confidence_cap':cfg['confidence_cap']}
    opts=[torch.optim.AdamW(n.parameters(),lr=cfg['main_lr'],weight_decay=cfg['main_weight_decay']) for n in nets]
    pre=encode(worker.checkpoint_payload(5,*nets,*opts,cfg,{}));FakeHistoryClient.action=action
    with patch.object(dispatch,'SKILL',self.skill),patch.object(history_audit,'fetch',side_effect=fetch):
     result=route(job,self.req,self.trusted,self.expected,pre,codex_binary='fake',skill_sha256=self.shash,
          verifier_sha256=self.verifier,reference_hashes=self.refs,client_class=FakeHistoryClient,allow_stop=True)
    self.assertEqual(result['decision']['action'],action)
    self.assertEqual(decode(result['post_checkpoint'])['config']['temperature'],2.0 if action=='ADJUST' else 1.8)
    self.assertEqual(result['decision']['continue_training'],action!='STOP')

if __name__=='__main__':unittest.main()
