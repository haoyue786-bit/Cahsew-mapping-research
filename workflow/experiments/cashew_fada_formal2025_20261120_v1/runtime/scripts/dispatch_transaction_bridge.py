"""Verify a sealed dispatcher receipt and attach it to a decision transaction.

 Caller pins/authorizes the skill, verifier, reference hashes and run provenance.
 Receipt/hash checks do not prove that training evidence was actually blind.
 Never issues a model call; it only accepts an already completed dispatcher turn.
"""
import hashlib,json,types
from pathlib import Path
from decision_transaction import canonical,digest
from request_reconstruction import verify_request
from review_contract import load_approved_references,validate_review_response,validate_checkpoint_selection

def read(stage,name):
 p=stage/name
 if p.is_symlink() or not p.is_file():raise ValueError('missing or unsafe dispatch artifact '+name)
 return p.read_bytes()
def attach_dispatch_receipt(transaction,stage,*,trusted,expected,skill_path,skill_sha256,verifier_sha256,reference_hashes,previous=None):
 stage=Path(stage);skill=Path(skill_path).resolve(strict=True);skill_bytes=skill.read_bytes()
 if digest(skill_bytes)!=skill_sha256:raise ValueError('skill hash mismatch')
 verifier=skill.parent/'scripts/verify_explicit_skill_binding.py';verifier_bytes=verifier.read_bytes()
 if digest(verifier_bytes)!=verifier_sha256:raise ValueError('verifier hash mismatch')
 module=types.ModuleType('sealed_transaction_binding');module.__file__=str(verifier)
 exec(compile(verifier_bytes,str(verifier),'exec'),module.__dict__)
 refs=load_approved_references(skill.parent,reference_hashes);refs_raw=canonical(refs)
 request_raw=read(stage,'request.json');request=json.loads(request_raw);verify_request(request,trusted,expected=expected,previous=previous)
 manifest,txn_req,_=transaction._require()
 if request_raw!=txn_req or digest(request_raw)!=manifest['request_sha256']:raise ValueError('transaction and dispatcher requests differ')
 if read(stage,'approved_references.json')!=refs_raw:raise ValueError('approved reference mismatch')
 thread=json.loads(read(stage,'thread.json'));turn=json.loads(read(stage,'actual_turn.json'));submission=json.loads(read(stage,'submission.json'));receipt=json.loads(read(stage,'receipt.json'))
 expected_thread={'baseInstructions':skill_bytes.decode('utf-8'),'cwd':str(stage.resolve()),'model':'gpt-6.2-sol','sandbox':'read-only','approvalPolicy':'never','ephemeral':False,
  'developerInstructions':'Use only this request and the supplied approved references. Do not use external tools, files, history or target-point evaluation. Return the supplied schema.',
  'allowProviderModelFallback':False, 'runtimeWorkspaceRoots':[], 'environments':[], 'config':{'model_reasoning_effort':'max','project_doc_max_bytes':0,'features.skip_host_skill_discovery':True,'features.multi_agent':False,'features.shell_tool':False,'features.unified_exec':False,'features.apps':False,'web_search':'disabled'}}
 if thread!=expected_thread:raise ValueError('unexpected thread configuration')
 audit=module.validate_binding(thread,turn,expected_model='gpt-6.2-sol',expected_effort='max')
 if audit.get('skill_sha256')!=skill_sha256:raise ValueError('actual binding does not match skill')
 inputs=turn.get('params',{}).get('input',[]);params=turn.get('params',{});prompt='$cashew-fada-training-expert\nApproved reference material:\n'+refs_raw.decode('utf-8')+'\nPoint-sealed request:\n'+request_raw.decode('utf-8')
 if len(inputs)!=2 or [x.get('type') for x in inputs]!=['skill','text'] or inputs[1].get('text')!=prompt or params.get('outputSchema')!=request['schema']:raise ValueError('actual turn differs from reviewed request')
 raw=json.loads(read(stage,'raw_result.json'));answer_raw=read(stage,'answer.json');answer=json.loads(answer_raw)
 if json.loads(raw['final_text'])!=answer:raise ValueError('answer does not match actual terminal response')
 if receipt!={'thread_id':submission.get('thread_id'),'turn_id':raw.get('turn_id'),'request_sha256':digest(request_raw),'skill_sha256':skill_sha256,'verifier_sha256':verifier_sha256,'answer_sha256':digest(answer_raw),'approved_references_sha256':digest(refs_raw)}:raise ValueError('dispatcher receipt mismatch')
 if submission.get('request_sha256')!=digest(request_raw) or submission.get('skill_sha256')!=skill_sha256 or params.get('threadId')!=receipt['thread_id']:raise ValueError('dispatcher submission mismatch')
 if request['contract_version']=='safe_checkpoint_selection_candidate_v1':validate_checkpoint_selection(request,answer)
 else:validate_review_response(request,answer)
 # Exact thread was already claimed before network by dispatcher before_turn.
 transaction.attach_completed_response(receipt['thread_id'],receipt['turn_id'],answer,reconciled=True)
 return {'thread_id':receipt['thread_id'],'turn_id':receipt['turn_id'],'request_sha256':receipt['request_sha256'],'answer_sha256':receipt['answer_sha256']}
