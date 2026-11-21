"""Isolation-only fd-safe transaction candidate. Not a production authorization.

Every artifact is opened relative to a no-symlink transaction directory fd,
with single-link checks; cooperating operations hold one transaction lock.
External evidence certification and raw-decision/transition guards are callers'
responsibility. Unknown submissions are never re-claimed.
"""
import json,os
from contextlib import contextmanager
from pathlib import Path
from decision_transaction import IDENT,canonical,digest

class SafeTransaction:
 def __init__(self,path,job_id,run_id,epoch,kind):
  valid_run=(isinstance(run_id,str) and len(run_id)<=160 and all(IDENT.fullmatch(part) and part not in ('.','..') for part in run_id.split('/')))
  if any(not isinstance(x,str) or not IDENT.fullmatch(x) for x in (job_id,kind)) or not valid_run or type(epoch) is not int or epoch<0:
   raise ValueError('invalid transaction identity')
  self.root=Path(os.path.abspath(path));self.identity={'job_id':job_id,'run_id':run_id,'epoch':epoch,'kind':kind}

 @contextmanager
 def locked(self):
  # Parent must already exist; SafeRoot rejects symlinks in every component.
  from sealed_safe_final_selection_v1 import SafeRoot
  parent=SafeRoot(self.root.parent)
  try:parent.mkdirs(self.root.name)
  finally:parent.close()
  fs=SafeRoot(self.root);fd=None
  try:
   fd=fs.lock('transaction.lock');yield fs
  finally:
   if fd is not None:os.close(fd)
   fs.close()

 def _require(self,fs=None):
  if fs is None:
   with self.locked() as owned:return self._require(owned)
  manifest=fs.json('manifest.json')
  if manifest.get('identity')!=self.identity:raise ValueError('transaction identity mismatch')
  req=fs.read('request.json');pre=fs.read('pre_checkpoint.bin')
  if digest(req)!=manifest['request_sha256'] or digest(pre)!=manifest['pre_checkpoint_sha256']:
   raise ValueError('request or checkpoint changed')
  return manifest,req,pre

 def prepare(self,request,pre_checkpoint,checkpoint_validate):
  if not isinstance(request,dict) or not isinstance(pre_checkpoint,bytes) or not callable(checkpoint_validate):raise ValueError('invalid prepare inputs')
  checkpoint_validate(pre_checkpoint,self.identity['epoch'])
  req=canonical(request);manifest={'version':'agent_transaction_v1','identity':self.identity,'request_sha256':digest(req),'pre_checkpoint_sha256':digest(pre_checkpoint)}
  with self.locked() as fs:
   fs.publish_relative('request.json',req);fs.publish_relative('pre_checkpoint.bin',pre_checkpoint);fs.publish_relative('manifest.json',canonical(manifest))
   self._require(fs)
  return manifest

 def claim_submission(self,thread_id):
  if not isinstance(thread_id,str) or not IDENT.fullmatch(thread_id):raise ValueError('invalid thread identity')
  with self.locked() as fs:
   manifest,_,_=self._require(fs)
   if fs.read('submission.json',missing_ok=True) is not None:raise RuntimeError('submission already claimed: reconcile exact prior turn; do not resend')
   fs.publish_relative('submission.json',canonical({'identity':self.identity,'thread_id':thread_id,'request_sha256':manifest['request_sha256'],'state':'intent_before_network'}))

 def attach_completed_response(self,thread_id,turn_id,response,*,reconciled):
  if reconciled is not True or not all(isinstance(x,str) and IDENT.fullmatch(x) for x in (thread_id,turn_id)) or not isinstance(response,dict):raise ValueError('verified terminal response required')
  with self.locked() as fs:
   manifest,_,_=self._require(fs);submission=fs.json('submission.json')
   if submission['identity']!=self.identity or submission['request_sha256']!=manifest['request_sha256'] or submission['thread_id']!=thread_id:raise ValueError('submission mismatch')
   blob=canonical(response);out={'identity':self.identity,'thread_id':thread_id,'turn_id':turn_id,'request_sha256':manifest['request_sha256'],'response_sha256':digest(blob),'state':'completed'}
   fs.publish_relative('terminal_identity.json',canonical(out));fs.publish_relative('response.json',blob);fs.publish_relative('completed.json',canonical(out))

 def _lineage(self,fs,decision_required):
  manifest,req,pre=self._require(fs);submission=fs.json('submission.json');terminal=fs.json('terminal_identity.json');completed=fs.json('completed.json')
  raw=fs.read('response.json');response=json.loads(raw)
  if (completed!=terminal or submission.get('identity')!=self.identity or submission.get('request_sha256')!=manifest['request_sha256'] or completed.get('identity')!=self.identity or completed.get('thread_id')!=submission.get('thread_id') or completed.get('request_sha256')!=manifest['request_sha256'] or completed.get('response_sha256')!=digest(raw)):
   raise ValueError('response lineage mismatch')
  if not decision_required:return manifest,req,pre,response,raw,completed
  decraw=fs.read('decision.json');decision=json.loads(decraw);receipt=fs.json('decision_receipt.json')
  if receipt!={'identity':self.identity,'request_sha256':manifest['request_sha256'],'pre_checkpoint_sha256':manifest['pre_checkpoint_sha256'],'response_sha256':digest(raw),'decision_sha256':digest(decraw)}:raise ValueError('decision lineage mismatch')
  return manifest,req,pre,response,raw,completed,decision,decraw

 def normalize(self,validator):
  if not callable(validator):raise ValueError('decision validator required')
  with self.locked() as fs:
   manifest,req,pre,response,raw,_=self._lineage(fs,False);decision=validator(json.loads(req),response,pre)
   if not isinstance(decision,dict) or decision.get('epoch')!=self.identity['epoch']:raise ValueError('invalid normalized decision')
   out=canonical(decision);meta={'identity':self.identity,'request_sha256':manifest['request_sha256'],'pre_checkpoint_sha256':manifest['pre_checkpoint_sha256'],'response_sha256':digest(raw),'decision_sha256':digest(out)}
   fs.publish_relative('decision.json',out);fs.publish_relative('decision_receipt.json',canonical(meta));return decision

 def commit(self,checkpoint_validate,transition,*,post_validate=None,require_commit=False):
  if not callable(checkpoint_validate) or not callable(transition) or (post_validate is not None and not callable(post_validate)):raise ValueError('worker validators required')
  with self.locked() as fs:
   if require_commit and fs.read('commit.json',missing_ok=True) is None:raise RuntimeError('committed transaction receipt missing')
   manifest,req,pre,response,raw,completed,decision,decraw=self._lineage(fs,True);checkpoint_validate(pre,self.identity['epoch'])
   post=fs.read('post_checkpoint.bin',missing_ok=True);readyraw=fs.read('post_ready.json',missing_ok=True)
   if post is not None and readyraw is None:raise RuntimeError('orphan post checkpoint: manual reconciliation required')
   if readyraw is not None:
    if post is None or json.loads(readyraw)!={'identity':self.identity,'pre_sha256':digest(pre),'decision_sha256':digest(decraw),'post_sha256':digest(post)}:raise ValueError('post checkpoint identity mismatch')
    checkpoint_validate(post,self.identity['epoch'])
    if post_validate is not None:post_validate(json.loads(req),response,pre,decision,post)
   else:
    post=transition(pre,decision)
    if not isinstance(post,bytes):raise ValueError('transition must return checkpoint bytes')
    checkpoint_validate(post,self.identity['epoch'])
    if post_validate is not None:post_validate(json.loads(req),response,pre,decision,post)
    fs.publish_relative('post_checkpoint.bin',post);fs.publish_relative('post_ready.json',canonical({'identity':self.identity,'pre_sha256':digest(pre),'decision_sha256':digest(decraw),'post_sha256':digest(post)}))
   checkpoint_validate(post,self.identity['epoch'])
   commit={'identity':self.identity,'request_sha256':digest(req),'pre_checkpoint_sha256':digest(pre),'response_sha256':digest(raw),'decision_sha256':digest(decraw),'post_checkpoint_sha256':digest(post),'thread_id':completed['thread_id'],'turn_id':completed['turn_id']}
   fs.publish_relative('commit.json',canonical(commit));return commit,post
