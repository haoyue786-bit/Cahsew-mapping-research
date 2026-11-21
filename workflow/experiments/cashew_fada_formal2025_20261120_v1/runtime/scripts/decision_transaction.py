"""Fail-closed candidate Agent decision transaction; not a live worker.

The controller authenticates request evidence and skill binding independently.
A submission intent is durable *before* transport; ambiguous turns cannot be
resubmitted. Completed-response attachment must be backed by reconciliation of
the exact thread/turn by a trusted caller. Applying a decision is performed on
an independently deserialized checkpoint through a side-effect-free callback;
no callback may mutate a running model, run a branch, or call an external tool.
A crash may lose work but must not duplicate a model call or an applied action.
"""
import fcntl,hashlib,json,os,re,tempfile
from contextlib import contextmanager
from pathlib import Path

IDENT=re.compile(r'^[A-Za-z0-9_.-]{1,100}$')
def digest(data):return hashlib.sha256(data).hexdigest()
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def sync_dir(path):
 fd=os.open(path,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def immutable(path,data):
 fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=str(path.parent))
 try:
  with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  try:os.link(tmp,path,follow_symlinks=False)
  except FileExistsError:
   if path.is_symlink() or path.read_bytes()!=data:raise RuntimeError('conflicting immutable artifact: '+path.name)
  sync_dir(path.parent)
 finally:
  try:os.unlink(tmp)
  except FileNotFoundError:pass

def read_json(path):
 if path.is_symlink():raise ValueError('symlink not permitted')
 data=path.read_bytes();return json.loads(data),data

class Transaction:
 def __init__(self,path,job_id,run_id,epoch,kind):
  if any(not isinstance(x,str) or not IDENT.fullmatch(x) for x in (job_id,run_id,kind)) or type(epoch) is not int or epoch<0:raise ValueError('invalid transaction identity')
  self.root=Path(path);self.identity={'job_id':job_id,'run_id':run_id,'epoch':epoch,'kind':kind}
 @contextmanager
 def locked(self):
  self.root.mkdir(parents=True,exist_ok=True)
  fd=os.open(self.root/'transaction.lock',os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
  try:
   fcntl.flock(fd,fcntl.LOCK_EX);yield
  finally:os.close(fd)
 def _require(self):
  manifest,_=read_json(self.root/'manifest.json')
  if manifest.get('identity')!=self.identity:raise ValueError('transaction identity mismatch')
  req=(self.root/'request.json').read_bytes();pre=(self.root/'pre_checkpoint.bin').read_bytes()
  if digest(req)!=manifest['request_sha256'] or digest(pre)!=manifest['pre_checkpoint_sha256']:raise ValueError('request or checkpoint changed')
  return manifest,req,pre
 def prepare(self,request,pre_checkpoint,checkpoint_validate):
  if not isinstance(request,dict) or not isinstance(pre_checkpoint,bytes) or not callable(checkpoint_validate):raise ValueError('invalid prepare inputs')
  checkpoint_validate(pre_checkpoint,self.identity['epoch'])
  req=canonical(request);manifest={'version':'agent_transaction_v1','identity':self.identity,'request_sha256':digest(req),'pre_checkpoint_sha256':digest(pre_checkpoint)}
  with self.locked():
   # No request is sent until all three artifacts exist and the manifest is verified.
   immutable(self.root/'request.json',req);immutable(self.root/'pre_checkpoint.bin',pre_checkpoint);immutable(self.root/'manifest.json',canonical(manifest));self._require()
  return manifest
 def claim_submission(self,thread_id):
  if not isinstance(thread_id,str) or not IDENT.fullmatch(thread_id):raise ValueError('invalid thread identity')
  with self.locked():
   manifest,_,_=self._require();p=self.root/'submission.json'
   if p.exists():raise RuntimeError('submission already claimed: reconcile exact prior turn; do not resend')
   immutable(p,canonical({'identity':self.identity,'thread_id':thread_id,'request_sha256':manifest['request_sha256'],'state':'intent_before_network'}))
 def attach_completed_response(self,thread_id,turn_id,response,*,reconciled):
  if reconciled is not True or not all(isinstance(x,str) and IDENT.fullmatch(x) for x in (thread_id,turn_id)) or not isinstance(response,dict):raise ValueError('verified terminal response required')
  with self.locked():
   manifest,_,_=self._require();submission,_=read_json(self.root/'submission.json')
   if submission['identity']!=self.identity or submission['request_sha256']!=manifest['request_sha256'] or submission['thread_id']!=thread_id:raise ValueError('submission mismatch')
   blob=canonical(response);out={'identity':self.identity,'thread_id':thread_id,'turn_id':turn_id,'request_sha256':manifest['request_sha256'],'response_sha256':digest(blob),'state':'completed'}
   # Persist exact turn identity *before* attaching a response. If interrupted,
   # a response from a different turn cannot be paired with an orphaned file.
   immutable(self.root/'terminal_identity.json',canonical(out));immutable(self.root/'response.json',blob);immutable(self.root/'completed.json',canonical(out))
 def normalize(self,validator):
  if not callable(validator):raise ValueError('decision validator required')
  with self.locked():
   manifest,req,pre=self._require();submission,_=read_json(self.root/'submission.json');terminal,_=read_json(self.root/'terminal_identity.json');completed,_=read_json(self.root/'completed.json');response,raw=read_json(self.root/'response.json')
   if completed!=terminal or submission.get('identity')!=self.identity or submission.get('request_sha256')!=manifest['request_sha256'] or completed.get('identity')!=self.identity or completed.get('thread_id')!=submission.get('thread_id') or completed.get('request_sha256')!=manifest['request_sha256'] or completed.get('response_sha256')!=digest(raw):raise ValueError('response lineage mismatch')
   decision=validator(json.loads(req),response,pre)
   if not isinstance(decision,dict) or decision.get('epoch')!=self.identity['epoch']:raise ValueError('invalid normalized decision')
   out=canonical(decision);meta={'identity':self.identity,'request_sha256':manifest['request_sha256'],'pre_checkpoint_sha256':manifest['pre_checkpoint_sha256'],'response_sha256':digest(raw),'decision_sha256':digest(out)}
   immutable(self.root/'decision.json',out);immutable(self.root/'decision_receipt.json',canonical(meta));return decision
 def commit(self,checkpoint_validate,transition,*,post_validate=None):
  if not callable(checkpoint_validate) or not callable(transition) or (post_validate is not None and not callable(post_validate)):raise ValueError('worker validators required')
  with self.locked():
   manifest,req,pre=self._require();submission,_=read_json(self.root/'submission.json');terminal,_=read_json(self.root/'terminal_identity.json');completed,_=read_json(self.root/'completed.json');response,raw=read_json(self.root/'response.json');decision,decraw=read_json(self.root/'decision.json');receipt,_=read_json(self.root/'decision_receipt.json')
   if completed!=terminal or submission.get('identity')!=self.identity or submission.get('thread_id')!=completed.get('thread_id') or submission.get('request_sha256')!=manifest['request_sha256'] or completed.get('identity')!=self.identity or completed.get('request_sha256')!=manifest['request_sha256'] or completed.get('response_sha256')!=digest(raw) or receipt!={'identity':self.identity,'request_sha256':manifest['request_sha256'],'pre_checkpoint_sha256':manifest['pre_checkpoint_sha256'],'response_sha256':digest(raw),'decision_sha256':digest(decraw)}:raise ValueError('decision lineage mismatch')
   checkpoint_validate(pre,self.identity['epoch'])
   postpath=self.root/'post_checkpoint.bin';ready=self.root/'post_ready.json'
   if postpath.exists() and not ready.exists():raise RuntimeError('orphan post checkpoint: manual reconciliation required')
   if ready.exists():
    ready_data,_=read_json(ready);post=postpath.read_bytes()
    if ready_data!={'identity':self.identity,'pre_sha256':digest(pre),'decision_sha256':digest(decraw),'post_sha256':digest(post)}:raise ValueError('post checkpoint identity mismatch')
    checkpoint_validate(post,self.identity['epoch'])
    if post_validate is not None:post_validate(json.loads(req),response,pre,decision,post)
   else:
    post=transition(pre,decision)
    if not isinstance(post,bytes):raise ValueError('transition must return checkpoint bytes')
    checkpoint_validate(post,self.identity['epoch'])
    if post_validate is not None:post_validate(json.loads(req),response,pre,decision,post)
    immutable(postpath,post)
    immutable(ready,canonical({'identity':self.identity,'pre_sha256':digest(pre),'decision_sha256':digest(decraw),'post_sha256':digest(post)}))
   checkpoint_validate(post,self.identity['epoch'])
   commit={'identity':self.identity,'request_sha256':digest(req),'pre_checkpoint_sha256':digest(pre),'response_sha256':digest(raw),'decision_sha256':digest(decraw),'post_checkpoint_sha256':digest(post),'thread_id':completed['thread_id'],'turn_id':completed['turn_id']}
   immutable(self.root/'commit.json',canonical(commit));return commit,post
