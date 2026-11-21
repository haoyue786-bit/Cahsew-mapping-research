"""Candidate checkpoint freeze, not yet integrated into the worker.

 This validates a worker-owned stop receipt, then freezes candidate identities
 before selection. It never reads target-point evaluation data. Hashes do not
 prove that an old trajectory was blind: the caller must authenticate job,
 receipt and run lineage before use.
"""
import fcntl,hashlib,json,os,tempfile
from pathlib import Path

def hash_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def canonical(x):return (json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def path_under(root,p):
 p=(root/p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
 try:p.relative_to(root)
 except ValueError:raise ValueError('artifact outside run')
 if not p.is_file():raise ValueError('artifact missing')
 return p
def publish(path,payload):
 data=canonical(payload)
 fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=str(path.parent))
 try:
  with os.fdopen(fd,'wb',closefd=False) as f:
   f.write(data);f.flush();os.fsync(fd)
  try:os.link(tmp,path,follow_symlinks=False)
  except FileExistsError:
   if path.is_symlink() or path.read_bytes()!=data:raise RuntimeError('frozen artifact already exists with different content')
 finally:
  os.close(fd)
  os.unlink(tmp)
 dirfd=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(dirfd)
 finally:os.close(dirfd)
def validate_checkpoint(path,epoch,loader):
 if not callable(loader):raise ValueError('checkpoint loader is required')
 payload=loader(path)
 required={'epoch','state_dict','local_discriminator','global_discriminator','model_optimizer','local_optimizer','global_optimizer','config','rng_state'}
 if not isinstance(payload,dict) or not required.issubset(payload) or type(payload['epoch']) is not int or payload['epoch']!=epoch:raise ValueError('checkpoint identity or state is incomplete')

def freeze_candidates(run_root,job_id,run_id,stop_receipt_path,candidates,*,max_epoch=None,checkpoint_loader=None,allow_epoch_zero=False):
 root=Path(run_root).resolve();lockpath=root/'selection_freeze.lock'
 if not root.is_dir() or not isinstance(job_id,str) or not isinstance(run_id,str):raise ValueError('invalid run identity')
 with lockpath.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);receipt_path=path_under(root,stop_receipt_path);receipt=json.loads(receipt_path.read_text())
  if receipt.get('job_id')!=job_id or receipt.get('run_id')!=run_id or receipt.get('state')!='stopped':raise ValueError('unverified stopped run identity')
  stop=receipt.get('epoch')
  if type(stop) is not int or stop<1 or receipt.get('reason') not in ('agent_stop','max_epoch'):raise ValueError('invalid stopping record')
  if receipt['reason']=='max_epoch' and (type(max_epoch) is not int or stop!=max_epoch):raise ValueError('maximum epoch stop lacks verified job cap')
  decision=path_under(root,receipt['decision_path'])
  if hash_file(decision)!=receipt.get('decision_sha256'):raise ValueError('stop decision hash mismatch')
  d=json.loads(decision.read_text())
  if (d.get('epoch')!=stop or
      (receipt['reason']=='agent_stop' and (d.get('action')!='STOP' or d.get('continue_training') is not False)) or
      (receipt['reason']=='max_epoch' and (d.get('action') not in ('KEEP','ADJUST','CONTINUE','PAUSE_ADV','RESUME_ADV') or d.get('continue_training') is not True))):
   raise ValueError('stop decision semantics mismatch')
  if not isinstance(candidates,list) or not candidates:raise ValueError('no candidate checkpoints')
  out=[]
  for c in candidates:
   if not isinstance(c,dict) or set(c)!={'epoch','path'} or type(c['epoch']) is not int or not (0 if allow_epoch_zero else 1)<=c['epoch']<=stop:raise ValueError('invalid candidate')
   if c['epoch']==0 and path_under(root,c['path'])!=root/'journal/initial_checkpoint.bin':raise ValueError('E0 must be committed run initialization')
   p=path_under(root,c['path']);before=hash_file(p);validate_checkpoint(p,c['epoch'],checkpoint_loader)
   if hash_file(p)!=before:raise ValueError('checkpoint changed during validation')
   out.append({'epoch':c['epoch'],'path':str(p.relative_to(root)),'sha256':before})
  epochs=[x['epoch'] for x in out]
  if epochs!=sorted(set(epochs)):raise ValueError('duplicate or unordered candidates')
  manifest={'version':'candidate_freeze_v1','job_id':job_id,'run_id':run_id,'stop_epoch':stop,'stop_receipt':str(receipt_path.relative_to(root)),'stop_receipt_sha256':hash_file(receipt_path),'candidates':out}
  if allow_epoch_zero:
   from checkpoint_selection_contract_v2 import VERSION
   if epochs[0]!=0:raise ValueError('missing E0 candidate')
   manifest['selection_policy']=VERSION
  publish(root/'frozen_candidates.json',manifest);return manifest

def commit_selection(run_root,selected_epoch,decision_path,expected_manifest_sha256,*,checkpoint_loader=None):
 root=Path(run_root).resolve();lockpath=root/'selection_freeze.lock'
 with lockpath.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);manifest_path=path_under(root,'frozen_candidates.json')
  if hash_file(manifest_path)!=expected_manifest_sha256:raise ValueError('candidate manifest changed')
  manifest=json.loads(manifest_path.read_text());matches=[x for x in manifest['candidates'] if x['epoch']==selected_epoch]
  stop_receipt=path_under(root,manifest['stop_receipt'])
  if hash_file(stop_receipt)!=manifest['stop_receipt_sha256']:raise ValueError('frozen stop receipt changed')
  stop_info=json.loads(stop_receipt.read_text());stop_decision=path_under(root,stop_info['decision_path'])
  if hash_file(stop_decision)!=stop_info.get('decision_sha256'):raise ValueError('frozen stop decision changed')
  if type(selected_epoch) is not int or len(matches)!=1:raise ValueError('selected epoch not eligible')
  chosen=matches[0];checkpoint=path_under(root,chosen['path'])
  if hash_file(checkpoint)!=chosen['sha256']:raise ValueError('candidate checkpoint changed')
  validate_checkpoint(checkpoint,selected_epoch,checkpoint_loader)
  if hash_file(checkpoint)!=chosen['sha256']:raise ValueError('checkpoint changed during validation')
  decision=path_under(root,decision_path);answer=json.loads(decision.read_text())
  if type(answer.get('selected_epoch')) is not int or answer.get('selected_epoch')!=selected_epoch or answer.get('action')!='SELECT_CHECKPOINT':raise ValueError('selection decision mismatch')
  record={'version':'selection_freeze_v1','job_id':manifest['job_id'],'run_id':manifest['run_id'],'stop_epoch':manifest['stop_epoch'],
   'candidate_manifest_sha256':expected_manifest_sha256,'selected_epoch':selected_epoch,'checkpoint':chosen['path'],'checkpoint_sha256':chosen['sha256'],
   'selection_decision_sha256':hash_file(decision)}
  publish(root/'frozen_selection.json',record);return record
