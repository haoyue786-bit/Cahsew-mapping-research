"""Experimental committed STOP/selection bridge; no target-point evaluation.

 A transaction receipt binds bytes and identities, not the provenance or
 isolation of the training trajectory. The caller still owns authenticating
 the job, stop policy and same-run checkpoint eligibility.
"""
import fcntl,json
from pathlib import Path
from selection_freeze import hash_file,path_under,publish,freeze_candidates,commit_selection,validate_checkpoint

def verify_committed_transaction(root,transaction_dir,job_id,run_id,checkpoint_loader):
 root=Path(root).resolve();directory=(root/transaction_dir).resolve() if not Path(transaction_dir).is_absolute() else Path(transaction_dir).resolve()
 try:directory.relative_to(root)
 except ValueError:raise ValueError('transaction outside run')
 if not directory.is_dir():raise ValueError('missing transaction')
 commit_path=path_under(root,directory/'commit.json');commit=json.loads(commit_path.read_text());identity=commit.get('identity',{})
 if identity.get('job_id')!=job_id or identity.get('run_id')!=run_id or type(identity.get('epoch')) is not int or identity['epoch']<1:raise ValueError('transaction identity mismatch')
 for name,key in [('request.json','request_sha256'),('pre_checkpoint.bin','pre_checkpoint_sha256'),('response.json','response_sha256'),('decision.json','decision_sha256'),('post_checkpoint.bin','post_checkpoint_sha256')]:
  if hash_file(path_under(root,directory/name))!=commit.get(key):raise ValueError('transaction artifact mismatch: '+name)
 submission=json.loads(path_under(root,directory/'submission.json').read_text());completed=json.loads(path_under(root,directory/'completed.json').read_text());terminal=json.loads(path_under(root,directory/'terminal_identity.json').read_text());ready=json.loads(path_under(root,directory/'post_ready.json').read_text());decision_receipt=json.loads(path_under(root,directory/'decision_receipt.json').read_text())
 if completed!=terminal or submission.get('identity')!=identity or completed.get('identity')!=identity or submission.get('thread_id')!=commit.get('thread_id') or completed.get('thread_id')!=commit.get('thread_id') or completed.get('turn_id')!=commit.get('turn_id') or submission.get('request_sha256')!=commit['request_sha256'] or completed.get('request_sha256')!=commit['request_sha256'] or completed.get('response_sha256')!=commit['response_sha256']:raise ValueError('turn identity mismatch')
 if ready!={'identity':identity,'pre_sha256':commit['pre_checkpoint_sha256'],'decision_sha256':commit['decision_sha256'],'post_sha256':commit['post_checkpoint_sha256']} or decision_receipt!={'identity':identity,'request_sha256':commit['request_sha256'],'pre_checkpoint_sha256':commit['pre_checkpoint_sha256'],'response_sha256':commit['response_sha256'],'decision_sha256':commit['decision_sha256']}:raise ValueError('transaction receipt mismatch')
 validate_checkpoint(directory/'pre_checkpoint.bin',identity['epoch'],checkpoint_loader);validate_checkpoint(directory/'post_checkpoint.bin',identity['epoch'],checkpoint_loader)
 decision=json.loads((directory/'decision.json').read_text())
 if decision.get('epoch')!=identity['epoch']:raise ValueError('stop epoch mismatch')
 return directory,commit_path,commit,decision

def publish_stop_from_transaction(run_root,transaction_dir,job_id,run_id,*,max_epoch,checkpoint_loader):
 root=Path(run_root).resolve()
 with (root/'stop_freeze.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);directory,commit_path,commit,decision=verify_committed_transaction(root,transaction_dir,job_id,run_id,checkpoint_loader);epoch=commit['identity']['epoch']
  if commit['identity'].get('kind')!='full_review':raise ValueError('stop must originate in full review')
  if decision.get('action')=='STOP' and decision.get('continue_training') is False:reason='agent_stop'
  elif type(max_epoch) is int and epoch==max_epoch and decision.get('action') not in ('STOP','REQUEST_EXPERIMENT') and decision.get('continue_training') is True:reason='max_epoch'
  else:raise ValueError('transaction does not establish a valid stop')
  record={'job_id':job_id,'run_id':run_id,'state':'stopped','epoch':epoch,'reason':reason,'decision_path':str(directory/'decision.json'),'decision_sha256':commit['decision_sha256'],
   'transaction_path':str(directory.relative_to(root)),'transaction_commit_sha256':hash_file(commit_path),'post_checkpoint_sha256':commit['post_checkpoint_sha256']}
  publish(root/'stop_receipt.json',record);return record

def freeze_from_transaction(run_root,job_id,run_id,candidates,*,max_epoch,checkpoint_loader,allow_epoch_zero=False):
 root=Path(run_root).resolve();stop=json.loads(path_under(root,'stop_receipt.json').read_text())
 directory,commit_path,commit,decision=verify_committed_transaction(root,stop['transaction_path'],job_id,run_id,checkpoint_loader)
 if hash_file(commit_path)!=stop.get('transaction_commit_sha256') or commit['post_checkpoint_sha256']!=stop.get('post_checkpoint_sha256') or commit['identity']['epoch']!=stop.get('epoch') or commit['decision_sha256']!=stop.get('decision_sha256'):raise ValueError('stop transaction changed')
 return freeze_candidates(root,job_id,run_id,root/'stop_receipt.json',candidates,max_epoch=max_epoch,checkpoint_loader=checkpoint_loader,allow_epoch_zero=allow_epoch_zero)

def commit_frozen_selection(run_root,selected_epoch,decision_path,expected_manifest_sha256,*,checkpoint_loader):
 root=Path(run_root).resolve();manifest=json.loads(path_under(root,'frozen_candidates.json').read_text());stop=json.loads(path_under(root,manifest['stop_receipt']).read_text())
 _,commit_path,commit,_=verify_committed_transaction(root,stop['transaction_path'],manifest['job_id'],manifest['run_id'],checkpoint_loader)
 if hash_file(commit_path)!=stop.get('transaction_commit_sha256') or commit['post_checkpoint_sha256']!=stop.get('post_checkpoint_sha256'):raise ValueError('stop transaction changed')
 return commit_selection(root,selected_epoch,decision_path,expected_manifest_sha256,checkpoint_loader=checkpoint_loader)

def commit_from_selection_transaction(run_root,transaction_dir,expected_manifest_sha256,*,checkpoint_loader):
 """Bind an already committed selector turn to frozen STOP and candidates."""
 root=Path(run_root).resolve();manifest=json.loads(path_under(root,'frozen_candidates.json').read_text());stop=json.loads(path_under(root,manifest['stop_receipt']).read_text())
 directory,_,commit,decision=verify_committed_transaction(root,transaction_dir,manifest['job_id'],manifest['run_id'],checkpoint_loader)
 if (commit['identity'].get('kind')!='checkpoint_selection' or commit['identity']['epoch']!=stop['epoch'] or
     commit['pre_checkpoint_sha256']!=stop['post_checkpoint_sha256'] or commit['post_checkpoint_sha256']!=commit['pre_checkpoint_sha256']):
  raise ValueError('selection transaction does not match frozen stop state')
 selected=decision.get('selected_epoch')
 if decision.get('action')!='SELECT_CHECKPOINT' or type(selected) is not int:raise ValueError('invalid selection decision')
 return commit_frozen_selection(root,selected,directory/'decision.json',expected_manifest_sha256,checkpoint_loader=checkpoint_loader)
