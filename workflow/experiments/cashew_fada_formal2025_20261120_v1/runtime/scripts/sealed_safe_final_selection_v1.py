"""Descriptor-relative, locked final selection for new-format checkpoints.

No target point file is a valid input. Each path is fixed or checked against a
fixed same-run location and opened component by component with O_NOFOLLOW.
This is not a sandbox, a provenance attestation or a signature for the ledger.
"""
import errno,fcntl,hashlib,json,os,secrets,stat
from pathlib import Path,PurePosixPath
from decision_bridge import normalize_selection
from native_safe_checkpoint_codec_v1 import decode
from request_reconstruction import verify_request
from sealed_safe_router_sim_v1 import _decision,_post
from sealed_safe_epoch_journal_v1 import _scheduled
from unified_epoch_pool_bridge_sim_v1 import _equal

def sha(raw):return hashlib.sha256(raw).hexdigest()

class SafeRoot:
    def __init__(self,root):
        self.path=Path(os.path.abspath(root));fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
        try:
            for name in self.path.parts[1:]:
                nxt=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        except OSError as exc:
            os.close(fd);raise ValueError('unsafe run root component') from exc
        self.fd=fd
    def close(self):os.close(self.fd)
    def rel(self,path):
        p=PurePosixPath(str(path))
        if p.is_absolute():
            try:p=p.relative_to(PurePosixPath(str(self.path)))
            except ValueError:raise ValueError('artifact outside run')
        if not p.parts or any(x in ('','.','..') for x in p.parts):raise ValueError('invalid artifact path')
        return str(p)
    def _parent(self,path):
        parts=PurePosixPath(self.rel(path)).parts;fd=os.dup(self.fd)
        try:
            for name in parts[:-1]:
                nxt=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
            return fd,parts[-1]
        except OSError as exc:
            os.close(fd);raise ValueError('unsafe artifact parent') from exc
    def read(self,path,*,missing_ok=False):
        try:parent,name=self._parent(path)
        except ValueError as exc:
            if missing_ok and isinstance(exc.__cause__,OSError) and exc.__cause__.errno==errno.ENOENT:return None
            raise
        try:
            try:fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
            except OSError as exc:
                if missing_ok and exc.errno==errno.ENOENT:return None
                raise ValueError('missing or unsafe artifact') from exc
            try:
                before=os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1:raise ValueError('artifact is not a private regular file')
                with os.fdopen(os.dup(fd),'rb') as file:raw=file.read()
                after=os.fstat(fd);linked=os.stat(name,dir_fd=parent,follow_symlinks=False)
                fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
                if any(getattr(before,k)!=getattr(after,k) or getattr(after,k)!=getattr(linked,k) for k in fields):
                    raise ValueError('artifact changed while being read')
                return raw
            finally:os.close(fd)
        finally:os.close(parent)
    def listdir(self,path):
        parent,name=self._parent(path)
        try:
            try:fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
            except OSError as exc:raise ValueError('unsafe journal directory') from exc
            try:return os.listdir(fd)
            finally:os.close(fd)
        finally:os.close(parent)
    def mkdirs(self,path):
        parts=PurePosixPath(self.rel(path)).parts;fd=os.dup(self.fd)
        try:
            for name in parts:
                try:os.mkdir(name,0o700,dir_fd=fd);os.fsync(fd)
                except FileExistsError:pass
                try:nxt=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                except OSError as exc:raise ValueError('unsafe artifact directory') from exc
                os.close(fd);fd=nxt
        finally:os.close(fd)
    def publish_relative(self,path,data):
        rel=self.rel(path);parent,name=self._parent(rel);temp='.'+name+'.'+secrets.token_hex(12)
        try:
            fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent)
            try:
                with os.fdopen(os.dup(fd),'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
                try:os.link(temp,name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
                except FileExistsError:
                    if self.read(rel)!=data:raise RuntimeError('immutable asset differs')
                os.fsync(parent)
            finally:os.close(fd);os.unlink(temp,dir_fd=parent)
        finally:os.close(parent)
    def json(self,path):return json.loads(self.read(path))
    def lock(self,name,*,nonblocking=False):
        try:fd=os.open(name,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK,0o600,dir_fd=self.fd)
        except OSError as exc:raise ValueError('unsafe lock') from exc
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:os.close(fd);raise ValueError('unsafe lock type')
        try:fcntl.flock(fd,fcntl.LOCK_EX|(fcntl.LOCK_NB if nonblocking else 0))
        except BaseException:os.close(fd);raise
        return fd
    def publish(self,name,data):
        temp='.'+name+'.'+secrets.token_hex(12)
        fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=self.fd)
        try:
            with os.fdopen(os.dup(fd),'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
            try:os.link(temp,name,src_dir_fd=self.fd,dst_dir_fd=self.fd,follow_symlinks=False)
            except FileExistsError:
                if self.read(name)!=data:raise RuntimeError('frozen selection differs')
            os.fsync(self.fd)
        finally:
            os.close(fd);os.unlink(temp,dir_fd=self.fd)

def _transaction(fs,directory,job,kind,epoch):
    def raw(name):return fs.read(directory+'/'+name)
    def obj(name):return json.loads(raw(name))
    commit=obj('commit.json');identity={'job_id':job['job_id'],'run_id':job['run_id'],'epoch':epoch,'kind':kind}
    if commit.get('identity')!=identity:raise ValueError('transaction run identity mismatch')
    blobs={name:raw(name) for name in ('request.json','pre_checkpoint.bin','response.json','decision.json','post_checkpoint.bin')}
    for name,key in (('request.json','request_sha256'),('pre_checkpoint.bin','pre_checkpoint_sha256'),
                     ('response.json','response_sha256'),('decision.json','decision_sha256'),('post_checkpoint.bin','post_checkpoint_sha256')):
        if sha(blobs[name])!=commit.get(key):raise ValueError('transaction artifact changed')
    manifest=obj('manifest.json');submission=obj('submission.json');terminal=obj('terminal_identity.json');completed=obj('completed.json')
    ready=obj('post_ready.json');receipt=obj('decision_receipt.json')
    if (manifest.get('identity')!=identity or manifest.get('request_sha256')!=commit['request_sha256']
        or manifest.get('pre_checkpoint_sha256')!=commit['pre_checkpoint_sha256'] or terminal!=completed
        or submission.get('identity')!=identity or completed.get('identity')!=identity
        or submission.get('thread_id')!=commit.get('thread_id') or completed.get('thread_id')!=commit.get('thread_id')
        or completed.get('turn_id')!=commit.get('turn_id') or submission.get('request_sha256')!=commit['request_sha256']
        or completed.get('request_sha256')!=commit['request_sha256'] or completed.get('response_sha256')!=commit['response_sha256']):
        raise ValueError('transaction identity or turn changed')
    if ready!={'identity':identity,'pre_sha256':commit['pre_checkpoint_sha256'],'decision_sha256':commit['decision_sha256'],'post_sha256':commit['post_checkpoint_sha256']}:
        raise ValueError('transaction post receipt changed')
    if receipt!={'identity':identity,'request_sha256':commit['request_sha256'],'pre_checkpoint_sha256':commit['pre_checkpoint_sha256'],
                 'response_sha256':commit['response_sha256'],'decision_sha256':commit['decision_sha256']}:
        raise ValueError('transaction decision receipt changed')
    if decode(blobs['pre_checkpoint.bin'])['epoch']!=epoch or decode(blobs['post_checkpoint.bin'])['epoch']!=epoch:
        raise ValueError('transaction checkpoint epoch changed')
    return commit,json.loads(blobs['request.json']),json.loads(blobs['response.json']),json.loads(blobs['decision.json']),blobs

def _default(fs,d,epoch,pre,phase):
    meta=fs.json(d+'/default_'+phase+'.json');expected,event=_scheduled(pre,epoch)
    post=pre if event is None else fs.read(d+'/default_'+phase+'.bin')
    if (meta!={'epoch':epoch,'phase':phase,'pre_sha256':sha(pre),'post_sha256':sha(post),'event':event}
        or not _equal(decode(expected),decode(post))):raise ValueError('default transition changed')
    return post

def _journal(fs,journal,job,stop_epoch):
    """Replay same-run checkpoint/decision chain without pathname opens."""
    if not journal._lease or journal._poisoned:raise RuntimeError('active healthy journal lease required')
    base=fs.rel(journal.root)
    if fs.json(base+'/identity.json')!=journal.identity:raise ValueError('journal identity changed')
    actual=sorted(name for name in fs.listdir(base) if name.startswith('epoch_'))
    if actual!=['epoch_%06d'%e for e in range(1,stop_epoch+1)]:raise ValueError('unexpected epochs around STOP')
    initial=fs.read(base+'/initial_checkpoint.bin')
    if fs.json(base+'/initial.json')!={'epoch':0,'checkpoint_sha256':sha(initial)} or decode(initial)['epoch']!=0:
        raise ValueError('initial checkpoint changed')
    previous=sha(initial);last=None
    for epoch in range(1,stop_epoch+1):
        d=base+'/epoch_%06d'%epoch;intent=fs.json(d+'/intent.json')
        if intent!={'epoch':epoch,'previous_checkpoint_sha256':previous}:raise ValueError('journal chain changed')
        trained=fs.read(d+'/trained_checkpoint.bin');meta=fs.json(d+'/trained.json');entry=fs.json(d+'/commit.json')
        if (meta.get('epoch')!=epoch or meta.get('checkpoint_sha256')!=sha(trained) or decode(trained)['epoch']!=epoch):
            raise ValueError('trained checkpoint changed')
        post=fs.read(d+'/committed_checkpoint.bin')
        if decode(post)['epoch']!=epoch:raise ValueError('committed checkpoint epoch changed')
        route=meta.get('route');action=entry.get('action')
        if (epoch%5==0 or epoch==journal.max_epoch) and route!='FULL_REVIEW':raise ValueError('required full review missing')
        expected={'epoch':epoch,'route':route,'previous_checkpoint_sha256':previous,'trained_checkpoint_sha256':sha(trained),
                  'post_checkpoint_sha256':sha(post),'decision_kind':'machine' if route=='SKIP' else 'agent','action':action}
        if entry!=expected:raise ValueError('journal commit changed')
        if route=='SKIP':
            if action!='KEEP' or post!=_default(fs,d,epoch,trained,'skip'):raise ValueError('skip changed')
        elif route in ('LIGHT_REVIEW','FULL_REVIEW'):
            kind='light_review' if route=='LIGHT_REVIEW' else 'full_review'
            tx='agent/sealed_v1/epoch_%03d/%s/transaction'%(epoch,'light' if route=='LIGHT_REVIEW' else 'full')
            receipt,q,a,decision,blobs=_transaction(fs,tx,job,kind,epoch)
            if decision!=_decision(job,q,a,blobs['pre_checkpoint.bin'],allow_stop=True) or not _equal(decode(blobs['post_checkpoint.bin']),decode(_post(blobs['pre_checkpoint.bin'],decision))):
                raise ValueError('journal Agent response or transition changed')
            pre=trained if route=='LIGHT_REVIEW' else _default(fs,d,epoch,trained,'before_agent')
            candidate=blobs['post_checkpoint.bin'] if route=='FULL_REVIEW' else _default(fs,d,epoch,blobs['post_checkpoint.bin'],'after_agent')
            if blobs['pre_checkpoint.bin']!=pre or candidate!=post or decision.get('action')!=action:
                raise ValueError('journal Agent checkpoint changed')
            if action=='STOP' and epoch!=stop_epoch:raise ValueError('later epochs after STOP')
        else:raise ValueError('unsafe journal route')
        previous=sha(post);last=(route,action,post)
    if not last or last[0]!='FULL_REVIEW' or (last[1]!='STOP' and stop_epoch!=journal.max_epoch):raise ValueError('STOP not committed')
    return last[2]

def commit(root,transaction_dir,manifest_sha256,*,job,journal):
    fs=SafeRoot(root);locks=[]
    try:
        locks.append(fs.lock('stop_freeze.lock'));locks.append(fs.lock('selection_freeze.lock'))
        manifest_raw=fs.read('frozen_candidates.json')
        if sha(manifest_raw)!=manifest_sha256:raise ValueError('candidate manifest changed')
        manifest=json.loads(manifest_raw);epoch=manifest.get('stop_epoch')
        if (type(epoch) is not int or not 1<=epoch<=journal.max_epoch or manifest.get('job_id')!=job['job_id'] or manifest.get('run_id')!=job['run_id']
            or manifest.get('stop_receipt')!='stop_receipt.json'):raise ValueError('manifest run or STOP mismatch')
        stop_raw=fs.read('stop_receipt.json')
        if sha(stop_raw)!=manifest.get('stop_receipt_sha256'):raise ValueError('stop receipt changed')
        stop=json.loads(stop_raw)
        stop_dir='agent/sealed_v1/epoch_%03d/full/transaction'%epoch
        if (stop.get('job_id')!=job['job_id'] or stop.get('run_id')!=job['run_id'] or stop.get('epoch')!=epoch
            or stop.get('state')!='stopped' or fs.rel(stop.get('transaction_path',''))!=stop_dir
            or fs.rel(stop.get('decision_path',''))!=stop_dir+'/decision.json'):
            raise ValueError('STOP transaction is not owned by this run')
        stopped=_journal(fs,journal,job,epoch)
        stop_commit,q,a,stop_decision,stop_blobs=_transaction(fs,stop_dir,job,'full_review',epoch)
        if (sha(fs.read(stop_dir+'/commit.json'))!=stop.get('transaction_commit_sha256') or stop_commit['decision_sha256']!=stop.get('decision_sha256')
            or stop_commit['post_checkpoint_sha256']!=stop.get('post_checkpoint_sha256') or stop_blobs['post_checkpoint.bin']!=stopped):
            raise ValueError('STOP receipt differs from committed journal')
        reason=stop.get('reason')
        if not ((reason=='agent_stop' and stop_decision.get('action')=='STOP' and stop_decision.get('continue_training') is False)
                or (reason=='max_epoch' and epoch==journal.max_epoch and stop_decision.get('continue_training') is True)):
            raise ValueError('STOP semantics differ')
        select_dir='agent/sealed_v1/selection/transaction'
        if fs.rel(transaction_dir)!=select_dir:raise ValueError('selection transaction is not owned by this run')
        selection,request,answer,decision,blobs=_transaction(fs,select_dir,job,'checkpoint_selection',epoch)
        if blobs['pre_checkpoint.bin']!=stopped or blobs['post_checkpoint.bin']!=stopped:raise ValueError('selector changed STOP checkpoint')
        candidates=manifest.get('candidates');observations=request.get('candidate_observations')
        if not isinstance(candidates,list) or not candidates or not isinstance(observations,list):raise ValueError('missing candidate records')
        if [x.get('epoch') for x in observations]!=[x.get('epoch') for x in candidates]:raise ValueError('candidate identities differ')
        trust={('candidate',x['epoch'],x['sha256']) for x in observations}
        expected={'stop_epoch':epoch}
        if request.get('capabilities',{}).get('history_query_required') is True:expected['history_required']=True
        from checkpoint_selection_contract_v2 import enabled as complete_selection,VERSION
        complete=complete_selection(job)
        if request.get('selection_policy')!=(VERSION if complete else None):raise ValueError('final selection policy mismatch')
        if complete:
            details=[{'kind':'selection_candidate',**x} for x in request.get('candidate_details',[])]
            expected.update(selection_policy=VERSION,candidate_details=details)
            trust.update(('selection_candidate',x['epoch'],x['sha256']) for x in details)
        verify_request(request,trust,expected=expected,previous=None)
        payload=decode(stopped)
        if decision!=normalize_selection(request,answer,payload['config'],job,frozen_stop_epoch=epoch):
            raise ValueError('selection raw response and normalized decision differ')
        selected=decision.get('selected_epoch');matches=[];base=fs.rel(journal.root)
        verified=None
        if complete:
            from checkpoint_selection_evidence_v2 import verify_frozen_candidates
            verified=verify_frozen_candidates(job,journal,request,manifest,verify_models=True,selected_epoch=selected)
            matches=[verified['candidate']]
        for candidate,observation in ([] if complete else zip(candidates,observations)):
            e=candidate.get('epoch')
            if type(e) is not int or not 1<=e<=epoch:raise ValueError('invalid candidate epoch')
            d=base+'/epoch_%06d'%e;expected_path=d+'/trained_checkpoint.bin'
            if fs.rel(candidate.get('path',''))!=expected_path:raise ValueError('candidate is not same-run checkpoint')
            raw=fs.read(expected_path);cp=decode(raw)
            if sha(raw)!=candidate.get('sha256') or cp['epoch']!=e:raise ValueError('frozen candidate changed')
            values=observation.get('values')
            if not isinstance(values,dict):raise ValueError('missing candidate observations')
            obs={}
            if any(k in values for k in ('source_class2_f1','target_unlabelled_class2_fraction')):
                obs=fs.json(d+'/review_observation.json');trained=fs.json(d+'/trained.json');entry=fs.json(d+'/commit.json')
                tx='agent/sealed_v1/epoch_%03d/full/transaction'%e
                _,original,_,_,_=_transaction(fs,tx,job,'full_review',e);evidence=original.get('evidence') or {}
                if (trained.get('route')!='FULL_REVIEW' or entry.get('decision_kind')!='agent' or obs.get('epoch')!=e
                    or obs.get('checkpoint_sha256')!=candidate['sha256']
                    or (evidence.get('source_validation') or {}).get('values')!=obs.get('source')
                    or (evidence.get('unlabelled_panel') or {}).get('values')!=obs.get('panel')):
                    raise ValueError('candidate review is not checkpoint-bound')
            available={'training_cls_loss':cp['metrics'].get('cls'),'source_class2_f1':(obs.get('source') or {}).get('class2_f1'),
                       'target_unlabelled_class2_fraction':(obs.get('panel') or {}).get('predicted_class2_fraction')}
            if any(k not in available or available[k] is None or available[k]!=v for k,v in values.items()):raise ValueError('candidate facts differ')
            if e==selected:matches.append(candidate)
        if decision.get('action')!='SELECT_CHECKPOINT' or type(selected) is not int or len(matches)!=1:raise ValueError('invalid frozen selection')
        chosen=matches[0]
        record={'version':'selection_freeze_v1','job_id':job['job_id'],'run_id':job['run_id'],'stop_epoch':epoch,
                'candidate_manifest_sha256':manifest_sha256,'selected_epoch':selected,'checkpoint':chosen['path'],
                'checkpoint_sha256':chosen['sha256'],'selection_decision_sha256':selection['decision_sha256']}
        if complete:
            record.update(selection_policy=VERSION,predictor=verified['predictor'],selected_evidence_sha256=verified['selected_evidence_sha256'],selection_comparison=decision['selection_comparison'])
        fs.publish('frozen_selection.json',(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
        return record
    finally:
        for fd in reversed(locks):os.close(fd)
        fs.close()
