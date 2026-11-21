"""Resume an owned journal from its latest usable checkpoint.

The small journal/transaction receipts are checked in sequence.  Older large
checkpoint contents are not rehashed: this relies on the existing journal and
worker-produced history, and is not a signed or retrospective full audit.
"""
import json
import re
from decision_transaction import digest

_SHA = re.compile(r'^[0-9a-f]{64}$')

def _hash(value):
    if not isinstance(value,str) or not _SHA.fullmatch(value):
        raise ValueError('invalid recorded checkpoint hash')
    return value

def _small(fs,path):
    raw=fs.read(path)
    if len(raw)>5_000_000:raise ValueError('oversized recovery metadata')
    return json.loads(raw)

def _transaction(j,path,epoch,kind):
    fs=j._out_fs
    def obj(name):return _small(fs,path+'/'+name)
    commit=obj('commit.json')
    identity={'job_id':j.job['job_id'],'run_id':j.job['run_id'],'epoch':epoch,'kind':kind}
    if commit.get('identity')!=identity:raise ValueError('recovery transaction identity mismatch')
    values={}
    for name,key in (('request','request_sha256'),('response','response_sha256'),('decision','decision_sha256')):
        raw=fs.read(path+'/'+name+'.json')
        if len(raw)>5_000_000 or digest(raw)!=_hash(commit.get(key)):raise ValueError('recovery transaction changed')
        values[name]=json.loads(raw)
    manifest=obj('manifest.json');submission=obj('submission.json');terminal=obj('terminal_identity.json');completed=obj('completed.json')
    ready=obj('post_ready.json');receipt=obj('decision_receipt.json')
    pre=_hash(commit.get('pre_checkpoint_sha256'));post=_hash(commit.get('post_checkpoint_sha256'))
    if (manifest.get('identity')!=identity or manifest.get('request_sha256')!=commit['request_sha256']
        or manifest.get('pre_checkpoint_sha256')!=pre or terminal!=completed
        or submission.get('identity')!=identity or completed.get('identity')!=identity
        or submission.get('thread_id')!=commit.get('thread_id') or completed.get('thread_id')!=commit.get('thread_id')
        or completed.get('turn_id')!=commit.get('turn_id') or submission.get('request_sha256')!=commit['request_sha256']
        or completed.get('request_sha256')!=commit['request_sha256'] or completed.get('response_sha256')!=commit['response_sha256']):
        raise ValueError('recovery transaction lineage mismatch')
    if ready!={'identity':identity,'pre_sha256':pre,'decision_sha256':commit['decision_sha256'],'post_sha256':post}:
        raise ValueError('recovery transaction post mismatch')
    if receipt!={'identity':identity,'request_sha256':commit['request_sha256'],'pre_checkpoint_sha256':pre,
                 'response_sha256':commit['response_sha256'],'decision_sha256':commit['decision_sha256']}:
        raise ValueError('recovery transaction decision mismatch')
    if values['request'].get('capabilities',{}).get('target_point_evidence') is not False:
        raise ValueError('recovery transaction lacks target-point isolation')
    if values['response'].get('action')!=values['decision'].get('action'):
        raise ValueError('recovery transaction action mismatch')
    return commit,values

def _default(j,prefix,epoch,phase,pre):
    meta=_small(j._fs,prefix+'default_'+phase+'.json')
    if (set(meta)!={'epoch','phase','pre_sha256','post_sha256','event'} or meta['epoch']!=epoch
        or meta['phase']!=phase or meta['pre_sha256']!=pre):raise ValueError('recovery default mismatch')
    post=_hash(meta['post_sha256']);event=meta['event']
    if event is None and post!=pre:raise ValueError('silent recovery default transition')
    if event is not None:
        if not isinstance(event,dict) or event.get('epoch')!=epoch or event.get('state') not in {'activated','no_samples'}:
            raise ValueError('unknown recovery default transition')
        # Verify existence/type without loading an old checkpoint.
        parent,name=j._fs._parent(prefix+'default_'+phase+'.bin')
        try:
            import os,stat
            st=os.stat(name,dir_fd=parent,follow_symlinks=False)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1:raise ValueError('unsafe default checkpoint')
        finally:os.close(parent)
    return post

def verify_initial(j,baseline=None):
    j._held()
    initial=_small(j._fs,'initial.json')
    if set(initial)!={'epoch','checkpoint_sha256'} or initial['epoch']!=0:raise ValueError('invalid recovery initial record')
    txn,values=_transaction(j,'agent/sealed_v1/initialization/transaction',0,'initialization')
    if txn['post_checkpoint_sha256']!=_hash(initial['checkpoint_sha256']):raise ValueError('recovery initialization mismatch')
    transfer=j.job.get('transfer_source')
    if transfer is not None:
        x=values['request'].get('evidence',{}).get('transfer_identity',{})
        expected={'source_job_id':transfer['source_job_id'],'source_run_id':transfer['source_run_id'],
                  'source_epoch':transfer['selected_epoch'],'source_checkpoint_sha256':transfer['checkpoint_sha256'],
                  'selection_decision_sha256':transfer['selection_decision_sha256'],
                  'target_job_id':j.job['job_id'],'target_run_id':j.job['run_id']}
        # The baseline and transfer record intentionally use different
        # model-state digest algorithms; compare the checkpoint identity,
        # not those two non-interchangeable model digests.
        _hash(x.get('model_state_sha256'))
        if baseline is not None and baseline.get('transfer_checkpoint_sha256')!=transfer['checkpoint_sha256']:
            raise ValueError('recovery baseline transfer mismatch')
        if any(x.get(k)!=v for k,v in expected.items()):raise ValueError('recovery transfer identity mismatch')
    return initial

def bootstrap(j):
    """Verify metadata chain and load exactly the last required model state."""
    initial=verify_initial(j);previous=initial['checkpoint_sha256'];last_epoch=0;action=None;snapshots=[]
    dirs=sorted(n for n in __import__('os').listdir(j._fs.fd) if n.startswith('epoch_'))
    pending=None
    for dirname in dirs:
        epoch=last_epoch+1;prefix=dirname+'/'
        if dirname!='epoch_%06d'%epoch or epoch>j.max_epoch:raise ValueError('recovery epoch gap')
        if _small(j._fs,prefix+'intent.json')!={'epoch':epoch,'previous_checkpoint_sha256':previous}:
            raise ValueError('recovery epoch chain mismatch')
        trained_bytes=j._fs.read(prefix+'trained.json',missing_ok=True)
        commit_bytes=j._fs.read(prefix+'commit.json',missing_ok=True)
        if trained_bytes is None or commit_bytes is None:
            if dirname!=dirs[-1]:raise ValueError('unfinished recovery epoch has successors')
            pending=(epoch,trained_bytes is not None)
            if trained_bytes is None and commit_bytes is not None:raise ValueError('commit without trained metadata')
            break
        trained=json.loads(trained_bytes);commit=json.loads(commit_bytes);route=trained.get('route')
        th=_hash(trained.get('checkpoint_sha256'));post=_hash(commit.get('post_checkpoint_sha256'))
        if (set(trained)!={'epoch','checkpoint_sha256','route'} or trained['epoch']!=epoch
            or route not in {'SKIP','LIGHT_REVIEW','FULL_REVIEW'} or
            ((epoch%5==0 or epoch==j.max_epoch) and route!='FULL_REVIEW')):
            raise ValueError('invalid recovery trained metadata')
        action=commit.get('action');kind='machine' if route=='SKIP' else 'agent'
        if commit!={'epoch':epoch,'route':route,'previous_checkpoint_sha256':previous,'trained_checkpoint_sha256':th,
                    'post_checkpoint_sha256':post,'decision_kind':kind,'action':action}:
            raise ValueError('invalid recovery commit metadata')
        if route=='SKIP':
            if action!='KEEP' or _default(j,prefix,epoch,'skip',th)!=post:raise ValueError('invalid recovery skip')
        else:
            typ='full' if route=='FULL_REVIEW' else 'light';tk='full_review' if typ=='full' else 'light_review'
            tx,values=_transaction(j,'agent/sealed_v1/epoch_%03d/%s/transaction'%(epoch,typ),epoch,tk)
            pre=th if typ=='light' else _default(j,prefix,epoch,'before_agent',th)
            result=tx['post_checkpoint_sha256'] if typ=='full' else _default(j,prefix,epoch,'after_agent',tx['post_checkpoint_sha256'])
            if tx['pre_checkpoint_sha256']!=pre or result!=post or values['decision'].get('action')!=action:
                raise ValueError('invalid recovery agent chain')
        if action=='STOP' and dirname!=dirs[-1]:raise ValueError('epochs follow recovery STOP')
        snapshots.append((epoch,route,j._epoch_snapshot(epoch,route)))
        previous=post;last_epoch=epoch
    if pending and pending[1]:
        epoch=pending[0];raw=j._read_pending('epoch_%06d/'%epoch);meta=_small(j._fs,'epoch_%06d/trained.json'%epoch)
        if meta.get('epoch')!=epoch or meta.get('checkpoint_sha256')!=digest(raw):raise ValueError('latest trained checkpoint changed')
        j.check(raw,epoch);head=None
    else:
        path='initial_checkpoint.bin' if last_epoch==0 else 'epoch_%06d/committed_checkpoint.bin'%last_epoch
        head=j._fs.read(path)
        if digest(head)!=previous:raise ValueError('latest committed checkpoint changed')
        j.check(head,last_epoch)
    return {'initial':j._snapshot_initial(),'snapshots':snapshots,'epoch':last_epoch,'checkpoint':head,'hash':previous,'action':action}
