"""Simulation-only single-authority bridge for epoch and automatic pool.

The sealed EpochJournal is the sole advancing journal. Pool generation, cursor
and retired set are fields of its checkpoint; assets/receipts are immutable and
have no independent head. Legacy pool head paths must be forbidden. Every worker
operation must use this bridge under the EpochJournal lease; direct legacy calls
cannot be prevented by a Python wrapper and are rejected on the next audit.

Receipt hashes detect mismatches, not a dishonest producer. Trusted independent
checks of input isolation, live optimizer updates, complete model/optimizer/RNG
snapshot and true contiguous exposure remain required before any real use.
"""
import copy
import errno
import fcntl
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from decision_transaction import canonical, digest, immutable, read_json, sync_dir
from automatic_pool_transaction_v1 import _state

_NAME=re.compile(r'^[A-Za-z0-9_.-]{1,96}$')

def _secure_directory(path):
    """Create/open each component without following a symlink."""
    path=Path(os.path.abspath(path));fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            try:os.mkdir(part,0o700,dir_fd=fd);os.fsync(fd)
            except FileExistsError:pass
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=nxt
        return fd
    except BaseException:
        os.close(fd);raise


def _equal(a,b):
    import torch
    import numpy as np
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b,equal_nan=True)
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and set(a)==set(b) and all(_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b

class UnifiedEpochPool:
    def __init__(self,journal,asset_root,*,decode,encode,attest,forbidden_heads):
        if not all(callable(x) for x in (decode,encode,attest)) or not isinstance(forbidden_heads,(tuple,list)) or not forbidden_heads:
            raise ValueError('callbacks and legacy-head exclusions required')
        self.journal=journal;self.assets=Path(asset_root);self.decode=decode;self.encode=encode
        self.attest=attest;self.forbidden_heads=tuple(Path(x) for x in forbidden_heads)
        self._active=False;self._legacy_fds={}

    @contextmanager
    def lease(self):
        if self._active:raise RuntimeError('bridge already leased')
        fds=[]
        with self.journal.lease():
            try:
                # Lock the same file used by each retired PoolTransaction for
                # the entire worker lease, not just while checking its head.
                # This coordinates cooperating workers; uncooperative writes
                # outside these locks still require OS-level isolation.
                for root in sorted({p.parent for p in self.forbidden_heads},key=str):
                    dirfd=_secure_directory(root);fds.append(dirfd);self._legacy_fds[root]=dirfd
                    try:fd=os.open('.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK,0o600,dir_fd=dirfd)
                    except OSError as exc:raise ValueError('unsafe legacy lock') from exc
                    fds.append(fd);info=os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:raise ValueError('unsafe legacy lock type')
                    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError as exc:raise RuntimeError('legacy pool worker holds lock') from exc
                if getattr(self.journal,'_active_pool_bridge',None) is not None:
                    raise RuntimeError('journal already has an active pool bridge')
                self._active=True;self.journal._active_pool_bridge=self;self._guard();yield self
            finally:
                if getattr(self.journal,'_active_pool_bridge',None) is self:
                    del self.journal._active_pool_bridge
                self._active=False
                for fd in fds:os.close(fd)
                self._legacy_fds.clear()

    def _guard(self):
        self.journal._held()
        if not self._active:raise RuntimeError('unified worker lease required')
        for p in self.forbidden_heads:
            if p.name in ('','.','..') or p.parent not in self._legacy_fds:raise ValueError('unsafe legacy head path')
            try:os.stat(p.name,dir_fd=self._legacy_fds[p.parent],follow_symlinks=False)
            except FileNotFoundError:continue
            raise RuntimeError('legacy independent pool head detected')

    def _cp_state(self,raw):
        obj=self.decode(raw)
        if not isinstance(obj,dict): raise ValueError('invalid checkpoint payload')
        return _state({k:obj.get(k) for k in ('epoch','pool','retired')})

    def _assets(self,state):
        pool=state['pool']
        if pool is None:return
        import json
        from sealed_safe_final_selection_v1 import SafeRoot
        root=SafeRoot(self.assets)
        try:
            prefix=pool['generation'];raw=root.read(prefix+'/manifest.json');manifest=json.loads(raw)
            if digest(raw)!=pool['manifest_sha256'] or set(manifest)!={'generation','horizon','weight','masks'}:
                raise ValueError('pool manifest changed')
            if (manifest['generation'],manifest['horizon'],manifest['weight']) != (pool['generation'],pool['horizon'],pool['weight']):
                raise ValueError('pool checkpoint/manifest mismatch')
            if not isinstance(manifest['masks'],dict) or not manifest['masks']:raise ValueError('no pool masks')
            for name,sha in manifest['masks'].items():
                if not isinstance(name,str) or not _NAME.fullmatch(name) or name in ('.','..'):raise ValueError('invalid mask name')
                if digest(root.read(prefix+'/masks/'+name))!=sha:raise ValueError('pool mask changed')
        finally:root.close()

    def initialize(self,checkpoint):
        self._guard(); state=self._cp_state(checkpoint);self._assets(state)
        if state['epoch']!=0 or self.attest(operation='initialize',checkpoint=checkpoint,state=state) is not True:
            raise ValueError('initial provenance not attested')
        self.journal.initialize(checkpoint);return self.recover()

    def _expected(self,previous,raw,exposure,epoch):
        before=self._cp_state(previous);current=self._cp_state(raw)
        if before['epoch']!=epoch-1 or current!={**before,'epoch':epoch}:raise ValueError('trained checkpoint pool state/epoch mismatch')
        if not isinstance(exposure,dict) or set(exposure)!={'mode','positions','completed_g','auxiliary_steps','patch_occurrences','pixel_occurrences'}:
            raise ValueError('invalid exposure schema')
        if exposure['mode'] not in ('active','paused','no_pool'):
            raise ValueError('invalid pool mode')
        for k in ('positions','completed_g','auxiliary_steps','patch_occurrences','pixel_occurrences'):
            if type(exposure[k]) is not int or exposure[k]<0:raise ValueError('invalid exposure count')
        if exposure['completed_g']<1 or exposure['auxiliary_steps']>exposure['completed_g']:
            raise ValueError('invalid completed updates')
        pool=before['pool']; result=copy.deepcopy(current)
        if pool is None:
            if exposure['mode']!='no_pool' or any(exposure[k] for k in ('positions','auxiliary_steps','patch_occurrences','pixel_occurrences')):
                raise ValueError('no pool cannot have exposure')
        else:
            if exposure['mode']=='no_pool':raise ValueError('active pool cannot disappear silently')
            expected_positions=min(pool['horizon']-pool['cursor'],exposure['completed_g']) if exposure['mode']=='active' else 0
            if exposure['positions']!=expected_positions:raise ValueError('cursor positions do not match completed G updates')
            if exposure['mode']=='paused' and any(exposure[k] for k in ('auxiliary_steps','patch_occurrences','pixel_occurrences')):
                raise ValueError('paused pool cannot have exposure')
            result['pool']['cursor']+=expected_positions
            if result['pool']['cursor']==result['pool']['horizon']:
                result['retired'].append(pool['generation']);result['pool']=None
        return before,_state(result)

    def _post(self,raw,expected):
        old=self.decode(raw);new=copy.deepcopy(old);new.update(expected);post=self.encode(new)
        decoded=self.decode(post)
        if self._cp_state(post)!=expected or not _equal({k:v for k,v in old.items() if k not in ('epoch','pool','retired')},
                                                       {k:v for k,v in decoded.items() if k not in ('epoch','pool','retired')}):
            raise ValueError('pool transition modified non-pool checkpoint state')
        return post

    def _check_exposure_metrics(self,raw,before,exposure):
        """Match the receipt against the saved epoch's measured pool use.

        This establishes consistency, not independent data provenance: the
        native producer and checkpoint lineage still need a real attestor.
        """
        import math
        payload=self.decode(raw);metrics=payload.get('metrics') if isinstance(payload,dict) else None
        if not isinstance(metrics,dict):raise ValueError('missing checkpoint training metrics')
        recorder=metrics.get('experimental_actual_updates')
        steps=recorder.get('steps') if isinstance(recorder,dict) else None
        if not isinstance(steps,dict) or type(steps.get('G')) is not int or steps['G']!=exposure['completed_g']:
            raise ValueError('checkpoint generator updates do not match pool receipt')
        pool=before['pool'];auto=metrics.get('automatic_negative')
        if pool is None:
            if auto is not None:raise ValueError('unexpected automatic pool metrics without a pool')
            return
        if not isinstance(auto,dict):raise ValueError('missing measured automatic pool exposure')
        for measured,recorded in (('processed_contiguous_positions','positions'),('completed_G_updates','completed_g'),
          ('completed_G_auxiliary_steps','auxiliary_steps'),('completed_G_patch_occurrences','patch_occurrences'),
          ('completed_G_pixel_occurrences','pixel_occurrences')):
            if type(auto.get(measured)) is not int or auto[measured]!=exposure[recorded]:
                raise ValueError('checkpoint and pool exposure count mismatch')
        if type(auto.get('updates')) is not int or auto['updates']!=exposure['auxiliary_steps']:
            raise ValueError('automatic pool update count mismatch')
        weight=pool['weight'];effective=auto.get('effective_weight')
        if (type(effective) not in (int,float) or not math.isfinite(effective)
            or auto.get('configured_weight')!=weight
            or effective!=(weight if exposure['mode']=='active' else 0)):
            raise ValueError('checkpoint automatic pool weight mismatch')
        start=pool['cursor'];end=start+exposure['positions']
        if (auto.get('generation_start_cursor')!=start or auto.get('generation_end_cursor')!=end
            or auto.get('generation_horizon_updates')!=pool['horizon']):
            raise ValueError('checkpoint automatic pool cursor mismatch')

    def _receipt(self,d,previous,epoch):
        from sealed_safe_final_selection_v1 import SafeRoot
        if d!=self.journal._dir(epoch):raise ValueError('pool receipt directory mismatch')
        store=SafeRoot(self.journal.root);prefix='epoch_%06d/'%epoch
        try:
            r=store.json(prefix+'pool_receipt.json');raw=store.read(prefix+'pool_raw.bin');post=store.read(prefix+'pool_post.bin')
        finally:store.close()
        if set(r)!={'version','epoch','route','previous_sha256','raw_sha256','post_sha256','exposure','pool_before','pool_after'}:
            raise ValueError('invalid pool receipt schema')
        before,after=self._expected(previous,raw,r['exposure'],epoch)
        self._check_exposure_metrics(raw,before,r['exposure'])
        if (r['version']!='unified_epoch_pool_receipt_v1' or r['epoch']!=epoch or r['route'] not in ('SKIP','LIGHT_REVIEW','FULL_REVIEW','SAFETY_STOP')
            or r['previous_sha256']!=digest(previous) or r['raw_sha256']!=digest(raw) or r['post_sha256']!=digest(post)
            or r['pool_before']!=before or r['pool_after']!=after or self._cp_state(post)!=after):raise ValueError('pool receipt changed')
        old=self.decode(raw);new=self.decode(post)
        if not _equal({k:v for k,v in old.items() if k not in ('epoch','pool','retired')},
                      {k:v for k,v in new.items() if k not in ('epoch','pool','retired')}):raise ValueError('pool post checkpoint changed non-pool state')
        self._assets(before);self._assets(after)
        if self.attest(operation='epoch',previous=previous,trained=raw,post=post,exposure=r['exposure'],epoch=epoch) is not True:
            raise ValueError('epoch exposure provenance not attested')
        return r,post

    def _agent_transaction(self,epoch):
        from decision_transaction import Transaction
        from sealed_safe_final_selection_v1 import SafeRoot
        root=SafeRoot(self.journal.root)
        try:meta=root.json('epoch_%06d/trained.json'%epoch);route=meta.get('route')
        finally:root.close()
        if route=='FULL_REVIEW':kind='full_review';contract='safe_full_review_candidate_v1'
        elif route=='LIGHT_REVIEW':kind='light_review';contract='safe_anomaly_review_candidate_v1'
        else:raise ValueError('pool action requires a reviewed epoch')
        if type(epoch) is not int or epoch<1:raise ValueError('invalid review epoch')
        stage=Path(self.journal.job['output_root'])/'agent'/'sealed_v1'/('epoch_%03d'%epoch)/('light' if kind=='light_review' else 'full')
        return Transaction(stage/'transaction',self.journal.job['job_id'],self.journal.job['run_id'],epoch,kind)

    def _validated_agent_decision(self,epoch,pre=None,decision=None):
        """Require one fully received and normalized turn, never an intent alone."""
        import json
        from sealed_safe_final_selection_v1 import SafeRoot
        txn=self._agent_transaction(epoch);root=SafeRoot(self.journal.job['output_root']);prefix=root.rel(txn.root)
        try:
            manifest=root.json(prefix+'/manifest.json');req=root.read(prefix+'/request.json');raw_pre=root.read(prefix+'/pre_checkpoint.bin')
            submission=root.json(prefix+'/submission.json');terminal=root.json(prefix+'/terminal_identity.json')
            completed=root.json(prefix+'/completed.json');response_raw=root.read(prefix+'/response.json')
            decraw=root.read(prefix+'/decision.json');stored=json.loads(decraw);receipt=root.json(prefix+'/decision_receipt.json')
        finally:root.close()
        if (manifest.get('identity')!=txn.identity or manifest.get('request_sha256')!=digest(req)
            or manifest.get('pre_checkpoint_sha256')!=digest(raw_pre)):
            raise ValueError('transaction request or checkpoint changed')
        expected={'identity':txn.identity,'request_sha256':manifest['request_sha256'],
                  'pre_checkpoint_sha256':manifest['pre_checkpoint_sha256'],'response_sha256':digest(response_raw),
                  'decision_sha256':digest(decraw)}
        if (terminal!=completed or submission.get('identity')!=txn.identity or submission.get('thread_id')!=completed.get('thread_id')
            or submission.get('request_sha256')!=manifest['request_sha256'] or completed.get('identity')!=txn.identity
            or completed.get('request_sha256')!=manifest['request_sha256'] or completed.get('response_sha256')!=digest(response_raw)
            or receipt!=expected or stored.get('epoch')!=epoch or (pre is not None and pre!=raw_pre)
            or (decision is not None and stored!=decision)):
            raise ValueError('unresolved or mismatched Agent decision')
        auto=stored.get('automatic_negative')
        if isinstance(auto,dict) and auto.get('action')=='ACTIVATE':
            from decision_bridge import normalize_automatic_selection,normalize_review
            request=json.loads(req);response=json.loads(response_raw)
            if (request.get('contract_version')!='safe_full_review_candidate_v1'
                or response.get('action') not in ('KEEP','ADJUST','CONTINUE','PAUSE_ADV','RESUME_ADV')
                or response.get('action')!=stored.get('action')):
                raise ValueError('pool activation conflicts with raw review action')
            reconstructed=normalize_automatic_selection(request,response,
                actual_candidate_count=request.get('automatic_candidate_limit'),
                actual_max_weight=request.get('automatic_max_weight'))
            if reconstructed!=auto:raise ValueError('pool selection differs from raw review response')
            # For the restricted continuing actions accepted above, stopping
            # and experiment allowances do not apply. Reconstruct the native
            # ordinary decision as well: a faulty callback must not quietly
            # modify temperature, another parameter or a schedule during KEEP.
            ordinary=normalize_review(request,response,self.decode(raw_pre)['config'],self.journal.job,
                minimum_stop_epoch=0,stop_allowed=True,source_collapsed=False,experiment_attempts=0,experiment_limit=0)
            if {k:v for k,v in stored.items() if k!='automatic_negative'}!=ordinary:
                raise ValueError('ordinary decision differs from raw review response')
        return txn,stored,raw_pre,digest(decraw)

    def _pool_action(self,epoch,pre,decision,change,native_transition):
        """Prepare one local checkpoint transition during an exact Agent commit.

        Assets can be staged, but only the journal Agent post checkpoint can
        activate/deactivate the pool. Native transition must be pure and may
        only be replayed before any base checkpoint was durably recorded.
        """
        self._guard();status=self.recover()
        if status['state'] not in {'LIGHT_REVIEW','FULL_REVIEW'} or status['epoch']!=epoch:raise RuntimeError('pool change needs a live review')
        txn,stored,raw_pre,decision_sha=self._validated_agent_decision(epoch,pre,decision)
        if not isinstance(change,dict) or change.get('action') not in ('ACTIVATE','DISABLE'):raise ValueError('invalid pool change')
        action=change['action'];primary=stored.get('action')
        if stored.get('continue_training') is not True:raise ValueError('STOP or SELECT cannot change pools')
        if status['state']=='LIGHT_REVIEW':
            if action!='DISABLE' or primary!='DISABLE_AUTOMATIC_NEGATIVE':raise ValueError('light review can only disable an active pool')
        elif action!='ACTIVATE' or primary not in ('KEEP','ADJUST','CONTINUE','PAUSE_ADV','RESUME_ADV'):
            raise ValueError('full review can only activate a pool with a continuing decision')
        auto=stored.get('automatic_negative')
        if not isinstance(auto,dict) or auto.get('action')!=action:raise ValueError('pool action not approved in normalized decision')
        before=self._cp_state(pre);after=copy.deepcopy(before)
        if status['state']=='LIGHT_REVIEW':
            from sealed_safe_final_selection_v1 import SafeRoot
            root=SafeRoot(self.journal.job['output_root']);prefix=root.rel(txn.root)
            try:req=root.json(prefix+'/request.json');response=root.json(prefix+'/response.json')
            finally:root.close()
            active=(req.get('evidence') or {}).get('active_automatic_pool') or {};advertised=(active.get('values') or {}).get('weight')
            config=self.decode(pre).get('config') or {};legacy=config.get('_automatic_negative_state')
            configured=(legacy or {}).get('weight') if legacy is not None else (before['pool'] or {}).get('weight')
            if (before['pool'] is None or response.get('action')!='DISABLE_AUTOMATIC_NEGATIVE'
                    or before['pool'].get('weight')!=advertised or configured!=advertised):
                raise ValueError('light disable pool identity or dose mismatch')
        if action=='ACTIVATE':
            if set(change)!={'action','manifest','masks'} or before['pool'] is not None:
                raise ValueError('activation requires no existing pool')
            manifest=change['manifest'];masks=change['masks']
            if not isinstance(manifest,dict) or set(manifest)!={'generation','horizon','weight','masks'} or not isinstance(masks,dict) or not masks:
                raise ValueError('invalid activation manifest')
            if (type(manifest['horizon']) is not int or manifest['horizon']!=128
                or type(manifest['weight']) not in (int,float) or not 0 < manifest['weight'] <= .05):
                raise ValueError('activation differs from legacy pool horizon/dose bounds')
            gen=manifest['generation'];hashes=manifest['masks']
            if not isinstance(gen,str) or not re.fullmatch(r'[0-9a-f]{64}',gen) or gen in before['retired']:
                raise ValueError('invalid or retired generation')
            if not isinstance(hashes,dict) or set(hashes)!=set(masks) or type(auto.get('selected_count')) is not int or auto['selected_count']!=len(masks) or auto.get('automatic_weight')!=manifest['weight']:
                raise ValueError('selection/asset mismatch')
            for name,body in masks.items():
                if not isinstance(name,str) or not _NAME.fullmatch(name) or name in ('.','..') or not isinstance(body,bytes) or not body or hashes[name]!=digest(body):
                    raise ValueError('invalid or changed mask')
            after['pool']={'generation':gen,'manifest_sha256':digest(canonical(manifest)),'cursor':0,'horizon':manifest['horizon'],'weight':manifest['weight']}
            _state(after)
            plan={'action':action,'manifest_sha256':digest(canonical(manifest)),'generation':gen,'mask_sha256':dict(hashes)}
        else:
            if set(change)!={'action'} or before['pool'] is None:raise ValueError('disable requires one active pool')
            self._assets(before);after['retired'].append(before['pool']['generation']);after['pool']=None
            plan={'action':action,'generation':before['pool']['generation']}
        _state(after)
        if self.attest(operation='pool_action',epoch=epoch,pre=pre,decision=stored,change=plan,expected_state=after) is not True:
            raise ValueError('pool action provenance not attested')
        from sealed_safe_final_selection_v1 import SafeRoot
        store=SafeRoot(self.journal.root);prefix='epoch_%06d/pool_action'%epoch
        intent={'version':'unified_epoch_pool_action_v1','epoch':epoch,'pre_sha256':digest(pre),
                'decision_sha256':decision_sha,'plan':plan,'pool_before':before,'pool_after':after}
        try:
            store.mkdirs(prefix);store.publish_relative(prefix+'/intent.json',canonical(intent))
            if action=='ACTIVATE':
                asset_parent=SafeRoot(self.assets.parent)
                try:asset_parent.mkdirs(self.assets.name)
                finally:asset_parent.close()
                assets=SafeRoot(self.assets)
                try:
                    assets.mkdirs(gen+'/masks');assets.publish_relative(gen+'/manifest.json',canonical(manifest))
                    for name,body in masks.items():assets.publish_relative(gen+'/masks/'+name,body)
                finally:assets.close()
                self._assets(after)
            base=store.read(prefix+'/native_post.bin',missing_ok=True)
            if base is not None:
                if store.read(prefix+'/receipt.json',missing_ok=True) is None:
                    if not callable(native_transition):raise ValueError('native checkpoint transition required')
                    check_base=native_transition(pre,stored)
                    if not isinstance(check_base,bytes) or not _equal(self.decode(base),self.decode(check_base)):
                        raise ValueError('orphan native checkpoint differs from pure transition')
            else:
                if not callable(native_transition):raise ValueError('native checkpoint transition required')
                base=native_transition(pre,stored)
                if not isinstance(base,bytes) or self._cp_state(base)!=before:raise ValueError('native transition changed pool state')
                store.publish_relative(prefix+'/native_post.bin',base)
            if self._cp_state(base)!=before:raise ValueError('native pool state changed')
            post=store.read(prefix+'/pool_post.bin',missing_ok=True)
            if post is None:post=self._post(base,after);store.publish_relative(prefix+'/pool_post.bin',post)
            b=self.decode(base);p=self.decode(post)
            if self._cp_state(post)!=after or not _equal({k:v for k,v in b.items() if k not in ('epoch','pool','retired')},
                                                          {k:v for k,v in p.items() if k not in ('epoch','pool','retired')}):
                raise ValueError('pool action changed non-pool state')
            receipt={'intent_sha256':digest(canonical(intent)),'native_post_sha256':digest(base),'post_sha256':digest(post)}
            store.publish_relative(prefix+'/receipt.json',canonical(receipt));return post
        finally:store.close()

    def _audit_pool_action(self,epoch,previous_pool,committed):
        from sealed_safe_final_selection_v1 import SafeRoot
        store=SafeRoot(self.journal.root);prefix='epoch_%06d/pool_action'%epoch
        try:
            intent=store.json(prefix+'/intent.json');receipt=store.json(prefix+'/receipt.json')
            base=store.read(prefix+'/native_post.bin');post=store.read(prefix+'/pool_post.bin')
            route=store.json('epoch_%06d/trained.json'%epoch).get('route')
        finally:store.close()
        txn,decision,pre,decision_sha=self._validated_agent_decision(epoch)
        if (intent.get('version')!='unified_epoch_pool_action_v1' or intent.get('epoch')!=epoch
            or intent.get('pre_sha256')!=digest(pre) or intent.get('decision_sha256')!=decision_sha
            or intent.get('pool_before')!=previous_pool or self._cp_state(pre)!=previous_pool
            or receipt!={'intent_sha256':digest(canonical(intent)),'native_post_sha256':digest(base),'post_sha256':digest(post)}
            or self._cp_state(post)!=intent.get('pool_after')):
            raise ValueError('committed pool action evidence mismatch')
        expected_committed=post
        if route=='LIGHT_REVIEW' and post!=committed:
            expected_committed,_=self.journal._default_transition(epoch,post,'after_agent',require_existing=True)
        if committed!=expected_committed:raise ValueError('pool checkpoint differs after default policy')
        valid=(route=='FULL_REVIEW' and decision.get('action') in ('KEEP','ADJUST','CONTINUE','PAUSE_ADV','RESUME_ADV')
               and intent.get('plan',{}).get('action')=='ACTIVATE') or (route=='LIGHT_REVIEW'
               and decision.get('action')=='DISABLE_AUTOMATIC_NEGATIVE' and intent.get('plan',{}).get('action')=='DISABLE')
        if not valid or decision.get('continue_training') is not True:raise ValueError('ineligible committed pool decision')
        if decision.get('automatic_negative',{}).get('action')!=intent.get('plan',{}).get('action'):
            raise ValueError('committed pool decision mismatched')
        plan=intent['plan'];after=intent['pool_after'];expected=copy.deepcopy(previous_pool)
        if plan['action']=='ACTIVATE':
            pool=after.get('pool')
            if (previous_pool['pool'] is not None or not isinstance(pool,dict) or pool.get('cursor')!=0
                or pool.get('generation')!=plan.get('generation') or pool.get('manifest_sha256')!=plan.get('manifest_sha256')
                or pool['generation'] in previous_pool['retired'] or after['retired']!=previous_pool['retired']
                or after['epoch']!=previous_pool['epoch']):raise ValueError('invalid committed pool activation')
            assets=SafeRoot(self.assets)
            try:manifest=assets.json(pool['generation']+'/manifest.json')
            finally:assets.close()
            auto=decision['automatic_negative']
            if (manifest['masks']!=plan.get('mask_sha256') or auto.get('selected_count')!=len(manifest['masks'])
                or auto.get('automatic_weight')!=pool['weight']):raise ValueError('committed pool selection differs')
        elif plan['action']=='DISABLE':
            if previous_pool['pool'] is None or plan.get('generation')!=previous_pool['pool']['generation']:
                raise ValueError('invalid committed pool rollback')
            expected['retired'].append(previous_pool['pool']['generation']);expected['pool']=None
            if after!=expected:raise ValueError('committed rollback state differs')
        else:raise ValueError('unknown committed pool action')
        b=self.decode(base);p=self.decode(post)
        if self._cp_state(base)!=previous_pool or not _equal({k:v for k,v in b.items() if k not in ('epoch','pool','retired')},
                                                             {k:v for k,v in p.items() if k not in ('epoch','pool','retired')}):
            raise ValueError('committed pool action altered unrelated state')
        self._assets(previous_pool);self._assets(self._cp_state(post))
        if self.attest(operation='pool_action',epoch=epoch,pre=pre,decision=decision,change=intent['plan'],expected_state=intent['pool_after']) is not True:
            raise ValueError('committed pool action provenance not attested')

    def _repair_action_agent_post(self,epoch,previous_pool):
        """Recover an exact Agent post written before its ready receipt."""
        from sealed_safe_final_selection_v1 import SafeRoot
        journal=SafeRoot(self.journal.root);prefix='epoch_%06d/pool_action/'%epoch
        try:receipt=journal.read(prefix+'receipt.json',missing_ok=True)
        finally:journal.close()
        if receipt is None:return
        txn,_,pre,decision_sha=self._validated_agent_decision(epoch)
        root=SafeRoot(self.journal.job['output_root']);prefix=root.rel(txn.root)+'/'
        try:
            post=root.read(prefix+'post_checkpoint.bin',missing_ok=True)
            ready=root.read(prefix+'post_ready.json',missing_ok=True)
            if post is None or ready is not None:return
            self._audit_pool_action(epoch,previous_pool,post)
            root.publish_relative(prefix+'post_ready.json',canonical({'identity':txn.identity,'pre_sha256':digest(pre),
                      'decision_sha256':decision_sha,'post_sha256':digest(post)}))
        finally:root.close()

    def recover(self):
        from sealed_safe_final_selection_v1 import SafeRoot
        self._guard();store=SafeRoot(self.journal.root)
        try:
            previous=store.read('initial_checkpoint.bin');initial=store.json('initial.json')
            if initial!={'epoch':0,'checkpoint_sha256':digest(previous)}:raise ValueError('initial pool checkpoint changed')
            self.journal.check(previous,0);self._assets(self._cp_state(previous))
            dirs=sorted(name for name in os.listdir(store.fd) if name.startswith('epoch_'))
            for epoch,name in enumerate(dirs,1):
                if name!='epoch_%06d'%epoch or epoch>self.journal.max_epoch:raise ValueError('epoch gap')
                entries=set(store.listdir(name));prefix=name+'/';d=self.journal._dir(epoch)
                intent=store.json(prefix+'intent.json')
                if intent!={'epoch':epoch,'previous_checkpoint_sha256':digest(previous)}:raise ValueError('epoch chain mismatch')
                if 'pool_receipt.json' not in entries:
                    if entries.intersection({'pool_raw.bin','pool_post.bin','trained_checkpoint.bin','trained.json','commit.json'}):
                        raise RuntimeError('partial or bypassed pool receipt requires reconciliation')
                    return self.journal.recover()
                r,post=self._receipt(d,previous,epoch)
                has_trained='trained_checkpoint.bin' in entries;has_meta='trained.json' in entries;has_commit='commit.json' in entries
                if has_trained and not has_meta:
                    if store.read(prefix+'trained_checkpoint.bin')!=post:raise ValueError('partial trained checkpoint differs')
                    store.publish_relative(prefix+'trained.json',canonical({'epoch':epoch,'checkpoint_sha256':digest(post),'route':r['route']}))
                    has_meta=True
                if not has_trained and not has_meta:
                    if has_commit:raise ValueError('commit lacks trained checkpoint')
                    return {'state':'RECONCILE_POOL','epoch':epoch,'checkpoint':previous}
                if not has_trained or not has_meta or store.read(prefix+'trained_checkpoint.bin')!=post:
                    raise ValueError('trained checkpoint does not match pool receipt')
                if store.json(prefix+'trained.json')!={'epoch':epoch,'checkpoint_sha256':digest(post),'route':r['route']}:
                    raise ValueError('trained metadata does not match pool receipt')
                if not has_commit:
                    if r['route'] in ('LIGHT_REVIEW','FULL_REVIEW'):self._repair_action_agent_post(epoch,r['pool_after'])
                    return self.journal.recover()
                committed=store.read(prefix+'committed_checkpoint.bin')
                has_action=store.read(prefix+'pool_action/intent.json',missing_ok=True) is not None
                if has_action:
                    if r['route'] not in ('LIGHT_REVIEW','FULL_REVIEW'):raise ValueError('post-review pool state changed outside pool transaction')
                    self._audit_pool_action(epoch,r['pool_after'],committed)
                elif self._cp_state(committed)!=r['pool_after']:
                    raise ValueError('post-review pool state changed outside pool transaction')
                elif r['route'] in ('LIGHT_REVIEW','FULL_REVIEW'):
                    txn,decision,_,_=self._validated_agent_decision(epoch)
                    if (decision.get('action')=='DISABLE_AUTOMATIC_NEGATIVE' or
                        (decision.get('automatic_negative') or {}).get('action') in ('ACTIVATE','DISABLE')):
                        raise RuntimeError('committed pool action has no pool receipt')
                previous=committed
            return self.journal.recover()
        finally:store.close()

    def begin(self,epoch):
        self._guard();s=self.recover()
        if s['state']=='TRAIN_REQUIRED' and s['epoch']==epoch:return s
        if s['state']!='READY' or s['epoch']+1!=epoch:raise RuntimeError('epoch not ready')
        return self.journal.begin(epoch)

    def finish_training(self,epoch,raw,route,exposure):
        self._guard();s=self.recover()
        if s['state']!='TRAIN_REQUIRED' or s['epoch']!=epoch:raise RuntimeError('epoch is not awaiting training')
        if route not in ('SKIP','LIGHT_REVIEW','FULL_REVIEW','SAFETY_STOP') or ((epoch%5==0 or epoch==self.journal.max_epoch) and route not in ('FULL_REVIEW','SAFETY_STOP')):
            raise ValueError('invalid epoch review route')
        before,after=self._expected(s['checkpoint'],raw,exposure,epoch);self._assets(before);self._assets(after)
        self._check_exposure_metrics(raw,before,exposure)
        post=self._post(raw,after)
        if self.attest(operation='epoch',previous=s['checkpoint'],trained=raw,post=post,exposure=exposure,epoch=epoch) is not True:
            raise ValueError('epoch exposure provenance not attested')
        from sealed_safe_final_selection_v1 import SafeRoot
        store=SafeRoot(self.journal.root);prefix='epoch_%06d/'%epoch
        try:
            store.publish_relative(prefix+'pool_raw.bin',raw);store.publish_relative(prefix+'pool_post.bin',post)
            store.publish_relative(prefix+'pool_receipt.json',canonical({'version':'unified_epoch_pool_receipt_v1','epoch':epoch,'route':route,
              'previous_sha256':digest(s['checkpoint']),'raw_sha256':digest(raw),'post_sha256':digest(post),
              'exposure':exposure,'pool_before':before,'pool_after':after}))
        finally:store.close()
        return self.reconcile_training(epoch)

    def reconcile_training(self,epoch):
        self._guard();s=self.recover()
        if s['state']!='RECONCILE_POOL' or s['epoch']!=epoch:raise RuntimeError('no recoverable pool receipt')
        r,post=self._receipt(self.journal._dir(epoch),s['checkpoint'],epoch)
        self.journal.finish_training(epoch,post,r['route']);return self.recover()

    def finish_skip(self,epoch):
        self._guard();s=self.recover()
        if s['state']!='SKIP' or s['epoch']!=epoch:raise RuntimeError('no machine skip')
        self.journal.finish_skip(epoch);return self.recover()

    def finish_agent(self,epoch,*args,pool_change=None,**kwargs):
        self._guard();s=self.recover()
        if s['state'] not in ('LIGHT_REVIEW','FULL_REVIEW') or s['epoch']!=epoch:raise RuntimeError('no Agent review')
        if pool_change is not None:
            if s['state']=='LIGHT_REVIEW' and pool_change.get('action')!='DISABLE':raise ValueError('light review cannot activate a pool')
            if s['state']=='FULL_REVIEW' and pool_change.get('action')!='ACTIVATE':raise ValueError('full review cannot disable a pool')
            native=kwargs.get('transition')
            if not callable(native):raise ValueError('native transition required for pool change')
            kwargs['transition']=lambda pre,decision:self._pool_action(epoch,pre,decision,pool_change,native)
            previous_pool=self._cp_state(s['checkpoint'])
            def verify_pool_receipt(checked_epoch,decision,post):
                if checked_epoch!=epoch:return False
                self._audit_pool_action(epoch,previous_pool,post);return True
            kwargs['pool_receipt_verify']=verify_pool_receipt
        else:
            native=kwargs.get('transition')
            if not callable(native):raise ValueError('native transition required for review')
            def no_unhandled_action(pre,decision):
                if (decision.get('action')=='DISABLE_AUTOMATIC_NEGATIVE' or
                    (decision.get('automatic_negative') or {}).get('action') in ('ACTIVATE','DISABLE')):
                    raise RuntimeError('approved pool action requires its checkpoint transaction')
                return native(pre,decision)
            kwargs['transition']=no_unhandled_action
        self.journal.finish_agent(epoch,*args,**kwargs);return self.recover()
