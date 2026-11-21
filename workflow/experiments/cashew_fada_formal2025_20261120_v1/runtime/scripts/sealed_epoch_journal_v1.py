"""Experimental epoch journal: durable state hand-off; not a training permit.

The owner must hold ``lease`` for the complete worker lifetime, independently
prove input/lineage isolation, and create sealed review requests from authentic
worker observations. Hash chains detect changed artifacts, not fabricated data.
No target evaluation information, names, job objects or Agent prose is stored.
Unknown/incomplete network submissions stay in the transaction's fail-closed
state. This module does not yet execute pool publication, branches or SELECT.
"""
import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

from decision_transaction import canonical, digest, immutable, read_json, sync_dir
from sealed_transaction_router_v1 import route_prepared

ROUTES = {'SKIP', 'LIGHT_REVIEW', 'FULL_REVIEW', 'SAFETY_STOP'}

class EpochJournal:
    def __init__(self, root, job, checkpoint_validate, max_epoch):
        if (not isinstance(job, dict) or not isinstance(job.get('job_id'), str)
                or not isinstance(job.get('run_id'), str) or not callable(checkpoint_validate)
                or type(max_epoch) is not int or max_epoch < 1):
            raise ValueError('invalid journal identity')
        policy = job.get('experimental_gate')
        if not isinstance(policy, dict) or policy.get('version') != 'experimental_gate_v1' or policy.get('experimental_only') is not True:
            raise ValueError('explicit experimental policy required')
        self.root, self.job, self.check, self.max_epoch = Path(root), job, checkpoint_validate, max_epoch
        self.identity = {'version':'sealed_epoch_journal_v1','job_id':job['job_id'],
                         'run_id':job['run_id'],'max_epoch':max_epoch}
        self._lease = False
        self._lease_token = None
        self._restored = set()
        self._poisoned = False
        self._review_scope = None

    @contextmanager
    def lease(self):
        if self._lease: raise RuntimeError('journal already leased')
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink(): raise ValueError('journal root cannot be a symlink')
        fd = os.open(self.root/'worker.lock', os.O_RDWR | os.O_CREAT | getattr(os,'O_NOFOLLOW',0), 0o600)
        try:
            try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc: raise RuntimeError('another worker holds the journal lease') from exc
            self._lease = True
            self._lease_token = object()
            self._restored.clear()
            self._poisoned = False
            immutable(self.root/'identity.json', canonical(self.identity))
            yield self
        finally:
            self._lease = False
            self._lease_token = None
            os.close(fd)

    @contextmanager
    def verified_worker_scope(self,state):
        """Reuse only current-call, already verified review state and projections.

        This scope never survives a call, process or lease. It retains one
        current checkpoint already held by the caller plus small numeric
        projections, not five full checkpoints. Any journal commit invalidates
        the snapshot before writing; crash recovery uses the disk audit.
        An external writer remains outside this process-local guarantee.
        """
        import copy
        self._held()
        if self._review_scope is not None:raise RuntimeError('nested review scope')
        if not isinstance(state,dict) or state.get('state') not in {'READY','TRAIN_REQUIRED','SKIP','LIGHT_REVIEW','FULL_REVIEW'}:
            raise ValueError('verified Agent review state required')
        original=self.recover;original_check=self.check
        checked=original()
        if checked!=state:raise ValueError('review state differs from verified worker head')
        scope={'valid':True,'state':state,'window':None,'prior':None,'trend':None}
        self._review_scope=scope
        # Cache only successful validation digests, never deserialized model
        # objects or additional checkpoint buffers. Any changed bytes/epoch
        # necessarily trigger the original complete validator again.
        checkpoint_epoch=state['epoch']-1 if state['state']=='TRAIN_REQUIRED' else state['epoch']
        checked_hashes={(digest(state['checkpoint']),checkpoint_epoch)}
        def check_once(raw,epoch):
            if not isinstance(raw,bytes) or type(epoch) is not int:raise ValueError('invalid checkpoint validation input')
            key=(digest(raw),epoch)
            if key not in checked_hashes:
                original_check(raw,epoch);checked_hashes.add(key)
        self.check=check_once
        def recover_snapshot():
            if not scope['valid']:return original()
            return dict(scope['state'])
        self.recover=recover_snapshot
        try:yield self
        finally:
            self.recover=original;self.check=original_check;checked_hashes.clear();self._review_scope=None

    def _held(self):
        if not self._lease: raise RuntimeError('worker journal lease required')
        if self._poisoned: raise RuntimeError('worker state poisoned; terminate and restore in a fresh process')
        identity,_ = read_json(self.root/'identity.json')
        if identity != self.identity: raise ValueError('journal identity changed')

    def _dir(self, epoch): return self.root / ('epoch_%06d' % epoch)

    def initialize(self, checkpoint):
        self._held()
        if not isinstance(checkpoint, bytes): raise ValueError('checkpoint bytes required')
        self.check(checkpoint, 0)
        immutable(self.root/'initial_checkpoint.bin',checkpoint)
        immutable(self.root/'initial.json',canonical({'epoch':0,'checkpoint_sha256':digest(checkpoint)}))
        return self.recover()

    def _initial(self):
        meta,_ = read_json(self.root/'initial.json')
        p = self.root/'initial_checkpoint.bin'
        if p.is_symlink(): raise ValueError('unsafe initial checkpoint')
        data = p.read_bytes()
        if meta != {'epoch':0,'checkpoint_sha256':digest(data)}: raise ValueError('initial checkpoint changed')
        self.check(data,0)
        return data

    def _trained(self, directory, epoch, previous_hash):
        intent,_ = read_json(directory/'intent.json')
        if intent != {'epoch':epoch,'previous_checkpoint_sha256':previous_hash}:
            raise ValueError('epoch chain mismatch')
        p = directory/'trained_checkpoint.bin'
        if p.is_symlink(): raise ValueError('unsafe trained checkpoint')
        data = p.read_bytes(); meta,_ = read_json(directory/'trained.json')
        if (set(meta) != {'epoch','checkpoint_sha256','route'} or meta['epoch'] != epoch
                or meta['checkpoint_sha256'] != digest(data) or meta['route'] not in ROUTES):
            raise ValueError('trained epoch metadata changed')
        if (epoch % 5 == 0 or epoch == self.max_epoch) and meta['route'] not in {'FULL_REVIEW','SAFETY_STOP'}:
            raise ValueError('five-epoch and final reviews cannot be skipped')
        self.check(data,epoch)
        return data,meta

    def recover(self):
        """Return next operation and exact committed state; never issues model calls."""
        self._held(); data=self._initial(); previous=digest(data); previous_epoch=0
        dirs = sorted(p for p in self.root.glob('epoch_*') if p.is_dir())
        for directory in dirs:
            epoch=previous_epoch+1
            if epoch>self.max_epoch or directory != self._dir(epoch) or directory.is_symlink():
                raise ValueError('unexpected epoch directory or gap')
            intent,_=read_json(directory/'intent.json')
            if intent != {'epoch':epoch,'previous_checkpoint_sha256':previous}: raise ValueError('epoch chain mismatch')
            has_data=(directory/'trained_checkpoint.bin').exists(); has_meta=(directory/'trained.json').exists()
            if has_data != has_meta: raise RuntimeError('partially published trained epoch requires reconciliation')
            if not has_data:
                if directory != dirs[-1]: raise ValueError('unfinished epoch has later epochs')
                return {'state':'TRAIN_REQUIRED','epoch':epoch,'checkpoint':data}
            trained,meta=self._trained(directory,epoch,previous)
            if not (directory/'commit.json').exists():
                if directory != dirs[-1]: raise ValueError('unfinished review has later epochs')
                return {'state':meta['route'],'epoch':epoch,'checkpoint':trained}
            commit,_=read_json(directory/'commit.json'); post_path=directory/'committed_checkpoint.bin'
            if post_path.is_symlink(): raise ValueError('unsafe committed checkpoint')
            post=post_path.read_bytes()
            if (set(commit) != {'epoch','route','previous_checkpoint_sha256','trained_checkpoint_sha256','post_checkpoint_sha256','decision_kind','action'}
                    or commit['epoch']!=epoch or commit['route']!=meta['route'] or commit['previous_checkpoint_sha256']!=previous
                    or commit['trained_checkpoint_sha256']!=digest(trained) or commit['post_checkpoint_sha256']!=digest(post)):
                raise ValueError('epoch commit changed')
            self.check(post,epoch)
            if meta['route']=='SKIP':
                expected,_=self._default_transition(epoch,trained,'skip',require_existing=True)
                if commit['decision_kind']!='machine' or commit['action']!='KEEP' or post!=expected:
                    raise ValueError('invalid skip commit')
            if meta['route'] in {'LIGHT_REVIEW','FULL_REVIEW'}:
                if commit['decision_kind']!='agent': raise ValueError('invalid Agent commit')
                from sealed_worker_adapter_v1 import stage_for
                from decision_transaction import Transaction
                kind='light_review' if meta['route']=='LIGHT_REVIEW' else 'full_review'
                contract='safe_anomaly_review_candidate_v1' if kind=='light_review' else 'safe_full_review_candidate_v1'
                stage=stage_for(self.job,{'contract_version':contract,'epoch':epoch})
                txn=Transaction(stage/'transaction',self.job['job_id'],self.job['run_id'],epoch,kind)
                if not (txn.root/'commit.json').is_file():raise ValueError('Agent transaction commit missing')
                def no_transition(*args):raise RuntimeError('committed Agent transaction cannot be repeated')
                receipt,agent_post=txn.commit(self.check,no_transition)
                expected_pre=trained
                if kind=='full_review': expected_pre,_=self._default_transition(epoch,trained,'before_agent',require_existing=True)
                if receipt['pre_checkpoint_sha256']!=digest(expected_pre):raise ValueError('Agent transaction input mismatch')
                decision,_=read_json(txn.root/'decision.json')
                if decision.get('action')!=commit['action']:raise ValueError('Agent action mismatch')
                expected_post=agent_post
                if kind=='light_review':expected_post,_=self._default_transition(epoch,agent_post,'after_agent',require_existing=True)
                if post!=expected_post:raise ValueError('Agent transaction post checkpoint mismatch')
            if meta['route']=='SAFETY_STOP': raise ValueError('safety stop must not be committed as continuation')
            data,previous,previous_epoch=post,digest(post),epoch
            if commit['action']=='STOP':
                if directory != dirs[-1]: raise ValueError('epochs follow a STOP')
                return {'state':'STOPPED','epoch':epoch,'checkpoint':data}
        if previous_epoch==self.max_epoch:
            return {'state':'STOPPED_AT_LIMIT','epoch':previous_epoch,'checkpoint':data}
        return {'state':'READY','epoch':previous_epoch,'checkpoint':data}

    def begin(self,epoch):
        self._held(); status=self.recover()
        if status['state']=='TRAIN_REQUIRED' and status['epoch']==epoch: return status
        if status['state']!='READY' or epoch!=status['epoch']+1:
            raise RuntimeError('preceding state not ready for this epoch')
        d=self._dir(epoch); d.mkdir(exist_ok=False); sync_dir(self.root)
        immutable(d/'intent.json',canonical({'epoch':epoch,'previous_checkpoint_sha256':digest(status['checkpoint'])}))
        return {'state':'TRAIN_REQUIRED','epoch':epoch,'checkpoint':status['checkpoint']}

    def finish_training(self,epoch,checkpoint,route):
        self._held(); state=self.recover()
        if state['state']!='TRAIN_REQUIRED' or state['epoch']!=epoch or not isinstance(checkpoint,bytes):
            raise RuntimeError('epoch is not ready for trained checkpoint')
        if route not in ROUTES or ((epoch%5==0 or epoch==self.max_epoch) and route not in {'FULL_REVIEW','SAFETY_STOP'}):
            raise ValueError('invalid or missing required review route')
        self.check(checkpoint,epoch); d=self._dir(epoch)
        if self._review_scope is not None:self._review_scope['valid']=False
        immutable(d/'trained_checkpoint.bin',checkpoint)
        immutable(d/'trained.json',canonical({'epoch':epoch,'checkpoint_sha256':digest(checkpoint),'route':route}))
        return self.recover()

    def _commit(self,epoch,post,decision_kind,action):
        state=self.recover(); route=state['state']; trained=state['checkpoint']
        if state['epoch']!=epoch or route not in {'SKIP','LIGHT_REVIEW','FULL_REVIEW'}:
            raise RuntimeError('epoch is not awaiting a commit')
        d=self._dir(epoch); intent,_=read_json(d/'intent.json')
        self.check(post,epoch)
        meta={'epoch':epoch,'route':route,'previous_checkpoint_sha256':intent['previous_checkpoint_sha256'],
              'trained_checkpoint_sha256':digest(trained),'post_checkpoint_sha256':digest(post),
              'decision_kind':decision_kind,'action':action}
        if self._review_scope is not None:self._review_scope['valid']=False
        immutable(d/'committed_checkpoint.bin',post)
        immutable(d/'commit.json',canonical(meta))
        return self.recover()

    def _default_transition(self,epoch,checkpoint,phase,*,require_existing=False):
        """Persist declared baseline schedule separately from Agent decisions."""
        if phase not in {'skip','before_agent','after_agent'}: raise ValueError('invalid default phase')
        from scheduled_default_transition_v1 import transition
        d=self._dir(epoch); meta_path=d/('default_'+phase+'.json'); post_path=d/('default_'+phase+'.bin')
        if meta_path.exists():
            meta,_=read_json(meta_path)
            if (set(meta)!={'epoch','phase','pre_sha256','post_sha256','event'} or meta['epoch']!=epoch
                    or meta['phase']!=phase or meta['pre_sha256']!=digest(checkpoint)):
                raise ValueError('scheduled transition identity mismatch')
            if meta['event'] is None: post=checkpoint
            else:
                if post_path.is_symlink():raise ValueError('unsafe scheduled checkpoint')
                post=post_path.read_bytes()
            if digest(post)!=meta['post_sha256']:raise ValueError('scheduled checkpoint changed')
            self.check(post,epoch)
            return post,meta['event']
        if require_existing:raise RuntimeError('committed scheduled transition metadata missing')
        if post_path.exists():raise RuntimeError('orphan scheduled checkpoint requires reconciliation')
        post,event=transition(checkpoint,epoch)
        self.check(post,epoch)
        if event is not None:immutable(post_path,post)
        immutable(meta_path,canonical({'epoch':epoch,'phase':phase,'pre_sha256':digest(checkpoint),
                                     'post_sha256':digest(post),'event':event}))
        return post,event

    def finish_skip(self,epoch):
        self._held(); state=self.recover()
        if state['state']!='SKIP' or state['epoch']!=epoch: raise RuntimeError('not a machine skip')
        post,event=self._default_transition(epoch,state['checkpoint'],'skip')
        return self._commit(epoch,post,'machine','KEEP')

    def finish_agent(self,epoch,request,trusted,expected,*,pool_receipt_verify=None,**options):
        """Reconstruct request, bind exact trained state, claim/commit Agent once."""
        self._held(); state=self.recover()
        kind={'LIGHT_REVIEW':'safe_anomaly_review_candidate_v1','FULL_REVIEW':'safe_full_review_candidate_v1'}.get(state['state'])
        if state['epoch']!=epoch or kind is None or request.get('contract_version')!=kind or request.get('epoch')!=epoch:
            raise RuntimeError('review request does not match epoch route')
        self._verify_review_request(epoch,request,state['state'],expected,trusted,options.get('previous'))
        checkpoint=state['checkpoint']
        if state['state']=='FULL_REVIEW': checkpoint,_=self._default_transition(epoch,checkpoint,'before_agent')
        if 'pool_journal' in options:raise ValueError('worker owns pool journal binding')
        result=route_prepared(self.job,request,trusted,expected,checkpoint,checkpoint_validate=self.check,pool_journal=self,**options)
        action=result['decision'].get('action')
        # Optional actions require their own transactional executor and must not
        # silently advance the main worker before those are completed.
        if action not in {'KEEP','ADJUST','CONTINUE','PAUSE_ADV','RESUME_ADV','STOP','DISABLE_AUTOMATIC_NEGATIVE'}:
            raise RuntimeError('optional action requires separate reconciliation')
        if state['state']=='LIGHT_REVIEW' and action=='STOP': raise RuntimeError('light review cannot normally STOP')
        pool_action=(result['decision'].get('automatic_negative') or {}).get('action')
        if action=='DISABLE_AUTOMATIC_NEGATIVE' or pool_action in {'ACTIVATE','DISABLE'}:
            if not callable(pool_receipt_verify) or pool_receipt_verify(epoch,result['decision'],result['post_checkpoint']) is not True:
                raise RuntimeError('pool action requires verified single-authority receipt')
        post=result['post_checkpoint']
        if state['state']=='LIGHT_REVIEW': post,_=self._default_transition(epoch,post,'after_agent')
        return self._commit(epoch,post,'agent',action)

    def collect_full_review(self,epoch,gan,common,model,device):
        """Measure source and fixed unlabelled imagery at the journal's saved state.

        The caller must own the live model while this worker lease is held.
        The result still needs a trusted evidence journal and request builder;
        it is not itself authorization or proof of clean training lineage.
        """
        self._held(); state=self.recover()
        if state['state']!='FULL_REVIEW' or state['epoch']!=epoch:
            raise RuntimeError('full review requires the current trained epoch')
        checkpoint=self._dir(epoch)/'trained_checkpoint.bin'
        from live_review_evidence_v1 import collect
        import random
        import numpy as np
        import torch
        device=torch.device(device)
        # Validation must not advance the training worker's random streams.
        py_state=random.getstate(); np_state=np.random.get_state(); torch_state=torch.get_rng_state()
        cuda_state=torch.cuda.get_rng_state(device) if device.type=='cuda' else None
        try:
            source,panel,audit=collect(self.job,gan,common,model,device,checkpoint,epoch)
            if (not isinstance(audit,dict) or audit.get('epoch')!=epoch
                    or audit.get('checkpoint_sha256')!=digest(state['checkpoint'])
                    or audit.get('training_authorized') is not False):
                raise ValueError('review does not bind the journal checkpoint')
            if self.recover()['state']!='FULL_REVIEW': raise RuntimeError('journal changed during review')
            immutable(self._dir(epoch)/'review_observation.json',canonical(
                {'epoch':epoch,'checkpoint_sha256':audit['checkpoint_sha256'],'source':source,'panel':panel,
                 'input_snapshot_sha256':audit.get('input_snapshot_sha256'),'input_snapshot_count':audit.get('input_snapshot_count')}))
            return source,panel,audit
        except BaseException:
            self._poisoned=True
            raise
        finally:
            random.setstate(py_state);np.random.set_state(np_state);torch.set_rng_state(torch_state)
            if cuda_state is not None: torch.cuda.set_rng_state(cuda_state,device)

    def restore_live(self,*,device,model,local_d,global_d,model_opt,local_opt,global_opt):
        """Restore exact latest journal state before any additional training.

        Only one restore per checkpoint in a worker lease is permitted. Worker
        code must terminate the process if strict restoration raises: native
        module/optimizer loading may otherwise be partially applied. Restore is
        not a provenance attestation or permission to train.
        """
        self._held()
        import io
        import torch
        from healthy_recovery_runtime_v4 import restore_full_state
        status=self.recover(); raw=status['checkpoint']; sha=digest(raw)
        if sha in self._restored: raise RuntimeError('checkpoint already restored in this worker lease')
        saved_epoch=status['epoch']-1 if status['state']=='TRAIN_REQUIRED' else status['epoch']
        self.check(raw,saved_epoch)
        payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        if not isinstance(payload,dict) or payload.get('epoch')!=saved_epoch:
            raise ValueError('live restore checkpoint epoch mismatch')
        rng=payload.get('rng_state')
        if not isinstance(rng,dict): raise ValueError('checkpoint RNG state missing')
        ordinal=rng.get('active_cuda_ordinal')
        if type(ordinal) is not int or ordinal<0: raise ValueError('saved active CUDA identity required')
        # Native trained-checkpoint validator historically rejects epoch zero.
        # All other checks apply unchanged; adapt only its positivity sentinel.
        checked_payload=payload if saved_epoch>0 else dict(payload,epoch=1)
        try:
            config,receipt=restore_full_state(checked_payload,expected_epoch=max(1,saved_epoch),
                saved_cuda_ordinal=ordinal,device=device,model=model,local_d=local_d,global_d=global_d,
                model_opt=model_opt,local_opt=local_opt,global_opt=global_opt)
        except BaseException:
            self._poisoned=True
            raise
        self._restored.add(sha)
        receipt.update(epoch=saved_epoch,checkpoint_sha256=sha,journal_state=status['state'],
                       training_authorized=False)
        return config,receipt

    def training_window_evidence(self,epoch):
        """Project up to five consecutive authentic saved epoch facts, or stop.

        The hashes bind local artifacts but do not attest clean lineage. Numeric
        projection never copies target point metrics, filenames or free text.
        """
        self._held(); state=self.recover()
        if state['state'] not in {'LIGHT_REVIEW','FULL_REVIEW'} or state['epoch']!=epoch:
            raise RuntimeError('training evidence requires the current Agent review')
        import copy
        if self._review_scope is not None and self._review_scope['valid'] and self._review_scope['window'] is not None:
            return copy.deepcopy(self._review_scope['window'])
        import io
        import torch
        from worker_projection import project_training,project_controls
        rows=[]; bindings=[]
        for e in range(max(1,epoch-4),epoch+1):
            d=self._dir(e); intent,_=read_json(d/'intent.json')
            raw,meta=self._trained(d,e,intent['previous_checkpoint_sha256'])
            payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
            metrics=payload.get('metrics') if isinstance(payload,dict) else None
            if not isinstance(metrics,dict): raise ValueError('missing saved epoch metrics')
            for key in ('source_target_actual_exposure','supplementary_actual_exposure','experimental_actual_updates'):
                if not isinstance(metrics.get(key),dict):raise ValueError('missing saved evidence: '+key)
            training=project_training(metrics); controls=project_controls(metrics)
            if (type(training.get('steps')) is not int or training['steps']<1
                    or type(training.get('cls')) not in (int,float)
                    or any(type(controls.get(k)) not in (int,float) for k in
                           ('main_lr','D_lr','lambda_adv','temperature','confidence_cap'))):
                raise ValueError('missing essential per-epoch training facts')
            rows.append({'epoch':e,'training':training,'controls':controls})
            bindings.append({'epoch':e,'trained_checkpoint_sha256':meta['checkpoint_sha256']})
        result=(rows,bindings)
        if self._review_scope is not None and self._review_scope['valid']:self._review_scope['window']=copy.deepcopy(result)
        return result

    def five_epoch_training_evidence(self,epoch):
        state=self.recover()
        if state['state']!='FULL_REVIEW' or state['epoch']!=epoch or epoch<5:
            raise RuntimeError('five-epoch evidence requires the current full review')
        return self.training_window_evidence(epoch)

    def _verify_review_request(self,epoch,request,route,expected,trusted,previous):
        """Bind Agent-visible numeric observations to saved worker measurements."""
        rows,_=self.training_window_evidence(epoch)
        prior,hypotheses,prior_response=self.prior_action_context(epoch)
        if previous!=prior:raise ValueError('previous action does not match committed worker history')
        if prior is None:
            if request.get('prior_committed_action') is not None:raise ValueError('unregistered prior action')
        else:
            from worker_projection import project_committed_action
            if request.get('prior_committed_action')!=project_committed_action(prior[0],prior[1],epoch):
                raise ValueError('prior action differs from committed worker history')
        if route=='FULL_REVIEW':
            if expected.get('prior_response')!=prior_response:raise ValueError('prior response differs from committed worker history')
            if expected.get('hypotheses',()) not in (hypotheses,tuple(hypotheses)):
                raise ValueError('response expectations differ from committed structured evidence')
            if request.get('prior_structured_hypotheses')!=hypotheses:
                raise ValueError('Agent hypotheses differ from committed structured evidence')
        evidence=request.get('evidence') if isinstance(request,dict) else None
        if route=='LIGHT_REVIEW':
            actual_pool=self.active_pool_evidence(epoch)
            advertised=evidence.get('active_automatic_pool') if isinstance(evidence,dict) else None
            if actual_pool is None:
                if advertised is not None:raise ValueError('unverified or inactive automatic pool evidence')
            elif not isinstance(advertised,dict) or advertised.get('epoch')!=epoch or advertised.get('values')!=actual_pool:
                raise ValueError('automatic pool evidence differs from checkpoint and exposure receipt')
        window=evidence.get('training_window') if isinstance(evidence,dict) else None
        if not isinstance(window,list) or len(window)!=len(rows):
            raise ValueError('Agent request does not contain the complete measured window')
        for actual,expected_row in zip(window,rows):
            if not isinstance(actual,dict):raise ValueError('invalid Agent training window')
            for kind in ('training','controls'):
                item=actual.get(kind)
                if not isinstance(item,dict) or item.get('epoch')!=expected_row['epoch'] or item.get('values')!=expected_row[kind]:
                    raise ValueError('Agent training window differs from saved observations')
        if route=='FULL_REVIEW':
            observation,_=read_json(self._dir(epoch)/'review_observation.json')
            trained=self.recover()['checkpoint']
            if (observation.get('epoch')!=epoch or observation.get('checkpoint_sha256')!=digest(trained)
                    or evidence.get('source_validation',{}).get('values')!=observation.get('source')
                    or evidence.get('unlabelled_panel',{}).get('values')!=observation.get('panel')):
                raise ValueError('Agent review differs from current source and panel measurements')
            ctx,required=self.trend_context(epoch)
            if ctx is None:
                if evidence.get('prior_review_trend') is not None or expected.get('trend_context') is not None:
                    raise ValueError('unregistered prior review trend')
            else:
                if expected.get('trend_context')!=ctx or not required.issubset(set(trusted)):
                    raise ValueError('trend context differs from worker-registered reviews')
                from review_trends_v1 import compare
                current_source=self._trend_artifact('source',epoch,observation['source'])
                current_panel=self._trend_artifact('panel',epoch,observation['panel'])
                import hashlib,json
                binding_sha=hashlib.sha256(json.dumps(ctx['current_binding'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                if evidence.get('current_review_binding_sha256')!=binding_sha:raise ValueError('current review binding differs')
                value=compare(ctx['previous_review'],{'source':current_source,'panel':current_panel,'binding':ctx['current_binding']},
                    job_id=ctx['job_id'],run_id=ctx['run_id'],trusted_values=trusted,
                    trusted_bindings=set(tuple(x) for x in ctx['trusted_bindings']))
                if evidence.get('prior_review_trend')!=value:raise ValueError('trend values differ from registered reviews')

    def prepare_full_review(self,epoch):
        """Expose post-default config before building the full Agent request."""
        self._held(); state=self.recover()
        if state['state']!='FULL_REVIEW' or state['epoch']!=epoch:
            raise RuntimeError('no current full review')
        post,event=self._default_transition(epoch,state['checkpoint'],'before_agent')
        return {'epoch':epoch,'checkpoint':post,'default_event':event,'training_authorized':False}

    @staticmethod
    def _trend_artifact(kind,epoch,values):
        from sealed_packet import values_digest
        return {'kind':kind,'epoch':epoch,'values':values,'sha256':values_digest(kind,epoch,values)}

    def trend_context(self,epoch):
        """Rebuild same-run trend bindings exclusively from owned review files.

        File-stat snapshots compare frame metadata, never equal masks, accuracy,
        causal effects or proven blind provenance. Missing identity gives unknown.
        """
        self._held();state=self.recover()
        if state['state']!='FULL_REVIEW' or state['epoch']!=epoch:
            raise RuntimeError('trend context requires current full review')
        import copy
        if self._review_scope is not None and self._review_scope['valid'] and self._review_scope['trend'] is not None:
            return copy.deepcopy(self._review_scope['trend'])
        from review_trends_v1 import bind
        def record(e):
            obs,_=read_json(self._dir(e)/'review_observation.json')
            meta,_=read_json(self._dir(e)/'trained.json')
            if obs.get('epoch')!=e or obs.get('checkpoint_sha256')!=meta.get('checkpoint_sha256'):
                raise ValueError('review observation checkpoint mismatch')
            snapshot=obs.get('input_snapshot_sha256');count=obs.get('input_snapshot_count')
            if not isinstance(snapshot,str) or len(snapshot)!=64 or type(count) is not int or count<1:
                return None
            source=self._trend_artifact('source',e,obs['source']);panel=self._trend_artifact('panel',e,obs['panel'])
            audit={'version':'live_review_evidence_v1','epoch':e,'checkpoint_sha256':obs['checkpoint_sha256'],
                   'input_snapshot_sha256':snapshot}
            binding,signature=bind(self.job['job_id'],self.job['run_id'],source,panel,audit)
            return {'source':source,'panel':panel,'binding':binding},signature
        current=record(epoch)
        if current is None:return None,set()
        prior_epochs=[]
        for e in range(1,epoch):
            d=self._dir(e)
            if not (d/'review_observation.json').exists() or not (d/'commit.json').exists():continue
            commit,_=read_json(d/'commit.json')
            if commit.get('route')=='FULL_REVIEW':prior_epochs.append(e)
        previous=None
        if prior_epochs:
            e=prior_epochs[-1];index=getattr(self,'_active_head_index',None);pool_bridge=getattr(self,'_active_pool_bridge',None)
            if index is None:
                if pool_bridge is None:raise RuntimeError('previous full review requires active same-run head audit')
                pool_bridge.recover()  # full single-authority pool and journal audit
            else:
                intent,_=read_json(self._dir(e)/'intent.json')
                validated=index._epoch(e,intent['previous_checkpoint_sha256'],None)
                if validated['state']!='READY':raise ValueError('prior review was not committed as continuing')
            from sealed_worker_adapter_v1 import stage_for
            stage=stage_for(self.job,{'contract_version':'safe_full_review_candidate_v1','epoch':e})
            prior_request,_=read_json(stage/'transaction'/'request.json')
            prior_obs,_=read_json(self._dir(e)/'review_observation.json')
            evidence=prior_request.get('evidence') or {}
            if (evidence.get('source_validation',{}).get('values')!=prior_obs.get('source')
                    or evidence.get('unlabelled_panel',{}).get('values')!=prior_obs.get('panel')):
                raise ValueError('prior committed review differs from live observations')
            previous=record(e)
            if previous is None:return None,set()
            if evidence.get('current_review_binding_sha256')!=previous[1]:
                raise ValueError('prior review snapshot binding differs from committed turn')
        records=[current]+([previous] if previous else [])
        trusted={(item['kind'],item['epoch'],item['sha256']) for r,_ in records for item in (r['source'],r['panel'])}
        bindings=sorted((self.job['job_id'],self.job['run_id'],r['binding']['epoch'],signature) for r,signature in records)
        ctx={'job_id':self.job['job_id'],'run_id':self.job['run_id'],
             'previous_review':previous[0] if previous else None,'current_binding':current[0]['binding'],
             'trusted_bindings':bindings,'current_checkpoint_sha256':current[0]['binding']['checkpoint_sha256']}
        result=(ctx,trusted)
        if self._review_scope is not None and self._review_scope['valid']:self._review_scope['trend']=copy.deepcopy(result)
        return result

    def prior_action_context(self,epoch):
        """Recover a committed adjustment or declared KEEP monitoring, no prose.

        Expected responses are included only if present as validated structured
        fields in the original committed turn. Missing structured fields are an
        empty list (unknown), never inferred from the Agent's narrative.
        """
        self._held();state=self.recover()
        if state['state'] not in {'LIGHT_REVIEW','FULL_REVIEW'} or state['epoch']!=epoch:
            raise RuntimeError('prior action requires current review')
        import copy
        if self._review_scope is not None and self._review_scope['valid'] and self._review_scope['prior'] is not None:
            return copy.deepcopy(self._review_scope['prior'])
        from sealed_worker_adapter_v1 import stage_for
        from worker_projection import project_committed_hypotheses
        index=getattr(self,'_active_head_index',None);pool_bridge=getattr(self,'_active_pool_bridge',None);monitor=None
        for e in range(epoch-1,max(0,epoch-6),-1):
            directory=self._dir(e)
            if not (directory/'commit.json').is_file():continue
            commit,_=read_json(directory/'commit.json')
            if commit.get('decision_kind')!='agent' or commit.get('action') not in {'ADJUST','PAUSE_ADV','RESUME_ADV','KEEP','CONTINUE'}:continue
            if index is None:
                if pool_bridge is None:raise RuntimeError('prior action requires active same-run head audit')
                pool_bridge.recover()  # verifies all committed pool/journal epochs
            else:
                intent,_=read_json(directory/'intent.json');verified=index._epoch(e,intent['previous_checkpoint_sha256'],None)
                if verified['state']!='READY':raise ValueError('prior action was not committed for continuation')
            kind='safe_anomaly_review_candidate_v1' if commit['route']=='LIGHT_REVIEW' else 'safe_full_review_candidate_v1'
            stage=stage_for(self.job,{'contract_version':kind,'epoch':e})/'transaction'
            rec,_=read_json(stage/'commit.json');decision_path=stage/'decision.json'
            if decision_path.is_symlink():raise ValueError('unsafe previous decision')
            raw=decision_path.read_bytes()
            if digest(raw)!=rec['decision_sha256']:raise ValueError('previous decision changed')
            response_path=stage/'response.json'
            if response_path.is_symlink():raise ValueError('unsafe previous response')
            response_bytes=response_path.read_bytes();decision_receipt={'epoch':e,'decision_sha256':rec['decision_sha256']}
            hypotheses=project_committed_hypotheses(response_bytes,rec,raw,decision_receipt,
                job_id=self.job['job_id'],run_id=self.job['run_id'],cutoff=epoch)
            prior_response={'response_bytes':response_bytes,'commit':rec,'job_id':self.job['job_id'],'run_id':self.job['run_id']}
            item=((raw,decision_receipt),hypotheses,prior_response)
            if commit['action'] in {'KEEP','CONTINUE'}:
                if hypotheses and monitor is None:monitor=item
                continue
            if self._review_scope is not None and self._review_scope['valid']:self._review_scope['prior']=copy.deepcopy(item)
            return item
        result=monitor if monitor is not None else (None,[],None)
        if self._review_scope is not None and self._review_scope['valid']:self._review_scope['prior']=copy.deepcopy(result)
        return result

    def active_pool_evidence(self,epoch):
        """Rebuild numeric light-review pool facts from same-epoch artifacts.

        This relies on the bridge's exact checkpoint/receipt/asset validation,
        not on a caller-supplied trusted set or boolean. Hashes do not establish
        real-data provenance; the separate production attestor remains required.
        Missing/inconsistent exposure blocks any advertised active-pool claim.
        """
        import io,math
        import torch
        self._held();bridge=getattr(self,'_active_pool_bridge',None)
        if bridge is None:return None
        state=bridge.recover()
        if state['state']!='LIGHT_REVIEW' or state['epoch']!=epoch:
            raise RuntimeError('pool evidence requires the owning light review')
        checkpoint=state['checkpoint'];logical=bridge.decode(checkpoint)
        if not isinstance(logical,dict) or logical.get('epoch')!=epoch:raise ValueError('pool checkpoint epoch mismatch')
        receipt,_=read_json(self._dir(epoch)/'pool_receipt.json')
        if receipt.get('post_sha256')!=digest(checkpoint) or receipt.get('epoch')!=epoch:
            raise ValueError('pool receipt not bound to current checkpoint')
        after=receipt.get('pool_after');before=receipt.get('pool_before');exposure=receipt.get('exposure')
        if not isinstance(after,dict) or not isinstance(before,dict) or not isinstance(exposure,dict):
            raise ValueError('incomplete pool exposure receipt')
        pool=after.get('pool')
        if pool is None:
            if logical.get('pool') is not None:raise ValueError('checkpoint and receipt pool mismatch')
            return None
        if logical.get('pool')!=pool or not isinstance(before.get('pool'),dict):raise ValueError('checkpoint and receipt pool mismatch')
        metrics=logical.get('metrics');auto=metrics.get('automatic_negative') if isinstance(metrics,dict) else None
        if not isinstance(auto,dict):raise ValueError('active pool exposure not measured')
        checks=(('processed_contiguous_positions','positions'),('completed_G_updates','completed_g'),
                ('completed_G_auxiliary_steps','auxiliary_steps'),('completed_G_patch_occurrences','patch_occurrences'),
                ('completed_G_pixel_occurrences','pixel_occurrences'))
        for measured,recorded in checks:
            if type(auto.get(measured)) is not int or type(exposure.get(recorded)) is not int or auto[measured]!=exposure[recorded]:
                raise ValueError('checkpoint and pool exposure count mismatch')
        updates=auto.get('updates');weight=pool.get('weight');effective=auto.get('effective_weight')
        if type(updates) is not int or updates<0 or updates!=exposure['auxiliary_steps']:
            raise ValueError('automatic pool update count mismatch')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (weight,effective)) or not 0<weight<=.005 or effective not in (0,weight):
            raise ValueError('invalid effective or configured pool weight')
        if auto.get('configured_weight')!=weight:
            raise ValueError('checkpoint configured pool weight mismatch')
        cursor=pool.get('cursor');horizon=pool.get('horizon');start=before['pool'].get('cursor')
        if (type(cursor) is not int or type(horizon) is not int or type(start) is not int or not 0<=start<=cursor<=horizon
                or auto.get('generation_start_cursor')!=start or auto.get('generation_end_cursor')!=cursor
                or auto.get('generation_horizon_updates')!=horizon or cursor-start!=exposure['positions']):
            raise ValueError('pool cursor or horizon mismatch')
        return {'weight':weight,'effective_weight':effective,'updates':updates,'cursor':cursor,'horizon':horizon}
