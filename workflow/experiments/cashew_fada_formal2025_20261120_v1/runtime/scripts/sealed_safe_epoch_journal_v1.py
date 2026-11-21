"""Experimental new-format-only training journal; Agent/STOP/SELECT blocked.

Only SKIP and its deterministic default schedule are implemented. This is not
authorization, provenance, a security sandbox or legacy checkpoint migration.
"""
import copy,io,json,os
from contextlib import contextmanager
from pathlib import Path
import torch
from decision_transaction import canonical,digest
from sealed_epoch_journal_v1 import EpochJournal
from native_safe_checkpoint_codec_v1 import decode,encode
from unified_epoch_pool_bridge_sim_v1 import _equal

def safe_check(raw,epoch):
    payload=decode(raw)
    if payload['epoch']!=epoch:raise ValueError('safe checkpoint epoch mismatch')
    return payload

def _scheduled(raw,epoch):
    from fixed_supplementary_default_v1 import POLICY,advance
    payload=safe_check(raw,epoch);config=payload['config'];policy=config.get('supplementary_default')
    if policy is None:return raw,None
    if not isinstance(policy,dict) or policy.get('version')!=POLICY or policy.get('state') not in {'pending','activated','no_samples','agent_override'}:
        raise ValueError('unknown supplementary default')
    if policy['state']=='pending' and (type(policy.get('activation_after_epoch')) is not int or policy['activation_after_epoch']<0):
        raise ValueError('invalid default activation epoch')
    before=copy.deepcopy(config);event=advance(config,epoch)
    if event is None:
        if config!=before:raise ValueError('silent default modification')
        return raw,None
    if not isinstance(event,dict) or event.get('state') not in {'activated','no_samples'} or event.get('epoch')!=epoch:
        raise ValueError('invalid default transition')
    return encode(payload),event

class SafeTrainingJournal(EpochJournal):
    def __init__(self,root,job,max_epoch):
        super().__init__(root,job,safe_check,max_epoch)
        self.identity['version']='sealed_safe_epoch_journal_v1'

    @contextmanager
    def lease(self):
        from sealed_safe_final_selection_v1 import SafeRoot
        if self._lease:raise RuntimeError('journal already leased')
        parent=SafeRoot(self.root.parent)
        try:parent.mkdirs(self.root.name)
        finally:parent.close()
        fs=SafeRoot(self.root);out=None;fd=None
        try:
            output=Path(os.path.abspath(self.job['output_root']))
            output_parent=SafeRoot(output.parent)
            try:output_parent.mkdirs(output.name)
            finally:output_parent.close()
            out=SafeRoot(output);fd=fs.lock('worker.lock',nonblocking=True)
            self._fs,self._out_fs=fs,out;self._lease=True;self._lease_token=object()
            self._restored.clear();self._poisoned=False;self._prefix_cache=None;self._pending_checkpoint_cache=None
            fs.publish_relative('identity.json',canonical(self.identity))
            yield self
        finally:
            self._lease=False;self._lease_token=None;self._prefix_cache=None;self._pending_checkpoint_cache=None
            if fd is not None:os.close(fd)
            if out is not None:out.close()
            fs.close();self._fs=None;self._out_fs=None

    def _held(self):
        if not self._lease or getattr(self,'_fs',None) is None:raise RuntimeError('worker journal lease required')
        if self._poisoned:raise RuntimeError('worker state poisoned; terminate and restore in a fresh process')
        if self._fs.json('identity.json')!=self.identity:raise ValueError('journal identity changed')

    def _initial(self):
        raw=self._fs.read('initial_checkpoint.bin');meta=self._fs.json('initial.json')
        if meta!={'epoch':0,'checkpoint_sha256':digest(raw)}:raise ValueError('initial checkpoint changed')
        self.check(raw,0);return raw

    def initialize(self,checkpoint):
        self._held();self.check(checkpoint,0)
        self._fs.publish_relative('initial_checkpoint.bin',checkpoint)
        self._fs.publish_relative('initial.json',canonical({'epoch':0,'checkpoint_sha256':digest(checkpoint)}))
        if (Path(self.job['output_root'])/'agent/sealed_v1/initialization/transaction/commit.json').is_file():
            from complete_history_store import save_initialization
            save_initialization(self.job['output_root'],self.job['job_id'],self.job['run_id'])
        return self.recover()

    def _trained(self,directory,epoch,previous_hash,*,trained_raw=None):
        if directory!=self._dir(epoch):raise ValueError('trained epoch directory mismatch')
        prefix='epoch_%06d/'%epoch;intent=self._fs.json(prefix+'intent.json')
        if intent!={'epoch':epoch,'previous_checkpoint_sha256':previous_hash}:raise ValueError('epoch chain mismatch')
        raw=trained_raw if trained_raw is not None else self._fs.read(prefix+'trained_checkpoint.bin')
        meta=self._fs.json(prefix+'trained.json')
        if (set(meta)!={'epoch','checkpoint_sha256','route'} or meta['epoch']!=epoch
            or meta['checkpoint_sha256']!=digest(raw) or meta['route'] not in {'SKIP','LIGHT_REVIEW','FULL_REVIEW','SAFETY_STOP'}):
            raise ValueError('trained epoch metadata changed')
        if (epoch%5==0 or epoch==self.max_epoch) and meta['route'] not in {'FULL_REVIEW','SAFETY_STOP'}:
            raise ValueError('five-epoch or final review required')
        from human_negative_epoch10 import review_due
        if review_due(self.job,epoch) and meta['route'] not in {'FULL_REVIEW','SAFETY_STOP'}:
            raise ValueError('Agent human-supervision decision required before epoch 10')
        self.check(raw,epoch);return raw,meta

    def begin(self,epoch):
        self._held();status=self.recover()
        if status['state']=='TRAIN_REQUIRED' and status['epoch']==epoch:return status
        if status['state']!='READY' or epoch!=status['epoch']+1:raise RuntimeError('preceding state not ready')
        prefix='epoch_%06d'%epoch
        if prefix in os.listdir(self._fs.fd):raise RuntimeError('epoch directory already exists')
        self._fs.mkdirs(prefix);self._fs.publish_relative(prefix+'/intent.json',canonical({'epoch':epoch,'previous_checkpoint_sha256':digest(status['checkpoint'])}))
        return {'state':'TRAIN_REQUIRED','epoch':epoch,'checkpoint':status['checkpoint']}

    def finish_training(self,epoch,checkpoint,route):
        self._held();state=self.recover()
        if state['state']!='TRAIN_REQUIRED' or state['epoch']!=epoch or not isinstance(checkpoint,bytes):raise RuntimeError('epoch not ready')
        if route not in {'SKIP','LIGHT_REVIEW','FULL_REVIEW','SAFETY_STOP'} or ((epoch%5==0 or epoch==self.max_epoch) and route not in {'FULL_REVIEW','SAFETY_STOP'}):
            raise ValueError('invalid epoch route')
        from human_negative_epoch10 import review_due
        if review_due(self.job,epoch) and route not in {'FULL_REVIEW','SAFETY_STOP'}:
            raise ValueError('epoch-10 decision cannot be skipped')
        self.check(checkpoint,epoch);prefix='epoch_%06d/'%epoch
        if self._review_scope is not None:self._review_scope['valid']=False
        self._fs.publish_relative(prefix+'trained_checkpoint.bin',checkpoint)
        self._fs.publish_relative(prefix+'trained.json',canonical({'epoch':epoch,'checkpoint_sha256':digest(checkpoint),'route':route}))
        from complete_history_store import save_epoch
        save_epoch(self.job['output_root'],self.job['job_id'],self.job['run_id'],epoch,checkpoint)
        return self.recover()

    def _commit(self,epoch,post,decision_kind,action):
        state=self.recover();route=state['state'];trained=state['checkpoint'];prefix='epoch_%06d/'%epoch
        if state['epoch']!=epoch or route not in {'SKIP','LIGHT_REVIEW','FULL_REVIEW'}:raise RuntimeError('epoch not awaiting commit')
        intent=self._fs.json(prefix+'intent.json');self.check(post,epoch)
        meta={'epoch':epoch,'route':route,'previous_checkpoint_sha256':intent['previous_checkpoint_sha256'],
              'trained_checkpoint_sha256':digest(trained),'post_checkpoint_sha256':digest(post),'decision_kind':decision_kind,'action':action}
        if self._review_scope is not None:self._review_scope['valid']=False
        self._fs.publish_relative(prefix+'committed_checkpoint.bin',post);self._fs.publish_relative(prefix+'commit.json',canonical(meta))
        from complete_history_store import review_data,observations
        review_data(self.job['output_root'],self.job['job_id'],self.job['run_id'],epoch,meta)
        if route=='FULL_REVIEW':
            observations(self.job['output_root'],epoch,meta['trained_checkpoint_sha256'],
                         job_id=self.job['job_id'],run_id=self.job['run_id'],commit=meta)
        return self.recover()

    def _default_transition(self,epoch,checkpoint,phase,*,require_existing=False):
        if phase not in {'skip','before_agent','after_agent'}:raise ValueError('invalid safe default phase')
        prefix='epoch_%06d/'%epoch;meta_path=prefix+'default_'+phase+'.json';post_path=prefix+'default_'+phase+'.bin'
        expected,event=_scheduled(checkpoint,epoch)
        meta_raw=self._fs.read(meta_path,missing_ok=True)
        if meta_raw is not None:
            meta=json.loads(meta_raw)
            if (set(meta)!={'epoch','phase','pre_sha256','post_sha256','event'} or meta['epoch']!=epoch
                or meta['phase']!=phase or meta['pre_sha256']!=digest(checkpoint) or meta['event']!=event):
                raise ValueError('safe default identity mismatch')
            if event is None:post=checkpoint
            else:
                post=self._fs.read(post_path)
            # _scheduled already safely decoded and verified the input. If
            # there is no event, both post and expected are that exact input;
            # decoding the same 400 MB payload twice adds no verification.
            if digest(post)!=meta['post_sha256'] or (event is not None and not _equal(safe_check(post,epoch),safe_check(expected,epoch))):
                raise ValueError('safe default checkpoint differs')
            return post,event
        if require_existing:raise RuntimeError('safe default metadata missing')
        if self._fs.read(post_path,missing_ok=True) is not None:raise RuntimeError('orphan safe default checkpoint')
        self.check(expected,epoch)
        if event is not None:self._fs.publish_relative(post_path,expected)
        self._fs.publish_relative(meta_path,canonical({'epoch':epoch,'phase':phase,'pre_sha256':digest(checkpoint),
                                       'post_sha256':digest(expected),'event':event}))
        return expected,event

    def _metadata_snapshot(self,fs,path):
        """Cheap change detector for immutable, already-verified artifacts.

        This is not a cryptographic revalidation. A new lease always performs
        a full replay. Only files inside the owning journal/transaction are
        inspected, and symlinks and nonregular artifacts are rejected.
        """
        import stat
        parent,name=fs._parent(path)
        try:
            fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
        finally:os.close(parent)
        try:
            fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_mode','st_nlink')
            ds=os.fstat(fd);out=[('.',tuple(getattr(ds,k) for k in fields))]
            for child in sorted(os.listdir(fd)):
                st=os.stat(child,dir_fd=fd,follow_symlinks=False)
                if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1:raise ValueError('unsafe cached journal artifact')
                out.append((child,tuple(getattr(st,k) for k in fields)))
            return tuple(out)
        finally:os.close(fd)

    def _snapshot_initial(self):
        fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_mode','st_nlink')
        return tuple((name,tuple(getattr(os.stat(name,dir_fd=self._fs.fd,follow_symlinks=False),k) for k in fields))
                     for name in ('initial.json','initial_checkpoint.bin'))

    def _epoch_snapshot(self,epoch,route):
        journal=self._metadata_snapshot(self._fs,'epoch_%06d'%epoch)
        if route in ('FULL_REVIEW','LIGHT_REVIEW'):
            typ='full' if route=='FULL_REVIEW' else 'light'
            txn='agent/sealed_v1/epoch_%03d/%s/transaction'%(epoch,typ)
            return journal,self._metadata_snapshot(self._out_fs,txn)
        return journal,None

    def _read_pending(self,prefix):
        """Cache the one current trained state, invalidating on file changes."""
        import stat
        parent,name=self._fs._parent(prefix+'trained_checkpoint.bin')
        try:
            try:st=os.stat(name,dir_fd=parent,follow_symlinks=False)
            except FileNotFoundError:return None
            if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1:raise ValueError('unsafe trained checkpoint')
            ident=tuple(getattr(st,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))
        finally:os.close(parent)
        cached=self._pending_checkpoint_cache
        if cached is not None and cached[:2]==(prefix,ident):return cached[2]
        raw=self._fs.read(prefix+'trained_checkpoint.bin')
        self._pending_checkpoint_cache=(prefix,ident,raw)
        return raw

    def recover(self):
        self._held()
        dirs=sorted(name for name in os.listdir(self._fs.fd) if name.startswith('epoch_'))
        cached=self._prefix_cache
        if cached is None:
            if os.environ.get('CASHEW_FAST_RESUME')=='1':
                from fast_resume_v1 import bootstrap
                cached=bootstrap(self);data=cached['checkpoint'];previous=cached['hash'];previous_epoch=cached['epoch']
            else:
                data=self._initial();previous=digest(data);previous_epoch=0
                cached={'initial':self._snapshot_initial(),'snapshots':[],'epoch':0,'checkpoint':data,'hash':previous,'action':None}
            self._prefix_cache=cached
            if cached['action']=='STOP':
                if len(dirs)!=previous_epoch:raise ValueError('epochs follow STOP')
                return {'state':'STOPPED','epoch':previous_epoch,'checkpoint':data}
        else:
            changed=self._snapshot_initial()!=cached['initial']
            changed=changed or dirs[:cached['epoch']]!=['epoch_%06d'%n for n in range(1,cached['epoch']+1)]
            if not changed:
                for e,route,snapshot in cached['snapshots']:
                    if self._epoch_snapshot(e,route)!=snapshot:
                        changed=True;break
            if changed:
                # A metadata change forces complete validation; it must never
                # roll back a previously committed in-lease prefix.
                required_epoch=cached['epoch']
                self._prefix_cache=None
                state=self.recover()
                if self._prefix_cache['epoch']<required_epoch:
                    self._prefix_cache=None
                    raise ValueError('previously committed journal prefix disappeared')
                return state
            data=cached['checkpoint'];previous=cached['hash'];previous_epoch=cached['epoch']
            if cached['action']=='STOP':
                if len(dirs)!=previous_epoch:raise ValueError('epochs follow STOP')
                return {'state':'STOPPED','epoch':previous_epoch,'checkpoint':data}
        for dirname in dirs[previous_epoch:]:
            epoch=previous_epoch+1
            if epoch>self.max_epoch or dirname!='epoch_%06d'%epoch:raise ValueError('unexpected epoch directory')
            directory=self._dir(epoch);prefix=dirname+'/'
            intent=self._fs.json(prefix+'intent.json')
            if intent!={'epoch':epoch,'previous_checkpoint_sha256':previous}:raise ValueError('safe epoch chain mismatch')
            trained_raw=(self._read_pending(prefix) if os.environ.get('CASHEW_FAST_RESUME')=='1'
                         else self._fs.read(prefix+'trained_checkpoint.bin',missing_ok=True))
            has_data=trained_raw is not None
            has_meta=self._fs.read(prefix+'trained.json',missing_ok=True) is not None
            if has_data!=has_meta:raise RuntimeError('partial trained epoch requires reconciliation')
            if not has_data:
                if dirname!=dirs[-1]:raise ValueError('unfinished epoch has later epochs')
                return {'state':'TRAIN_REQUIRED','epoch':epoch,'checkpoint':data}
            trained,meta=self._trained(directory,epoch,previous,trained_raw=trained_raw)
            commit_raw=self._fs.read(prefix+'commit.json',missing_ok=True)
            if commit_raw is None:
                if dirname!=dirs[-1]:raise ValueError('unfinished review has later epochs')
                return {'state':meta['route'],'epoch':epoch,'checkpoint':trained}
            commit=json.loads(commit_raw);post=self._fs.read(prefix+'committed_checkpoint.bin');self.check(post,epoch)
            if meta['route']=='SKIP':
                expected,_=self._default_transition(epoch,trained,'skip',require_existing=True);kind='machine';action='KEEP'
            elif meta['route'] in {'LIGHT_REVIEW','FULL_REVIEW'}:
                from fd_safe_decision_transaction_candidate_v1 import SafeTransaction
                route=meta['route'];contract='safe_anomaly_review_candidate_v1' if route=='LIGHT_REVIEW' else 'safe_full_review_candidate_v1'
                txn_path=Path(self.job['output_root'])/'agent'/'sealed_v1'/('epoch_%03d'%epoch)/('light' if route=='LIGHT_REVIEW' else 'full')/'transaction'
                txn=SafeTransaction(txn_path,self.job['job_id'],self.job['run_id'],epoch,
                                'light_review' if route=='LIGHT_REVIEW' else 'full_review')
                def no_transition(*args):raise RuntimeError('committed Agent turn cannot be repeated')
                def validate(q,a,p,d,result):
                    self._verify_committed_agent(epoch,q,a,p,d,result)
                receipt,agent_post=txn.commit(self.check,no_transition,post_validate=validate,require_commit=True)
                pre=trained
                if route=='FULL_REVIEW':pre,_=self._default_transition(epoch,trained,'before_agent',require_existing=True)
                if receipt['pre_checkpoint_sha256']!=digest(pre):raise ValueError('Agent pre checkpoint mismatch')
                decision=self._out_fs.json(self._out_fs.rel(txn.root/'decision.json'));action=decision['action'];kind='agent'
                expected=agent_post
                if route=='LIGHT_REVIEW':expected,_=self._default_transition(epoch,agent_post,'after_agent',require_existing=True)
            else:raise RuntimeError('SAFETY_STOP/STOP/SELECT commit is not certified')
            if commit!={'epoch':epoch,'route':meta['route'],'previous_checkpoint_sha256':previous,'trained_checkpoint_sha256':digest(trained),
                        'post_checkpoint_sha256':digest(post),'decision_kind':kind,'action':action} or post!=expected:
                raise ValueError('invalid safe epoch commit')
            data,previous,previous_epoch=post,digest(post),epoch
            cached['snapshots'].append((epoch,meta['route'],self._epoch_snapshot(epoch,meta['route'])))
            cached.update(epoch=epoch,checkpoint=data,hash=previous,action=action)
            if action=='STOP':
                if dirname!=dirs[-1]:raise ValueError('epochs follow STOP')
                return {'state':'STOPPED','epoch':epoch,'checkpoint':post}
        if previous_epoch==self.max_epoch:return {'state':'STOPPED_AT_LIMIT','epoch':previous_epoch,'checkpoint':data}
        return {'state':'READY','epoch':previous_epoch,'checkpoint':data}

    def restore_live(self,*,device,model,local_d,global_d,model_opt,local_opt,global_opt,purpose=None):
        self._held();from healthy_recovery_runtime_v4 import restore_full_state
        status=self.recover();raw=status['checkpoint'];sha=digest(raw)
        if purpose not in (None,'review','training'):raise ValueError('invalid checkpoint restore purpose')
        if purpose=='review' and status['state']!='FULL_REVIEW':raise RuntimeError('review restore requires a full review')
        if purpose=='training' and status['state']!='TRAIN_REQUIRED':raise RuntimeError('training restore requires a pending epoch')
        key=(sha,purpose)
        if key in self._restored or (sha,None) in self._restored or (purpose is None and any(item[0]==sha for item in self._restored)):
            raise RuntimeError('checkpoint already restored for this purpose in this worker lease')
        epoch=status['epoch']-1 if status['state']=='TRAIN_REQUIRED' else status['epoch']
        payload=safe_check(raw,epoch);ordinal=payload['rng_state']['active_cuda_ordinal']
        if type(ordinal) is not int or ordinal<0:raise ValueError('saved active CUDA identity required')
        checked=payload if epoch>0 else dict(payload,epoch=1)
        try:
            config,receipt=restore_full_state(checked,expected_epoch=max(1,epoch),saved_cuda_ordinal=ordinal,device=device,
                model=model,local_d=local_d,global_d=global_d,model_opt=model_opt,local_opt=local_opt,global_opt=global_opt)
        except BaseException:self._poisoned=True;raise
        self._restored.add(key);receipt.update(epoch=epoch,checkpoint_sha256=sha,journal_state=status['state'],restore_purpose=purpose,training_authorized=False)
        return config,receipt

    def training_window_evidence(self,epoch,*,current_only=False):
        from worker_projection import project_training,project_controls
        self._held();state=self.recover()
        if state['state'] not in {'LIGHT_REVIEW','FULL_REVIEW'} or state['epoch']!=epoch:
            raise RuntimeError('training evidence requires current review')
        if type(current_only) is not bool:raise ValueError('invalid current-only evidence flag')
        rows=[];bindings=[]
        for e in range(epoch if current_only else max(1,epoch-4),epoch+1):
            if os.environ.get('CASHEW_FAST_RESUME')=='1' and e<epoch:
                from fast_history_v1 import committed
                record=committed(self,e);metrics=record['training'];config=record['controls']
                meta=self._fs.json('epoch_%06d/trained.json'%e)
            else:
                d=self._dir(e);intent=self._fs.json('epoch_%06d/intent.json'%e)
                raw,meta=self._trained(d,e,intent['previous_checkpoint_sha256'],
                                       trained_raw=state['checkpoint'] if e==epoch else None)
                payload=safe_check(raw,e);metrics=payload['metrics'];config=payload['config']
            for key in ('source_target_actual_exposure','supplementary_actual_exposure','experimental_actual_updates'):
                if not isinstance(metrics.get(key),dict):raise ValueError('missing saved evidence: '+key)
            observed=metrics['experimental_actual_updates'].get('training_controls')
            if not isinstance(observed,dict):raise ValueError('actual training controls are unknown')
            # These are not governed by the per-epoch LR/weight schedules.
            # A disagreement means execution is unverified, not that a
            # requested intervention succeeded or that a cause is known.
            for key in ('temperature','confidence_cap','hard_threshold','main_weight_decay','D_weight_decay',
                        'D_steps_per_G_step','adversarial_projection_strength','supplementary_positive_scale',
                        'supplementary_occurrence_cap','supplementary_interval_steps','supplementary_horizon_epochs',
                        'hard_sample_weight','hard_sample_repeats',
                        'main_lr_finish_epochs','refiner_lr_multiplier','encoder_lr_multiplier','pyramid_lr_multiplier','fusion_lr_multiplier','decoder_lr_multiplier','domain_norm_lr_multiplier','bn_update_mode'):
                if key in observed and (key not in config or observed[key]!=config[key]):
                    raise ValueError('actual training control differs from checkpoint: '+key)
            training=project_training(metrics);controls=project_controls(metrics)
            if (type(training.get('steps')) is not int or training['steps']<1 or type(training.get('cls')) not in (int,float)
                or any(type(controls.get(k)) not in (int,float) for k in ('main_lr','D_lr','lambda_adv','temperature','confidence_cap'))):
                raise ValueError('missing essential per-epoch training facts')
            rows.append({'epoch':e,'training':training,'controls':controls})
            bindings.append({'epoch':e,'trained_checkpoint_sha256':meta['checkpoint_sha256']})
        return rows,bindings

    def collect_full_review(self,epoch,gan,common,model,device):
        import random,numpy as np
        from live_safe_review_evidence_v1 import collect
        self._held();state=self.recover()
        if state['state']!='FULL_REVIEW' or state['epoch']!=epoch:raise RuntimeError('full review requires current trained epoch')
        device=torch.device(device);py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state()
        cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None
        try:
            source,panel,audit=collect(self.job,gan,common,model,device,self._dir(epoch)/'trained_checkpoint.bin',epoch,
                                       checkpoint_reader=lambda:self._fs.read('epoch_%06d/trained_checkpoint.bin'%epoch))
            if (not isinstance(audit,dict) or audit.get('epoch')!=epoch or audit.get('checkpoint_sha256')!=digest(state['checkpoint'])
                or audit.get('training_authorized') is not False):raise ValueError('review is not bound to safe checkpoint')
            if self.recover()['state']!='FULL_REVIEW':raise RuntimeError('journal changed during review')
            from human_training_evidence_v1 import collect as collect_human
            from human_training_evidence_contract import bind
            human=collect_human(self.job,gan,model,device,epoch)
            bind(self.job,human,epoch,decode(state['checkpoint'])['state_dict'])
            observation={'epoch':epoch,'checkpoint_sha256':audit['checkpoint_sha256'],
                'source':source,'panel':panel,'input_snapshot_sha256':audit.get('input_snapshot_sha256'),'input_snapshot_count':audit.get('input_snapshot_count')}
            if human is not None:observation['human_training']=human
            from hard_sample_training import enabled as annotated_training
            if annotated_training(self.job):
                from hard_sample_evidence import review as annotation_review
                fit = annotation_review(self.job,gan,common,model,device,epoch,audit['checkpoint_sha256'])
                if fit is not None:
                    observation['annotated_training'] = fit
            self._fs.publish_relative('epoch_%06d/review_observation.json'%epoch,canonical(observation))
            return source,panel,audit
        except BaseException:self._poisoned=True;raise
        finally:
            random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)
            if cuda is not None:torch.cuda.set_rng_state(cuda,device)

    def prior_action_context(self,epoch):
        """Rebuild prior structured expectations only from verified receipts."""
        from worker_projection import project_committed_hypotheses
        state=self.recover()
        if state['state'] not in {'LIGHT_REVIEW','FULL_REVIEW'} or state['epoch']!=epoch:
            raise RuntimeError('prior action requires current review')
        monitor=None
        for e in range(epoch-1,max(0,epoch-6),-1):
            data=self._fs.read('epoch_%06d/commit.json'%e,missing_ok=True)
            if data is None:continue
            commit=json.loads(data)
            if commit.get('decision_kind')!='agent' or commit.get('action') not in {'ADJUST','PAUSE_ADV','RESUME_ADV','KEEP','CONTINUE'}:continue
            contract='safe_anomaly_review_candidate_v1' if commit['route']=='LIGHT_REVIEW' else 'safe_full_review_candidate_v1'
            stage='agent/sealed_v1/epoch_%03d/%s/transaction/'%(e,'light' if commit['route']=='LIGHT_REVIEW' else 'full')
            rec=self._out_fs.json(stage+'commit.json');raw=self._out_fs.read(stage+'decision.json');response=self._out_fs.read(stage+'response.json')
            if digest(raw)!=rec['decision_sha256'] or digest(response)!=rec['response_sha256']:
                raise ValueError('previous decision or response changed')
            evidence={'epoch':e,'decision_sha256':rec['decision_sha256']}
            hypotheses=project_committed_hypotheses(response,rec,raw,evidence,
                job_id=self.job['job_id'],run_id=self.job['run_id'],cutoff=epoch)
            prior_response={'response_bytes':response,'commit':rec,'job_id':self.job['job_id'],'run_id':self.job['run_id']}
            item=((raw,evidence),hypotheses,prior_response)
            if commit['action'] in {'KEEP','CONTINUE'}:
                if hypotheses and monitor is None:monitor=item
                continue
            return item
        return monitor if monitor is not None else (None,[],None)

    def trend_context(self,epoch):
        """Compare registered full reviews only after safe journal replay."""
        from review_trends_v1 import bind
        state=self.recover()
        if state['state']!='FULL_REVIEW' or state['epoch']!=epoch:raise RuntimeError('trend context requires current review')
        def record(e):
            prefix='epoch_%06d/'%e;obs=self._fs.json(prefix+'review_observation.json');meta=self._fs.json(prefix+'trained.json')
            if obs.get('epoch')!=e or obs.get('checkpoint_sha256')!=meta.get('checkpoint_sha256'):
                raise ValueError('review observation checkpoint mismatch')
            snapshot=obs.get('input_snapshot_sha256');count=obs.get('input_snapshot_count')
            if not isinstance(snapshot,str) or len(snapshot)!=64 or type(count) is not int or count<1:return None
            source=self._trend_artifact('source',e,obs['source']);panel=self._trend_artifact('panel',e,obs['panel'])
            audit={'version':'live_review_evidence_v1','epoch':e,'checkpoint_sha256':obs['checkpoint_sha256'],'input_snapshot_sha256':snapshot}
            binding,signature=bind(self.job['job_id'],self.job['run_id'],source,panel,audit)
            return {'source':source,'panel':panel,'binding':binding},signature
        current=record(epoch)
        if current is None:return None,set()
        prior=[]
        for e in range(1,epoch):
            prefix='epoch_%06d/'%e;obs=self._fs.read(prefix+'review_observation.json',missing_ok=True);commit=self._fs.read(prefix+'commit.json',missing_ok=True)
            if obs is not None and commit is not None and json.loads(commit).get('route')=='FULL_REVIEW':prior.append(e)
        previous=None
        if prior:
            e=prior[-1];stage='agent/sealed_v1/epoch_%03d/full/transaction/'%e
            request=self._out_fs.json(stage+'request.json');obs=self._fs.json('epoch_%06d/review_observation.json'%e)
            evidence=request.get('evidence') or {}
            if (evidence.get('source_validation',{}).get('values')!=obs.get('source')
                or evidence.get('unlabelled_panel',{}).get('values')!=obs.get('panel')):
                raise ValueError('prior review differs from registered observation')
            previous=record(e)
            if previous is None:return None,set()
            if evidence.get('current_review_binding_sha256')!=previous[1]:raise ValueError('prior review binding changed')
        records=[current]+([previous] if previous else [])
        trusted={(item['kind'],item['epoch'],item['sha256']) for r,_ in records for item in (r['source'],r['panel'])}
        bindings=sorted((self.job['job_id'],self.job['run_id'],r['binding']['epoch'],signature) for r,signature in records)
        return {'job_id':self.job['job_id'],'run_id':self.job['run_id'],'previous_review':previous[0] if previous else None,
                'current_binding':current[0]['binding'],'trusted_bindings':bindings,
                'current_checkpoint_sha256':current[0]['binding']['checkpoint_sha256']},trusted

    def current_review_context(self,epoch):
        """Keep the authenticated current binding without sending prior values."""
        ctx,trusted=self.trend_context(epoch)
        if ctx is None:return None,set()
        ctx=copy.deepcopy(ctx);ctx['previous_review']=None
        ctx['trusted_bindings']=[x for x in ctx['trusted_bindings'] if x[2]==epoch]
        return ctx,{x for x in trusted if x[1]==epoch}

    def _verify_review_request(self,epoch,request,route,expected,trusted,previous):
        from paired_trial_v1 import capability as trial_capability
        declared_trial = trial_capability(self.job,epoch=epoch) if route=='FULL_REVIEW' else None
        if request.get('matched_trial_capability') != declared_trial or expected.get('matched_trial_capability') != declared_trial:
            raise ValueError('trial capability differs from the owned job')
        history_required=expected.get('history_required',False)
        if type(history_required) is not bool:raise ValueError('invalid worker history requirement')
        rows,_=(self.training_window_evidence(epoch,current_only=True) if history_required else self.training_window_evidence(epoch))
        if history_required:
            if previous is not None or request.get('prior_committed_action') is not None:
                raise ValueError('history-required request must not preload a previous action')
            if route=='FULL_REVIEW' and (expected.get('prior_response') is not None or expected.get('hypotheses',()) or request.get('prior_structured_hypotheses')):
                raise ValueError('history-required request must not preload past hypotheses')
        else:
            prior,hypotheses,prior_response=self.prior_action_context(epoch)
            if previous!=prior:raise ValueError('previous action does not match committed worker history')
            if prior is None:
                if request.get('prior_committed_action') is not None:raise ValueError('unregistered prior action')
            else:
                from worker_projection import project_committed_action
                if request.get('prior_committed_action')!=project_committed_action(prior[0],prior[1],epoch):raise ValueError('prior action differs')
            if route=='FULL_REVIEW':
                if expected.get('prior_response')!=prior_response or expected.get('hypotheses',()) not in (hypotheses,tuple(hypotheses)):
                    raise ValueError('prior response or expectations differ')
                if request.get('prior_structured_hypotheses')!=hypotheses:raise ValueError('Agent hypotheses differ')
        evidence=request.get('evidence') if isinstance(request,dict) else None
        if route=='LIGHT_REVIEW':
            actual_pool=self.active_pool_evidence(epoch);advertised=evidence.get('active_automatic_pool') if isinstance(evidence,dict) else None
            if actual_pool is None:
                if advertised is not None:raise ValueError('unverified or inactive automatic pool evidence')
            elif not isinstance(advertised,dict) or advertised.get('epoch')!=epoch or advertised.get('values')!=actual_pool:
                raise ValueError('automatic pool evidence differs from checkpoint and exposure receipt')
        window=evidence.get('training_window') if isinstance(evidence,dict) else None
        if not isinstance(window,list) or len(window)!=len(rows):raise ValueError('incomplete measured training window')
        for actual,row in zip(window,rows):
            if not isinstance(actual,dict):raise ValueError('invalid Agent training window')
            for kind in ('training','controls'):
                item=actual.get(kind)
                if not isinstance(item,dict) or item.get('epoch')!=row['epoch'] or item.get('values')!=row[kind]:
                    raise ValueError('Agent training window differs from saved observations')
        if route=='FULL_REVIEW':
            obs=self._fs.json('epoch_%06d/review_observation.json'%epoch);trained=self.recover()['checkpoint']
            if (obs.get('epoch')!=epoch or obs.get('checkpoint_sha256')!=digest(trained)
                or evidence.get('source_validation',{}).get('values')!=obs.get('source')
                or evidence.get('unlabelled_panel',{}).get('values')!=obs.get('panel')):
                raise ValueError('Agent review differs from current source and panel measurements')
            from human_training_evidence_contract import bind,artifact
            human=obs.get('human_training')
            bind(self.job,human,epoch,decode(trained)['state_dict'])
            projected={k:v for k,v in artifact(human).items() if k!='kind'} if human is not None else None
            if evidence.get('human_training')!=projected or expected.get('human_training')!=(artifact(human) if human is not None else None):
                raise ValueError('human training request differs from the saved measurement')
            from hard_sample_contract import REVIEW_KIND, artifact as annotation_artifact
            annotation = annotation_artifact(epoch,obs[REVIEW_KIND]) if REVIEW_KIND in obs else None
            if (expected.get(REVIEW_KIND) != annotation or evidence.get(REVIEW_KIND) !=
                    ({k:v for k,v in annotation.items() if k != 'kind'} if annotation else None)):
                raise ValueError('Annotated training evidence differs from the bound review measurement')
            from human_negative_epoch10 import describe as describe_human_default
            declared=describe_human_default(decode(self.prepare_full_review(epoch)['checkpoint'])['config'])
            if expected.get('human_default')!=declared or evidence.get('human_supplementary_default')!=declared:
                raise ValueError('human-negative default brief differs from the committed configuration')
            ctx,required=(self.current_review_context(epoch) if history_required else self.trend_context(epoch))
            if ctx is None:
                if evidence.get('prior_review_trend') is not None or expected.get('trend_context') is not None:raise ValueError('unregistered prior trend')
            else:
                if expected.get('trend_context')!=ctx or not required.issubset(set(trusted)):raise ValueError('trend context differs')
                from review_trends_v1 import compare
                import hashlib
                binding=hashlib.sha256(json.dumps(ctx['current_binding'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                if evidence.get('current_review_binding_sha256')!=binding:raise ValueError('current review binding differs')
                current={'source':self._trend_artifact('source',epoch,obs['source']),
                         'panel':self._trend_artifact('panel',epoch,obs['panel']),'binding':ctx['current_binding']}
                value=compare(ctx['previous_review'],current,job_id=ctx['job_id'],run_id=ctx['run_id'],trusted_values=trusted,
                              trusted_bindings=set(tuple(x) for x in ctx['trusted_bindings']))
                if evidence.get('prior_review_trend')!=value:raise ValueError('trend values differ')

    def finish_agent(self,epoch,request,trusted,expected,**options):
        from history_router_candidate import route
        self._held();state=self.recover()
        kind={'LIGHT_REVIEW':'safe_anomaly_review_candidate_v1','FULL_REVIEW':'safe_full_review_candidate_v1'}.get(state['state'])
        if state['epoch']!=epoch or kind is None or request.get('contract_version')!=kind or request.get('epoch')!=epoch:
            raise RuntimeError('safe review request does not match journal')
        self._verify_review_request(epoch,request,state['state'],expected,trusted,options.get('previous'))
        pre=state['checkpoint']
        if state['state']=='FULL_REVIEW':pre,_=self._default_transition(epoch,pre,'before_agent')
        result=route(self.job,request,trusted,expected,pre,**options)
        post=result['post_checkpoint']
        if state['state']=='LIGHT_REVIEW':post,_=self._default_transition(epoch,post,'after_agent')
        return self._commit(epoch,post,'agent',result['decision']['action'])

    def _verify_committed_agent(self,epoch,request,answer,pre,decision,post):
        from sealed_safe_router_sim_v1 import _decision,_post
        if decision!=_decision(self.job,request,answer,pre,allow_stop=True) or not _equal(decode(post),decode(_post(pre,decision))):
            raise ValueError('committed Agent decision or post mismatch')
