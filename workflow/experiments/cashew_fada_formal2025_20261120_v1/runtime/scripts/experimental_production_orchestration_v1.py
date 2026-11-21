"""Owned Cashew training operations with explicit release/evidence hooks.

The base class denies execution. An authorized cohort's GroupWorker implements
ownership, skill pins, source/panel identity and stopping checks. This does not
certify the ancestry of historical transfer weights. The current owned path
supports same-run history, declared module diagnostics and isolated trial tools;
automatic-pool execution retains its independent gate.
"""
from formal_training_account import codex_binary
import copy
import io
from dataclasses import dataclass
from pathlib import Path

# No job flag, environment variable, caller callback or disk file can enable
# this unreleased code. A reviewed release must replace the implementation and
# establish independently authenticated provenance, not flip a job setting.
_CERTIFIED_RELEASE = None

class ProductionNotAuthorized(RuntimeError):
    pass

@dataclass
class RuntimeHandles:
    """Owning worker's already-constructed real training objects, no callbacks."""
    gan: object
    common: object
    model: object
    local_d: object
    global_d: object
    model_opt: object
    local_opt: object
    global_opt: object
    dataset: object
    source_count: int
    target_count: int
    supplementary_names: tuple
    supplementary_image_root: object
    supplementary_label_root: object
    device: object

class ExperimentalProductionWorker:
    """Concrete operations for a future authenticated worker, currently shut."""
    def __init__(self,job,journal,handles,*,pool_bridge=None):
        if not isinstance(job,dict):raise ValueError('worker job must be a mapping')
        policy=job.get('conditional_negative_policy') or {}
        if policy.get('minimum_activation_fraction') is not None:
            maximum=job.get('max_epoch')
            interval=job.get('review_interval',5)
            if type(maximum) is not int or maximum <= 0:
                raise ValueError('conditional negative policy requires a positive integer job.max_epoch before training')
            if maximum != getattr(journal,'max_epoch',None):
                raise ValueError('job.max_epoch must match the owned training journal limit')
            from area_alert_v1 import late_start
            late_start(policy,maximum,interval)
        self.job,self.journal,self.handles,self.pool_bridge=job,journal,handles,pool_bridge
        self._human_library_check=None

    def _authorize(self):
        # This check comes before any external data, model construction, file
        # writes, callback or network operation. It cannot be set by a job.
        if _CERTIFIED_RELEASE is None:
            raise ProductionNotAuthorized('experimental Cashew production release and blind provenance are not certified')
        raise ProductionNotAuthorized('release authorization implementation is not complete')

    def _authorize_review_sources(self,epoch):
        """No certified source/panel/trajectory authorization exists yet."""
        raise ProductionNotAuthorized('review evidence provenance is not certified')

    def _authorize_stop(self,epoch):
        """Real stop eligibility is not yet independently established."""
        raise ProductionNotAuthorized('STOP eligibility is not certified')

    def _authorize_selection(self):
        raise ProductionNotAuthorized('selection evidence provenance is not certified')

    def _authorize_pool_sources(self):
        """No real catalog/mask/training lineage attestor is certified yet."""
        raise ProductionNotAuthorized('automatic pool provenance is not certified')

    def _validate_pool_resource_budget(self):
        budget=self.job.get('experimental_catalog_snapshot_budget')
        if (not isinstance(budget,dict) or set(budget)!={'max_images','max_bytes'}
            or any(type(budget[k]) is not int or budget[k]<1 for k in ('max_images','max_bytes'))):
            raise ProductionNotAuthorized('explicit catalog image and byte budget required')
        return budget

    def _materialize_pool_change(self,request,decision,pre):
        """Derive selection from the already completed, bound review turn."""
        action=(decision.get('automatic_negative') or {}).get('action')
        if action=='DISABLE':return {'action':'DISABLE'}
        if action!='ACTIVATE':raise ValueError('pool materialization requires an approved action')
        from protected_catalog_handoff_candidate_v1 import materialize
        return materialize(self.journal,pre,decision)

    def _pool_owner(self):
        import os
        from safe_pool_journal_router_sim_v1 import SafePoolJournal
        from unified_epoch_pool_bridge_sim_v1 import UnifiedEpochPool
        if (type(self.journal) is not SafePoolJournal or type(self.pool_bridge) is not UnifiedEpochPool
            or self.pool_bridge.journal is not self.journal):
            raise ProductionNotAuthorized('owning single-authority pool journal is required')
        actual=Path(os.path.abspath(self.pool_bridge.assets));expected=Path(os.path.abspath(self.job['output_root']))/'automatic_negative'/'assets'
        if actual!=expected:raise ProductionNotAuthorized('pool assets must use the fixed run-owned directory')
        return self.pool_bridge

    def initialize_from_evidence(self,request,trusted,expected,checkpoint,*,skill_sha256,verifier_sha256,reference_hashes):
        """Bind the real explicit-skill initialization before any epoch.

        The owning worker must already hold the journal lease and have built
        the point-sealed initialization from authenticated source/panel data.
        Existing transaction bytes are reused exactly on restart; an unknown
        claimed turn is never resent. No fresh authorization exists yet.
        """
        self._authorize()
        self._authorize_review_sources(0)
        from sealed_safe_epoch_journal_v1 import SafeTrainingJournal
        from safe_initialization_transaction_candidate_v1 import initialize,recover_pre
        if type(self.journal) is not SafeTrainingJournal or self.pool_bridge is not None:
            raise ProductionNotAuthorized('initialization requires the owned safe journal without an automatic pool')
        if checkpoint is None:
            checkpoint=recover_pre(self.job,self.journal,request,trusted,expected)
        return initialize(self.job,self.journal,request,trusted,expected,checkpoint,
            codex_binary=codex_binary(),skill_sha256=skill_sha256,
            verifier_sha256=verifier_sha256,reference_hashes=reference_hashes)

    def initialize_from_frozen_period(self,source_job,source_journal,manifest_sha256,request,trusted,expected,construct,*,recover=False,skill_sha256,verifier_sha256,reference_hashes):
        """Candidate transfer initialization; both journal owners must be held.

        The normal authorization gates still apply. Recovery requires the exact
        previously persisted epoch-zero transaction, never a new random state.
        The caller must attest source/target inputs before using this API.
        """
        self._authorize()
        self._authorize_review_sources(0)
        self.journal._held()
        if type(recover) is not bool:raise ValueError('recover must be a boolean')
        from two_period_frozen_handoff_candidate_v1 import prepare,initialize
        receipt=prepare(source_job,self.job,source_journal,manifest_sha256)['transfer_record']
        if expected.get('transfer_record')!=receipt:
            raise ValueError('initialization request does not match frozen source transfer')
        if recover:
            checkpoint=None
        else:
            if not callable(construct):raise ValueError('fresh model factory required')
            checkpoint,record=initialize(source_job,self.job,source_journal,manifest_sha256,construct)
            if record!=receipt:raise ValueError('source transfer changed while initializing')
        return self.initialize_from_evidence(request,trusted,expected,checkpoint,
            skill_sha256=skill_sha256,verifier_sha256=verifier_sha256,reference_hashes=reference_hashes)

    def run_until_review(self):
        """Advance persisted epochs until a real review or safety stop is due.

        Caller owns the journal lease and must authenticate the real inputs.
        This does not create review evidence, choose a model or authorize a run.
        """
        self._authorize()
        if self.pool_bridge is not None:
            raise ProductionNotAuthorized('automatic-pool loop not integrated')
        while True:
            state=self.journal.recover()
            kind=state['state'];epoch=state['epoch']
            if kind=='READY':
                self.journal.begin(epoch+1)
            elif kind=='TRAIN_REQUIRED':
                self.train_one_epoch(epoch)
            elif kind=='SKIP':
                self.journal.finish_skip(epoch)
            elif kind in ('LIGHT_REVIEW','FULL_REVIEW','SAFETY_STOP','STOPPED','STOPPED_AT_LIMIT'):
                return state
            else:
                raise RuntimeError('unrecognized journal state: '+str(kind))

    def run_review_cycles(self,*,skill_sha256,verifier_sha256,reference_hashes,allow_stop=False):
        """Advance training and route registered reviews until a terminal state.

        Requires owning the journal lease. Source authorization and the
        simulation-only router remain independent blockers. STOP is opt-in and
        separately gated; selection, pools and experiments remain separate.
        """
        self._authorize()
        if type(allow_stop) is not bool:raise ValueError('allow_stop must be a boolean')
        while True:
            state=self.run_until_review();epoch=state['epoch'];kind=state['state']
            if kind in ('SAFETY_STOP','STOPPED','STOPPED_AT_LIMIT'):
                return state
            if kind=='FULL_REVIEW':
                self._authorize_review_sources(epoch)
                self.journal._held()
                self.before_full_review(epoch, state)
                path='epoch_%06d/review_observation.json'%epoch
                if self.journal._fs.read(path,missing_ok=True) is None:
                    h=self.handles
                    self.journal.restore_live(device=h.device,model=h.model,local_d=h.local_d,global_d=h.global_d,
                        model_opt=h.model_opt,local_opt=h.local_opt,global_opt=h.global_opt,purpose='review')
                    self.collect_full_review(epoch)
                request,trusted,expected,previous=self.prepare_full_review_request(epoch,allow_stop=allow_stop)
            elif kind=='LIGHT_REVIEW':
                request,trusted,expected,previous=self.prepare_light_review_request(epoch)
            else:
                raise RuntimeError('unexpected review state: '+str(kind))
            self.route_review(epoch,request,trusted,expected,previous=previous,skill_sha256=skill_sha256,
                verifier_sha256=verifier_sha256,reference_hashes=reference_hashes)

    def before_full_review(self, epoch, state):
        """Hook for a worker-owned acquisition at a committed epoch boundary."""

    def train_one_epoch(self,epoch):
        """Restore, train and persist an actual epoch, once authorized.

        Requires the owner to hold the journal's full worker lease. The single
        existing full checkpoint is the exact pre-epoch state. Automatic pool
        execution remains behind a separate unconditional provenance gate.
        """
        self._authorize()
        from sealed_safe_epoch_journal_v1 import SafeTrainingJournal
        from native_safe_checkpoint_codec_v1 import encode,decode
        pool_mode=self.pool_bridge is not None
        if pool_mode:self._authorize_pool_sources();self._validate_pool_resource_budget();self._pool_owner()
        elif type(self.journal) is not SafeTrainingJournal:raise ProductionNotAuthorized('only the new-format safe training journal is supported')
        h=self.handles
        if h.supplementary_names:
            import hashlib
            from two_period_human_library_guard_v1 import validate as validate_human_library
            spec=self.job.get('period_library_manifest') or {}
            path=spec.get('path');expected=spec.get('sha256')
            if not isinstance(path,str) or not isinstance(expected,str) or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
                raise ValueError('period library manifest missing or changed')
            signature=(self.job.get('target_year'),path,expected,tuple(h.supplementary_names),str(h.supplementary_image_root),str(h.supplementary_label_root))
            if self._human_library_check is None:
                receipt=validate_human_library(self.job,h.supplementary_names,h.supplementary_image_root,h.supplementary_label_root)
                self._human_library_check=(signature,receipt)
            elif self._human_library_check[0]!=signature:
                raise ValueError('period library changed while training')
        import numpy as np
        import torch
        import train_expert_v4 as worker
        from full_state_finiteness_v1 import check as check_full_state
        from experimental_gate_v1 import (StepRecorder,classify,epoch_integrity,
            automatic_context,supplementary_context,fingerprint,stable_config,FIELDS,validate)
        from bounded_recorder_candidate_v1 import BoundedRecorder
        from schedule_v4 import apply_to_optimizers
        from sealed_pool_integration_contract_v1 import require_no_unintegrated_pool
        if not pool_mode:require_no_unintegrated_pool(self.job)
        policy=validate(self.job)
        if not policy or policy.get('mode')!='enforce':raise ValueError('explicit enforced policy required')
        state=self.pool_bridge.recover() if pool_mode else self.journal.recover()
        if state['state']!='TRAIN_REQUIRED' or state['epoch']!=epoch:raise RuntimeError('journal does not require this epoch')
        h=self.handles
        config,_=self.journal.restore_live(device=h.device,model=h.model,local_d=h.local_d,global_d=h.global_d,
            model_opt=h.model_opt,local_opt=h.local_opt,global_opt=h.global_opt,**({} if pool_mode else {'purpose':'training'}))
        if pool_mode:
            import safe_pool_checkpoint_codec_sim_v1 as poolcodec
            from unified_epoch_pool_bridge_sim_v1 import _equal
            logical=poolcodec.decode(state['checkpoint'])
            restored=dict(config);restored.pop(poolcodec.FIELD,None)
            if not _equal(restored,logical['config']):raise ValueError('restored pool training config differs from checkpoint')
            config=logical['config'];pool=logical['pool']
        else:require_no_unintegrated_pool(self.job,config)
        models={'G':h.model,'D_local':h.local_d,'D_global':h.global_d}
        optimizers={'G':h.model_opt,'D_local':h.local_opt,'D_global':h.global_opt}
        check_full_state(models,optimizers)
        scheduled=apply_to_optimizers(config,epoch,h.model_opt,h.local_opt,h.global_opt)
        from module_dynamics_v1 import EpochDynamics, VERSION as DYNAMICS_VERSION
        dynamics = (EpochDynamics(h.model,h.model_opt)
                    if self.job.get('module_dynamics_policy') == DYNAMICS_VERSION else None)
        detailed = None
        if self.job.get('module_dynamics_policy') == 'module_observation_v2':
            from module_observer_v2 import Observer, anchor
            from expert_policy_v4 import flat
            signature = (flat(config), config.get('schedule_state'))
            cached = getattr(self, '_module_anchor_cache', None)
            if cached is None or cached[0] != signature:
                initial, intervention = anchor(self.job['output_root'], models, epoch-1, config)
                self._module_anchor_cache = (copy.deepcopy(signature), initial, intervention)
            else: _, initial, intervention = cached
            detailed = Observer(models, optimizers, initial, intervention)
        epoch_config=copy.deepcopy(config);epoch_config.update(scheduled)
        seed=int(self.job['seed']);epoch_config['_negative_epoch_start']=epoch-1;epoch_config['_negative_seed']=seed+epoch*100000
        sampling=policy.get('recorder_sampling')
        recorder=BoundedRecorder(count=sampling['elements_per_tensor'],max_tensors=sampling['max_tensors_per_group']) if sampling else StepRecorder()
        train_args=(h.gan,h.common,h.model,h.local_d,h.global_d,h.model_opt,h.local_opt,h.global_opt,
            h.dataset,h.source_count,h.target_count,h.supplementary_names,h.supplementary_image_root,
            h.supplementary_label_root,epoch_config,h.device,np.random.RandomState(seed+epoch))
        if pool_mode and pool is not None:
            from safe_pool_epoch_adapter_candidate_v1 import run_epoch
            metrics,exposure=run_epoch(self.pool_bridge,epoch,state['checkpoint'],train_args,{'experimental_recorder':recorder})
        else:
            from hard_sample_training import EpochSamples, enabled as annotated_training
            extra = {}
            if annotated_training(self.job):
                steps = max((n + worker.BATCH_SIZE - 1) // worker.BATCH_SIZE for n in (h.source_count, h.target_count))
                extra['annotated_samples'] = EpochSamples(self.job, epoch, epoch_config, steps, h.common, h.device)
            metrics=worker.train_epoch(*train_args,automatic_negative_runtime=None,experimental_recorder=recorder,**extra)
            if pool_mode:
                count=metrics['experimental_actual_updates']['steps']['G']
                if type(count) is not int or count<1:raise ValueError('actual completed generator steps required')
                exposure={'mode':'no_pool','positions':0,'completed_g':count,'auxiliary_steps':0,'patch_occurrences':0,'pixel_occurrences':0}
        check_full_state(models,optimizers)
        metrics['scheduled_controls']=scheduled
        if detailed is not None: metrics['module_observation'] = detailed.finish()
        if dynamics is not None:
            metrics['module_dynamics']=dynamics.finish()
        metrics['experimental_actual_updates']['training_controls']={k:epoch_config[k] for k in (
            'temperature','confidence_cap','hard_threshold','lambda_adv','main_lr','D_lr','main_weight_decay',
            'D_weight_decay','D_steps_per_G_step','adversarial_projection_strength','supplementary_loss_weight',
            'supplementary_positive_scale','supplementary_occurrence_cap','supplementary_interval_steps',
            'supplementary_horizon_epochs','main_lr_finish_epochs','refiner_lr_multiplier',
            'encoder_lr_multiplier','pyramid_lr_multiplier','fusion_lr_multiplier','decoder_lr_multiplier','domain_norm_lr_multiplier','bn_update_mode',
            'representation_anchor_weight','refiner_anchor_weight','decoder_anchor_weight',
            'hard_sample_weight','hard_sample_repeats') if k in epoch_config}
        def pool_context(pool_state,observed):
            if pool_state is None:return {'mode':'no_pool'},observed.get('automatic_negative') is not None
            a=observed.get('automatic_negative')
            if not isinstance(a,dict) or a.get('configured_weight')!=pool_state['weight']:
                return None,True
            return {'mode':'active','generation':pool_state['generation'],'weight':pool_state['weight']},False
        auto,auto_changed=pool_context(pool,metrics) if pool_mode else automatic_context(False,False,config,metrics)
        if pool_mode and pool is not None:
            auto_changed=bool(auto_changed or exposure['mode']=='paused'
                              or pool['cursor']+exposure['positions']>=pool['horizon'])
        supp,supp_changed=supplementary_context(metrics)
        control_fingerprint=fingerprint(stable_config(config),{'automatic':auto,'supplementary':supp})
        history=[]
        import os
        fast_history=os.environ.get('CASHEW_FAST_RESUME')=='1' and not pool_mode
        if fast_history:
            from fast_history_v1 import committed,restore_config,verify_current_config
            verify_current_config(config,self.job)
        for e in range(max(1,epoch-policy['lookback']),epoch):
            if fast_history:
                record=committed(self.journal,e);old=record['training'];old_config=restore_config(record['controls'],self.job)
                payload=None
            else:
                d=self.journal._dir(e);intent=self.journal._fs.json('epoch_%06d/intent.json'%e)
                raw,_=self.journal._trained(d,e,intent['previous_checkpoint_sha256'])
                payload=poolcodec.decode(raw) if pool_mode else decode(raw)
                old=payload['metrics'];old_config=payload['config']
            a,ach=pool_context(payload['pool'],old) if pool_mode else automatic_context(False,False,old_config,old)
            s,sch=supplementary_context(old)
            history.append({'epoch':e,'facts':{k:old.get(k) for k in FIELDS},
                'config_hash':fingerprint(stable_config(old_config),{'automatic':a,'supplementary':s}),
                'stable':not (ach or sch)})
        gate=classify(epoch,metrics,control_fingerprint,history,policy,active_aux=bool(auto_changed or supp_changed),
            pending_action=False,adversarial_active=epoch_config.get('adversarial_state')=='active')
        integrity=epoch_integrity(metrics,epoch_config,scheduled)
        if integrity['route']=='SAFETY_STOP':gate={'route':'SAFETY_STOP'}
        elif integrity['route']=='LIGHT_REVIEW' and gate['route']!='FULL_REVIEW':gate={'route':'LIGHT_REVIEW'}
        if epoch==self.journal.max_epoch and gate['route']!='SAFETY_STOP':gate={'route':'FULL_REVIEW'}
        from human_negative_epoch10 import review_due
        if review_due(self.job,epoch) and gate['route']!='SAFETY_STOP':
            gate={'route':'FULL_REVIEW','reason':'Agent decides human supervision before epoch 10; zero weight is valid'}
        # Preserve the decision inputs and integrity details with the actual
        # epoch. A restart must not invent the reason an Agent was called.
        metrics['experimental_gate_decision']={'route':gate['route'],
            'reason':gate.get('reason'),'integrity':integrity}
        cp=worker.checkpoint_payload(epoch,h.model,h.local_d,h.global_d,h.model_opt,h.local_opt,h.global_opt,config,metrics)
        if pool_mode:
            cp.update(pool=pool,retired=logical['retired'])
            return self.pool_bridge.finish_training(epoch,poolcodec.encode(cp),gate['route'],exposure)
        return self.journal.finish_training(epoch,encode(cp),gate['route'])

    def collect_full_review(self,epoch):
        """Call the actual source and registered unlabelled-panel producer."""
        self._authorize()
        if self.pool_bridge is not None:
            self._authorize_pool_sources();self._validate_pool_resource_budget();self._pool_owner()
        self._authorize_review_sources(epoch)
        from sealed_safe_epoch_journal_v1 import SafeTrainingJournal
        if self.pool_bridge is None and type(self.journal) is not SafeTrainingJournal:raise ProductionNotAuthorized('safe journal required')
        h=self.handles
        return self.journal.collect_full_review(epoch,h.gan,h.common,h.model,h.device)

    def prepare_full_review_request(self,epoch,*,allow_stop=False):
        """Project current journal observations into a bound review request.

        The separate observation collector and source-authorization gate must
        have succeeded first. This does not itself call a model or authorize a
        live run; STOP requires its own authorization. Experiments and automatic
        pools remain unavailable.
        """
        self._authorize();self._authorize_review_sources(epoch)
        if type(allow_stop) is not bool:raise ValueError('allow_stop must be a boolean')
        if allow_stop:self._authorize_stop(epoch)
        if self.pool_bridge is not None:
            raise ProductionNotAuthorized('automatic-pool request not integrated')
        from expert_policy_v4 import parameter_bounds
        from native_safe_checkpoint_codec_v1 import decode
        from review_contract import build_full_review
        from sealed_packet import values_digest
        journal=self.journal;journal._held()
        state=journal.recover()
        if state['state']!='FULL_REVIEW' or state['epoch']!=epoch:
            raise RuntimeError('full review is not due')
        import os
        history_required=os.environ.get('CASHEW_HISTORY_REQUIRED','0')=='1'
        rows,_=(journal.training_window_evidence(epoch,current_only=True) if history_required else journal.training_window_evidence(epoch))
        obs=journal._fs.json('epoch_%06d/review_observation.json'%epoch)
        prepared=journal.prepare_full_review(epoch)
        prepared_config=decode(prepared['checkpoint'])['config']
        bounds=parameter_bounds(prepared_config)
        from human_negative_epoch10 import describe as describe_human_default
        human_default=describe_human_default(prepared_config)
        def art(kind,e,values):
            return {'kind':kind,'epoch':e,'values':values,'sha256':values_digest(kind,e,values)}
        items=[{k:art(k,r['epoch'],r[k]) for k in ('training','controls')} for r in rows]
        source=art('source',epoch,obs['source']);panel=art('panel',epoch,obs['panel'])
        trusted={(a['kind'],a['epoch'],a['sha256']) for row in items for a in row.values()}
        trusted|={(a['kind'],a['epoch'],a['sha256']) for a in (source,panel)}
        from human_training_evidence_contract import artifact as human_artifact
        human=human_artifact(obs['human_training']) if 'human_training' in obs else None
        if human is not None:trusted.add(('human_training',epoch,human['sha256']))
        trend,extra=(journal.current_review_context(epoch) if history_required else journal.trend_context(epoch));trusted|=extra
        previous,hypotheses,prior_response=((None,[],None) if history_required else journal.prior_action_context(epoch))
        actions=['KEEP','ADJUST']+(['STOP'] if allow_stop else [])
        from hard_sample_contract import REVIEW_KIND, artifact as annotation_artifact
        annotation = annotation_artifact(epoch, obs[REVIEW_KIND]) if REVIEW_KIND in obs else None
        if annotation is not None:
            trusted.add((REVIEW_KIND,epoch,annotation['sha256']))
        expected={'epoch':epoch,'bounds':bounds,'actions':actions,'trend_context':trend,
                  'hypotheses':hypotheses,'prior_response':prior_response,'human_default':human_default,'human_training':human}
        if annotation is not None:
            expected[REVIEW_KIND] = annotation
        if history_required:expected['history_required']=True
        expected['strict_schedule_schema']=True
        from paired_trial_v1 import capability as trial_capability
        trial_policy = trial_capability(self.job,epoch=epoch)
        if trial_policy is not None: expected['matched_trial_capability'] = trial_policy
        from cosine_finish_v1 import enabled as finish_enabled
        schedules={key:['hold','resume','set_phase','set_phase_hold'] for key in ('main_lr','D_lr','lambda_adv','supplementary_loss_weight') if key in prepared_config.get('schedule_state',{}) and key in bounds}
        if finish_enabled(prepared_config) and not prepared_config.get('cosine_finish_state'):
            schedules['main_lr'].append('start_cosine_finish')
        expected['schedules']=schedules
        request=build_full_review(epoch,items,source,panel,trusted,bounds,actions,
            previous=previous,hypotheses=hypotheses,prior_response=prior_response,trend_context=trend,human_default=human_default,history_required=history_required,strict_schedule_schema=True,human_training=human,matched_trial_capability=trial_policy,schedules=schedules,annotated_training=annotation)
        journal._verify_review_request(epoch,request,'FULL_REVIEW',expected,trusted,previous)
        return request,trusted,expected,previous

    def prepare_light_review_request(self,epoch):
        """Build an anomaly request from the saved epoch and its gate reason."""
        self._authorize();self._authorize_review_sources(epoch)
        if self.pool_bridge is not None:
            raise ProductionNotAuthorized('automatic-pool request not integrated')
        from expert_policy_v4 import parameter_bounds
        from native_safe_checkpoint_codec_v1 import decode
        from review_contract import build_anomaly_review
        from sealed_packet import values_digest
        journal=self.journal;journal._held();state=journal.recover()
        if state['state']!='LIGHT_REVIEW' or state['epoch']!=epoch:
            raise RuntimeError('light review is not due')
        cp=decode(state['checkpoint']);gate=cp['metrics'].get('experimental_gate_decision')
        if not isinstance(gate,dict) or gate.get('route')!='LIGHT_REVIEW':
            raise ValueError('saved light-review reason missing')
        integrity=gate.get('integrity') or {};reason=gate.get('reason') or ''
        if integrity.get('missing') or reason=='missing training metric':trigger='missing_training_metric'
        elif reason=='training metric outside physical range':trigger='metric_out_of_range'
        elif reason in ('unexpected missing or nonconsecutive history','invalid prior facts'):
            trigger='unexpected_missing_history'
        elif reason.startswith('deviation '):trigger='training_metric_deviation'
        elif reason in ('adversarial branch state unknown','auxiliary supervision, action response, or config state unknown/active'):
            trigger='unexpected_auxiliary_state'
        else:raise ValueError('unrecognized or missing light-review trigger')
        import os
        history_required=os.environ.get('CASHEW_HISTORY_REQUIRED','0')=='1'
        rows,_=(journal.training_window_evidence(epoch,current_only=True) if history_required else journal.training_window_evidence(epoch))
        def art(kind,e,values):
            return {'kind':kind,'epoch':e,'values':values,'sha256':values_digest(kind,e,values)}
        items=[{k:art(k,r['epoch'],r[k]) for k in ('training','controls')} for r in rows]
        trusted={(a['kind'],a['epoch'],a['sha256']) for row in items for a in row.values()}
        all_bounds=parameter_bounds(cp['config'])
        bounds={k:all_bounds[k] for k in ('main_lr','D_lr','lambda_adv') if k in all_bounds}
        previous=None if history_required else journal.prior_action_context(epoch)[0]
        expected={'epoch':epoch,'bounds':bounds,'trigger':trigger}
        if history_required:expected['history_required']=True
        expected['strict_schedule_schema']=True
        request=build_anomaly_review(epoch,items,trusted,bounds,trigger,previous=previous,history_required=history_required,strict_schedule_schema=True)
        journal._verify_review_request(epoch,request,'LIGHT_REVIEW',expected,trusted,previous)
        return request,trusted,expected,previous

    def route_review(self,epoch,request,trusted,expected,*,skill_sha256,verifier_sha256,reference_hashes,previous=None):
        """Use sealed structured Cashew binding; optional features stay blocked."""
        self._authorize()
        pool=self.pool_bridge is not None
        if pool:self._authorize_pool_sources();self._validate_pool_resource_budget();self._pool_owner()
        if (request.get('contract_version') not in ('safe_anomaly_review_candidate_v1','safe_full_review_candidate_v1')
            or (not pool and (request.get('automatic_selection_enabled') or (request.get('evidence') or {}).get('active_automatic_pool') is not None))):
            raise ProductionNotAuthorized('STOP/SELECT/branches/automatic pools are not certified')
        self._authorize_review_sources(epoch)
        from sealed_safe_epoch_journal_v1 import SafeTrainingJournal
        if not pool and type(self.journal) is not SafeTrainingJournal:raise ProductionNotAuthorized('safe journal required')
        actions=((request.get('schema') or {}).get('properties') or {}).get('action',{}).get('enum') or []
        allow_stop='STOP' in actions
        if allow_stop:self._authorize_stop(epoch)
        if pool:
            if allow_stop:raise ProductionNotAuthorized('STOP with automatic pool is not integrated')
            if request.get('automatic_selection_enabled'):
                import math
                summary=((request.get('evidence') or {}).get('automatic_negative_candidates') or {}).get('values') or {}
                count=request.get('automatic_candidate_limit');weight=request.get('automatic_max_weight')
                if (type(count) is not int or count<1 or summary.get('candidate_count')!=count
                    or type(weight) not in (int,float) or not math.isfinite(weight) or not 0<weight<=.005
                    or expected.get('automatic_max_weight')!=weight or expected.get('automatic_enabled') is not True):
                    raise ValueError('actual catalog count and allowed dose must be explicitly bound')
                from protected_catalog_handoff_candidate_v1 import bind_request
                bind_request(self.journal,request,trusted,expected,previous=previous)
            from safe_pool_journal_router_sim_v1 import ordinary_post
            bridge=self.pool_bridge;state=bridge.recover();before=bridge._cp_state(state['checkpoint'])
            def transition(pre,decision):
                action=(decision.get('automatic_negative') or {}).get('action')
                if action not in ('ACTIVATE','DISABLE'):return ordinary_post(pre,decision)
                change=self._materialize_pool_change(request,decision,pre)
                if not isinstance(change,dict) or change.get('action')!=action:raise ValueError('materialized pool action differs from decision')
                return bridge._pool_action(epoch,pre,decision,change,ordinary_post)
            def verify_pool(checked,decision,post):
                if checked!=epoch:return False
                bridge._audit_pool_action(epoch,before,post);return True
            return self.journal.finish_agent(epoch,request,trusted,expected,transition=transition,pool_receipt_verify=verify_pool,
                codex_binary=codex_binary(),skill_sha256=skill_sha256,verifier_sha256=verifier_sha256,
                reference_hashes=reference_hashes,previous=previous)
        from paired_trial_v1 import owned_service
        with owned_service(self,epoch,request):
            return self.journal.finish_agent(epoch,request,trusted,expected,codex_binary=codex_binary(),
                skill_sha256=skill_sha256,verifier_sha256=verifier_sha256,reference_hashes=reference_hashes,previous=previous,allow_stop=allow_stop)

    def freeze_stop_and_candidates(self,transaction_dir,candidates):
        """Bind STOP and candidate checkpoint hashes before any target eval."""
        self._authorize()
        self._authorize_review_sources(None);self._authorize_stop(None);self._authorize_selection()
        from sealed_safe_epoch_journal_v1 import SafeTrainingJournal
        from sealed_worker_adapter_v1 import stage_for
        from native_safe_checkpoint_codec_v1 import decode
        from stop_freeze_transaction import publish_stop_from_transaction,freeze_from_transaction
        if type(self.journal) is not SafeTrainingJournal:raise ProductionNotAuthorized('safe journal required')
        state=self.journal.recover()
        if state['state'] not in ('STOPPED','STOPPED_AT_LIMIT'):raise ValueError('same-run STOP not committed')
        epoch=state['epoch'];root=Path(self.job['output_root']).resolve()
        expected=(stage_for(self.job,{'contract_version':'safe_full_review_candidate_v1','epoch':epoch})/'transaction').resolve()
        supplied=(root/transaction_dir).resolve() if not Path(transaction_dir).is_absolute() else Path(transaction_dir).resolve()
        if supplied!=expected:raise ValueError('STOP transaction is not owned by current journal')
        from checkpoint_selection_contract_v2 import enabled as complete_selection
        from checkpoint_selection_evidence_v2 import checkpoint_relative,collect as candidate_evidence
        allow_zero=complete_selection(self.job)
        if not isinstance(candidates,list) or not candidates:raise ValueError('no candidate checkpoints')
        for item in candidates:
            if not isinstance(item,dict) or set(item)!={'epoch','path'} or type(item['epoch']) is not int or not (0 if allow_zero else 1)<=item['epoch']<=epoch:
                raise ValueError('invalid candidate')
            path=Path(item['path']);path=(root/path).resolve() if not path.is_absolute() else path.resolve()
            required=(root/checkpoint_relative(item['epoch'])) if allow_zero else self.journal._dir(item['epoch']).resolve()/'trained_checkpoint.bin'
            if path!=required:raise ValueError('candidate is not a same-run checkpoint')
            if allow_zero:candidate_evidence(root,self.job['job_id'],self.job['run_id'],item['epoch'],include_annotation_fit=False)
        def safe_loader(path):return decode(Path(path).read_bytes())
        publish_stop_from_transaction(root,supplied,self.job['job_id'],self.job['run_id'],max_epoch=self.journal.max_epoch,checkpoint_loader=safe_loader)
        from predictor_artifacts_v3 import enabled as compact_enabled,freeze_candidates as freeze_compact
        if compact_enabled(self.job):return freeze_compact(root,self.job,candidates,epoch)
        return freeze_from_transaction(root,self.job['job_id'],self.job['run_id'],candidates,max_epoch=self.journal.max_epoch,checkpoint_loader=safe_loader,allow_epoch_zero=allow_zero)

    def select_checkpoint(self,request,trusted,expected,*,manifest_sha256,skill_sha256,verifier_sha256,reference_hashes):
        """Select solely among frozen same-run source/unlabelled evidence."""
        self._authorize();self._authorize_review_sources(None);self._authorize_stop(None);self._authorize_selection()
        import json
        from decision_transaction import read_json,digest
        from native_safe_checkpoint_codec_v1 import decode
        from sealed_safe_epoch_journal_v1 import SafeTrainingJournal
        from sealed_safe_selection_router_candidate_v1 import route
        from selection_freeze import hash_file
        if type(self.journal) is not SafeTrainingJournal:raise ProductionNotAuthorized('safe journal required')
        state=self.journal.recover()
        if state['state'] not in ('STOPPED','STOPPED_AT_LIMIT'):raise ValueError('same-run STOP required')
        root=Path(self.job['output_root']);manifest_path=root/'frozen_candidates.json'
        if manifest_path.is_symlink() or hash_file(manifest_path)!=manifest_sha256:raise ValueError('candidate manifest changed')
        manifest=json.loads(manifest_path.read_text())
        if (manifest.get('job_id')!=self.job['job_id'] or manifest.get('run_id')!=self.job['run_id']
            or manifest.get('stop_epoch')!=state['epoch']):raise ValueError('candidate manifest belongs to another run or stop')
        observations=request.get('candidate_observations')
        if not isinstance(observations,list) or [x.get('epoch') for x in observations]!=[x['epoch'] for x in manifest.get('candidates',[])]:
            raise ValueError('selection evidence differs from frozen candidates')
        from checkpoint_selection_contract_v2 import enabled as complete_selection,VERSION
        if complete_selection(self.job):
            from checkpoint_selection_evidence_v2 import verify_frozen_candidates
            from request_reconstruction import verify_request
            if expected.get('selection_policy')!=VERSION:raise ValueError('worker selection policy missing')
            verify_request(request,trusted,expected=expected)
            verify_frozen_candidates(self.job,self.journal,request,manifest)
            return route(self.job,request,trusted,expected,state['checkpoint'],manifest_sha256=manifest_sha256,
                codex_binary=codex_binary(),skill_sha256=skill_sha256,verifier_sha256=verifier_sha256,reference_hashes=reference_hashes)
        if request.get('selection_policy') is not None:raise ValueError('unadvertised complete selection')
        for item,frozen in zip(observations,manifest['candidates']):
            epoch=item['epoch'];values=item.get('values');d=self.journal._dir(epoch)
            if not isinstance(values,dict):raise ValueError('missing candidate evidence')
            path=d/'trained_checkpoint.bin';expected_path=(root/frozen['path']).resolve()
            if (path.is_symlink() or expected_path!=path.resolve() or not path.is_file()):
                raise ValueError('frozen candidate is not the same-run checkpoint')
            raw=path.read_bytes();trained,_=read_json(d/'trained.json')
            if digest(raw)!=frozen.get('sha256') or trained.get('checkpoint_sha256')!=frozen.get('sha256'):
                raise ValueError('frozen candidate checkpoint changed')
            cp=decode(raw)
            if cp['epoch']!=epoch:raise ValueError('frozen candidate epoch changed')
            obs_path=d/'review_observation.json'
            obs=read_json(obs_path)[0] if obs_path.is_file() and not obs_path.is_symlink() else {}
            if any(k in values for k in ('source_class2_f1','target_unlabelled_class2_fraction')):
                from sealed_worker_adapter_v1 import stage_for
                commit,_=read_json(d/'commit.json')
                if (trained.get('route')!='FULL_REVIEW' or commit.get('decision_kind')!='agent'
                    or obs.get('epoch')!=epoch or obs.get('checkpoint_sha256')!=trained.get('checkpoint_sha256')):
                    raise ValueError('candidate review observation is not checkpoint-bound')
                txn=stage_for(self.job,{'contract_version':'safe_full_review_candidate_v1','epoch':epoch})/'transaction'
                request_path=txn/'request.json'
                if request_path.is_symlink() or not request_path.is_file():raise ValueError('candidate review transaction missing')
                recorded,_=read_json(request_path);evidence=recorded.get('evidence') or {}
                if ((evidence.get('source_validation') or {}).get('values')!=obs.get('source')
                    or (evidence.get('unlabelled_panel') or {}).get('values')!=obs.get('panel')):
                    raise ValueError('candidate evidence differs from committed review')
            available={'training_cls_loss':cp['metrics'].get('cls'),
              'source_class2_f1':(obs.get('source') or {}).get('class2_f1'),
              'target_unlabelled_class2_fraction':(obs.get('panel') or {}).get('predicted_class2_fraction')}
            if any(k not in available or available[k] is None or available[k]!=v for k,v in values.items()):
                raise ValueError('selection evidence is not measured for frozen candidate')
        if hash_file(manifest_path)!=manifest_sha256:raise ValueError('candidate manifest changed during validation')
        return route(self.job,request,trusted,expected,state['checkpoint'],manifest_sha256=manifest_sha256,
            codex_binary=codex_binary(),skill_sha256=skill_sha256,verifier_sha256=verifier_sha256,reference_hashes=reference_hashes)

    def commit_selection(self,transaction_dir,manifest_sha256):
        """Freeze an already sealed selector transaction, never open point data."""
        self._authorize();self._authorize_review_sources(None);self._authorize_stop(None);self._authorize_selection()
        from sealed_safe_epoch_journal_v1 import SafeTrainingJournal
        from sealed_safe_final_selection_v1 import commit
        if type(self.journal) is not SafeTrainingJournal:raise ProductionNotAuthorized('safe journal required')
        # The finalizer validates the fixed transaction path and complete
        # journal using descriptor-relative, no-symlink reads under its locks.
        return commit(Path(self.job['output_root']),transaction_dir,manifest_sha256,job=self.job,journal=self.journal)
