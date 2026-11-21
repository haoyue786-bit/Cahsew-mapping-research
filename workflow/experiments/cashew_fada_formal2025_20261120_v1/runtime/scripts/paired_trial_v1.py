"""Bounded same-state trials exposed only inside an owned full-review turn.

Trials train isolated copies sequentially on the owner's GPU. They never
publish a candidate or transplant branch state into the formal trajectory.
"""
import copy
import hashlib
import json
import math
import random
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import numpy as np
import torch

VERSION = 'matched_module_trial_v2'
POLICY = {'version': VERSION, 'steps': [63,126,315,630], 'max_per_review': 2,
          'source_validation_patches': 220, 'unlabelled_panel_patches': 128,
          'evaluation_sampling':'tile_balanced_spatial_spread_v2','updates_per_epoch':63,
          'observation_checkpoints':'each_complete_epoch_and_final','human_training_probe':True}
_ACTIVE = ContextVar('cashew_owned_trial_service', default=None)

def raw(value):
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()

def capability(job, *, epoch=None):
    p=job.get('paired_trial_policy')
    if p is None:return None
    if p != POLICY or job.get('point_validation_online') is not False:
        raise ValueError('unapproved trial capability')
    maximum = job.get('max_epoch',200)
    if epoch is not None and epoch >= maximum:
        return None
    raw_policy=copy.deepcopy(POLICY)
    from hard_sample_training import enabled as annotated
    if annotated(job):
        raw_policy.update(max_epoch=maximum, human_training_probe=False, annotation_training_probe=True)
    return raw_policy


def trial_max_epoch(policy):
    if policy == POLICY:
        return 200
    maximum = policy.get('max_epoch') if isinstance(policy,dict) else None
    if (type(maximum) is not int or not 5 <= maximum <= 200
            or policy != dict(POLICY,max_epoch=maximum,human_training_probe=False,annotation_training_probe=True)):
        raise ValueError('Invalid matched trial policy')
    return maximum

def spec(request):
    policy=request.get('matched_trial_capability')
    if policy is None:return []
    if request.get('contract_version')!='safe_full_review_candidate_v1':
        raise ValueError('trial unavailable in this review')
    budgets=[n for n in POLICY['steps'] if n <= (trial_max_epoch(policy)-request['epoch'])*POLICY['updates_per_epoch']]
    if not budgets:return []
    result=[{'type':'function','name':'run_matched_training_trial',
        'description':'Run a matched control/treatment mechanism test on isolated complete states. Choose 63/126/315/630 updates per arm (1/2/5/10 current epochs). Use the horizon needed to test the mechanism. Both arms use identical domain minibatches and the same accepted visual training library. Reports full source validation, fixed unlabelled imagery, fit, gradients and actual updates. Give the hypothesis, expected response and falsification.',
        'inputSchema':{'type':'object','additionalProperties':False,
          'required':['updates','steps','hypothesis','expected_response','falsification'],
          'properties':{'updates':{'type':'array','minItems':0,'maxItems':2,'items':{
             'type':'object','additionalProperties':False,'required':['parameter','value'],
             'properties':{'parameter':{'type':'string','enum':sorted(request['allowed_parameter_bounds'])},'value':{'type':'number'}}}},
             'steps':{'type':'integer','enum':budgets},
             'schedule':{'type':'object','additionalProperties':False,'required':['component','operation'],
                 'properties':{'component':{'type':'string','enum':['main_lr']},'operation':{'type':'string','enum':['start_cosine_finish']}}},
             **{k:{'type':'string'} for k in ('hypothesis','expected_response','falsification')}}}}]
    if 'start_cosine_finish' not in request.get('allowed_schedules',{}).get('main_lr',[]):
        result[0]['inputSchema']['properties'].pop('schedule')
    return result

def validate_args(args, bounds, config):
    required={'updates','steps','hypothesis','expected_response','falsification'}
    if not isinstance(args,dict) or not required<=set(args) or set(args)-required-{'schedule'}:
        raise ValueError('invalid trial arguments')
    if type(args['steps']) is not int or args['steps'] not in POLICY['steps']:raise ValueError('invalid trial budget')
    updates=args['updates']
    if not isinstance(updates,list) or len(updates)>2 or (not updates and 'schedule' not in args):raise ValueError('one mechanism uses up to two controls and/or one schedule')
    if 'schedule' in args:
        from cosine_finish_v1 import enabled
        if args['schedule']!={'component':'main_lr','operation':'start_cosine_finish'} or not enabled(config) or config.get('cosine_finish_state'):raise ValueError('cosine finish unavailable for this trial')
    from review_contract import INTEGER_CONTROLS
    seen=set()
    for u in updates:
        if not isinstance(u,dict) or set(u)!={'parameter','value'}:raise ValueError('invalid trial update')
        k=u['parameter'];v=u['value']
        if k not in bounds or k in seen:raise ValueError('unknown or duplicated control')
        if type(v) not in (int,float) or not math.isfinite(v) or not bounds[k][0]<=v<=bounds[k][1]:raise ValueError('control outside bounds')
        if k in INTEGER_CONTROLS and type(v) is not int:raise ValueError('integer control required')
        if config[k]==v:raise ValueError('trial treatment has no effect')
        seen.add(k)
    for key in ('hypothesis','expected_response','falsification'):
        if not isinstance(args[key],str) or not args[key].strip() or len(args[key])>4000:raise ValueError('trial needs bounded mechanism and response statements')
    from online_boundary import verify_online
    verify_online(args)

def stratified_subset(names,count):
    """Spread a fixed budget over all tiles and each tile's ordered locations."""
    from collections import defaultdict
    groups=defaultdict(list)
    for name in names:
        p=Path(name);parts=p.stem.rsplit('_',3)
        groups[parts[1] if len(parts)==4 else str(p.parent)].append(name)
    if len(groups)>count:raise ValueError('trial budget cannot cover all tiles')
    out=[];keys=sorted(groups)
    for index,key in enumerate(keys):
        values=sorted(groups[key]);n=min(len(values),count//len(keys)+(index<count%len(keys)))
        ids=np.linspace(0,len(values)-1,n).round().astype(int).tolist()
        out.extend(values[i] for i in ids)
    if len(out)<count:
        used=set(out);out.extend(x for x in names if x not in used and len(out)<count)
    return out,{'tiles':len(groups),'patches':len(out),'policy':POLICY['evaluation_sampling']}

@contextmanager
def owned_service(worker, epoch, request):
    if request.get('matched_trial_capability') is None:
        yield;return
    worker._authorize();worker._authorize_review_sources(epoch);worker.journal._held()
    if capability(worker.job)!=request['matched_trial_capability']:raise ValueError('trial policy differs from owner')
    service=TrialService(worker,epoch,request)
    token=_ACTIVE.set(service)
    try:yield
    finally:_ACTIVE.reset(token)

def invoke(args,request,root,epoch):
    service=_ACTIVE.get()
    if service is None or service.epoch!=epoch or service.request!=request or Path(root)!=Path(service.worker.job['output_root']):
        raise RuntimeError('no owning trial executor for this request')
    return service.run(args)

def _rng():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def _restore_rng(value):
    random.setstate(value[0]);np.random.set_state(value[1]);torch.set_rng_state(value[2])
    if value[3] is not None:torch.cuda.set_rng_state_all(value[3])

class TrialService:
    def __init__(self,worker,epoch,request):
        self.worker=worker;self.epoch=epoch;self.request=request
        self.root=Path(worker.job['output_root'])/'trial_evidence'/('epoch_%06d'%epoch)
        self.root.mkdir(parents=True,exist_ok=True)

    def run(self,args):
        from complete_history_store import _write_immutable
        from native_safe_checkpoint_codec_v1 import decode
        from expert_policy_v4 import parameter_bounds
        from online_boundary import verify_online
        w=self.worker;w.journal._held();h=w.handles
        prepared=w.journal.prepare_full_review(self.epoch);pre=prepared['checkpoint']
        cp=decode(pre);config=cp['config']
        validate_args(args,parameter_bounds(config),config)
        if self.epoch+math.ceil(args['steps']/POLICY['updates_per_epoch'])>w.job['max_epoch']:raise ValueError('trial exceeds remaining epoch budget')
        identity={'job_id':w.job['job_id'],'run_id':w.job['run_id'],'epoch':self.epoch,
                  'checkpoint_sha256':hashlib.sha256(pre).hexdigest(),
                  'request_sha256':hashlib.sha256(raw(self.request)).hexdigest(),'arguments':args}
        trial_id=hashlib.sha256(raw(identity)).hexdigest();directory=self.root/trial_id
        if directory.exists():
            if (directory/'result.json').is_file():return json.loads((directory/'result.json').read_text())
            return {'trial_id':trial_id,'status':'incomplete','formal_state_adopted':False,
                    'message':'This attempt was interrupted; no response or benefit is established. It is not rerun.'}
        if len(list(self.root.glob('*/intent.json')))>=POLICY['max_per_review']:
            return {'status':'budget_exhausted','formal_state_adopted':False}
        directory.mkdir();_write_immutable(directory/'intent.json',raw(identity))
        saved_rng=_rng();start=time.monotonic()
        try:
            result=self.execute(cp,args)
            result.update(trial_id=trial_id,status='completed',epoch=self.epoch,
                          formal_state_adopted=False,elapsed_seconds=time.monotonic()-start,
                          checkpoint_sha256=identity['checkpoint_sha256'])
        except Exception as exc:
            result={'trial_id':trial_id,'status':'failed','epoch':self.epoch,'formal_state_adopted':False,
                    'failure_type':type(exc).__name__,'elapsed_seconds':time.monotonic()-start,
                    'checkpoint_sha256':identity['checkpoint_sha256']}
            _write_immutable(directory/'failure.txt',str(exc).encode())
        finally:
            _restore_rng(saved_rng)
        verify_online(result)
        _write_immutable(directory/'result.json',raw(result))
        return result

    def execute(self, cp, args):
        import train_expert_v4 as train
        from architecture import model_parameter_groups
        from module_bn_v2 import domain as bn_domain
        from module_functional_v2 import FunctionalObserver
        from module_observer_v2 import Observer, snapshot, project
        from live_safe_review_evidence_v1 import _registered_panel
        from schedule_v4 import apply_to_optimizers
        from bounded_recorder_candidate_v1 import BoundedRecorder
        from healthy_recovery_runtime_v4 import restore_full_state
        from full_state_finiteness_v1 import check as finite
        from worker_projection import project_training
        h=self.worker.handles;job=self.worker.job
        from hard_sample_training import EpochSamples, PER_REVIEW, comparison, enabled as annotated, manifest, summary
        has_annotations = annotated(job)
        boundary = self.epoch // PER_REVIEW * PER_REVIEW
        owned_library = manifest(job,boundary) if has_annotations else None
        source=train.read_lines(Path(job['source']['source_valid']))
        panel=_registered_panel(job);target,coverage=stratified_subset(list(panel),POLICY['unlabelled_panel_patches'])
        if len(source)!=POLICY['source_validation_patches'] or len(target)!=POLICY['unlabelled_panel_patches']:raise ValueError('incomplete fixed trial panel')
        subset_hash=hashlib.sha256(raw({'source':source,'unlabelled':target})).hexdigest()
        # Copies stay sequential: one extra training copy on this same GPU.
        model=copy.deepcopy(h.model);local=copy.deepcopy(h.local_d);glob=copy.deepcopy(h.global_d)
        opt=torch.optim.AdamW(model_parameter_groups(model,cp['config']['main_lr'],cp['config']['main_weight_decay'],(.9,.999),1e-8,config=cp['config']))
        lo=torch.optim.AdamW(local.parameters());go=torch.optim.AdamW(glob.parameters())
        models={'G':model,'D_local':local,'D_global':glob};opts={'G':opt,'D_local':lo,'D_global':go}
        def restore():
            payload=dict(cp)
            for key in ('model_optimizer','local_optimizer','global_optimizer'):payload[key]=copy.deepcopy(cp[key])
            return restore_full_state(payload,expected_epoch=self.epoch,saved_cuda_ordinal=cp['rng_state']['active_cuda_ordinal'],
                device=h.device,model=model,local_d=local,global_d=glob,model_opt=opt,local_opt=lo,global_opt=go)[0]
        def evaluate(observed_epoch):
            model.eval()
            with torch.no_grad(),bn_domain('source'),FunctionalObserver(model) as fs:
                sm=h.common.old_diag.evaluate_source(model,source)
            with torch.no_grad(),bn_domain('target'),FunctionalObserver(model) as ft:
                tm=train.full_target_prediction_metrics(h.gan,h.common,model,Path(job['target_root']),target,h.device,batch_size=16,return_argmax=True)
            codes=tm.pop('_argmax_codes')
            from human_training_evidence_v1 import collect as human_probe
            human=human_probe(job,h.gan,model,h.device,observed_epoch,persist=False)
            fit = None if owned_library is None else dict(
                role='training_fit',independent=False,library_revision=owned_library['revision'],
                backend=owned_library['backend'],
                comparison=summary(comparison(h.gan,h.common,model,h.device,owned_library['entries'])))
            return {'source':{'class2_f1':float(sm['f1_per_class'][2]),'loss':float(sm['loss']),'patches':len(source),**fs.result()},
                    'unlabelled_panel':{'class2_fraction':float(tm['class_distribution'][2]),'entropy':float(tm['entropy']),
                                       'patches':len(target),'tile_coverage':coverage,**ft.result()},
                    'human_training':human,'human_training_scope':'current_trial_branch_predictor_only',
                    'annotated_training':fit,
                    'bn_update_mode':int(model.fuse_conv[1].domain_mode)},codes
        restore();baseline,base_codes=evaluate(self.epoch);out={}
        sample_seed=int(job['seed'])+(self.epoch+1)*100000
        for role in ('control','treatment'):
            config=restore()
            if role=='treatment':
                for u in args['updates']:config[u['parameter']]=u['value']
                if 'schedule' in args:
                    from cosine_finish_v1 import start
                    start(config,self.epoch,float(opt.param_groups[0]['lr']))
            class MeasuredGAN:
                def __init__(self):self.indices=[]
                def __getattr__(self,name):return getattr(h.gan,name)
                def batch_from(self,dataset,source_ids,target_ids,device):
                    self.indices.append([list(map(int,source_ids)),list(map(int,target_ids))])
                    return h.gan.batch_from(dataset,source_ids,target_ids,device)
            measured_gan=MeasuredGAN()
            observations=[];completed=0
            while completed<args['steps']:
                trial_epoch=self.epoch+completed//POLICY['updates_per_epoch']+1
                effective=apply_to_optimizers(config,trial_epoch,opt,lo,go)
                config.update(effective,_negative_epoch_start=trial_epoch-1,_negative_seed=int(job['seed'])+trial_epoch*100000)
                anchors={'epoch':self.epoch,'samples':snapshot(models)}
                observer=Observer(models,opts,anchors,anchors)
                count=min(POLICY['updates_per_epoch'],args['steps']-completed)
                extra = {}
                if has_annotations:
                    extra['annotated_samples'] = EpochSamples(job,trial_epoch,config,count,h.common,h.device,
                                                            current_round=boundary)
                metrics=train.train_epoch(measured_gan,h.common,model,local,glob,opt,lo,go,h.dataset,h.source_count,h.target_count,
                    h.supplementary_names,Path(h.supplementary_image_root),Path(h.supplementary_label_root),config,h.device,
                    np.random.RandomState(int(job['seed'])+trial_epoch),steps_override=count,
                    experimental_recorder=BoundedRecorder(count=64,max_tensors=8),**extra)
                finite(models,opts);metrics['module_observation']=observer.finish()
                completed+=metrics['experimental_actual_updates']['steps']['G']
                end,codes=evaluate(trial_epoch)
                end['unlabelled_panel']['pixel_agreement_with_start']=float(np.mean(codes==base_codes))
                observations.append({'completed_generator_updates':completed,'simulated_epoch':trial_epoch,'end':end,
                                     'training':project_training(metrics),
                                     'annotated_training_updates':metrics.get('annotated_training_updates')})
            from expert_policy_v4 import parameter_bounds
            out[role]={'source_target_batch_indices_sha256':hashlib.sha256(raw(measured_gan.indices)).hexdigest(),
                       'effective_controls':{k:config[k] for k in parameter_bounds(config)},
                       'completed_generator_updates':completed,
                       'end':end,'training':project_training(metrics),'observations':observations}
            if out[role]['completed_generator_updates']!=args['steps']:raise ValueError('incomplete trial updates')
        if out['control']['source_target_batch_indices_sha256']!=out['treatment']['source_target_batch_indices_sha256']:
            raise ValueError('trial source/target batches were not matched')
        del model,local,glob,opt,lo,go,models,opts
        return {'version':VERSION,'baseline':baseline,'arms':out,'subset_sha256':subset_hash,
                'source_target_minibatch_seed':int(job['seed'])+self.epoch+1,
                'scope':'Question-specific matched multi-epoch response; 220 source patches, 128 spatially spread patches covering all target tiles, and fixed training-polygon fit; not target accuracy. Each epoch and actual exposure are retained.',
                'all_initial_states_matched':True,'branch_checkpoint_selection_allowed':False}

def verify_transcripts(stage,root,epoch,request,thread_id,turn_ids):
    """Bind every tool result to the owned execution receipt before commit."""
    hashes=[];files=sorted(Path(stage).glob('trial_tool_call_*.json'))
    if files and not spec(request):raise ValueError('unadvertised trial invocation')
    for i,path in enumerate(files,1):
        if path.is_symlink() or path.name!='trial_tool_call_%02d.json'%i:raise ValueError('invalid trial transcript')
        record=json.loads(path.read_text());params=record['request']['params'];result=record['result']
        from training_tool_output_v1 import verify_delivery
        verify_delivery(record,'run_matched_training_trial',i)
        if params.get('threadId')!=thread_id or params.get('tool')!='run_matched_training_trial' or params.get('turnId') not in turn_ids:
            raise ValueError('trial belongs to another turn')
        if result.get('status') in ('completed','failed'):
            folder=Path(root)/'trial_evidence'/('epoch_%06d'%epoch)/result['trial_id']
            identity=json.loads((folder/'intent.json').read_text())
            if (identity['arguments']!=params['arguments'] or identity['request_sha256']!=hashlib.sha256(raw(request)).hexdigest()
                or hashlib.sha256(raw(identity)).hexdigest()!=result['trial_id']
                or json.loads((folder/'result.json').read_text())!=result):raise ValueError('trial receipt mismatch')
        elif result.get('status')=='incomplete':
            if set(result)!={'trial_id','status','formal_state_adopted','message'}:raise ValueError('invalid incomplete receipt')
            folder=Path(root)/'trial_evidence'/('epoch_%06d'%epoch)/result['trial_id']
            identity=json.loads((folder/'intent.json').read_text())
            if identity['arguments']!=params['arguments'] or (folder/'result.json').exists():raise ValueError('invalid interrupted trial')
        elif result.get('status') in ('budget_exhausted','invalid_arguments','history_required'):
            allowed={'status','formal_state_adopted'} | ({'message'} if result['status']!='budget_exhausted' else set())
            if set(result)!=allowed or result['formal_state_adopted'] is not False:raise ValueError('invalid empty trial response')
        else:raise ValueError('unknown trial outcome')
        from online_boundary import verify_online
        verify_online(result);hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return hashes
