"""Train, acquire, or select one registered Cashew group."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0,str(ROOT/'runtime/scripts'))

from hard_sample_training import PER_REVIEW, enabled, initialize as initialize_samples, mode, read, sha
from hard_sample_acquisition import WAIT_EXIT, WaitingForAnnotations, process_round

SKILL_HOME = PROJECT/'.codex_training_current_account'
SKILL_DIR = SKILL_HOME/'skills/cashew-fada-training-expert'
from formal_training_account import codex_binary

CODEX = codex_binary()


def load_job(path):
    path = Path(path).resolve(strict=True)
    job = read(path/'job_candidate.json')
    backend = mode(job)
    collection = ROOT/('verification/jobs' if backend=='simulated' else 'jobs')
    if (path.parent != collection or Path(job['output_root']) != path or path.name != job['job_id']
            or job['point_validation_online'] is not False):
        raise ValueError('Expected a fresh registered hard-example job in its correct mode')
    if backend == 'disabled':
        if job['target_year'] != 2019 or not job.get('initial_transfer_weights_v3'):
            raise ValueError('Only a 2019 job with a same-group selected parent disables auto-labeling')
    elif not enabled(job) or (backend == 'sample_assist' and
            (job['target_year'] != 2025 or job.get('initial_transfer_weights_v3') or
             job['source_profile'] != 'source2019_benin_e173')):
        raise ValueError('2025 starts from the canonical source E173 profile with auto-labeling')
    return job


def set_training_account():
    proxy = os.environ.get('CASHEW_CODEX_PROXY')
    # The annotation Skill launches Codex by name even under a headless queue.
    os.environ['PATH'] = os.pathsep.join((str(Path(CODEX).parent), os.environ.get('PATH', os.defpath)))
    os.environ.update(CODEX_HOME=os.environ.get('CASHEW_CODEX_ACCOUNT_HOME', str(SKILL_HOME)),
                      CASHEW_HISTORY_REQUIRED='1',CASHEW_FAST_RESUME='1')
    for key in ('https_proxy','HTTPS_PROXY','http_proxy','HTTP_PROXY'):
        if proxy:
            os.environ[key] = proxy


def skill_pins():
    from review_contract import load_approved_references
    pins = read(ROOT/'skill_identity.json')
    if (sha(SKILL_DIR/'SKILL.md') != pins['skill_sha256'] or
            sha(SKILL_DIR/'scripts/verify_explicit_skill_binding.py') != pins['verifier_sha256']):
        raise ValueError('Installed training Skill and binding must match the release')
    load_approved_references(SKILL_DIR,pins['reference_hashes'])
    return pins


def verify_annotation_skill():
    installed = SKILL_HOME / 'skills/cashew-sample-assist'
    pins = read(ROOT / 'annotation_identity.json')
    for name, digest in pins['files'].items():
        path = installed / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError('Installed annotation workflow differs from the release: ' + name)


def new_baseline(job, gan, common, model, panel, source, device):
    import torch
    from source_panel_attestation_safe_candidate_v1 import input_snapshot, assert_unchanged
    from train_expert_v4 import full_target_prediction_metrics, model_state_digest
    from live_safe_review_evidence_v1 import _project
    from module_bn_v2 import domain
    from module_functional_v2 import FunctionalObserver
    before = input_snapshot(job,source,panel)
    model_sha = model_state_digest(model)
    model.eval()
    with torch.no_grad(), domain('source'), FunctionalObserver(model) as sf:
        validation = common.old_diag.evaluate_source(model,source)
    with torch.no_grad(), domain('target'), FunctionalObserver(model) as pf:
        prediction = full_target_prediction_metrics(gan,common,model,Path(job['target_root']),
                                                    panel,device,batch_size=16,return_argmax=False)
    validation,prediction = _project(validation,prediction,sorted({r['tile_id'] for r in panel.locations}))
    validation.update(sf.result()); prediction.update(pf.result())
    assert_unchanged(before,input_snapshot(job,source,panel))
    if model_state_digest(model) != model_sha:
        raise ValueError('Baseline changed the model state')
    return dict(source=validation,panel=prediction,model_state_sha256=model_sha,target_point_data_read=False)


def worker_for(job, *, journal, handles, gan, common, valid, panel):
    from experimental_production_orchestration_v1 import ExperimentalProductionWorker
    from hard_sample_training import manifest, require_trainable_annotations, round_dir
    from hard_sample_scan import scan
    from source_panel_attestation_safe_candidate_v1 import check

    class GroupWorker(ExperimentalProductionWorker):
        def _authorize(self):
            self.journal._held()

        def _authorize_review_sources(self, epoch):
            if epoch is None:
                epoch = self.journal.recover()['epoch']
            if not 0 <= epoch <= self.journal.max_epoch:
                raise ValueError('Review is outside the job epoch bounds')
            check(job,gan,common,valid,panel,epoch=epoch,expected_epoch=epoch,hash_pretrained=False)

        def _authorize_stop(self, epoch):
            self.journal._held()
            if epoch is None:
                if self.journal.recover()['state'] not in ('STOPPED','STOPPED_AT_LIMIT'):
                    raise ValueError('Sealed STOP is required before selection')
            elif epoch < self.stop_floor:
                raise ValueError('STOP is outside the current review scope')

        def _authorize_selection(self):
            self._authorize_stop(None)

        @property
        def stop_floor(self):
            return 10 if mode(job)=='simulated' else 5

        def prepare_full_review_request(self, epoch, *, allow_stop=False):
            return super().prepare_full_review_request(epoch,allow_stop=allow_stop and epoch>=self.stop_floor)

        def before_full_review(self, epoch, state):
            if not enabled(job) or epoch % PER_REVIEW:
                return
            boundary = round_dir(job,epoch)
            checkpoint_sha = hashlib.sha256(state['checkpoint']).hexdigest()
            if (boundary/'ready.json').is_file():
                intake = manifest(job, epoch)
                if intake['checkpoint_sha256'] != checkpoint_sha:
                    raise ValueError('Sample intake does not match the waiting checkpoint')
                require_trainable_annotations(intake)
                return
            if not (boundary/'screen.json').is_file():
                h = self.handles
                self.journal.restore_live(device=h.device,model=h.model,local_d=h.local_d,
                    global_d=h.global_d,model_opt=h.model_opt,local_opt=h.local_opt,
                    global_opt=h.global_opt,purpose='review')
                scan(job,h,epoch,checkpoint_sha)
            # The queue releases the training GPU at this immutable epoch.
            raise WaitingForAnnotations(epoch)

    return GroupWorker(job,journal,handles)


def train(job, gpu, pins):
    import numpy as np
    import torch
    import train_expert_v4 as w
    from experimental_production_orchestration_v1 import RuntimeHandles
    from expert_policy_v4 import parameter_bounds, schedule_curves
    from live_safe_review_evidence_v1 import _registered_panel
    from native_safe_checkpoint_codec_v1 import encode, decode
    from review_contract import build_initialization_review
    from sealed_safe_epoch_journal_v1 import SafeTrainingJournal
    from sealed_packet import values_digest
    from source_panel_attestation_safe_candidate_v1 import check
    from safe_initialization_transaction_candidate_v1 import initialize
    from predictor_artifacts_v3 import publish, verify
    from hard_sample_training import save
    from transfer_parameter_policy import initialize as initialize_groups
    from module_controls_v2 import initialize as initialize_modules

    torch.cuda.set_device(gpu); device = torch.device('cuda',gpu)
    root = Path(job['output_root']); seed = job['seed']
    # A recovered job restores its existing optimizer and model journal.
    resumed = (root/'journal/initial.json').is_file()
    names = lambda path: Path(path).read_text().splitlines()
    source = names(job['source']['source_train']); valid = names(job['source']['source_valid'])
    targets = names(job['target_train_list']); panel = _registered_panel(job)
    gan,common = w.configure_runtime(job,device)
    check(job,gan,common,valid,panel,epoch=0,expected_epoch=0,hash_pretrained=not resumed)
    dataset = gan.StrategyDataset(source,targets,'none')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model,_ = w.make_architecture_model(gan,job,device,resume_uninitialized=resumed)
    local = w.FineGrainedDomainDiscriminator(768,4,(256,128)).to(device)
    glob = w.GlobalDomainDiscriminator(768).to(device)
    config = w.config_anchor(job['variant']); config.update(local_mode='soft',hard_threshold=None,
        class_knowledge=dict(temperature=config['temperature'],confidence_cap=config['confidence_cap']),
        supplementary_training=job['supplementary'])
    initialize_samples(config,job); w.gradient_control.initialize(config,job); w.finish.initialize(config,job)
    initialize_groups(config,job); initialize_modules(config,job)
    from transfer_regularization_v3 import initialize as initialize_regularization
    initialize_regularization(config,job)
    opt = torch.optim.AdamW(w.model_parameter_groups(model,config['main_lr'],
        config['main_weight_decay'],(.9,.999),1e-8,config=config))
    opts = [torch.optim.AdamW(d.parameters(),lr=config['D_lr'],weight_decay=config['D_weight_decay'])
            for d in (local,glob)]
    journal = SafeTrainingJournal(root/'journal',job,job['max_epoch'])
    handles = RuntimeHandles(gan,common,model,local,glob,opt,*opts,dataset,len(source),len(targets),(),
                             root,root,device)
    worker = worker_for(job,journal=journal,handles=handles,gan=gan,common=common,valid=valid,panel=panel)
    with journal.lease():
        baseline_path = root/'initial_baseline_bound.json'
        if not resumed:
            baseline = read(baseline_path) if baseline_path.is_file() else new_baseline(job,gan,common,model,panel,valid,device)
            save(baseline_path,baseline)
            if w.model_state_digest(model) != baseline['model_state_sha256']:
                raise ValueError('Starting model differs from measured E0')
            path = root/'initial_pre_checkpoint.bin'
            if not path.is_file():
                from complete_history_store import _write_immutable
                _write_immutable(path,encode(w.checkpoint_payload(0,model,local,glob,opt,*opts,config,{})))
            raw = path.read_bytes(); payload = decode(raw); config = payload['config']
            bounds = parameter_bounds(config)
            families = {k:sorted(v) for k,v in schedule_curves(config).items()}
            initial = {k:config['schedule_state'][k]['curve'] for k in families}
            def artifact(kind, values):
                return dict(kind=kind,epoch=0,values=values,sha256=values_digest(kind,0,values))
            a = artifact('initial_controls',{k:config[k] for k in bounds})
            s = artifact('source',baseline['source']); p = artifact('panel',baseline['panel'])
            trust = {(v['kind'],0,v['sha256']) for v in (a,s,p)}
            expected = dict(epoch=0,bounds=bounds,initial_families=families,initial_schedule_baseline=initial)
            request = build_initialization_review(a,s,p,trust,bounds,initial_families=families,
                                                  initial_schedule_baseline=initial)
            initialize(job,journal,request,trust,expected,raw,codex_binary=CODEX,**pins)
            del raw,payload
        else:
            from fast_resume_v1 import verify_initial
            verify_initial(journal,read(baseline_path))
        meta = read(root/'journal/initial.json')
        if (root/'predictors/epoch_000000.json').is_file():
            verify(root,job,0,meta['checkpoint_sha256'])
        else:
            initial = (root/'journal/initial_checkpoint.bin').read_bytes()
            if hashlib.sha256(initial).hexdigest() != meta['checkpoint_sha256']:
                raise ValueError('Initial checkpoint changed')
            publish(root,job,decode(initial),meta['checkpoint_sha256'],'journal/initial_checkpoint.bin')
            del initial
        status = worker.run_review_cycles(**pins,allow_stop=True)
        print(json.dumps(dict(phase=status['state'],epoch=status['epoch'])),flush=True)
        if status['state'] not in ('STOPPED','STOPPED_AT_LIMIT'):
            raise ValueError('Group returned without a committed STOP')


def select(job, gpu, pins):
    from types import SimpleNamespace
    import torch
    import train_expert_v4 as w
    from live_safe_review_evidence_v1 import _registered_panel
    from sealed_safe_epoch_journal_v1 import SafeTrainingJournal
    from point_sealed_finalizer_candidate_v1 import finalize

    torch.cuda.set_device(gpu); device = torch.device('cuda',gpu)
    gan,common = w.configure_runtime(job,device)
    model,_ = w.make_architecture_model(gan,job,device,resume_uninitialized=True)
    h = SimpleNamespace(gan=gan,common=common,model=model,device=device)
    path = Path(job['output_root']); journal = SafeTrainingJournal(path/'journal',job,job['max_epoch'])
    worker = worker_for(job,journal=journal,handles=h,gan=gan,common=common,
                        valid=w.read_lines(Path(job['source']['source_valid'])),panel=_registered_panel(job))
    with journal.lease():
        result = finalize(worker,**pins)
    print(json.dumps(dict(phase='complete',selected_epoch=result['selected_epoch'])),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('train','annotate','select'))
    parser.add_argument('job',type=Path)
    parser.add_argument('--gpu',type=int)
    parser.add_argument('--epoch',type=int)
    args = parser.parse_args()
    if args.phase == 'annotate' and (args.epoch is None or args.epoch < PER_REVIEW or args.epoch % PER_REVIEW):
        parser.error('Annotation requires a completed five-epoch boundary')
    if args.phase != 'annotate' and args.gpu is None:
        parser.error('Training and selection require an explicit GPU')
    job = load_job(args.job)
    if args.phase == 'annotate' and not enabled(job):
        parser.error('Automatic annotation is disabled for this job')
    from release_identity_v2 import verify
    verify(ROOT)
    set_training_account(); pins = skill_pins()
    if args.phase == 'annotate' and mode(job) == 'sample_assist':
        verify_annotation_skill()
    root = args.job.resolve(); lease = root/('annotation_owner.lock' if args.phase=='annotate' else 'gpu_owner.lock')
    with lease.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.phase=='annotate':
            path = root/'journal'/('epoch_%06d' % args.epoch)/'trained.json'
            from hard_sample_training import round_dir
            screen = read(round_dir(job,args.epoch)/'screen.json')
            if screen['checkpoint_sha256'] != read(path)['checkpoint_sha256']:
                raise ValueError('Annotation epoch lacks a bound trained checkpoint')
            ready = process_round(job,args.epoch,pins)
            print(json.dumps(dict(phase='annotation_ready',epoch=args.epoch,entries=len(ready['entries']))),flush=True)
        elif args.phase=='select':
            select(job,args.gpu,pins)
        else:
            try:
                train(job,args.gpu,pins)
            except WaitingForAnnotations as waiting:
                from hard_sample_training import save, round_dir
                epoch = waiting.args[0]
                save(round_dir(job,epoch)/'waiting.json',dict(epoch=epoch))
                print(json.dumps(dict(phase='waiting_annotations',epoch=epoch)),flush=True)
                return WAIT_EXIT
    return 0


if __name__=='__main__':
    sys.exit(main())
