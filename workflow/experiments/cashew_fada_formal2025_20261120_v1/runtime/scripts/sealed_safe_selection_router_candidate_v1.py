"""Candidate frozen-selector transaction for weights-only checkpoints.

No target points are opened. Source provenance, STOP authorization, candidate
eligibility and resource authorization must be enforced by the owning worker.
Simulation flags and injected clients are not security boundaries.
"""
import json
from contextlib import nullcontext
from pathlib import Path
from fd_safe_decision_transaction_candidate_v1 import SafeTransaction
from sealed_safe_final_selection_v1 import SafeRoot
from request_reconstruction import verify_request
from history_bridge_candidate import attach_dispatch_receipt
from decision_bridge import normalize_selection
from native_safe_checkpoint_codec_v1 import decode
from selection_freeze import hash_file,path_under
import history_dispatch_candidate as dispatcher
from agent_service_slots import call_in_slot

def _route(job,request,trusted,expected,pre,*,manifest_sha256,codex_binary,skill_sha256,verifier_sha256,
          reference_hashes,client_class=None,timeout=1800):
    policy=job.get('experimental_gate') if isinstance(job,dict) else None
    if (not isinstance(policy,dict) or policy.get('version')!='experimental_gate_v1'
        or policy.get('experimental_only') is not True or policy.get('mode')!='enforce'):
        raise ValueError('enforced candidate selector policy required')
    if request.get('contract_version')!='safe_checkpoint_selection_candidate_v1':raise ValueError('selector contract required')
    root=Path(job['output_root']);manifest_path=root/'frozen_candidates.json';stop_path=root/'stop_receipt.json'
    if manifest_path.is_symlink() or stop_path.is_symlink() or hash_file(manifest_path)!=manifest_sha256:
        raise ValueError('candidate freeze missing or changed')
    manifest=json.loads(manifest_path.read_text());stop=json.loads(stop_path.read_text());epoch=request.get('stop_epoch')
    if (manifest.get('job_id')!=job['job_id'] or manifest.get('run_id')!=job['run_id'] or manifest.get('stop_epoch')!=epoch
        or stop.get('epoch')!=epoch or manifest.get('stop_receipt_sha256')!=hash_file(stop_path)):
        raise ValueError('selector freeze identity mismatch')
    if [x['epoch'] for x in request.get('candidate_observations',[])]!=[x['epoch'] for x in manifest.get('candidates',[])]:
        raise ValueError('selector candidates differ from frozen manifest')
    cp=decode(pre)
    stop_post=path_under(root,Path(stop['transaction_path'])/'post_checkpoint.bin')
    if cp['epoch']!=epoch or hash_file(stop_post)!=stop.get('post_checkpoint_sha256'):
        raise ValueError('selector stop checkpoint mismatch')
    from decision_transaction import digest
    if digest(pre)!=stop.get('post_checkpoint_sha256'):raise ValueError('selector input differs from frozen stop')
    from checkpoint_selection_contract_v2 import enabled as complete_selection,VERSION
    if request.get('selection_policy')!=(VERSION if complete_selection(job) else None):raise ValueError('selector job policy mismatch')
    clean=verify_request(request,trusted,expected=expected,previous=None)
    prefix='agent/sealed_v1/selection';owner=SafeRoot(root)
    try:owner.mkdirs(prefix)
    finally:owner.close()
    stage=root/prefix;txn=SafeTransaction(stage/'transaction',job['job_id'],job['run_id'],epoch,'checkpoint_selection')
    if client_class is None:client_class=dispatcher.PersistentLunaClient
    def check(raw,e):
        if decode(raw)['epoch']!=e:raise ValueError('selector checkpoint epoch mismatch')
    def decision(q,a,p):return normalize_selection(q,a,decode(p)['config'],job,frozen_stop_epoch=epoch)
    def transition(p,d):
        if d.get('action')!='SELECT_CHECKPOINT' or d.get('continue_training') is not False:raise ValueError('invalid selection action')
        return p
    with nullcontext(txn):
        txn.prepare(clean,pre,check)
        reply=call_in_slot(lambda:dispatcher.dispatch(clean,trusted,expected,stage/'transport',codex_binary=codex_binary,
            skill_sha256=skill_sha256,verifier_sha256=verifier_sha256,reference_hashes=reference_hashes,
            before_turn=txn.claim_submission,client_class=client_class,timeout=timeout,
            history_root=job['output_root'],history_job_id=job['job_id'],history_run_id=job['run_id']))
        attach_dispatch_receipt(txn,stage/'transport',trusted=trusted,expected=expected,skill_path=dispatcher.SKILL,
            skill_sha256=skill_sha256,verifier_sha256=verifier_sha256,reference_hashes=reference_hashes,
            history_root=job['output_root'],history_job_id=job['job_id'],history_run_id=job['run_id'])
        result=txn.normalize(decision)
        def locked(q,a,p,d,post):
            if d!=decision(q,a,p) or post!=p:raise ValueError('selection decision or post changed')
        commit,post=txn.commit(check,transition,post_validate=locked)
        return {'decision':result,'commit':commit,'post_checkpoint':post,'replayed':reply['replayed'],'stage':str(stage)}


def route(job, request, *args, **kwargs):
    from query_recovery_v1 import with_recovery
    stage = Path(job['output_root']) / 'agent/sealed_v1/selection'
    return with_recovery(stage, lambda: _route(job, request, *args, **kwargs))
