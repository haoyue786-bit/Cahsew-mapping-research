"""Experimental sealed decision transaction orchestrator; NOT a launch permit.

This module combines the isolated request/dispatch/commit components. It is not
wired to main: the worker must first supply independently authenticated source,
lineage, frozen skill hashes, checkpoint bytes and appropriate native validator
and transition. A caller-supplied trusted set/hash does not prove provenance.
The global experimental launch block and all legacy fences remain mandatory.
"""
from pathlib import Path
from contextlib import nullcontext
from scoped_decision_transaction_v1 import ScopedTransaction
import sealed_dispatch_v1 as dispatcher
from sealed_worker_adapter_v1 import stage_for
from decision_transaction import Transaction
from dispatch_transaction_bridge import attach_dispatch_receipt
from request_reconstruction import verify_request

KINDS={'safe_initialization_candidate_v1':'initialization',
       'safe_anomaly_review_candidate_v1':'light_review',
       'safe_full_review_candidate_v1':'full_review',
       'safe_branch_review_candidate_v1':'experiment_review',
       'safe_checkpoint_selection_candidate_v1':'checkpoint_selection'}

def route_prepared(job,request,trusted,expected,pre_checkpoint,*,checkpoint_validate,
                   normalize_decision,transition,codex_binary,skill_sha256,
                   verifier_sha256,reference_hashes,previous=None,client_class=None,
                   timeout=1800,pool_journal=None):
    """Complete exactly one decision on a saved state; never starts training.

    A submitted but unconfirmed network turn remains blocked for exact external
    reconciliation. Restarting uses a valid existing receipt and committed post
    checkpoint without calling the model or applying an action again.
    """
    if not isinstance(job,dict) or not isinstance(request,dict):raise ValueError('invalid job or request')
    policy=job.get('experimental_gate')
    if not isinstance(policy,dict) or policy.get('version')!='experimental_gate_v1' or policy.get('experimental_only') is not True:
        raise ValueError('router requires explicitly declared experimental policy')
    if request.get('contract_version') not in KINDS:raise ValueError('unsupported sealed decision kind')
    if not isinstance(pre_checkpoint,bytes) or not all(callable(x) for x in (checkpoint_validate,normalize_decision,transition)):
        raise ValueError('checkpoint and native callbacks required')
    clean=verify_request(request,trusted,expected=expected,previous=previous)
    epoch=clean.get('stop_epoch') if clean['contract_version']=='safe_checkpoint_selection_candidate_v1' else clean.get('epoch')
    if type(epoch) is not int or epoch<0:raise ValueError('invalid worker epoch')
    pool=clean.get('evidence',{}).get('active_automatic_pool') if clean['contract_version']=='safe_anomaly_review_candidate_v1' else None
    if pool is not None:
        from sealed_epoch_journal_v1 import EpochJournal
        if type(pool_journal) is not EpochJournal:raise ValueError('pool review requires owning epoch journal')
        pool_journal._held()
        if pool_journal.job!=job:raise ValueError('pool review job mismatch')
        state=pool_journal.recover()
        if state['state']!='LIGHT_REVIEW' or state['epoch']!=epoch or state['checkpoint']!=pre_checkpoint:
            raise ValueError('pool review checkpoint mismatch')
        actual=pool_journal.active_pool_evidence(epoch)
        if actual is None or pool.get('epoch')!=epoch or pool.get('values')!=actual:
            raise ValueError('pool evidence differs from checkpoint and exposure receipt')
    stage=stage_for(job,clean)
    scoped=policy.get('simulation_scoped_transaction') is True and policy.get('simulation_only') is True
    transaction=(ScopedTransaction if scoped else Transaction)(stage/'transaction',job['job_id'],job['run_id'],epoch,KINDS[clean['contract_version']])
    with transaction.scope() if scoped else nullcontext(transaction):
        return _execute_scoped(job,pool_journal,transaction,clean,trusted,expected,pre_checkpoint,stage,checkpoint_validate,
            normalize_decision,transition,codex_binary,skill_sha256,verifier_sha256,reference_hashes,previous,client_class,timeout)

def _execute_scoped(job,pool_journal,transaction,clean,trusted,expected,pre_checkpoint,stage,checkpoint_validate,normalize_decision,transition,
                    codex_binary,skill_sha256,verifier_sha256,reference_hashes,previous,client_class,timeout):
    # The complete state is persisted and validated before any thread starts.
    transaction.prepare(clean,pre_checkpoint,checkpoint_validate)
    options={'codex_binary':codex_binary,'skill_sha256':skill_sha256,'verifier_sha256':verifier_sha256,
             'reference_hashes':reference_hashes,'previous':previous,'before_turn':transaction.claim_submission,'timeout':timeout}
    if client_class is not None:options['client_class']=client_class
    reply=dispatcher.dispatch(clean,trusted,expected,stage/'transport',**options)
    attach_dispatch_receipt(transaction,stage/'transport',trusted=trusted,expected=expected,
        skill_path=dispatcher.SKILL,skill_sha256=skill_sha256,verifier_sha256=verifier_sha256,
        reference_hashes=reference_hashes,previous=previous)
    def independently_checked(request,answer,pre):
        from sealed_ordinary_decision_guard_v1 import verify
        actual=normalize_decision(request,answer,pre)
        return verify(job,request,answer,pre,actual)
    decision=transaction.normalize(independently_checked)
    def validate_locked(request,answer,pre,decision,post):
        from sealed_ordinary_decision_guard_v1 import verify as verify_decision
        from sealed_transition_guard_v1 import verify as verify_post
        verify_decision(job,request,answer,pre,decision)
        verify_post(pre,decision,post,pool_journal=pool_journal)
    commit,post_checkpoint=transaction.commit(checkpoint_validate,transition,post_validate=validate_locked)
    return {'decision':decision,'commit':commit,'post_checkpoint':post_checkpoint,
            'replayed':reply['replayed'],'stage':str(stage)}
