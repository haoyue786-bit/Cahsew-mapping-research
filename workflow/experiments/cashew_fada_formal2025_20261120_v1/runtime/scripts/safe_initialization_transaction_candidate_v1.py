"""Candidate sealed initialization transaction for a new-format checkpoint.

Caller must hold the journal lease and independently certify data lineage and
release permissions. This helper does not authorize training or read target
point data. An unknown submitted turn is never automatically resubmitted.
"""
from pathlib import Path

from dispatch_transaction_bridge import attach_dispatch_receipt
from fd_safe_decision_transaction_candidate_v1 import SafeTransaction
from native_safe_checkpoint_codec_v1 import decode
from request_reconstruction import verify_request
from safe_initialization_transition_candidate_v1 import apply
from sealed_safe_final_selection_v1 import SafeRoot
from unified_epoch_pool_bridge_sim_v1 import _equal
import sealed_dispatch_v1 as dispatcher
from agent_service_slots import call_in_slot


def recover_pre(job, journal, request, trusted, expected):
    """Recover the exact persisted pre-state; caller must verify data lineage.

    This is for a restarted owner that no longer has the original in-memory
    random state. It does not reconstruct an absent or partial transaction.
    """
    from decision_transaction import canonical
    if not isinstance(job,dict) or journal.job != job:
        raise ValueError('initialization job/journal mismatch')
    journal._held()
    clean=verify_request(request,trusted,expected=expected)
    if clean.get('contract_version')!='safe_initialization_candidate_v1' or clean.get('epoch')!=0:
        raise ValueError('expected point-sealed initialization')
    stage=Path(job['output_root'])/'agent/sealed_v1/initialization'
    txn=SafeTransaction(stage/'transaction',job['job_id'],job['run_id'],0,'initialization')
    _,saved_request,checkpoint=txn._require()
    if saved_request!=canonical(clean):
        raise ValueError('persisted initialization request differs')
    payload=decode(checkpoint)
    if payload.get('epoch')!=0:
        raise ValueError('invalid persisted epoch-zero checkpoint')
    from human_training_evidence_contract import bind
    bind(job,(expected.get('human_training') or {}).get('values'),0,payload['state_dict'])
    from human_negative_epoch10 import describe as describe_human_default
    if expected.get('human_default')!=describe_human_default(payload['config']):
        raise ValueError('persisted human-negative default differs')
    record=expected.get('transfer_record')
    if payload['metrics'].get('two_period_transfer')!=record:
        raise ValueError('persisted transfer identity differs from epoch-zero checkpoint')
    if record is not None:
        from transfer_record_contract_v1 import model_digest
        if model_digest(payload['state_dict'])!=record['model_state_sha256']:raise ValueError('persisted transferred model weights differ')
    return checkpoint


def initialize(job, journal, request, trusted, expected, checkpoint, *,
               codex_binary, skill_sha256, verifier_sha256, reference_hashes,
               client_class=None, timeout=1800):
    if not isinstance(job,dict) or journal.job != job:
        raise ValueError('initialization job/journal mismatch')
    journal._held()
    clean=verify_request(request,trusted,expected=expected)
    if clean.get('contract_version')!='safe_initialization_candidate_v1' or clean.get('epoch')!=0:
        raise ValueError('expected point-sealed initialization')
    def check(raw,epoch):
        if epoch!=0 or decode(raw).get('epoch')!=0:
            raise ValueError('invalid epoch-zero safe checkpoint')
    check(checkpoint,0)
    payload=decode(checkpoint);record=expected.get('transfer_record')
    from human_training_evidence_contract import bind
    bind(job,(expected.get('human_training') or {}).get('values'),0,payload['state_dict'])
    from human_negative_epoch10 import describe as describe_human_default
    if expected.get('human_default')!=describe_human_default(payload['config']):
        raise ValueError('human-negative default differs from initialization checkpoint')
    if payload['metrics'].get('two_period_transfer')!=record:
        raise ValueError('initialization transfer identity differs from epoch-zero checkpoint')
    if record is not None:
        from transfer_record_contract_v1 import model_digest
        if model_digest(payload['state_dict'])!=record['model_state_sha256']:raise ValueError('transferred model weights differ from frozen selection')
    fs=SafeRoot(job['output_root'])
    try:fs.mkdirs('agent/sealed_v1/initialization')
    finally:fs.close()
    stage=Path(job['output_root'])/'agent/sealed_v1/initialization'
    txn=SafeTransaction(stage/'transaction',job['job_id'],job['run_id'],0,'initialization')
    txn.prepare(clean,checkpoint,check)
    opts={'codex_binary':codex_binary,'skill_sha256':skill_sha256,
          'verifier_sha256':verifier_sha256,'reference_hashes':reference_hashes,
          'before_turn':txn.claim_submission,'timeout':timeout}
    if client_class is not None:opts['client_class']=client_class
    result=call_in_slot(lambda:dispatcher.dispatch(clean,trusted,expected,stage/'transport',**opts))
    attach_dispatch_receipt(txn,stage/'transport',trusted=trusted,expected=expected,
        skill_path=dispatcher.SKILL,skill_sha256=skill_sha256,
        verifier_sha256=verifier_sha256,reference_hashes=reference_hashes)
    def normalized(q,a,p):return apply(q,a,p,job)[0]
    decision=txn.normalize(normalized)
    # Commit recomputes both the decision and post from the original response
    # under the transaction lock. No response-dependent mutation is trusted.
    def actual_transition(p,d):
        from sealed_safe_router_sim_v1 import _post
        return _post(p,d)
    def verify(q,a,p,d,post):
        expected_decision,expected_post=apply(q,a,p,job)
        if d!=expected_decision or not _equal(decode(post),decode(expected_post)):
            raise ValueError('initialization decision or checkpoint mismatch')
    commit,post=txn.commit(check,actual_transition,post_validate=verify)
    state=journal.initialize(post)
    return {'decision':decision,'commit':commit,'state':state,'replayed':result['replayed']}
