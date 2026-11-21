"""Required interface for a future joint epoch/pool transaction.

Declaration only: neither this protocol nor the synthetic driver implements an
integrated automatic pool. Two independent mutable heads are not an atomic
commit. An owner must publish one authoritative combined head, or prove exact
reconciliation of every partial state before permitting another epoch.
"""
from typing import Protocol, Any

class PoolEpochInterface(Protocol):
    def restore_verified(self, combined_checkpoint: bytes, epoch: int) -> Any:
        """Verify run, pool generation, masks, exact model/optimizer/RNG and cursor."""
        ...
    def bind_actual_generator(self, generator_optimizer: Any) -> Any:
        """Return context manager: count only successfully completed G steps."""
        ...
    def prepare_epoch(self, trained_checkpoint: bytes, *, epoch: int,
                      processed_contiguous_positions: int, execution_receipt: Any) -> Any:
        """Persist an idempotent pool intent bound to the exact trained checkpoint."""
        ...
    def commit_combined(self, *, epoch: int, trained_checkpoint: bytes,
                        prepared_pool: Any, sealed_decision_receipt: Any) -> bytes:
        """Commit one recoverable joint head; never permit repeated cursor advancement."""
        ...
    def recover_combined(self, epoch: int) -> bytes:
        """Verify an existing joint commit or stop on unknown/orphan intent."""
        ...

def require_no_unintegrated_pool(job, checkpoint_config=None):
    """Explicitly block pool-enabled jobs until that interface is implemented."""
    if not isinstance(job,dict):raise ValueError('invalid worker job')
    if job.get('automatic_negative_policy') is not None:
        raise RuntimeError('automatic-negative pool is not integrated into the sealed worker')
    if checkpoint_config is not None:
        if not isinstance(checkpoint_config,dict):raise ValueError('missing checkpoint config')
        if checkpoint_config.get('_automatic_negative_state') is not None:
            raise RuntimeError('checkpoint contains an unintegrated automatic-negative pool')
