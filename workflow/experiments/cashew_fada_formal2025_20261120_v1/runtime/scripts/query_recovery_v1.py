"""Retry only a conclusively interrupted query loop, retaining its full audit."""
import hashlib, json, os
from pathlib import Path
from query_progress_v1 import QueryStalled


class QueryRecoveryDeferred(RuntimeError):
    """Checkpoint is intact; repeated diagnostic failure needs inspection."""


def read(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('unsafe recovery artifact')
    return json.loads(path.read_text())


def archive_reconciled(stage):
    """Caller holds the run lease, and the old client has closed."""
    stage=Path(stage);tx=stage/'transaction';transport=stage/'transport'
    if any(p.is_symlink() for p in (stage,tx,transport)):raise ValueError('unsafe recovery path')
    marker=read(transport/'query_stall_reconciled.json')
    sub=read(transport/'submission.json');claim=read(tx/'submission.json');manifest=read(tx/'manifest.json')
    if (marker.get('version')!='query_stall_recovery_v1' or marker.get('remote_reconciled') is not True
        or marker.get('status') not in ('interrupted','failed') or not marker.get('turn_id')
        or marker.get('thread_id')!=sub.get('thread_id') or sub.get('thread_id')!=claim.get('thread_id')
        or manifest.get('identity')!=claim.get('identity')):
        raise ValueError('loop terminal status or transaction identity is unresolved')
    for p in (tx/'request.json',transport/'request.json'):
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest!=manifest.get('request_sha256') or digest!=sub.get('request_sha256') or digest!=claim.get('request_sha256'):
            raise ValueError('recovery request changed')
    calls=list(transport.glob('detail_tool_call_*.json'))+list(transport.glob('tool_call_*.json'))
    if not any(read(p).get('request',{}).get('params',{}).get('turnId')==marker['turn_id'] for p in calls):
        raise ValueError('interrupted turn does not own saved evidence calls')
    if any((tx/p).exists() for p in ('completed.json','response.json','decision.json','post_ready.json','commit.json')):
        raise ValueError('a completed or applied decision must not be retried')
    if any((transport/p).exists() for p in ('answer.json','raw_result.json','receipt.json')):
        raise ValueError('an existing answer must be reconciled, not replaced')
    archive=stage/'failed_attempts'/('query_stall_'+marker['thread_id'])
    archive.mkdir(parents=True,exist_ok=False)
    os.rename(transport,archive/'transport');os.rename(tx,archive/'transaction')
    for path in (archive,stage):
        fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    return archive


def with_recovery(stage, operation):
    """One automatic fresh diagnostic retry; never continue without a decision."""
    stage=Path(stage)
    while True:
        try:return operation()
        except QueryStalled:
            archives=list((stage/'failed_attempts').glob('query_stall_*'))
            archive_reconciled(stage)
            if archives:
                raise QueryRecoveryDeferred('Repeated query-loop recovery did not complete; pause this diagnostic with checkpoint preserved')
