"""Candidate single-call checkpoint reuse; not wired to any real worker.

The original transaction remains authoritative. This opt-in variant keeps the
caller's already-present immutable ``bytes`` only until one synchronous scope
ends; it never trusts file metadata, and hashes the complete durable pre-state
after preparation, before submission and before commit. Intermediate steps
recheck the manifest and request, but consume that same verified snapshot.

This is not a protection against an actor who can rewrite all unkeyed records,
not a source/target provenance check, and not a persistent cache. The owner must
hold the worker lease, and must not resume with a different train/checkpoint
while the decision is outstanding. On a new process, use normal full recovery.
"""
import hashlib
import os
import stat
import tempfile
from contextlib import contextmanager

from decision_transaction import Transaction, canonical, digest, immutable, read_json, sync_dir


def _stream_digest(path):
    """Hash one opened file, using O_NOFOLLOW and bounded buffering.

    Hashing verifies bytes as read; it does not promise that the backing file
    cannot subsequently be modified by a separate writer.
    """
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    fd = os.open(path, flags)
    try:
        # This checks file type only; identity/mtime/size never establish
        # integrity. O_NONBLOCK avoids hanging on a substituted FIFO.
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('checkpoint is not a regular file')
        sha = hashlib.sha256()
        with os.fdopen(fd, 'rb', closefd=False) as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                sha.update(block)
        return sha.hexdigest()
    finally:
        os.close(fd)


def _publish_checkpoint(path, data):
    """Publish without allocating a second full checkpoint on replay."""
    if path.is_symlink():
        raise ValueError('unsafe checkpoint symlink')
    if path.exists():
        return  # The following strict, streamed validation checks its bytes.
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(data); out.flush(); os.fsync(out.fileno())
        try:
            os.link(tmp, path, follow_symlinks=False)
        except FileExistsError:
            pass  # Validated, not trusted, below.
        sync_dir(path.parent)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


class ScopedTransaction(Transaction):
    """Transaction scoped to one decision and one immutable pre-state."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scoped = False
        self._snapshot = None
        self._scope_manifest = None
        self._strict = False
        self._failed = False

    @contextmanager
    def scope(self):
        if self._scoped:
            raise RuntimeError('nested transaction scope is forbidden')
        self._scoped = True
        try:
            yield self
        finally:
            self._snapshot = self._scope_manifest = None
            self._strict = self._failed = self._scoped = False

    def prepare(self, request, pre_checkpoint, checkpoint_validate):
        if not self._scoped:
            return super().prepare(request, pre_checkpoint, checkpoint_validate)
        if self._snapshot is not None or self._failed:
            raise RuntimeError('scope already prepared or failed')
        if not isinstance(request, dict) or not isinstance(pre_checkpoint, bytes) or not callable(checkpoint_validate):
            raise ValueError('invalid prepare inputs')
        checkpoint_validate(pre_checkpoint, self.identity['epoch'])
        req = canonical(request)
        manifest = {'version': 'agent_transaction_v1', 'identity': self.identity,
                    'request_sha256': digest(req), 'pre_checkpoint_sha256': digest(pre_checkpoint)}
        try:
            with self.locked():
                immutable(self.root / 'request.json', req)
                _publish_checkpoint(self.root / 'pre_checkpoint.bin', pre_checkpoint)
                immutable(self.root / 'manifest.json', canonical(manifest))
                self._snapshot = pre_checkpoint
                self._scope_manifest = manifest
                self._strict = True
                try:
                    self._require()
                finally:
                    self._strict = False
        except BaseException:
            self._failed = True
            raise
        return manifest

    def _require(self):
        if not self._scoped:
            return super()._require()
        if self._failed or self._snapshot is None or self._scope_manifest is None:
            raise RuntimeError('scope has no verified checkpoint')
        manifest, _ = read_json(self.root / 'manifest.json')
        if manifest != self._scope_manifest or manifest.get('identity') != self.identity:
            raise ValueError('transaction manifest changed')
        request_path = self.root / 'request.json'
        if request_path.is_symlink():
            raise ValueError('unsafe request symlink')
        req = request_path.read_bytes()
        if digest(req) != manifest['request_sha256']:
            raise ValueError('transaction request changed')
        # The cached object is the caller's immutable bytes, not a file-stat
        # shortcut. It has the digest established by prepare().
        if self._strict and _stream_digest(self.root / 'pre_checkpoint.bin') != manifest['pre_checkpoint_sha256']:
            raise ValueError('durable checkpoint changed')
        return manifest, req, self._snapshot

    def claim_submission(self, thread_id):
        if not self._scoped:
            return super().claim_submission(thread_id)
        self._strict = True
        try:
            return super().claim_submission(thread_id)
        except BaseException:
            self._failed = True
            raise
        finally:
            self._strict = False

    def attach_completed_response(self, thread_id, turn_id, response, *, reconciled):
        try:
            return super().attach_completed_response(thread_id, turn_id, response, reconciled=reconciled)
        except BaseException:
            if self._scoped:
                self._failed = True
            raise

    def normalize(self, validator):
        try:
            return super().normalize(validator)
        except BaseException:
            if self._scoped:
                self._failed = True
            raise

    def commit(self, checkpoint_validate, transition, *, post_validate=None):
        if not self._scoped:
            return super().commit(checkpoint_validate, transition, post_validate=post_validate)
        self._strict = True
        try:
            return super().commit(checkpoint_validate, transition, post_validate=post_validate)
        except BaseException:
            self._failed = True
            raise
        finally:
            self._strict = False
