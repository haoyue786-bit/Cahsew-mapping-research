"""Candidate automatic-negative transaction. NOT wired to the live trainer.

The checkpoint is authoritative for pool generation, cursor and retirement.
This module does not mine or validate negative pixels. A trusted worker must
independently attest the sealed review, actual mask provenance, checkpoint
lineage, and (for advance) the exact contiguous positions attempted by the
training loop. Hashes alone do not establish those facts. Checkpoint callbacks
must be deterministic, side-effect-free and preserve model/optimizer/RNG.

A crash with an orphan checkpoint or intent requires manual reconciliation;
recovery never silently regenerates a potentially different checkpoint.
"""
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from decision_transaction import canonical, digest, immutable, read_json, sync_dir

HEX = re.compile(r'^[a-f0-9]{64}$')
NAME = re.compile(r'^[A-Za-z0-9_.-]{1,96}$')


def _sha(value):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError('invalid sha256')
    return value


def _name(value):
    if not isinstance(value, str) or not NAME.fullmatch(value) or value in ('.', '..'):
        raise ValueError('invalid identifier')
    return value


def _atomic(path, data):
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path); sync_dir(path.parent)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _state(value):
    if not isinstance(value, dict) or set(value) != {'epoch', 'pool', 'retired'}:
        raise ValueError('invalid checkpoint pool state')
    epoch, pool, retired = value['epoch'], value['pool'], value['retired']
    if type(epoch) is not int or epoch < 0 or not isinstance(retired, list):
        raise ValueError('invalid epoch or retired generations')
    if len(set(retired)) != len(retired): raise ValueError('duplicate retired generation')
    for item in retired: _sha(item)
    if pool is not None:
        if not isinstance(pool, dict) or set(pool) != {'generation', 'manifest_sha256', 'cursor', 'horizon', 'weight'}:
            raise ValueError('invalid active generation')
        _sha(pool['generation']); _sha(pool['manifest_sha256'])
        if pool['generation'] in retired: raise ValueError('retired generation cannot be active')
        if type(pool['cursor']) is not int or type(pool['horizon']) is not int or not 0 <= pool['cursor'] < pool['horizon'] <= 100000:
            raise ValueError('invalid active cursor')
        if type(pool['weight']) not in (int, float) or not math.isfinite(pool['weight']) or pool['weight'] <= 0:
            raise ValueError('invalid pool weight')
    return value


class PoolTransaction:
    def __init__(self, root, *, checkpoint_state, transition, attest):
        if not all(callable(v) for v in (checkpoint_state, transition, attest)):
            raise ValueError('trusted worker callbacks required')
        self.root = Path(root)
        self.checkpoint_state = checkpoint_state
        self.transition = transition
        self.attest = attest

    def _lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink(): raise ValueError('pool journal symlink forbidden')
        sync_dir(self.root.parent)
        fd = os.open(self.root / '.lock', os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd

    def initialize(self, checkpoint):
        if not isinstance(checkpoint, bytes): raise ValueError('checkpoint bytes required')
        state = _state(self.checkpoint_state(checkpoint))
        if state['pool'] is not None or state['retired']:
            raise ValueError('initial checkpoint must have no pool history; resume from existing journal')
        if self.attest(operation='initialize', input_checkpoint=checkpoint, expected_state=state) is not True:
            raise ValueError('initial checkpoint provenance attestation failed')
        fd = self._lock()
        try:
            path = self.root / 'initial.bin'
            immutable(path, checkpoint)
            head = {'event': None, 'checkpoint_sha256': digest(checkpoint), 'checkpoint': 'initial.bin', 'state': state}
            if not (self.root / 'head.json').exists(): immutable(self.root / 'head.json', canonical(head))
            found, _ = read_json(self.root / 'head.json')
            if found.get('event') is None and found != head: raise ValueError('initial checkpoint mismatch')
            return self._head()
        finally: os.close(fd)

    def _head(self):
        h, _ = read_json(self.root / 'head.json')
        if set(h) != {'event', 'checkpoint_sha256', 'checkpoint', 'state'}: raise ValueError('bad head')
        _sha(h['checkpoint_sha256']); _state(h['state'])
        if h['event'] is None:
            if h['checkpoint'] != 'initial.bin': raise ValueError('bad initial head')
        else:
            _name(h['event'])
            if h['checkpoint'] != 'events/' + h['event'] + '/post.bin': raise ValueError('bad head path')
        p = self.root / h['checkpoint']
        if p.is_symlink(): raise ValueError('symlink forbidden')
        data = p.read_bytes()
        if digest(data) != h['checkpoint_sha256'] or _state(self.checkpoint_state(data)) != h['state']:
            raise ValueError('head checkpoint mismatch')
        if h['state']['pool'] is not None:
            self._verify_active_assets(h['state']['pool'])
        return h

    def _verify_active_assets(self, pool):
        # The event owning the assets remains immutable even after later epochs.
        # Refuse to resume training on missing, substituted or partial masks.
        events = self.root / 'events'
        matches = []
        if events.is_dir():
            for event in events.iterdir():
                path = event / 'manifest.json'
                if event.is_symlink() or not event.is_dir() or not path.is_file() or path.is_symlink():
                    continue
                raw = path.read_bytes()
                if digest(raw) == pool['manifest_sha256']:
                    matches.append((event, json.loads(raw)))
        if len(matches) != 1:
            raise ValueError('active pool manifest missing or ambiguous')
        event, manifest = matches[0]
        if manifest.get('generation') != pool['generation'] or manifest.get('horizon') != pool['horizon'] or manifest.get('weight') != pool['weight'] or not isinstance(manifest.get('masks'), dict) or not manifest['masks']:
            raise ValueError('active pool manifest mismatch')
        for name, sha in manifest['masks'].items():
            _name(name); _sha(sha); p = event / 'masks' / name
            if p.is_symlink() or not p.is_file() or digest(p.read_bytes()) != sha:
                raise ValueError('active pool mask missing or changed')

    def apply(self, event_id, operation, *, input_checkpoint=None, manifest=None,
              masks=None, consumed_positions=None):
        """Commit publish/advance/rollback, or recover an already-ready event.

        `advance` consumes contiguous schedule positions attempted by training,
        not merely planned iterations or number of successful gradient updates.
        The caller must attest them against the *saved* full training checkpoint.
        An interrupted unsaved step is replayed only from that earlier complete
        model+optimizer+RNG checkpoint, with no cursor advance.
        """
        _name(event_id)
        if operation not in ('publish', 'advance', 'rollback'): raise ValueError('bad operation')
        fd = self._lock()
        try:
            head = self._head(); events = self.root / 'events'
            if events.is_symlink(): raise ValueError('events symlink forbidden')
            events.mkdir(exist_ok=True); sync_dir(self.root)
            root = events / event_id
            if root.is_symlink(): raise ValueError('event symlink forbidden')
            root.mkdir(exist_ok=True); sync_dir(events)
            spec = {'version': 'automatic_pool_transaction_v1', 'event': event_id,
                    'operation': operation, 'head': head,
                    'input_checkpoint_sha256': digest(input_checkpoint) if isinstance(input_checkpoint, bytes) else None,
                    'manifest_sha256': digest(canonical(manifest)) if isinstance(manifest, dict) else None,
                    'mask_sha256': {k: digest(v) for k, v in sorted((masks or {}).items())} if isinstance(masks, dict) else None,
                    'consumed_positions': consumed_positions}
            # Once committed, repeat returns the same result only for the same
            # original request; head may have advanced since then.
            if (root / 'intent.json').exists():
                prior, _ = read_json(root / 'intent.json')
                comparison = dict(spec); comparison['head'] = prior['head']
                comparison['expected_state'] = prior.get('expected_state')
                if comparison != prior: raise ValueError('event identity reused for different request')
                if not (root / 'ready.json').exists(): raise RuntimeError('unresolved pool intent: reconcile manually')
                return self._finish(root, prior, head)
            old = head['state']; new = json.loads(json.dumps(old))
            if operation == 'publish':
                if old['pool'] is not None or input_checkpoint is not None or consumed_positions is not None:
                    raise ValueError('publish requires no active pool')
                if not isinstance(manifest, dict) or set(manifest) != {'generation', 'origin_checkpoint_sha256', 'review_post_checkpoint_sha256', 'catalog_sha256', 'decision_sha256', 'horizon', 'weight', 'masks'}:
                    raise ValueError('invalid manifest schema')
                for k in ('generation','origin_checkpoint_sha256','review_post_checkpoint_sha256','catalog_sha256','decision_sha256'): _sha(manifest[k])
                if manifest['review_post_checkpoint_sha256'] != head['checkpoint_sha256'] or manifest['generation'] in old['retired']:
                    raise ValueError('stale review or retired generation')
                if type(manifest['horizon']) is not int or not 1 <= manifest['horizon'] <= 100000 or type(manifest['weight']) not in (int,float) or not math.isfinite(manifest['weight']) or manifest['weight'] <= 0:
                    raise ValueError('invalid manifest schedule')
                if not isinstance(masks, dict) or not masks or not isinstance(manifest['masks'], dict) or set(masks) != set(manifest['masks']): raise ValueError('missing masks')
                for k,v in masks.items():
                    _name(k)
                    if not isinstance(v, bytes) or not v or digest(v) != _sha(manifest['masks'][k]): raise ValueError('mask content mismatch')
                new['pool'] = {'generation':manifest['generation'], 'manifest_sha256':spec['manifest_sha256'], 'cursor':0, 'horizon':manifest['horizon'], 'weight':manifest['weight']}
            elif operation == 'advance':
                pool = old['pool']
                if pool is None or not isinstance(input_checkpoint, bytes) or manifest is not None or masks is not None:
                    raise ValueError('advance requires active pool and saved training checkpoint')
                trained = _state(self.checkpoint_state(input_checkpoint))
                if trained['epoch'] != old['epoch'] + 1 or trained['pool'] != pool or trained['retired'] != old['retired']:
                    raise ValueError('training checkpoint state/epoch mismatch')
                if type(consumed_positions) is not int or not 0 <= consumed_positions <= pool['horizon']-pool['cursor']:
                    raise ValueError('invalid consumed schedule positions')
                new['epoch'] = trained['epoch']; new['pool']['cursor'] += consumed_positions
                if new['pool']['cursor'] == new['pool']['horizon']:
                    new['retired'].append(pool['generation']); new['pool'] = None
            else:
                if old['pool'] is None or input_checkpoint is not None or manifest is not None or masks is not None or consumed_positions is not None:
                    raise ValueError('rollback requires active pool')
                new['retired'].append(old['pool']['generation']); new['pool'] = None
            _state(new)
            if self.attest(operation=operation, head=head, input_checkpoint=input_checkpoint,
                           manifest=manifest, masks=masks, consumed_positions=consumed_positions, expected_state=new) is not True:
                raise ValueError('trusted provenance/lineage attestation failed')
            spec['expected_state'] = new
            immutable(root / 'intent.json', canonical(spec))
            if operation == 'publish':
                immutable(root / 'manifest.json', canonical(manifest)); (root/'masks').mkdir(exist_ok=True); sync_dir(root)
                if (root/'masks').is_symlink(): raise ValueError('masks symlink forbidden')
                for k, v in masks.items(): immutable(root/'masks'/k, v)
            base = input_checkpoint if operation == 'advance' else (self.root/head['checkpoint']).read_bytes()
            post = self.transition(base, new)
            if not isinstance(post, bytes) or _state(self.checkpoint_state(post)) != new:
                raise ValueError('transition did not embed expected state')
            immutable(root/'post.bin', post)
            immutable(root/'ready.json', canonical({'intent_sha256':digest(canonical(spec)), 'post_sha256':digest(post), 'state':new}))
            return self._finish(root, spec, head)
        finally: os.close(fd)

    def _finish(self, root, spec, head):
        ready, _ = read_json(root/'ready.json'); post = (root/'post.bin').read_bytes()
        if ready.get('intent_sha256') != digest(canonical(spec)) or ready.get('post_sha256') != digest(post) or _state(self.checkpoint_state(post)) != ready.get('state') or ready.get('state') != spec.get('expected_state'):
            raise ValueError('ready checkpoint mismatch')
        if spec['operation'] == 'publish':
            manifest, mb = read_json(root/'manifest.json')
            if digest(mb) != spec['manifest_sha256'] or set(manifest['masks']) != set(spec['mask_sha256']): raise ValueError('manifest changed')
            for key, val in spec['mask_sha256'].items():
                p = root/'masks'/key
                if p.is_symlink() or digest(p.read_bytes()) != val or manifest['masks'][key] != val: raise ValueError('mask changed')
        new_head = {'event':spec['event'], 'checkpoint_sha256':ready['post_sha256'], 'checkpoint':'events/'+spec['event']+'/post.bin', 'state':ready['state']}
        if head != spec['head'] and head != new_head:
            raise RuntimeError('head moved: refusing stale pool event')
        immutable(root/'commit.json', canonical({'intent_sha256':digest(canonical(spec)), 'head':new_head}))
        if head == spec['head']:
            _atomic(self.root/'head.json', canonical(new_head)); return new_head
        if head == new_head: return new_head
        raise AssertionError('unreachable head state')
