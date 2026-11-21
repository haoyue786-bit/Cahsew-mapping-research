"""Durable conversation identity, independent of numerical training lifetime.

Only an explicitly invoked review can start a model turn. A persisted pending
turn blocks a new submission until its immutable publication is reconciled.
"""
from pathlib import Path
from on_demand_memory import read, write, sha
from persistent_luna import PersistentLunaRecoveryRequired

LONG = 'persistent_group_on_demand_v1'
SHORT = 'fresh_review_on_demand_v1'

def output_root(packet):
    advertised = packet.get('runtime_artifacts', {}).get('output_root')
    if advertised:
        return Path(advertised)
    # Initialization packets precede the experiment capability which supplies
    # runtime_artifacts. Resolve their canonical local job, then verify identity;
    # never search another run or infer an owner's directory from history.
    job_id = packet['job']['job_id']
    if Path(job_id).name != job_id:
        raise ValueError('Invalid local job identity')
    root = Path(__file__).resolve().parents[2] / 'jobs' / job_id
    canonical = read(root / 'job.json')
    if any(canonical.get(k) != packet['job'].get(k)
           for k in ['job_id', 'run_id', 'thread_key']):
        raise ValueError('Initialization owner differs from canonical local job')
    if Path(canonical['output_root']).resolve() != root.resolve():
        raise ValueError('Canonical initialization output path mismatch')
    return root

def policy(packet):
    root = output_root(packet)
    value = read(root / 'job.json').get('session_memory_policy', SHORT)
    if value not in (SHORT, LONG):
        raise ValueError('Unsupported session memory policy')
    return value

def owner_path(packet):
    return output_root(packet) / 'conversation_owner.json'

def identity(packet, model, effort, skill):
    return dict(owner=packet['job']['thread_key'], model=model,
                reasoning_effort=effort, skill_sha256=sha(skill))

def reconciled_owner(packet, expected):
    path = owner_path(packet)
    if not path.exists():
        return None
    data = read(path)
    if any(data.get(k) != v for k, v in expected.items()):
        raise ValueError('Persistent group identity/model/Skill changed')
    if data['state'] != 'completed':
        request = Path(data['request'])
        pub = request.with_name('publication_receipt.json')
        if pub.exists():
            receipt = read(pub)
            valid = (request.is_file() and receipt['request_sha256'] == sha(request)
                     and all(Path(p).is_file() and sha(p) == h
                             for p, h in receipt['files'].items()))
            if valid:
                prov = read(request.with_name('provenance.json'))
                if prov['thread_id'] != data['thread_id']:
                    raise ValueError('Publication belongs to a different group thread')
                data.update(state='completed', turn_id=prov['turn_id'])
                write(path, data)
        if data['state'] != 'completed':
            raise PersistentLunaRecoveryRequired(
                'Previous group review has no verified publication; reconcile it before another turn')
    return data

def begin(client, packet, tools, request, model, effort, skill):
    if policy(packet) == SHORT:
        return client.start_owner_session(packet, tools), None, False
    expected = identity(packet, model, effort, skill)
    owner = reconciled_owner(packet, expected)
    if owner:
        client.resume_thread(thread_id=owner['thread_id'], cwd=request.parent/'memory_store',
                             model=model, reasoning_effort=effort, sandbox='read-only')
        tid, previous, resumed = owner['thread_id'], owner['turn_id'], True
    else:
        tid, previous, resumed = client.start_owner_session(packet, tools), None, False
    write(owner_path(packet), dict(**expected, thread_id=tid, state='started',
                                  request=str(request.resolve()), request_sha256=sha(request),
                                  previous_turn_id=previous, epoch=packet['epoch']))
    return tid, previous, resumed

def complete(packet, request, result, model, effort, skill):
    if policy(packet) != LONG:
        return
    expected = identity(packet, model, effort, skill)
    existing = read(owner_path(packet))
    if existing['thread_id'] != result['thread_id']:
        raise ValueError('Cannot publish another group conversation')
    write(owner_path(packet), dict(**expected, thread_id=result['thread_id'],
                                  state='completed', request=str(request.resolve()),
                                  request_sha256=sha(request), epoch=packet['epoch'],
                                  turn_id=result['turn_id']))
