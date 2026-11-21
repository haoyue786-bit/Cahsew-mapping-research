"""Run one host's trained, annotated, selected and exported Cashew groups."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_group import ROOT, WAIT_EXIT, load_job
from export_selected import PREDICTION_BATCH, YIELD_EXPORT
from hard_sample_training import enabled, read, save


PREDICTION_STARTUP_MIB = 4096


@dataclass
class Worker:
    phase: str
    gpu: object
    process: subprocess.Popen


def state(job):
    root = Path(job['output_root'])
    if (root / 'frozen_selection.json').is_file():
        return 'complete', read(root / 'frozen_selection.json')['stop_epoch']
    rounds = sorted((root / 'journal').glob('epoch_*'))
    if not rounds:
        return 'train', None
    latest = rounds[-1]
    number = int(latest.name.split('_')[1])
    trained, commit = latest / 'trained.json', latest / 'commit.json'
    if trained.is_file() and read(trained)['route'] == 'SAFETY_STOP':
        return 'safety_stop', number
    if commit.is_file():
        action = read(commit)['action']
        return ('select' if action == 'STOP' or number == job['max_epoch'] else 'train'), number
    batch = root / 'hard_samples' / latest.name
    if enabled(job) and (batch / 'screen.json').is_file() and not (batch / 'ready.json').is_file():
        return 'annotate', number
    return 'train', number


def write_status(path, value):
    next_path = path.with_suffix('.tmp')
    next_path.write_text(json.dumps(value, indent=2) + '\n')
    next_path.replace(path)


def opened_slots(slots, wave_size, interval, elapsed):
    wave = int(max(0, elapsed) // interval) + 1
    return {gpu: min(cap, wave_size * wave) for gpu, cap in slots.items()}


def devices(sam_unit):
    command = ['nvidia-smi', '--format=csv,noheader,nounits']
    gpus = subprocess.check_output(command + ['--query-gpu=index,uuid,memory.free'], text=True)
    compute = subprocess.check_output(command + ['--query-compute-apps=pid,gpu_uuid'], text=True)
    sam_pid = int(os.environ.get('CASHEW_SAM_PID', '0'))
    if sam_unit and not sam_pid:
        result = subprocess.run(
            ['systemctl', '--user', 'show', sam_unit, '-p', 'MainPID', '--value'],
            text=True, capture_output=True, check=False)
        if result.returncode == 0:
            sam_pid = int(result.stdout.strip() or 0)
    occupants = defaultdict(set)
    for row in compute.splitlines():
        if ',' in row:
            pid, uuid = (item.strip() for item in row.split(','))
            if int(pid) != sam_pid:
                occupants[uuid].add(int(pid))
    result = {}
    for row in gpus.splitlines():
        index, uuid, free = (item.strip() for item in row.split(','))
        result[int(index)] = (int(free), occupants[uuid])
    return result


def gpu_available(capacities, snapshot, workers, min_free):
    held = defaultdict(set)
    for item in workers:
        if item.gpu is not None:
            held[item.gpu].add(item.process.pid)
    options = []
    for gpu, limit in capacities.items():
        free, processes = snapshot[gpu]
        count = len(held[gpu] | processes)
        if count < limit and free >= min_free:
            options.append((count, -free, gpu))
    return min(options)[2] if options else None


def start(job, worker, phase, *, gpu=None, epoch=None, batch=None):
    folder = Path(job['output_root'])
    command = [sys.executable, '-B', '-u', str(ROOT / worker), phase, str(folder)]
    if gpu is not None:
        command += ['--gpu', str(gpu)]
    if epoch is not None:
        command += ['--epoch', str(epoch)]
    if batch is not None:
        command += ['--batch', str(batch)]
    with (folder / (phase + '.log')).open('a', buffering=1) as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    print(json.dumps(dict(job=job['job_id'], started=phase, epoch=epoch, gpu=gpu, pid=process.pid,
                          time=time.time())), flush=True)
    return Worker(phase, gpu, process)


def reap(live, failures, *, exports=False, training_gpus=None):
    for name, item in tuple(live.items()):
        code = item.process.poll()
        if code is None:
            continue
        allowed = {0}
        if exports and item.phase == 'predict':
            allowed.update((YIELD_EXPORT, -signal.SIGUSR1))
        if not exports and item.phase == 'train':
            allowed.add(WAIT_EXIT)
        if code not in allowed:
            failures[name] = dict(phase=item.phase, exit_code=code)
        if training_gpus is not None and item.phase == 'train' and code == 0:
            training_gpus[name] = item.gpu
        print(json.dumps(dict(job=name, finished=item.phase, gpu=item.gpu, exit_code=code)), flush=True)
        del live[name]


def last_training_gpus(log, previous, jobs):
    """Recover the GPU of the worker that finished each group's training."""
    assigned = {}
    if log.is_file():
        with log.open() as stream:
            for line in stream:
                if not line.startswith('{'):
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if (event.get('finished') == 'train' and event.get('exit_code') == 0
                        and event.get('job') in jobs and type(event.get('gpu')) is int):
                    assigned[event['job']] = event['gpu']
    assigned.update({name: row['training_gpu'] for name, row in previous.items()
                     if name in jobs and type(row.get('training_gpu')) is int})
    return assigned


def prediction_slots(active, requested, snapshot, minimum_free):
    """One prediction per assigned GPU, in the same order as the export queue."""
    running = {item.gpu for item in active.values() if item.phase == 'predict'}
    for name, gpu in requested.items():
        if gpu in snapshot and gpu not in running and snapshot[gpu][0] >= minimum_free:
            yield name, gpu
            running.add(gpu)


def reserve_starting_memory(snapshot, workers, reserve_mib):
    """Include launched workers whose CUDA allocations have yet to appear."""
    remaining = dict(snapshot)
    for worker in workers:
        if worker.gpu is not None:
            free, pids = remaining[worker.gpu]
            if worker.process.pid not in pids:
                remaining[worker.gpu] = (free - reserve_mib, pids)
    return remaining


def phase_progress(job, information, training, exports, failures, export_failures,
                   training_gpus, prediction_gpu):
    name = job['job_id']
    folder = Path(job['output_root'])
    phase, epoch = information[name]
    current = dict(phase=phase, epoch=epoch, job=name, host=job['assigned_host'], export='pending_select')
    if name in failures:
        current['failure'] = failures[name]
    if name in training:
        item = training[name]
        current.update(active=item.phase, gpu=item.gpu, pid=item.process.pid)
    if name in training_gpus:
        current['training_gpu'] = training_gpus[name]
    if phase == 'complete':
        current.update(selected_epoch=read(folder / 'frozen_selection.json')['selected_epoch'], export='queued_inference')
        if prediction_gpu is not None:
            current['prediction_gpu'] = prediction_gpu
        if (folder / 'maps/export_receipt.json').is_file():
            current['export'] = 'complete'
            current['export_receipt'] = str(folder / 'maps/export_receipt.json')
        elif (folder / 'prediction_cache/prediction.json').is_file():
            current['export'] = 'queued_packaging'
        if name in exports:
            item = exports[name]
            current.update(export=item.phase, export_gpu=item.gpu, export_pid=item.process.pid)
        if name in export_failures:
            current.update(export='failed', export_failure=export_failures[name])
    return current


def execute(plan, host, year, jobs, interval, min_free, handoff=None, retry_jobs=(), include_2019=False):
    profile = plan['hosts'][host]
    slots = {int(gpu): cap for gpu, cap in profile['gpu_slots'].items()}
    wave_size = profile['wave_size']
    wave_interval = plan['wave_interval_seconds']
    launch_gap = plan['between_group_starts_seconds']
    signature = ROOT / f'schedule_{host}_{year}.json'
    status = ROOT / f'queue_status_{host}_{year}.json'
    earlier_status = ROOT / f'queue_status_{host}_2019.json'
    earlier_assignments = {x['name'].replace('_y2025_', '_y2019_'): x['name']
                           for x in plan['jobs'] if include_2019 and x['host'] == host}
    anchor = read(signature)['first_gpu_start'] if signature.exists() else None
    previous = read(status) if status.exists() else {}
    previous_jobs = dict(previous.get('jobs', {}))
    if include_2019 and earlier_status.is_file():
        previous_jobs.update(read(earlier_status)['jobs'])
    training_gpus = last_training_gpus(ROOT / f'queue_{host}.log', previous_jobs,
                                       set(jobs) | earlier_assignments.keys())
    prediction_overrides = {name: gpu for name, gpu in plan.get('prediction_gpu_overrides', {}).items()
                            if name in jobs}
    if any(type(gpu) is not int or gpu not in slots for gpu in prediction_overrides.values()):
        raise ValueError('Prediction GPU override is outside the assigned host')
    last_start = previous.get('last_gpu_start', 0)
    last_recovery_annotation = previous.get('last_recovery_annotation_start', 0)
    failures = {name: row['failure'] for name, row in previous_jobs.items()
                if 'failure' in row and name not in retry_jobs}
    export_failures = {name: row['export_failure'] for name, row in previous_jobs.items()
                       if 'export_failure' in row}
    training = dict(handoff.workers) if handoff else {}
    exports = dict(handoff.exports) if handoff else {}
    with ExitStack() as stack:
        def claim(path):
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def claim_regular_ownership():
            claim(ROOT / f'scheduler_{host}_{year}.lock')
            for name, job in sorted(jobs.items()):
                if name not in earlier_assignments:
                    claim(Path(job['output_root']) / 'coordinator.lock')

        if include_2019:
            claim(ROOT / f'scheduler_{host}_2019.lock')

        if handoff:
            claim(ROOT / handoff.lock_name)
        else:
            claim_regular_ownership()
        while True:
            now = time.time()
            if handoff:
                handoff.verify()
            reap(training, failures, training_gpus=training_gpus)
            reap(exports, export_failures, exports=True)
            if handoff and not any(training.get(name) is item for name, item in handoff.workers.items()) \
                    and not any(exports.get(name) is item for name, item in handoff.exports.items()):
                handoff.retire()
                claim_regular_ownership()
                handoff = None
            if include_2019:
                for name, parent in earlier_assignments.items():
                    folder = ROOT / 'jobs' / name
                    if name in jobs or not (folder / 'job_candidate.json').is_file():
                        continue
                    child = load_job(folder)
                    if (child['assigned_host'], child['target_year'],
                        child['initial_transfer_weights_v3']['parent_job']) != (host, 2019, parent):
                        raise ValueError(f'Conflicting earlier-period job assignment: {name}')
                    claim(folder / 'coordinator.lock')
                    jobs[name] = child
            information = {name: state(job) for name, job in jobs.items()}
            available = opened_slots(slots, wave_size, wave_interval, now - anchor if anchor else 0)
            snapshot = devices(profile['sam_unit'])
            predictors = [item for item in exports.values() if item.phase == 'predict']
            prediction_pids = {item.process.pid for item in predictors}
            training_snapshot = {gpu: (free, pids - prediction_pids)
                                 for gpu, (free, pids) in reserve_starting_memory(
                                     snapshot, predictors, PREDICTION_STARTUP_MIB).items()}

            # Label workers release their training reservation while the
            # registered candidate cases and the Agent are being processed.
            for name, job in jobs.items():
                phase, epoch = information[name]
                if phase == 'annotate' and name not in training and name not in failures:
                    recovery = (Path(job['output_root']) / 'hard_samples/acquisition_generation.json').is_file()
                    if recovery and epoch == 5 and now - last_recovery_annotation < launch_gap:
                        continue
                    training[name] = start(job, 'run_group.py', 'annotate', epoch=epoch)
                    if recovery and epoch == 5:
                        last_recovery_annotation = now

            demand = [name for name, (phase, _) in information.items()
                      if phase in ('train', 'select') and name not in training and name not in failures]
            demand.sort(key=lambda name: (jobs[name]['target_year'] != 2025, information[name][0] != 'select',
                                         information[name][1] is None))
            if demand and now - last_start >= launch_gap:
                gpu = gpu_available(available, training_snapshot, training.values(), min_free)
                if gpu is not None:
                    name = demand.pop(0)
                    training[name] = start(jobs[name], 'run_group.py', information[name][0], gpu=gpu)
                    last_start = now
                    if anchor is None:
                        anchor = now
                        save(signature, dict(host=host, year=year, first_gpu_start=anchor, wave_seconds=wave_interval))

            # Export uses a frozen predictor and can proceed beside the training queue.
            ready = [name for name, (phase, _) in information.items() if phase == 'complete'
                     and name not in exports and name not in training and name not in export_failures
                     and not (Path(jobs[name]['output_root']) / 'maps/export_receipt.json').is_file()]
            waiting_for_package = [name for name in ready
                                   if (Path(jobs[name]['output_root']) / 'prediction_cache/prediction.json').is_file()]
            if waiting_for_package and not any(item.phase == 'package' for item in exports.values()):
                name = waiting_for_package[0]
                exports[name] = start(jobs[name], 'export_selected.py', 'package')
            prediction_gpus = {**training_gpus, **prediction_overrides}
            pending_prediction = {name: prediction_gpus.get(name) for name in ready
                                  if name not in waiting_for_package}
            prediction_snapshot = reserve_starting_memory(snapshot, training.values(), min_free)
            for name, gpu in prediction_slots(exports, pending_prediction, prediction_snapshot, max(10000, min_free)):
                exports[name] = start(jobs[name], 'export_selected.py', 'predict',
                                      gpu=gpu, batch=PREDICTION_BATCH)

            progress = {name: phase_progress(job, information, training, exports, failures, export_failures,
                                             training_gpus, prediction_gpus.get(name))
                        for name, job in jobs.items()}
            early = {}
            if include_2019:
                prep = ROOT / f'prepare_status_{host}_2019.json'
                preparation = read(prep).get('failures', {}) if prep.is_file() else {}
                for name, parent in earlier_assignments.items():
                    if name in progress:
                        early[name] = progress.pop(name)
                    else:
                        frozen = (ROOT / 'jobs' / parent / 'frozen_selection.json').is_file()
                        phase = 'prepare_failed' if parent in preparation else ('preparing' if frozen else 'awaiting_2025_selection')
                        early[name] = dict(job=name, phase=phase, parent_job=parent, error=preparation.get(parent))
                write_status(earlier_status, dict(host=host, year=2019, timestamp=now, jobs=early))
            write_status(status, dict(host=host, year=year, timestamp=now, first_gpu_start=anchor,
                                      last_gpu_start=last_start, last_recovery_annotation_start=last_recovery_annotation,
                                      active_slots=available, total_slots=slots, jobs=progress))
            deliveries = (*progress.values(), *early.values())
            if not training and not exports and all(
                    'failure' in row or row['phase'] in ('safety_stop', 'prepare_failed') or
                    (row['phase'] == 'complete' and row['export'] in ('complete', 'failed'))
                    for row in deliveries):
                return int(bool(failures or export_failures or any(
                    row['phase'] in ('safety_stop', 'prepare_failed') for row in deliveries)))
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True, choices=('local', '124'))
    parser.add_argument('--year', type=int, default=2025, choices=(2025, 2019))
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--minimum-free-mib', type=int, default=8000)
    parser.add_argument('--include-2019', action='store_true')
    parser.add_argument('--retry-job', action='append', default=[])
    args = parser.parse_args()
    if args.interval <= 0 or args.minimum_free_mib <= 0:
        parser.error('Polling and resource limits must be positive')
    if args.include_2019 and args.year != 2025:
        parser.error('Combined mode uses the current 2025 coordinator')
    from release_identity_v2 import verify
    verify(ROOT)
    plan = read(ROOT / 'cohort_2025.json')
    jobs = {}
    for assignment in plan['jobs']:
        if assignment['host'] == args.host:
            name = assignment['name'].replace('_y2025_', f'_y{args.year}_')
            job = load_job(ROOT / 'jobs' / name)
            if job['assigned_host'] != args.host or job['target_year'] != args.year:
                raise ValueError(f'Conflicting job assignment: {name}')
            jobs[name] = job
    if not jobs:
        raise ValueError('No jobs registered for this host')
    retry_scope = set(jobs)
    if args.include_2019:
        retry_scope.update(name.replace('_y2025_', '_y2019_') for name in jobs)
    if set(args.retry_job) - retry_scope:
        parser.error('Each retry must name a job assigned to this host and year')
    return execute(plan, args.host, args.year, jobs, args.interval, args.minimum_free_mib,
                   retry_jobs=args.retry_job, include_2019=args.include_2019)


if __name__ == '__main__':
    sys.exit(main())
