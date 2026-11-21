"""Durable queue for image annotation, with a separate simulated backend."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from formal_training_account import codex_binary
import fcntl
import os
import subprocess
import sys

from hard_sample_scan import CATEGORIES, decode_mask, inventory
from hard_sample_training import (NoTrainableAnnotations, identity, manifest, mode, read,
                                  require_trainable_annotations, round_dir, save, sha, work_root)

PROJECT = Path('@WORKSPACE@')
ANNOTATOR = PROJECT / '.codex_training_current_account/skills/cashew-sample-assist'
LIBRARY = PROJECT / 'label_pools/cashew_sample_assist_accepted_v1'
WAIT_EXIT = 76


class WaitingForAnnotations(Exception):
    pass


class AnnotationDatesIneligible(ValueError):
    pass


def grid(candidate, year):
    import rasterio
    with rasterio.open(candidate['aef']) as raster:
        if raster.shape != (128,128) or not raster.crs:
            raise ValueError('Expected original georeferenced AEF128')
        return dict(label_period=year, native_crs=raster.crs.to_string(),
                    native_transform=list(raster.transform)[:6], aef_path=str(Path(candidate['aef']).resolve()))


def grid_key(grid_record):
    return identity({k:grid_record[k] for k in ('label_period','native_crs','native_transform')})


def directory_for(job):
    # Mock labels may be reused in this release's isolated tests only.
    return work_root(job).parent.parent.parent / 'simulated_sample_library' if mode(job) == 'simulated' else LIBRARY


def checked_label(candidate, year, receipt_path, backend):
    import numpy as np
    import rasterio
    receipt = read(receipt_path)
    if receipt['status'] != 'completed':
        return None
    original = grid(candidate, year)
    if any(receipt['grid'].get(k) != original[k] for k in ('label_period','native_crs','native_transform')):
        raise ValueError('Annotation receipt identifies a different AEF grid or period')
    label = Path(receipt['label_geotiff']).resolve(strict=True)
    if receipt_path.parent.resolve() not in label.parents or suffix_not_tiff(label):
        raise ValueError('Annotation must be a case-owned GeoTIFF')
    dates = sorted(date.fromisoformat(d) for d in receipt['acquisition_dates'])
    if not dates or dates[0] < date(2017 if year == 2019 else 2024,1,1) or dates[-1] > date.today():
        raise AnnotationDatesIneligible('Annotation imagery dates not eligible')
    if year == 2019 and dates[-1] > date(2020,12,31):
        raise ValueError('Annotation imagery outside the 2019 window')
    if backend == 'sample_assist':
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location('pinned_imagery_rules', ANNOTATOR/'scripts/imagery_policy.py')
        rules = module_from_spec(spec); spec.loader.exec_module(rules)
        if (receipt.get('simulated') or (receipt.get('model'),receipt.get('effort')) != ('gpt-6.2-sol','max')
                or receipt.get('imagery_provider') not in ('Google','Google Earth','Esri')
                or not (receipt.get('annotation_id') or receipt.get('reviewed_preview_id'))):
            raise ValueError('Formal visual review identity missing')
        paths = [Path(receipt[key]).resolve(strict=True) for key in ('selection','vision_audit')]
        if any(receipt_path.parent.resolve() not in p.parents for p in paths):
            raise ValueError('Visual audits must be case-owned')
        selection, audit = map(read, paths)
        if (sha(paths[0]) != receipt['selection_sha256'] or selection['status'] != 'selected'
                or selection['grid'] != receipt['grid'] or audit['status'] != 'verified'
                or rules.review_rejection(year, selection['frame'], selection['review'])):
            raise ValueError('Selected image or visual delivery failed verification')
        if receipt.get('annotation_id') or selection.get('selection_method') == 'code_preflight':
            turn = receipt.get('decision_turn_id')
            if (not receipt.get('annotation_id') or not turn or audit.get('decision_turn_id') != turn
                    or not audit.get('verified_image_ids') or audit.get('missing_image_ids')):
                raise ValueError('Visual audit does not verify the annotation decision turn')
        if (selection['frame']['provider'] != receipt['imagery_provider']
                or sorted(selection['frame']['acquisition_dates']) != sorted(receipt['acquisition_dates'])):
            raise ValueError('Selection and annotation imagery identity differ')
    elif receipt.get('simulated') is not True:
        raise ValueError('Simulation cannot consume a real sample receipt')
    if sha(label) != receipt['label_sha256']:
        raise ValueError('Annotation TIFF bytes differ from receipt')
    with rasterio.open(label) as raster, rasterio.open(candidate['aef']) as aef:
        if (raster.count != 1 or raster.shape != (128,128) or raster.crs != aef.crs
                or raster.transform != aef.transform or raster.dtypes != ('uint8',)):
            raise ValueError('Annotation shape, CRS, transform, or pixel type differs')
        if backend == 'sample_assist' and raster.tags().get('simulated') == 'true':
            raise ValueError('Simulated GeoTIFF cannot enter the formal library')
        labels = raster.read(1)
        if ((raster.read_masks(1) == 0) & (labels != 10)).any():
            raise ValueError('GeoTIFF marks a training label outside its valid image mask')
        image = aef.read()
        supported = np.isfinite(image).all(0) & (image != 0).any(0)
    if not np.isin(labels,[0,1,2,3,10]).all():
        raise ValueError('Unexpected annotation class')
    trained = supported & (labels != 10)
    counts = [int(((labels == k) & trained).sum()) for k in range(4)]
    if not sum(counts):
        return None
    sample = dict(key=grid_key(original), path=candidate['path'], aef=candidate['aef'],
                  label=str(label), label_period=year, backend=backend, counts=counts,
                  label_sha256=receipt['label_sha256'], receipt=str(receipt_path),
                  receipt_sha256=sha(receipt_path), acquisition_dates=receipt['acquisition_dates'],
                  imagery_provider=receipt['imagery_provider'])
    return sample, trained


def suffix_not_tiff(path):
    return path.suffix.lower() not in ('.tif', '.tiff')


def sample_assist_command(candidate, year, case, case_root):
    return [sys.executable, str(ANNOTATOR/'scripts/run_sample.py'), '--aef', candidate['aef'],
            '--case', case, '--country', candidate['country'], '--period', str(year),
            '--run-dir', str(case_root)]


def case_prefix(job, epoch):
    prefix = identity(job['run_id'])[:12]
    recovery = work_root(job) / 'acquisition_generation.json'
    if recovery.is_file():
        record = read(recovery)
        if record['run_id'] != job['run_id']:
            raise ValueError('Annotation retry generation belongs to another run')
        generation = record['generation']
        if not generation.isascii() or not generation.isalnum() or not 1 <= len(generation) <= 12:
            raise ValueError('Invalid annotation retry generation')
        if epoch >= record['from_epoch']:
            prefix += '_' + generation
    return prefix


def simulate(candidate, year, case_dir, behavior):
    import numpy as np
    import rasterio
    case_dir.mkdir(parents=True, exist_ok=True)
    receipt = case_dir/'RECEIPT.json'; record = grid(candidate, year)
    if behavior == 'no_imagery':
        save(receipt, dict(status='skipped', simulated=True, grid=record,
                           reason='no_usable_date_qualified_image'))
        return
    if behavior == 'retry_once':
        raise OSError('Simulated transient annotation transport failure')
    label = case_dir/'labels_native128.tif'
    y,x = np.indices((128,128))
    labels = ((x // 64) + 2*(y // 64)).astype('uint8')
    labels[abs(x - 64) < 4] = 10
    if behavior == 'partial':
        labels[:80] = 10
    with rasterio.open(candidate['aef']) as original:
        with rasterio.open(label, 'w', driver='GTiff', count=1, width=128, height=128,
                           dtype='uint8', crs=original.crs, transform=original.transform,
                           compress='deflate') as output:
            output.write(labels, 1)
            output.update_tags(simulated='true', label_period=year, unknown_class=10)
    save(receipt, dict(status='completed', simulated=True, grid=record,
                       imagery_provider='simulated', acquisition_dates=[f'{year}-02-01'],
                       label_geotiff=str(label), label_sha256=sha(label)))


def acquire(job, epoch, candidate, *, ordinal=99):
    year = job['target_year']; backend = mode(job)
    key = grid_key(grid(candidate, year)); bank = directory_for(job) / key[:2]
    bank.mkdir(parents=True, exist_ok=True)
    with (bank / (key+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        published = bank / (key+'.json')
        if published.is_file():
            record = read(published)
            receipt = Path(record['receipt'])
            if record['backend'] != backend or sha(receipt) != record['receipt_sha256']:
                raise ValueError('Shared annotation registry identity differs')
            try:
                reusable = checked_label(candidate, year, receipt, backend)
            except AnnotationDatesIneligible:
                # Preserve the old registry for its original cohort. A late-period
                # replacement is reusable under the new floor on either host.
                published = bank / (key + '.from_2024.json')
                if published.is_file():
                    replacement = read(published)
                    receipt = Path(replacement['receipt'])
                    if replacement['backend'] != backend or sha(receipt) != replacement['receipt_sha256']:
                        raise ValueError('Replacement annotation registry identity differs')
                    reusable = checked_label(candidate, year, receipt, backend)
                else:
                    reusable = None
            if published.is_file():
                if not reusable or reusable[0]['key'] != key:
                    raise ValueError('Shared sample does not match the requested grid')
                return 'reused', reusable
        # Persist per-attempt intent before real submission. Inherited lease prevents
        # concurrent retry while a child from an earlier coordinator is still alive.
        case_root = bank / key / 'cases'; case_root.mkdir(parents=True,exist_ok=True)
        run_prefix = case_prefix(job, epoch)
        for attempt in (1,2):
            case = f'{run_prefix}_e{epoch:03d}_{key[:32]}_{attempt}'
            receipt = case_root/case/'RECEIPT.json'
            intent = round_dir(job,epoch)/'attempts'/(case+'.json')
            if not intent.is_file():
                save(intent, dict(key=key, ordinal=ordinal, attempt=attempt, backend=backend))
                try:
                    if backend == 'sample_assist':
                        env = dict(os.environ)
                        subprocess.run(sample_assist_command(candidate,year,case,case_root),
                                       check=True, env=env, pass_fds=(lock.fileno(),), timeout=7200)
                    else:
                        behavior = ('no_imagery' if ordinal == 0 else 'partial' if ordinal == 1 else
                                    'retry_once' if ordinal == 2 and attempt == 1 else 'success')
                        simulate(candidate, year, case_root/case, behavior)
                except (subprocess.SubprocessError,OSError) as error:
                    save(intent.with_suffix('.error.json'), dict(error=type(error).__name__, message=str(error)))
            if receipt.is_file():
                if read(receipt)['status'] == 'skipped':
                    return 'no_imagery', None
                result = checked_label(candidate,year,receipt,backend)
                if not result:
                    return 'no_trainable_pixels', None
                sample,_ = result
                save(published,{k:sample[k] for k in ('backend','receipt','receipt_sha256')})
                return 'accepted', result
            if not intent.with_suffix('.error.json').is_file():
                return 'interrupted_unresolved', None
        return 'transport_error', None


def choose_plan(job, epoch, screen, pins):
    from hard_sample_contract import planning_request
    from history_dispatch_candidate import dispatch
    from agent_service_slots import call_in_slot
    request = planning_request(epoch, inventory(screen), mode(job))
    stage = round_dir(job,epoch)/'plan_turn'
    return call_in_slot(lambda:dispatch(request,set(),
                        dict(epoch=epoch,history_required=True,hard_sample_inventory=inventory(screen),
                             annotation_backend=mode(job)), stage,
                        codex_binary=codex_binary(), **pins,
                        before_turn=lambda tid:save(stage/'claim.json',dict(thread_id=tid)),
                        history_root=job['output_root'],history_job_id=job['job_id'],history_run_id=job['run_id']))['answer']


def collect_entries(previous_entries, results):
    entries = list(previous_entries)
    keys = {entry['key'] for entry in entries}
    resolved = Counter()
    for ordinal in sorted(results):
        attempt = results[ordinal]
        sample = attempt['entry']
        if sample and sample['key'] not in keys:
            entries.append(sample)
            keys.add(sample['key'])
            resolved[attempt['category']] += int(attempt['resolved'])
    return entries, resolved


def select_samples(screen, remaining, visited, limit):
    selected = []
    for category in CATEGORIES:
        available = min(remaining[category], limit - len(selected))
        for candidate in screen['candidates'][category]:
            if available <= 0:
                break
            path = candidate['path']
            if path not in visited:
                selected.append(dict(category=category, path=path))
                visited.add(path)
                available -= 1
    return selected


def restore_dispatch(root, job, screen, plan, previous_entries, results, candidates):
    assigned = []
    for path in sorted((root / 'dispatch').glob('[0-9][0-9][0-9].json')):
        if int(path.stem) != len(assigned):
            raise ValueError('Annotation dispatch journal has a gap')
        assigned.extend(read(path))

    legacy = not assigned
    if legacy:
        # Older rounds wrote results and case intents in waves of three.
        intents = [read(p) for p in (root / 'attempts').glob('*.json') if not p.name.endswith('.error.json')]
        last = max([*results, *(row['ordinal'] for row in intents)], default=-1)
        while len(assigned) <= last:
            finished = {i: results[i] for i in range(len(assigned)) if i in results}
            if len(finished) != len(assigned):
                raise ValueError('Unfinished legacy annotation wave overlaps a later wave')
            _, resolved = collect_entries(previous_entries, finished)
            used = Counter(row['category'] for row in assigned)
            remaining = {k: min(plan[k] - resolved[k], max(2 * plan[k], plan[k] + 2) - used[k])
                         for k in CATEGORIES}
            visited = {row['path'] for row in previous_entries + assigned}
            wave = select_samples(screen, remaining, visited, limit=3)
            if not wave:
                raise ValueError('Legacy annotation record exceeds its acquisition plan')
            assigned.extend(wave)
        assigned = assigned[:last + 1]
        for entry in intents:
            row = assigned[entry['ordinal']]
            candidate = candidates[row['category'], row['path']]
            if entry['key'] != grid_key(grid(candidate, job['target_year'])):
                raise ValueError('Legacy annotation intent belongs to a different grid')

    paths = {row['path'] for row in previous_entries}
    for row in assigned:
        if (row['category'], row['path']) not in candidates or row['path'] in paths:
            raise ValueError('Dispatch journal contains an ineligible or duplicate sample')
        paths.add(row['path'])
    if any(ordinal >= len(assigned) or
           any(row[key] != assigned[ordinal][key] for key in ('category', 'path'))
           for ordinal, row in results.items()):
        raise ValueError('Annotation result differs from its dispatched sample')
    if legacy and assigned:
        save(root / 'dispatch' / '000.json', assigned)
    return assigned


def record_result(path, row, screen, epoch, future):
    try:
        status, result = future.result()
    except (KeyError, TypeError, ValueError) as exc:
        status, result = 'invalid_receipt', None
        save(path.with_suffix('.rejected.json'), dict(error=str(exc)))
    attempt = dict(row, status=status, issue_coverage=None, resolved=False, entry=None)
    if result:
        sample, valid = result
        category = row['category']
        encoded = screen['masks'].get(row['path'], {}).get(category)
        coverage = None
        if encoded:
            mask = decode_mask(encoded)
            coverage = float((mask & valid).sum() / mask.sum())
        attempt.update(issue_coverage=coverage, resolved=coverage is None or coverage >= .50,
                       entry=dict(sample, round=epoch, origin=status, category=category, hard_area=encoded))
    save(path, attempt)
    return attempt


def process_round(job, epoch, pins, *, planner=choose_plan):
    """Run the selected annotations together and close the round after they finish."""
    from hard_sample_contract import planning_request, validate_plan
    from hard_sample_training import PER_REVIEW
    root = round_dir(job,epoch)
    with (root/'acquisition.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        if (root/'ready.json').is_file():
            document = manifest(job, epoch)
            require_trainable_annotations(document)
            return document
        screen = read(root/'screen.json'); previous = manifest(job,epoch-PER_REVIEW)
        plan_path = root/'plan.json'
        plan = read(plan_path) if plan_path.is_file() else planner(job,epoch,screen,pins)
        validate_plan(planning_request(epoch,inventory(screen),mode(job)),plan)
        save(plan_path,plan)
        candidates = {(category, row['path']): row for category in CATEGORIES
                      for row in screen['candidates'][category]}
        results = {int(path.stem): read(path) for path in (root / 'results').glob('[0-9][0-9][0-9].json')}
        assigned = restore_dispatch(root, job, screen, plan, previous['entries'], results, candidates)
        submitted = Counter(row['category'] for row in assigned)
        remaining = {category: max(0, plan[category] - submitted[category]) for category in CATEGORIES}
        visited = {row['path'] for row in previous['entries'] + assigned}
        selected = select_samples(screen, remaining, visited, limit=sum(remaining.values()))
        if selected:
            save(root / 'dispatch' / ('%03d.json' % len(assigned)), selected)
            assigned.extend(selected)

        queued = [(i, candidates[row['category'], row['path']])
                  for i, row in enumerate(assigned) if i not in results]
        with ThreadPoolExecutor(max_workers=max(1, len(queued))) as workers:
            pending = {workers.submit(acquire, job, epoch, candidate, ordinal=i): i for i, candidate in queued}
            for future in as_completed(pending):
                ordinal = pending[future]
                path = root / 'results' / ('%03d.json' % ordinal)
                results[ordinal] = record_result(path, assigned[ordinal], screen, epoch, future)
        entries, resolved = collect_entries(previous['entries'], results)
        wins = {category: resolved[category] for category in CATEGORIES}
        attempts = [{k: v for k, v in results[i].items() if k != 'entry'} for i in sorted(results)]
        report = dict(requested={k:plan[k] for k in CATEGORIES}, resolved=wins,
                      attempted=len(attempts), added=len(entries)-len(previous['entries']), attempts=attempts)
        document = dict(epoch=epoch, backend=mode(job), entries=entries,
                        revision=identity(entries), checkpoint_sha256=screen['checkpoint_sha256'], report=report)
        try:
            require_trainable_annotations(document)
        except NoTrainableAnnotations as exc:
            save(root/'label_intake_failed.json', dict(epoch=epoch, reason=str(exc), report=report,
                                                      usable_labels=len(entries)))
            raise
        save(root/'ready.json',document)
        return document
