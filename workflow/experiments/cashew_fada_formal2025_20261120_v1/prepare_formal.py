"""Register the memorized Cashew group scope and host-local data identities."""

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import time

from prepare_job import COUNTRY_CODES, PROJECT, ROOT, acquisition_frame, protect_geometry
from hard_sample_training import VERSION, read, save, sha
from evaluation_sample_v1 import build
from point_sealed_panel_loader_candidate_v1 import IMAGE, VERSION as PANEL_VERSION


COUNTRY_NAME = {'CIV': 'Ivoire', 'NGA': 'Nigeria', 'GSG': 'Gambia_Senegal_Guinea-Bissau',
                'BEN_TGO': 'Benin_Togo', 'GHA': 'Ghana', 'GIN': 'Guinea',
                'BFA': 'Burkina_Faso', 'MLI': 'Mali'}
TARGET_ROOT = Path('@ASSETS@/alphaearth_data_split')
DATABASE = Path('@ASSETS@/country_shp/ADM1/patch_index/patch_index.sqlite')
CANONICAL = 'source2019_benin_e173'
ASSIGNMENT = ROOT / 'cohort_2025.json'
YEAR_CHOICES = ROOT / 'config/tile-years.csv'


def registered_years():
    with YEAR_CHOICES.open(encoding='utf-8', newline='') as stream:
        return {row['tile_id']: (int(row['early_aef_year']), int(row['late_aef_year']))
                for row in csv.DictReader(stream)}


def imagery_path(indexed_path):
    # The database may have been produced on another mount. Its final inventory
    # directory and acquisition-year name retain the same scientific identity.
    parts = Path(indexed_path).parts
    idx = parts.index('alphaearth_data_split')
    suffix = parts[idx + 1:]
    if len(suffix) != 3 or '..' in suffix:
        raise ValueError(f'Unexpected indexed imagery directory: {indexed_path}')
    return TARGET_ROOT.joinpath(*suffix)

sys.path.insert(0, str(PROJECT / 'fada_expert_runs/BEN_TGO_GHA_fixed200_2019_2025_124_20261101_v001/tools'))
from tile_random_sample import sample_by_tile


def directory_catalog(db):
    # Read the small inventory table once, then look up each tile in memory.
    catalog = defaultdict(list)
    for row in db.execute('SELECT * FROM directories'):
        image_year = row['file_name_year']
        if image_year not in ('2018', '2019', '2024', '2025'):
            continue
        path = imagery_path(row['path'])
        period = int(path.parents[1].name)
        if period in (2019, 2025):
            catalog[row['tile_id'], period, int(image_year)].append((path, row))
    return catalog


def period_directory(catalog, tile, late_path, year, countries, choice):
    # 'actual_year' is the inventory period. 'file_name_year' is the AEF capture year.
    candidates = catalog.get((tile, year, choice), ())
    rows = [row for path, row in candidates if COUNTRY_CODES.get(row['country']) in countries
            and (year != 2025 or path == Path(late_path))]
    if not rows:
        raise ValueError(f'Missing registered AEF {choice} imagery for {tile} in period {year}')
    preferred = Path(late_path).parent.name
    return min(rows, key=lambda row: (row['country'] != preferred, row['country']))


def read_locations(db, catalog, group, year):
    locations = []
    years_by_tile = registered_years()
    for tile, late_path in sorted(group['members'].items()):
        early, late = years_by_tile[tile]
        choice = late if year == 2025 else early
        directory = period_directory(catalog, tile, late_path, year, group['countries'], choice)
        files = db.execute('SELECT p.patch_id,p.west,p.south,p.east,p.north,l.file_name '
                           'FROM patches p JOIN locations l ON p.id=l.patch_no '
                           'WHERE p.patch_id>=? AND p.patch_id<? AND l.directory_id=? ORDER BY p.patch_id',
                           (tile + '_', tile + '_~', directory['id']))
        tile_rows = []
        for row in files:
            path = imagery_path(directory['path']) / row['file_name']
            relative = path.relative_to(TARGET_ROOT).as_posix()
            match = IMAGE.fullmatch(relative)
            if (match is None or match['id'] != tile or int(match['year']) != year
                    or match['image_year'] != directory['file_name_year']):
                raise ValueError(f'Inventory identity differs from the registered name: {path}')
            tile_rows.append(dict(tile_id=tile, patch_id=row['patch_id'], country=directory['country'],
                                  actual_year=int(directory['file_name_year']), tif_path=str(path),
                                  **{key: row[key] for key in ('west', 'south', 'east', 'north')}))
        if not tile_rows or not Path(tile_rows[0]['tif_path']).is_file() or not Path(tile_rows[-1]['tif_path']).is_file():
            raise ValueError(f'Indexed imagery files missing for {tile}: {directory["path"]}')
        locations.extend(tile_rows)
    if len(locations) < 4000 or len({row['patch_id'] for row in locations}) != len(locations):
        raise ValueError(f'Inadequate or duplicate target footprint for {group["name"]}')
    return locations


def new_job(group, year, destination):
    # Start from the versioned policy defaults, then register this specific period and group.
    job = read(ROOT / 'config/job_defaults.json')
    source = read(ROOT / 'runtime/references/model_registry.json')['profiles'][CANONICAL]
    job.update(job_id=destination.name, run_id=f'{ROOT.name}/{destination.name}', output_root=str(destination),
               target_year=year, country=COUNTRY_NAME.get(group['family'], 'West_Africa'), source_profile=CANONICAL, source=source,
               target_root=str(TARGET_ROOT), point_validation_online=False, max_epoch=200,
               hard_sample_policy={'version': VERSION, 'annotation_backend': 'sample_assist' if year == 2025 else 'disabled'},
               supplementary={'enabled': False, 'count': 0, 'mode': 'none', 'loss_weight': 0.},
               assigned_host=group['host'])
    card = destination / 'no_online_prior.md'
    card.write_text('No geographic prior was supplied for online decisions.\n')
    job['geographic_prior'] = {'card': str(card)}
    return job


def prepare_panel(job, destination, locations, group):
    patch_csv = destination / 'patch_locations.csv'
    with patch_csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(locations[0]))
        writer.writeheader()
        writer.writerows(locations)
    items = [(row['patch_id'], Path(row['tif_path']).relative_to(TARGET_ROOT).as_posix())
             for row in locations]
    rng = random.Random(916)
    tiles = defaultdict(list)
    for item in items:
        tiles[item[0].rsplit('_', 2)[0]].append(item)
    # Reserve unlabelled monitoring chips before choosing the training split.
    reserve = {patch for key in sorted(tiles)
               for patch, _ in rng.sample(tiles[key], min(8, len(tiles[key])))}
    training, report = sample_by_tile([item for item in items if item[0] not in reserve], total=1000, seed=16)
    if len(training) != 1000:
        raise ValueError('Training sample budget not met')
    report['monitor_reserve'] = {'seed': 916, 'count': len(reserve), 'target_labels_or_scores_used': False}
    sampling = destination / 'sampling_by_tile.json'
    save(sampling, report)
    full, selected = destination / 'target_full.txt', destination / 'target_train_seed16.txt'
    full.write_text(''.join(name + '\n' for _, name in sorted(items)))
    selected.write_text(''.join(name + '\n' for name in training))
    job.update(target_full_list=str(full), target_train_list=str(selected), target_patch_count=len(items),
               target_train_count=len(training))
    job['target_domain'] = {'kind': 'TILE_GROUP', 'gid': f'{group["family"]}_joint_group_{group["group"]:02d}',
                            'name': job['job_id'], 'patch_csv': str(patch_csv), 'patch_csv_sha256': sha(patch_csv),
                            'member_tile_ids': sorted(tiles), 'unique_patch_count': len(items),
                            'actual_image_year_counts': dict(Counter(row['actual_year'] for row in locations)),
                            'sampling_policy': report['policy'], 'sampling_report': str(sampling),
                            'sampling_report_sha256': sha(sampling), 'previously_trained_tiles_excluded': False}
    panel = build(job, destination / 'monitor', ratio=3, seed=916)
    if panel['sample_count'] != 3000:
        raise ValueError('Unlabelled measurement panel did not reach its fixed sampling budget')
    roles = {'names': Path(panel['names_file']), 'locations': Path(panel['locations_csv']), 'report': Path(panel['report']),
             'full': full, 'training': selected}
    artifacts = {}
    for role, path in roles.items():
        info = path.stat()
        artifacts[role] = dict(path=path.relative_to(destination).as_posix(), sha256=sha(path), dev=info.st_dev,
                               ino=info.st_ino, size=info.st_size, max_bytes=max(1, info.st_size))
    registration = dict(version=PANEL_VERSION, job_id=job['job_id'], run_id=job['run_id'],
                        target_year=job['target_year'], seed=panel['seed'], sample_count=panel['sample_count'],
                        population_count=panel['population_count'], tile_count=panel['tile_count'], artifacts=artifacts)
    path = destination / 'panel_registration_candidate.json'
    save(path, registration)
    job.update(target_evaluation=panel, panel_registration_sha256=sha(path))
    return len(items)


def prepare_transfer(job, parent):
    import hashlib
    import torch
    from predictor_artifacts_v3 import VERSION as PREDICTOR, predictor_identity, verify
    from transfer_record_contract_v1 import model_digest

    parent_job = read(parent / 'job_candidate.json')
    freeze_path = parent / 'frozen_selection.json'
    freeze = read(freeze_path)
    if (parent_job['target_year'] != 2025 or freeze['job_id'] != parent_job['job_id']
            or freeze['run_id'] != parent_job['run_id']
            or parent_job['target_domain']['member_tile_ids'] != job['target_domain']['member_tile_ids']):
        raise ValueError('The selected 2025 parent must have the same group membership')
    selected = freeze['selected_epoch']
    receipt_path = parent / 'predictors' / f'epoch_{selected:06d}.json'
    previous_receipt = read(receipt_path)
    receipt = verify(parent, parent_job, selected, previous_receipt['full_checkpoint_sha256'])
    if predictor_identity(receipt) != freeze['predictor']:
        raise ValueError('2025 frozen selection and inference receipt differ')
    raw = (parent / receipt['predictor']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt['predictor_sha256']:
        raise ValueError('Frozen 2025 predictor hash mismatch')
    state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if state['format'] != PREDICTOR or state['epoch'] != selected or model_digest(state['state_dict']) != receipt['model_state_sha256']:
        raise ValueError('Invalid 2025 selected inference state')
    # Transfer just the frozen model weights into the earlier period.
    destination = Path(job['output_root']) / 'selected_2025_initialization.pt'
    torch.save({'format': 'cashew_initial_transfer_weights_v3', 'state_dict': state['state_dict']}, destination)
    job['initial_transfer_weights_v3'] = {'path': str(destination), 'sha256': sha(destination),
                                         'parent_job': parent_job['job_id'], 'parent_selected_epoch': selected,
                                         'parent_selection_sha256': sha(freeze_path),
                                         'parent_checkpoint_sha256': receipt['full_checkpoint_sha256']}


def prepare_group(db, catalog, group, year, protection):
    name = group['name'].replace('_y2025_', f'_y{year}_')
    destination = ROOT / 'jobs' / name
    if (destination / 'job_candidate.json').is_file():
        from run_group import load_job
        job = load_job(destination)
        registration = read(destination / 'panel_registration_candidate.json')
        if registration['artifacts']['full']['dev'] != (destination / 'target_full.txt').stat().st_dev:
            raise ValueError('The job is registered on a different NAS client')
        return dict(name=name, reused_registration=True, patches=job['target_patch_count'])
    if destination.exists():
        destination.rename(destination.with_name('.incomplete_' + name + '_' + str(time.time_ns())))
    locations = read_locations(db, catalog, group, year)
    destination.mkdir()
    job = new_job(group, year, destination)
    count = prepare_panel(job, destination, locations, group)
    if year == 2025:
        frame, exclusions = acquisition_frame(job, 360, Path, protection=protection)
        path = destination / 'hard_sample_frame.json'
        save(path, frame)
        job['hard_sample_frame'] = dict(path=str(path), sha256=sha(path))
        save(destination / 'hard_sample_protection_receipt.json', exclusions)
    else:
        prepare_transfer(job, ROOT / 'jobs' / group['name'])
    from live_safe_review_evidence_v1 import _registered_panel
    panel = _registered_panel(job)
    if len(panel) != 3000 or set(panel) & set(Path(job['target_train_list']).read_text().splitlines()):
        raise ValueError('Training and monitoring frames are inconsistent')
    save(destination / 'job_candidate.json', job)
    return dict(name=name, patches=count, tiles=len(group['members']))


def prepare_host(host, year, first):
    plan = read(ASSIGNMENT)
    scope = [row for row in plan['jobs'] if row['host'] == host]
    if first:
        scope = scope[:first]
    database = Path(os.environ.get('CASHEW_PATCH_INDEX', str(DATABASE))).resolve(strict=True)
    (ROOT / 'jobs').mkdir(exist_ok=True)
    lock = ROOT / f'prepare_{year}_{host}.lock'
    with lock.open('a') as handle, sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db.row_factory = sqlite3.Row
        catalog = directory_catalog(db)
        protect_codes = {code for group in plan['jobs'] for code in group['countries']}
        protection = protect_geometry(protect_codes, (year,)) if year == 2025 else None
        if year == 2025:
            for group in scope:
                record = prepare_group(db, catalog, group, year, protection)
                print(json.dumps(record, ensure_ascii=False), flush=True)
            return
        waiting = list(scope)
        failures = {}
        status = ROOT / f'prepare_status_{host}_2019.json'
        while waiting:
            for group in waiting[:]:
                parent = ROOT / 'jobs' / group['name']
                if not (parent / 'frozen_selection.json').is_file():
                    continue
                try:
                    record = prepare_group(db, catalog, group, year, protection)
                except (ValueError, FileNotFoundError) as error:
                    failures[group['name']] = str(error)
                    print(json.dumps({'job': group['name'], 'error': str(error)}), flush=True)
                else:
                    failures.pop(group['name'], None)
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                waiting.remove(group)
                write_prepare_status(status, waiting, failures)
            write_prepare_status(status, waiting, failures)
            if waiting:
                time.sleep(30)


def write_prepare_status(path, waiting, failures):
    from training_queue import write_status
    write_status(path, dict(timestamp=time.time(), waiting_for_2025=[g['name'] for g in waiting],
                            failures=failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True, choices=('local', '124'))
    parser.add_argument('--year', type=int, choices=(2019, 2025), default=2025)
    parser.add_argument('--first', type=int)
    args = parser.parse_args()
    if args.first is not None and args.first < 1:
        parser.error('--first must be a positive integer')
    prepare_host(args.host, args.year, args.first)
