"""Generate four-class maps from an already frozen SELECT checkpoint."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import rasterio
from rasterio.transform import Affine

from run_group import load_job, ROOT
from hard_sample_training import read, save, sha
from geometry import coordinates, inspect_name_grid
from predictor_artifacts_v3 import VERSION as PREDICTOR_VERSION, predictor_identity, verify


CHIP = 128
WIDTH, HEIGHT = 57 * CHIP, 55 * CHIP
NODATA = 255
CASHEW, UNKNOWN = 2, 10
YIELD_EXPORT = 75
PREDICTION_BATCH = 16
PALETTE = {0: (231, 185, 56), 1: (48, 145, 66), 2: (239, 51, 90),
           3: (45, 147, 239), 10: (160, 160, 160), 255: (0, 0, 0)}


def frozen_model_identity(job, root, frozen):
    if (frozen['job_id'], frozen['run_id']) != (job['job_id'], job['run_id']):
        raise ValueError('Frozen selection belongs to a different run')
    receipt = verify(root, job, frozen['selected_epoch'], frozen['checkpoint_sha256'])
    if predictor_identity(receipt) != frozen['predictor']:
        raise ValueError('Frozen selection and verified predictor identity differ')
    return receipt


def patch_inventory(job):
    target_root = Path(job['target_root'])
    paths = Path(job['target_full_list']).read_text().splitlines()
    registered = {str(target_root / name) for name in paths}
    if len(registered) != job['target_patch_count'] or len(registered) != len(paths):
        raise ValueError('Incomplete or duplicated target image list')
    table = Path(job['target_domain']['patch_csv'])
    blob = table.read_bytes()
    if hashlib.sha256(blob).hexdigest() != job['target_domain']['patch_csv_sha256']:
        raise ValueError('Registered geometry differs from the training manifest')
    years = {2024, 2025} if job['target_year'] == 2025 else {2018, 2019}
    grouped = defaultdict(list)
    for row in csv.DictReader(io.StringIO(blob.decode('utf-8-sig'))):
        name = row['tif_path']
        if name not in registered or int(row['actual_year']) not in years:
            raise ValueError(f'Unexpected registered image: {name}')
        registered.remove(name)
        x, y = coordinates(name)
        grouped[row['tile_id']].append((y, x, Path(name), row))
    if registered or set(grouped) != set(job['target_domain']['member_tile_ids']):
        raise ValueError('Tile inventory differs from the selected group')
    return {tile: sorted(rows) for tile, rows in grouped.items()}


def tile_transform(rows):
    y, x, sample, _ = rows[0]
    with rasterio.open(sample) as src:
        if src.crs.to_epsg() != 4326 or src.shape != (CHIP, CHIP) or src.count != 64:
            raise ValueError(f'Unexpected AEF source grid: {sample}')
        a, b, c, d, e, f = tuple(src.transform)[:6]
    if not (a > 0 and e < 0) or b or d:
        raise ValueError('Expected a north-up imagery grid')
    west, north = c - x * CHIP * a, f - y * CHIP * e
    for y, x, path, metadata in rows:
        w, n = west + x * CHIP * a, north + y * CHIP * e
        recorded = [float(metadata[key]) for key in ('west', 'south', 'east', 'north')]
        if not np.allclose((w, n + CHIP * e, w + CHIP * a, n), recorded, rtol=0, atol=1e-8):
            raise ValueError(f'Off-grid inventory patch: {path}')
    return Affine(a, 0, west, 0, e, north)


def decode(probabilities, available):
    # The 0.5 variant only changes pixels already classified as Cashew.
    """Apply the requested confidence rule only to argmax Cashew."""
    raw = np.asarray(probabilities.argmax(axis=1), dtype=np.uint8)
    raw[~available] = NODATA
    uncertain = np.where((raw == CASHEW) & (probabilities[:, CASHEW] < 0.5), UNKNOWN, raw)
    return raw, uncertain.astype(np.uint8, copy=False)


def verify_cache(workspace, tile, checkpoint, count):
    path = workspace / (tile + '.json')
    if not path.is_file():
        return None
    record = read(path)
    if record['checkpoint_sha256'] != checkpoint or record['patches'] != count:
        raise ValueError(f'Cached inference belongs to another job: {tile}')
    for kind, identity in record['arrays'].items():
        path = workspace / f'{tile}_{kind}.npy'
        if sha(path) != identity:
            raise ValueError(f'Cached inference bytes changed: {path}')
    return record


def predict(job, root, frozen, workspace, gpu, batch):
    import torch
    from architecture import make_model
    from module_bn_v2 import domain
    from train_tile import configure_runtime
    from transfer_record_contract_v1 import model_digest

    receipt = frozen_model_identity(job, root, frozen)
    raw = (root / receipt['predictor']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt['predictor_sha256']:
        raise ValueError('Frozen predictor bytes changed')
    checkpoint = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    del raw
    epoch = frozen['selected_epoch']
    if (checkpoint['epoch'] != epoch or checkpoint['format'] != PREDICTOR_VERSION
            or checkpoint['architecture'] != job['architecture']
            or model_digest(checkpoint['state_dict']) != receipt['model_state_sha256']):
        raise ValueError('Compact predictor does not preserve the frozen model')
    device = torch.device('cuda', gpu)
    torch.cuda.set_device(device)
    gan, common = configure_runtime(job, device)
    model, _ = make_model(gan, job, device, resume_uninitialized=True)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    model.eval()
    groups = patch_inventory(job)
    print(json.dumps(dict(phase='loaded', epoch=epoch, tiles=len(groups), batch=batch, total=job['target_patch_count'])),
          flush=True)
    yielding = False

    def request_yield(signum, frame):
        nonlocal yielding
        yielding = True

    signal.signal(signal.SIGUSR1, request_yield)

    def load_batch(pool, rows):
        images = list(pool.map(common.read_raster, [row[2] for row in rows]))
        result = np.stack(images).astype(np.float32, copy=False)
        if result.shape != (len(rows), 64, CHIP, CHIP):
            raise ValueError('Invalid AEF prediction batch dimensions')
        supported = np.isfinite(result).all(axis=1) & np.any(result != 0, axis=1)
        np.copyto(result, 0, where=~supported[:, None])
        return result, supported

    tiles = {}
    with ThreadPoolExecutor(max_workers=6) as reads, ThreadPoolExecutor(max_workers=1) as prefetch:
        for tile, rows in sorted(groups.items()):
            if yielding:
                return YIELD_EXPORT
            cached = verify_cache(workspace, tile, frozen['checkpoint_sha256'], len(rows))
            if cached:
                tiles[tile] = cached
                continue
            inspect_name_grid([str(row[2]) for row in rows], 3135, allow_partial=len(rows) != 3135)
            transform = tile_transform(rows)
            masks = {name: np.full((HEIGHT, WIDTH), NODATA, dtype=np.uint8) for name in ('argmax', 'p05')}
            start_time = time.monotonic()
            upcoming = prefetch.submit(load_batch, reads, rows[:batch])
            with torch.inference_mode(), domain('target'):
                for start in range(0, len(rows), batch):
                    if yielding:
                        upcoming.cancel()
                        return YIELD_EXPORT
                    images, supported = upcoming.result()
                    following = start + batch
                    if following < len(rows):
                        upcoming = prefetch.submit(load_batch, reads, rows[following:following+batch])
                    logits, _ = gan.forward_fuse(model, torch.from_numpy(images).to(device))
                    actual = rows[start:following]
                    if tuple(logits.shape) != (len(actual), 4, CHIP, CHIP) or not torch.isfinite(logits).all().item():
                        raise ValueError('Invalid model output')
                    argmax, p05 = decode(logits.float().softmax(dim=1).cpu().numpy(), supported)
                    for i, (y, x, _, _) in enumerate(actual):
                        window = np.s_[y*CHIP:(y+1)*CHIP, x*CHIP:(x+1)*CHIP]
                        masks['argmax'][window] = argmax[i]
                        masks['p05'][window] = p05[i]
                    del logits, images, argmax, p05
            counts = np.bincount(masks['argmax'].ravel(), minlength=256)
            if counts[4:255].any() or not counts[:4].sum():
                raise ValueError('Unexpected classes or no valid source support')
            changes = masks['p05'] == UNKNOWN
            if not (np.all(masks['argmax'][changes] == CASHEW)
                    and np.array_equal(masks['argmax'][~changes], masks['p05'][~changes])):
                raise ValueError('Confidence export changed an unrelated class')
            arrays = {}
            for kind, mask in masks.items():
                pending = workspace / f'.{tile}_{kind}.npy'
                np.save(pending, mask, allow_pickle=False)
                final = pending.with_name(pending.name[1:])
                pending.replace(final)
                arrays[kind] = sha(final)
            coords = rasterio.transform.array_bounds(HEIGHT, WIDTH, transform)
            record = dict(tile=tile, patches=len(rows), transform=list(transform)[:6], bounds=list(coords),
                          class_counts=counts[:4].tolist(), uncertain_pixels=int(changes.sum()),
                          nodata_pixels=int(counts[NODATA]), checkpoint_sha256=frozen['checkpoint_sha256'],
                          arrays=arrays)
            save(workspace / (tile + '.json'), record)
            tiles[tile] = record
            print(json.dumps(dict(phase='tile_complete', tile=tile, count=len(tiles),
                                  elapsed_seconds=round(time.monotonic() - start_time, 1))), flush=True)
    save(workspace / 'prediction.json', dict(job_id=job['job_id'], run_id=job['run_id'], epoch=epoch,
                                            year=job['target_year'], checkpoint_sha256=frozen['checkpoint_sha256'],
                                            tiles=tiles))
    return 0


def tiff(workspace, staging, path, report, kind):
    from osgeo import gdal

    gdal.UseExceptions()
    order = sorted(report['tiles'], key=lambda tile: (report['tiles'][tile]['patches'], tile))
    sources = []
    with TemporaryDirectory(prefix='tile_tiffs_', dir=staging) as folder:
        for tile in order:
            row = report['tiles'][tile]
            mask = np.load(workspace / f'{tile}_{kind}.npy', mmap_mode='r', allow_pickle=False)
            tif = Path(folder) / (tile + '.tif')
            with rasterio.open(tif, 'w', driver='GTiff', height=HEIGHT, width=WIDTH, count=1, dtype='uint8',
                               crs='EPSG:4326', transform=Affine(*row['transform']), nodata=NODATA,
                               tiled=True, blockxsize=256, blockysize=256, compress='deflate') as output:
                output.write(mask, 1)
                output.write_colormap(1, PALETTE)
            sources.append(str(tif))
        mosaic = gdal.BuildVRT('', sources, options=gdal.BuildVRTOptions(
            srcNodata=NODATA, VRTNodata=NODATA, resolution='highest', resampleAlg='nearest'))
        if mosaic is None:
            raise RuntimeError('GDAL could not form the group mosaic')
        raster = gdal.Translate(str(path), mosaic, options=gdal.TranslateOptions(
            format='GTiff', creationOptions=['TILED=YES', 'COMPRESS=DEFLATE', 'BIGTIFF=IF_SAFER']))
        if raster is None:
            raise RuntimeError('GDAL could not write the group TIFF')
        raster = mosaic = None
    with rasterio.open(path, 'r+') as dataset:
        if dataset.count != 1 or dataset.crs.to_epsg() != 4326 or dataset.nodata != NODATA or dataset.dtypes != ('uint8',):
            raise ValueError('Invalid published group TIFF')
        dataset.update_tags(job_id=report['job_id'], selected_epoch=report['epoch'],
                            checkpoint_sha256=report['checkpoint_sha256'], classification=kind)


def package(job, root, frozen, workspace):
    from export_selected_kmz import render, verify_archive

    frozen_model_identity(job, root, frozen)
    output = root / 'maps'
    if output.exists():
        receipt = read(output / 'export_receipt.json')
        if receipt['checkpoint_sha256'] != frozen['checkpoint_sha256'] or any(
                sha(output / name) != digest for name, digest in receipt['files'].items()):
            raise ValueError('Published export differs from the requested SELECT result')
        return
    report = read(workspace / 'prediction.json')
    if (report['job_id'], report['run_id'], report['epoch'], report['checkpoint_sha256']) != (
            job['job_id'], job['run_id'], frozen['selected_epoch'], frozen['checkpoint_sha256']):
        raise ValueError('Prediction inventory identity differs')
    if set(report['tiles']) != set(job['target_domain']['member_tile_ids']):
        raise ValueError('Prediction inventory does not cover the requested members')
    for tile, row in report['tiles'].items():
        if verify_cache(workspace, tile, frozen['checkpoint_sha256'], row['patches']) != row:
            raise ValueError(f'Missing verified prediction for tile {tile}')
    stem = f"{job['job_id']}_E{report['epoch']:03d}"
    with TemporaryDirectory(prefix='.maps-build-', dir=root) as temporary:
        staging = Path(temporary)
        identities = {}
        for kind, label in (('argmax', '4class_argmax'), ('p05', '4class_argmax_cashew_p05_uncertain10')):
            path = staging / (stem + '_' + label + '.tif')
            tiff(workspace, staging, path, report, kind)
            identities[path.name] = sha(path)
            archive = path.with_suffix('.kmz')
            render(workspace, report, job['job_id'], kind == 'p05', archive)
            verify_archive(archive, report, kind == 'p05')
            identities[archive.name] = sha(archive)
        save(staging / 'export_receipt.json', dict(
            job_id=job['job_id'], selected_epoch=report['epoch'], checkpoint_sha256=report['checkpoint_sha256'],
            tiles=len(report['tiles']), files=identities, decoder='four_class_argmax',
            uncertainty='CASHEW argmax and p(cashew)<0.5 -> 10', nodata=255,
            raw_cashew_pixels=sum(row['class_counts'][CASHEW] for row in report['tiles'].values()),
            uncertain_pixels=sum(row['uncertain_pixels'] for row in report['tiles'].values())))
        staging.rename(output)
    print(str(output / 'export_receipt.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('predict', 'package'))
    parser.add_argument('job', type=Path)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--batch', type=int, default=PREDICTION_BATCH)
    args = parser.parse_args()
    if args.phase == 'predict' and (args.gpu is None or args.batch < 1):
        parser.error('Prediction requires an allocated GPU and a positive batch size')
    from release_identity_v2 import verify as verify_release
    verify_release(ROOT)
    root = args.job.resolve(strict=True)
    job = load_job(root)
    frozen = read(root / 'frozen_selection.json')
    workspace = root / 'prediction_cache'
    workspace.mkdir(exist_ok=True)
    if args.phase == 'predict':
        return predict(job, root, frozen, workspace, args.gpu, args.batch)
    package(job, root, frozen, workspace)
    return 0


if __name__ == '__main__':
    sys.exit(main())
