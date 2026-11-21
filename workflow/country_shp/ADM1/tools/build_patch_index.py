#!/usr/bin/env python3
"""Index existing regular-grid TIFF patches by administrative polygon and year.

Only reads filenames and sampled GeoTIFF headers, never imagery or training jobs.
Every directory's corners/interior validate the grid; irregular grids are rejected.
Spatial identities are checked across country copies and all available years.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import fiona
import numpy as np
import shapely
import tifffile
from pyproj import CRS, Transformer
from shapely.ops import transform

PATCH_RE = re.compile(r"^(?P<prefix>.+_(?P<tile>[0-9a-fA-F]{20}))_(?P<col>\d+)_(?P<row>\d+)\.(?:tif|tiff)$", re.I)
TILE_RE = re.compile(r"(?P<tile>[0-9a-fA-F]{20})$")
TOLERANCE = 1e-9


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def slug(value):
    ascii_value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode()
    return re.sub(r'[^A-Za-z0-9.-]+', '_', ascii_value).strip('_') or 'unnamed'


def raster_header(path):
    with tifffile.TiffFile(path) as src:
        page = src.pages[0]
        geo = src.geotiff_metadata or {}
        if (int(geo.get('GeographicTypeGeoKey', 0)) != 4326
                or int(geo.get('GTModelTypeGeoKey', 0)) != 2
                or int(geo.get('GTRasterTypeGeoKey', 0)) != 1
                or 'ModelTransformation' in geo):
            raise ValueError(f'Unsupported CRS/raster geometry: {path}')
        scale, tie = geo.get('ModelPixelScale'), geo.get('ModelTiepoint')
        if not scale or not tie or len(tie) != 6 or scale[0] <= 0 or scale[1] <= 0:
            raise ValueError(f'Unsupported GeoTIFF georeferencing: {path}')
        return dict(width=page.imagewidth, height=page.imagelength,
                    dx=float(scale[0]), dy=float(scale[1]),
                    west=float(tie[3] - tie[0] * scale[0]),
                    north=float(tie[4] + tie[1] * scale[1]))


def bounds(grid, col, row):
    west = grid['origin_west'] + col * grid['width'] * grid['dx']
    north = grid['origin_north'] - row * grid['height'] * grid['dy']
    return (west, north - grid['height'] * grid['dy'],
            west + grid['width'] * grid['dx'], north)


def same_bounds(a, b):
    return all(math.isclose(x, y, rel_tol=0, abs_tol=TOLERANCE) for x, y in zip(a, b))


def scan_directory(item):
    year, country, path, tile_id = item
    patches = {}
    for entry in os.scandir(path):
        if not entry.name.lower().endswith(('.tif', '.tiff')):
            continue
        match = PATCH_RE.fullmatch(entry.name)
        if not match or match['tile'].lower() != tile_id:
            raise ValueError(f'Unexpected patch filename: {entry.path}')
        key = (int(match['col']), int(match['row']))
        if key in patches:
            raise ValueError(f'Duplicate suffix in directory: {entry.path}')
        patches[key] = entry.name
    if not patches:
        raise ValueError(f'No TIFF patches in tile directory: {path}')
    ordered = sorted(patches)
    first = ordered[0]
    header = raster_header(path / patches[first])
    grid = dict(header, origin_west=header['west'] - first[0] * header['width'] * header['dx'],
                origin_north=header['north'] + first[1] * header['height'] * header['dy'])
    # Corners and quartiles cover edge dimensions and both filename axes.
    candidates = {ordered[0], ordered[-1]}
    for key in (lambda p: (p[1], p[0]), lambda p: (p[0], -p[1]), lambda p: (-p[0], p[1])):
        candidates.add(min(patches, key=key))
        candidates.add(max(patches, key=key))
    for fraction in (.25, .5, .75):
        candidates.add(ordered[int((len(ordered) - 1) * fraction)])
    audit = []
    for col, row in sorted(candidates):
        filename = patches[(col, row)]
        h = header if (col, row) == first else raster_header(path / filename)
        measured = (h['west'], h['north'] - h['height'] * h['dy'],
                    h['west'] + h['width'] * h['dx'], h['north'])
        if h['width'] != grid['width'] or h['height'] != grid['height'] or not same_bounds(measured, bounds(grid, col, row)):
            raise ValueError(f'Irregular patch grid; full-metadata indexing needed: {path / filename}')
        audit.append(dict(file=filename, width=h['width'], height=h['height'], bounds=list(measured)))
    name_year = re.search(r'(20\d{2})_', path.name)
    return dict(year=year, country=country, path=str(path), tile_id=tile_id,
                file_name_year=name_year.group(1) if name_year else '', patches=patches,
                grid=grid, header_samples=audit)


def discover(root):
    groups = defaultdict(list)
    for year in sorted(root.iterdir()):
        if not year.is_dir() or not re.fullmatch(r'20\d{2}', year.name):
            continue
        for country in sorted(year.iterdir()):
            if not country.is_dir():
                continue
            for entry in sorted(os.scandir(country), key=lambda e: e.name):
                if not entry.is_dir():
                    continue
                match = TILE_RE.search(entry.name)
                if not match:
                    raise ValueError(f'Unexpected tile directory: {entry.path}')
                tile = match['tile'].lower()
                groups[tile].append((year.name, country.name, Path(entry.path), tile))
    if not groups:
        raise ValueError('No tile directories found')
    return groups


def load_admins(path):
    result, seen = [], set()
    with fiona.open(path) as src:
        crs = CRS.from_user_input(src.crs_wkt or src.crs)
        projector = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform
        for feature in src:
            p = dict(feature['properties'])
            if int(p.get('ADM_LEVEL', -1)) != 1 or not p.get('GID_1') or not p.get('NAME_1'):
                raise ValueError('Expected ADM1 features with GID_1 and NAME_1')
            if p['GID_1'] in seen:
                raise ValueError(f"Duplicate ADM1 identifier: {p['GID_1']}")
            seen.add(p['GID_1'])
            original = transform(projector, shapely.geometry.shape(feature['geometry']))
            geometry = shapely.make_valid(original) if not original.is_valid else original
            if geometry.is_empty or geometry.area <= 0:
                raise ValueError(f"Empty/non-area administrative geometry: {p['GID_1']}")
            result.append(dict(properties=p, geometry=geometry, geometry_repaired=not original.is_valid,
                               filename=f"{slug(p['NAME_0']).upper()}_{slug(p['NAME_1'])}__{slug(p['GID_1'])}.csv"))
    return result


def create_database(path):
    db = sqlite3.connect(path)
    db.executescript('''
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE directories(id INTEGER PRIMARY KEY, actual_year TEXT, file_name_year TEXT,
          country TEXT, tile_id TEXT, path TEXT UNIQUE, grid_json TEXT, header_samples_json TEXT);
        CREATE TABLE patches(id INTEGER PRIMARY KEY, patch_id TEXT UNIQUE, tile_id TEXT,
          patch_col INTEGER, patch_row INTEGER, west REAL, south REAL, east REAL, north REAL);
        CREATE TABLE locations(patch_no INTEGER, directory_id INTEGER, file_name TEXT);
        CREATE TABLE admins(id INTEGER PRIMARY KEY, adm1_id TEXT UNIQUE, country TEXT, name TEXT,
          csv_path TEXT, geometry_repaired INTEGER);
        CREATE TABLE memberships(admin_no INTEGER, patch_no INTEGER, overlap_fraction REAL);
    ''')
    return db


def run(shp, root, output, workers=4):
    started = time.monotonic()
    if output.exists():
        raise ValueError(f'Output already exists; use a new destination: {output}')
    output.mkdir(parents=True)
    (output / 'by_admin').mkdir()
    status = dict(state='building', started_at=stamp(), shapefile=str(shp), data_root=str(root),
                  output=str(output), administrative_level=1, patch_identity='tile_id_patch_col_patch_row',
                  spatial_rule='positive-area intersection; boundary-only touches excluded',
                  geometry_method='regular grid inferred from TIFF names; corners and quartiles header-verified in EVERY directory',
                  geometry_tolerance_degrees=TOLERANCE, headers_only=True)
    save_json(output / 'build_status.json', status)
    db = None
    handles = []
    try:
        admins = load_admins(shp)
        groups = discover(root)
        years = sorted({i[0] for group in groups.values() for i in group})
        fields = ['patch_id', 'tile_id', 'patch_col', 'patch_row', 'adm_level', 'adm1_id',
                  'adm1_name', 'country', 'iso3', 'west', 'south', 'east', 'north',
                  'overlap_fraction', 'available_years']
        for year in years:
            fields += [f'tif_paths_{year}', f'tile_dirs_{year}', f'location_count_{year}']
        db = create_database(output / 'patch_index.sqlite')
        writers, counts, year_counts = [], Counter(), defaultdict(Counter)
        for i, admin in enumerate(admins):
            f = (output / 'by_admin' / admin['filename']).open('w', newline='', encoding='utf-8-sig')
            handles.append(f)
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writers.append(writer)
            p = admin['properties']
            db.execute('INSERT INTO admins VALUES (?,?,?,?,?,?)', (i, p['GID_1'], p['NAME_0'], p['NAME_1'],
                       str(output / 'by_admin' / admin['filename']), int(admin['geometry_repaired'])))
        geoms = np.asarray([a['geometry'] for a in admins], dtype=object)
        tree = shapely.STRtree(geoms)
        directory_no, patch_no, memberships, physical_files, unmatched = 0, 0, 0, 0, 0
        location_writer_file = (output / 'tile_locations.csv').open('w', newline='', encoding='utf-8-sig')
        handles.append(location_writer_file)
        directory_writer = csv.DictWriter(location_writer_file, ['directory_id', 'tile_id', 'actual_data_year',
                          'file_name_year', 'country', 'tile_directory', 'patch_file_count', 'sampled_headers'])
        directory_writer.writeheader()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for processed, (tile_id, items) in enumerate(sorted(groups.items()), 1):
                directories = list(pool.map(scan_directory, items))
                by_suffix = defaultdict(list)
                location_rows = []
                for directory in directories:
                    directory_no += 1
                    directory['id'] = directory_no
                    db.execute('INSERT INTO directories VALUES (?,?,?,?,?,?,?,?)',
                               (directory_no, directory['year'], directory['file_name_year'], directory['country'],
                                tile_id, directory['path'], json.dumps(directory['grid']),
                                json.dumps(directory['header_samples'])))
                    directory_writer.writerow(dict(directory_id=directory_no, tile_id=tile_id,
                        actual_data_year=directory['year'], file_name_year=directory['file_name_year'],
                        country=directory['country'], tile_directory=directory['path'],
                        patch_file_count=len(directory['patches']), sampled_headers=len(directory['header_samples'])))
                    physical_files += len(directory['patches'])
                    for suffix, filename in directory['patches'].items():
                        by_suffix[suffix].append((directory, filename))
                suffixes = sorted(by_suffix)
                boxes = []
                patch_rows, csv_bases, year_locations = [], [], []
                for col, row in suffixes:
                    locations = by_suffix[(col, row)]
                    extent = bounds(locations[0][0]['grid'], col, row)
                    for directory, filename in locations[1:]:
                        other = bounds(directory['grid'], col, row)
                        if not same_bounds(extent, other):
                            conflict = dict(patch_id=f'{tile_id}_{col}_{row}',
                                            locations=[dict(path=str(Path(d['path']) / fn),
                                                            bounds=list(bounds(d['grid'], col, row))) for d, fn in locations])
                            save_json(output / 'identity_conflict.json', conflict)
                            raise ValueError(f"Patch identity has different coordinates: {conflict['patch_id']}")
                    patch_no += 1
                    patch_id = f'{tile_id}_{col}_{row}'
                    patch_rows.append((patch_no, patch_id, tile_id, col, row, *extent))
                    base = dict(patch_id=patch_id, tile_id=tile_id, patch_col=col, patch_row=row,
                                west=f'{extent[0]:.12f}', south=f'{extent[1]:.12f}',
                                east=f'{extent[2]:.12f}', north=f'{extent[3]:.12f}')
                    by_year = defaultdict(list)
                    for directory, filename in locations:
                        by_year[directory['year']].append((directory['path'], filename))
                        location_rows.append((patch_no, directory['id'], filename))
                    base['available_years'] = '|'.join(sorted(by_year))
                    for year in years:
                        present = sorted(by_year.get(year, []))
                        base[f'tif_paths_{year}'] = '|'.join(str(Path(d) / fn) for d, fn in present)
                        base[f'tile_dirs_{year}'] = '|'.join(d for d, fn in present)
                        base[f'location_count_{year}'] = len(present)
                    csv_bases.append(base)
                    year_locations.append(set(by_year))
                    boxes.append(shapely.box(*extent))
                db.executemany('INSERT INTO patches VALUES (?,?,?,?,?,?,?,?,?)', patch_rows)
                db.executemany('INSERT INTO locations VALUES (?,?,?)', location_rows)
                boxes = np.asarray(boxes, dtype=object)
                pairs = tree.query(boxes, predicate='intersects')
                members, covered = [], set()
                if pairs.size:
                    intersections = shapely.intersection(boxes[pairs[0]], geoms[pairs[1]])
                    areas = shapely.area(intersections)
                    fractions = areas / shapely.area(boxes[pairs[0]])
                    for patch_index, admin_index, area, fraction in zip(pairs[0], pairs[1], areas, fractions):
                        if area <= 0:
                            continue
                        ai, pi = int(admin_index), int(patch_index)
                        p = admins[ai]['properties']
                        writers[ai].writerow(dict(csv_bases[pi], adm_level=1, adm1_id=p['GID_1'],
                            adm1_name=p['NAME_1'], country=p['NAME_0'], iso3=p['ISO3'],
                            overlap_fraction=f'{float(fraction):.10f}'))
                        counts[ai] += 1
                        for year in year_locations[pi]:
                            year_counts[ai][year] += 1
                        covered.add(pi)
                        members.append((ai, patch_rows[pi][0], float(fraction)))
                db.executemany('INSERT INTO memberships VALUES (?,?,?)', members)
                memberships += len(members)
                unmatched += len(suffixes) - len(covered)
                db.commit()
                status.update(updated_at=stamp(), tile_groups_processed=processed, tile_groups_total=len(groups),
                              tile_directories_processed=directory_no, unique_patches=patch_no,
                              physical_tif_locations=physical_files, administrative_memberships=memberships,
                              patches_without_adm1=unmatched, elapsed_seconds=round(time.monotonic()-started, 1))
                save_json(output / 'build_status.json', status)
                if processed % 10 == 0 or processed == len(groups):
                    print(json.dumps(status, ensure_ascii=False), flush=True)
        for handle in handles:
            handle.close()
        handles.clear()
        db.executescript('CREATE INDEX location_patch ON locations(patch_no); '
                        'CREATE INDEX membership_admin ON memberships(admin_no); '
                        'CREATE INDEX membership_patch ON memberships(patch_no);')
        db.commit()
        check = db.execute('PRAGMA quick_check').fetchone()[0]
        if check != 'ok':
            raise ValueError(f'SQLite check: {check}')
        summary_fields = ['adm_level', 'adm1_id', 'country', 'iso3', 'adm1_name', 'unique_patch_count',
                          *[f'patch_count_{year}' for year in years], 'csv_path', 'geometry_repaired']
        with (output / 'admin_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, summary_fields)
            writer.writeheader()
            for i, admin in enumerate(admins):
                p = admin['properties']
                writer.writerow(dict(adm_level=1, adm1_id=p['GID_1'], country=p['NAME_0'], iso3=p['ISO3'],
                    adm1_name=p['NAME_1'], unique_patch_count=counts[i],
                    **{f'patch_count_{year}': year_counts[i][year] for year in years},
                    csv_path=str(output / 'by_admin' / admin['filename']), geometry_repaired=admin['geometry_repaired']))
        status.update(state='complete', completed_at=stamp(), administrative_regions=len(admins),
                      regions_with_patches=sum(counts[i] > 0 for i in range(len(admins))), available_years=years,
                      sqlite_quick_check=check, elapsed_seconds=round(time.monotonic()-started, 1),
                      shapefile_sha256=hashlib.sha256(shp.read_bytes()).hexdigest())
        save_json(output / 'build_status.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        return status
    except BaseException as exc:
        status.update(state='failed', updated_at=stamp(), error=f'{type(exc).__name__}: {exc}')
        save_json(output / 'build_status.json', status)
        raise
    finally:
        for handle in handles:
            handle.close()
        if db:
            db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shp', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.shp.resolve(), args.data_root.resolve(), args.output.resolve(), args.workers)


if __name__ == '__main__':
    main()
