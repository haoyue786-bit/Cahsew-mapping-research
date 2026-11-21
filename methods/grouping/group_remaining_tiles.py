"""Regroup the 2025 West African AEF inventory outside the formal 36 groups."""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from heapq import heappop, heappush
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import fiona
import numpy as np
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import box, mapping
from shapely.ops import unary_union
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

import cluster_tiles as method


ROOT = Path(__file__).resolve().parent
DIAMETER_KM = 350
PREFERRED_TILES = 3
PREFERRED_PATCHES = 4000
GEOGRAPHIC_WEIGHTS = (0.2, 0.35, 0.5)
SILHOUETTE_TOLERANCE = 0.02


@dataclass
class WardGroup:
    indices: np.ndarray
    center: np.ndarray
    patches: int
    neighbors: set

    @property
    def size(self):
        return len(self.indices)

    @property
    def small(self):
        return self.size < PREFERRED_TILES or self.patches < PREFERRED_PATCHES


def ward_partitions(values, adjacency, distance, populations):
    """Make Ward merges only along edges and within the group diameter."""
    groups = {i: WardGroup(np.array([i]), values[i], int(populations[i]),
                           set(np.flatnonzero(adjacency[i]))) for i in range(len(values))}
    queue = []
    labels = np.arange(len(values))
    partitions = {}

    def offer(left_id, right_id):
        left, right = groups[left_id], groups[right_id]
        if distance[np.ix_(left.indices, right.indices)].max() <= DIAMETER_KM:
            weight = left.size * right.size / (left.size + right.size)
            cost = weight * float(np.square(left.center-right.center).sum())
            heappush(queue, (not (left.small or right.small), cost, left_id, right_id))

    for left_id, group in groups.items():
        for right_id in sorted(group.neighbors):
            if left_id < right_id:
                offer(left_id, right_id)
    next_id = len(values)
    while queue:
        _, _, left_id, right_id = heappop(queue)
        if left_id not in groups or right_id not in groups:
            continue
        left, right = groups.pop(left_id), groups.pop(right_id)
        members = np.concatenate((left.indices, right.indices))
        neighbors = (left.neighbors | right.neighbors) - {left_id, right_id}
        center = (left.size * left.center + right.size * right.center) / len(members)
        groups[next_id] = WardGroup(members, center, left.patches + right.patches, neighbors)
        for neighbor in sorted(neighbors):
            groups[neighbor].neighbors.difference_update((left_id, right_id))
            groups[neighbor].neighbors.add(next_id)
            offer(neighbor, next_id)
        labels[members] = next_id
        # Each active group and its internal span remain valid after every merge.
        if 1 < len(groups) < len(values):
            partitions[len(groups)] = (labels.copy(), sum(g.patches < PREFERRED_PATCHES for g in groups.values()),
                                       sum(g.size < PREFERRED_TILES for g in groups.values()))
        next_id += 1
    return partitions


def group_tiles(method, records, features):
    bounds = np.asarray([feature['bounds'] for feature in features])
    centers = np.asarray([feature['center'] for feature in features])
    west, south, east, north = bounds.T
    tolerance = 1e-7
    adjacent = ((west[:, None] <= east + tolerance) & (east[:, None] >= west - tolerance)
                & (south[:, None] <= north + tolerance) & (north[:, None] >= south - tolerance))
    np.fill_diagonal(adjacent, False)
    # Restrict Ward's candidate merges to adjacent regions with an acceptable span.
    population = np.array([row['indexed_patches'] for row in records])
    n = len(centers)
    first, second = np.triu_indices(n, k=1)
    geo = np.zeros((n, n))
    distances = np.abs(method.GEOD.inv(*centers[first].T, *centers[second].T)[2]) / 1000
    geo[first, second] = geo[second, first] = distances
    descriptors = [method.feature_vectors(features, half) for half in (None, 0, 1)]
    feature_distances = np.maximum(cosine_distances(descriptors[0]), 0)
    np.fill_diagonal(feature_distances, 0)
    feature_scale = np.median(feature_distances[first, second])
    scale = np.median(distances)
    lat, lon = centers[:, 1].mean(), centers[:, 0].mean()
    projection = Transformer.from_crs('EPSG:4326',
        f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m', always_xy=True)
    xy = np.column_stack(projection.transform(*centers.T)) / 1000
    candidates, partitions = [], {}
    # Independently sampled half-panels test the stability of each candidate partition.
    for alpha in GEOGRAPHIC_WEIGHTS:
        vectors = [np.column_stack((d * np.sqrt((1-alpha)/(2*feature_scale)),
                                   xy * np.sqrt(alpha)/scale)) for d in descriptors]
        joint = euclidean_distances(vectors[0])
        np.fill_diagonal(joint, 0)
        full, split_a, split_b = [ward_partitions(matrix, adjacent, geo, population) for matrix in vectors]
        eligible_counts = full.keys() & split_a.keys() & split_b.keys()
        if not eligible_counts:
            raise ValueError(f'No common split-sample group counts for geographic weight {alpha}')
        best_size = min(full[k][1:] for k in eligible_counts)
        for k in sorted(eligible_counts):
            labels, under_patches, under_tiles = full[k]
            if (under_patches, under_tiles) != best_size:
                continue
            stability = float(np.mean([adjusted_rand_score(labels, split[k][0]) for split in (split_a, split_b)]))
            candidates.append(dict(groups=k, geographic_weight=alpha, groups_below_4000_patches=under_patches,
                                   groups_below_three_tiles=under_tiles,
                                   joint_silhouette=float(silhouette_score(joint, labels, metric='precomputed')),
                                   feature_silhouette=float(silhouette_score(feature_distances, labels, metric='precomputed')),
                                   split_sample_ari=stability))
            partitions[alpha, k] = labels
    size_rank = lambda row: (row['groups_below_4000_patches'], row['groups_below_three_tiles'])
    target_size = min(map(size_rank, candidates))
    eligible = [row for row in candidates if size_rank(row) == target_size]
    best = max(row['joint_silhouette'] for row in eligible)
    chosen = min((row for row in eligible if row['joint_silhouette'] >= best-SILHOUETTE_TOLERANCE),
                 key=lambda row: (-row['split_sample_ari'], row['groups'], -row['joint_silhouette']))
    labels = method.stable_ids(partitions[chosen['geographic_weight'], chosen['groups']], centers)
    return labels, bounds, centers, adjacent, geo, candidates, chosen


def shapefile(path, rows, polygons):
    schema = {'geometry': 'Polygon', 'properties': {
        'tile_id': 'str:20', 'group_id': 'int', 'img_year': 'int',
        'patches': 'int', 'country': 'str:32', 'feat_src': 'str:20'}}
    with fiona.open(path, 'w', driver='ESRI Shapefile', crs='EPSG:4326', schema=schema, encoding='UTF-8') as layer:
        for row, geometry in zip(rows, polygons):
            properties = dict(tile_id=row['tile_id'], group_id=row['group'], img_year=row['image_year'],
                              patches=row['indexed_patches'], country=row['country'],
                              feat_src='AEF64')
            layer.write(dict(geometry=mapping(geometry), properties=properties))
    with fiona.open(path) as layer:
        expected = {row['tile_id']: row['group'] for row in rows}
        if len(layer) != len(expected) or layer.crs.to_epsg() != 4326:
            raise ValueError('Shapefile lost rows or its CRS')
        for feature in layer:
            props = feature['properties']
            if expected.pop(props['tile_id']) != props['group_id']:
                raise ValueError('Shapefile does not preserve tile membership')


def overview(path, labels, polygons, adjacency):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    ids = sorted(set(labels))
    neighbor_colors = {}
    palette = list(plt.get_cmap('tab20').colors)
    for group in ids:
        neighbors = set(labels[np.any(adjacency[labels == group], axis=0)]) - {group}
        used = {neighbor_colors[other] for other in neighbors if other in neighbor_colors}
        neighbor_colors[group] = next(i for i in range(len(palette)) if i not in used)
    fig, ax = plt.subplots(figsize=(15, 6.5), layout='constrained')
    ax.add_collection(PolyCollection([polygon.exterior.coords for polygon in polygons],
                      facecolors=[palette[neighbor_colors[group]] for group in labels],
                      edgecolors='#f8f8f8', linewidths=0.40, alpha=0.93))
    for group in ids:
        geometry = unary_union([poly for poly, value in zip(polygons, labels) if value == group])
        parts = [geometry] if geometry.geom_type == 'Polygon' else list(geometry.geoms)
        for poly in parts:
            ax.plot(*poly.exterior.xy, color='#24354b', linewidth=0.52)
            for hole in poly.interiors:
                ax.plot(*hole.xy, color='#24354b', linewidth=0.52)
        x, y = max(parts, key=lambda poly: poly.area).representative_point().coords[0]
        ax.text(x, y, str(group), ha='center', va='center', fontsize=6,
                bbox=dict(boxstyle='round,pad=.14', facecolor='white', alpha=.78, edgecolor='none'))
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set(xlabel='Longitude (°E)', ylabel='Latitude (°N)')
    ax.set_title(f'West Africa 2025  |  {len(polygons)} remaining tiles in {len(ids)} feature groups',
                 loc='left', fontweight='bold')
    ax.grid(alpha=.12)
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)


def publish(method, output, rows, features, result):
    labels, bounds, centers, adjacency, geo, candidates, selected = result
    polygons = [box(*b) for b in bounds]
    for i, row in enumerate(rows):
        row.update(group=int(labels[i]), center_lon=float(centers[i, 0]), center_lat=float(centers[i, 1]))
    with TemporaryDirectory(prefix='.grouping-', dir=output.parent) as temp:
        staging = Path(temp) / 'results'
        staging.mkdir()
        (staging / 'groups').mkdir()
        summaries = []
        for group in range(1, selected['groups']+1):
            indices = np.flatnonzero(labels == group)
            diameter = float(geo[np.ix_(indices, indices)].max())
            if diameter > DIAMETER_KM or connected_components(csr_matrix(adjacency[np.ix_(indices, indices)]))[0] != 1:
                raise ValueError(f'Group {group} violates geographic connectivity or diameter')
            members = [rows[i] for i in indices]
            countries = sorted({row['country'] for row in members})
            method.write_csv(staging / 'groups' / f'group_{group:03d}.csv', members)
            summaries.append(dict(group=group, tiles=len(members), patches=sum(r['indexed_patches'] for r in members),
                                  countries=';'.join(countries), country_count=len(countries),
                                  diameter_km=round(diameter, 6)))
        if len({row['tile_id'] for row in rows}) != 437 or sum(row['tiles'] for row in summaries) != 437:
            raise ValueError('Regrouping must assign all 437 unique tiles exactly once')
        method.write_csv(staging / 'tile_groups.csv', sorted(rows, key=lambda row: (row['group'], row['tile_id'])))
        method.write_csv(staging / 'group_summary.csv', summaries)
        method.write_csv(staging / 'candidate_scores.csv', candidates)
        shapefile(staging / 'tile_groups.shp', rows, polygons)
        overview(staging / 'group_overview.png', labels, polygons, adjacency)
        method.dump(staging / 'result.json', dict(selected=selected, groups=summaries))
        staging.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--descriptors', type=Path, default=ROOT / 'aef64-descriptors.json.gz')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory already exists')
    with gzip.open(args.descriptors, 'rt', encoding='utf-8') as stream:
        inputs = json.load(stream)
    rows = [entry['record'] for entry in inputs]
    expected = [row.pop('group') for row in rows]
    features = [entry['aef'] for entry in inputs]
    result = group_tiles(method, rows, features)
    if result[0].tolist() != expected:
        raise ValueError('Reconstructed groups differ from the frozen membership')
    print('Reconstructed grouping:', result[-1], flush=True)
    publish(method, args.output, rows, features, result)


if __name__ == '__main__':
    main()
