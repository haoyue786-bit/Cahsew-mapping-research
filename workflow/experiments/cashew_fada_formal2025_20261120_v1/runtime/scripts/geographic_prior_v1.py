"""Match attributed text evidence to administrative polygons, entirely offline."""
from __future__ import annotations
import argparse
import hashlib
import json
import unicodedata
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data/cashew_geo_prior_v1'
VERSION = 'cashew_geographic_text_prior_v1_20261005'
COUNTRIES = {'Benin':'BEN', 'Ivoire':'CIV', 'Ghana':'GHA', 'Guinea-Bissau':'GNB',
             'Guinea':'GIN', 'Nigeria':'NGA', 'Mali':'MLI', 'Gambia':'GMB', 'Senegal':'SEN', 'Togo':'TGO'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value):
    return ''.join(x.lower() for x in unicodedata.normalize('NFKD', value) if x.isalnum())


@lru_cache(maxsize=16)
def boundary_features(path, expected_hash):
    from shapely.geometry import shape
    if digest(path) != expected_hash:
        raise ValueError('boundary hash mismatch: '+path.name)
    return [(feature['properties']['shapeName'], shape(feature['geometry']))
            for feature in json.loads(path.read_text())['features']]


def match(bounds, countries, target_year, root=ROOT):
    from shapely.geometry import box
    import math
    if len(bounds) != 4 or any(type(x) not in (int,float) or not math.isfinite(x) for x in bounds):
        raise ValueError('finite WGS84 bounds required')
    if not (-180 <= bounds[0] < bounds[2] <= 180 and -90 <= bounds[1] < bounds[3] <= 90):
        raise ValueError('invalid WGS84 bounds')
    countries = sorted(set(COUNTRIES.get(x,x) for x in countries))
    catalog_path = root / 'catalog.json'
    catalog = json.loads(catalog_path.read_text())
    sources = {x['id']:x for x in catalog['sources']}
    footprint = box(*bounds)
    admins, missing, geometry_hashes = {}, [], {}
    for iso in countries:
        for level in ('ADM1','ADM2'):
            path = root / 'boundaries' / (iso+'-'+level+'.geojson')
            if not path.is_file():
                if level == 'ADM1' or iso == 'BEN': missing.append(iso+'-'+level)
                continue
            metadata_path = path.with_name(iso+'-'+level+'.metadata.json')
            metadata = json.loads(metadata_path.read_text())
            geometry_hashes[iso+'-'+level] = metadata['geometry_sha256']
            members = []
            for name, polygon in boundary_features(path, metadata['geometry_sha256']):
                overlap = polygon.intersection(footprint)
                if not overlap.is_empty and overlap.area > 0:
                    members.append({'name':name,
                                    'overlap_kind':'footprint_intersection',
                                    'degree_area_share_for_ranking_only':overlap.area/footprint.area})
            admins[iso+'-'+level] = sorted(members,key=lambda x:-x['degree_area_share_for_ranking_only'])
    records = []
    for record in catalog['records']:
        applicable = set(record['countries']).intersection(countries)
        if not applicable: continue
        matched = []
        if record['spatial_scope'] == 'named_administrative_areas':
            names = {norm(x) for x in record['admin_names']}
            for iso in applicable:
                matched += [x['name'] for x in admins.get(iso+'-'+record['admin_level'],[]) if norm(x['name']) in names]
            if not matched: continue
        source = sources[record['source_id']]
        date = source.get('published_date')
        temporal = 'publication_date_unknown' if not date else 'published_after_target_year' if int(date[:4]) > target_year else 'published_by_target_year'
        records.append({**record, 'matched_admin_names':matched, 'source':source,
                        'target_year_alignment':temporal})
    return {'version':VERSION, 'target_year':target_year, 'bounds_wgs84':bounds,
            'countries':countries, 'admin_matches':admins, 'missing_boundary_layers':missing,
            'catalog_sha256':digest(catalog_path), 'boundary_sha256':geometry_hashes,
            'evidence':records, 'tile_fraction_estimate':None,
            'guidance':'Text evidence is regional context, not pixel labels or measured tile prevalence. Include contradictory and coastal-cashew evidence. Missing evidence means unknown. Do not apply a global latitude cutoff. Broad ecological records need local habitat confirmation. Future-dated publications cannot establish target-year area. These priors may motivate diagnostics and late reversible negative trials; they never force a prevalence quota or replace source safety.'}


def prepare(job):
    if job.get('geographic_prior_policy',{}).get('version','').startswith('cashew_spatial_prior_v2_'):
        from spatial_prior_v2 import prepare as prepare_spatial
        return prepare_spatial(job)
    report_path = Path(job['supplementary']['selection']['report'])
    report = json.loads(report_path.read_text())
    context = match(report['target_bounds'], [job['country']], int(job['target_year']))
    path = Path(job['output_root']) / 'data/geographic_prior.json'
    path.write_text(json.dumps(context, indent=2, ensure_ascii=False)+'\n')
    return {'version':VERSION, 'report':str(path.resolve()), 'sha256':digest(path)}


def packet_evidence(job, compact=False):
    if job.get('geographic_prior_policy',{}).get('version')=='WestAfrica_approved_regional_prior_target2019_v1':
        from geographic_prior_v3 import packet_evidence as packet_v3
        return packet_v3(job, compact=compact)
    pinned = job.get('geographic_prior')
    if not pinned: return None
    path = Path(pinned['report'])
    if digest(path) != pinned['sha256']: raise ValueError('geographic prior report hash changed')
    result = json.loads(path.read_text())
    if result['version'] != pinned['version']: raise ValueError('geographic prior version mismatch')
    if result['version'].startswith('cashew_spatial_prior_v2_'):
        from spatial_prior_v2 import VERSION as SPATIAL_VERSION, packet
        if result['version']!=SPATIAL_VERSION or result['prior_card']['version']!=SPATIAL_VERSION:
            raise ValueError('spatial prior card version mismatch')
        policy=job.get('geographic_prior_policy',{})
        if result['package_sha256']!=pinned.get('package_sha256') or result['package_sha256']!=policy.get('package_sha256'):
            raise ValueError('spatial prior package identity mismatch')
        if not pinned.get('card') or digest(Path(pinned['card']))!=pinned.get('card_sha256'):
            raise ValueError('spatial prior card hash changed')
        return packet(result,pinned) if compact else result
    if compact:
        # Keep repeated review packets small; the full attributed report stays
        # pinned and readable without repeating polygon metadata or sources.
        evidence = result.get('evidence', [])
        return {'version':result['version'], 'context_report':str(path),
                'context_sha256':pinned['sha256'], 'bounds_wgs84':result.get('bounds_wgs84'),
                'countries':result.get('countries'), 'tile_fraction_estimate':None,
                'evidence':[{'id':x['id'],'scope':x['spatial_scope'],
                             'admin_names':x['matched_admin_names'],
                             'summary':x['summary_en'] if 'summary_en' in x else x['summary_zh'],
                             'source_id':x['source_id'],'confidence':x['source_confidence'],
                             'time_alignment':x['target_year_alignment']} for x in evidence],
                'sources':{x['source_id']:{'publisher':x['source']['publisher'], 'url':x['source']['url'],
                                         'date':x['source'].get('published_date')} for x in evidence},
                'guidance':result.get('guidance')}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox',type=float,nargs=4,required=True)
    parser.add_argument('--country',action='append',required=True)
    parser.add_argument('--year',type=int,default=2025)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    value = json.dumps(match(args.bbox,args.country,args.year),ensure_ascii=False,indent=2)+'\n'
    if args.output: args.output.write_text(value)
    else: print(value,end='')
