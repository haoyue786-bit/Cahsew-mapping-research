"""Offline evidence -> retained ADM1/ADM2 geometry -> tile context and prior cards.

Text priors never produce pixel labels, a prevalence probability or loss changes.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import unicodedata
from functools import lru_cache
from pathlib import Path

VERSION = 'cashew_spatial_prior_v2_20261006'
ROOT = Path(__file__).resolve().parents[1] / 'data/cashew_spatial_prior_v2'
COUNTRIES = {'Benin':'BEN','Burkina Faso':'BFA','Burkina_Faso':'BFA','Ivoire':'CIV',
             "Côte d'Ivoire":'CIV','Ghana':'GHA','Nigeria':'NGA','Togo':'TGO','Guinea':'GIN',
             'Guinea-Bissau':'GNB','Senegal':'SEN','Gambia':'GMB','The Gambia':'GMB',
             'Mali':'MLI','Sierra Leone':'SLE','Liberia':'LBR'}
GUIDANCE = ('Regional evidence is context, never pixel truth or a measured cashew fraction. '
            'Keep contradictory presence and competitors. Unknown is not absence. '
            'Unlocated villages matched through a parent are not known to lie inside the tile. '
            'No latitude/coast cutoff, prevalence quota, logit offset, pseudo-label or mask from text. '
            'Late controlled negative trials may test a background-confusion hypothesis; '
            'remote-sensing evidence and source safety take precedence over the prior.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(text):
    return ''.join(c.lower() for c in unicodedata.normalize('NFKD', str(text)) if c.isalnum())


def area(geometry):
    """WGS84 geodesic area; context overlap is not crop area."""
    from pyproj import Geod
    geod = Geod(ellps='WGS84')
    if geometry.is_empty: return 0.0
    if geometry.geom_type == 'Polygon':
        exterior = abs(geod.polygon_area_perimeter(*zip(*geometry.exterior.coords))[0])
        holes = sum(abs(geod.polygon_area_perimeter(*zip(*ring.coords))[0]) for ring in geometry.interiors)
        return max(0.0, exterior-holes)
    return sum(area(g) for g in getattr(geometry,'geoms',[]) if g.geom_type in ('Polygon','MultiPolygon','GeometryCollection'))


@lru_cache(maxsize=4)
def _load(root_string, manifest_sha):
    root=Path(root_string); manifest=json.loads((root/'manifest.json').read_text())
    if manifest['version'] != VERSION: raise ValueError('unsupported spatial prior package')
    for name,sha in manifest['file_sha256'].items():
        path=root/name
        if not path.is_file() or digest(path) != sha: raise ValueError('spatial prior package hash changed: '+name)
    records=[json.loads(line) for line in (root/'knowledge/evidence.jsonl').read_text().splitlines() if line.strip()]
    supplement=root/'knowledge/supplemental_evidence_20261008.jsonl'
    if supplement.is_file():
        records += [json.loads(line) for line in supplement.read_text().splitlines() if line.strip()]
    annotations=root/'knowledge/background_details_20261008.json'
    if annotations.is_file():
        details=json.loads(annotations.read_text())
        for record in records:
            if record['evidence_id'] in details:record['background_detail']=details[record['evidence_id']]
    ids={r['evidence_id'] for r in records}
    if len(ids)!=len(records): raise ValueError('duplicate evidence IDs')
    with (root/'knowledge/contradictions.csv').open() as handle:
        contradictions=list(csv.DictReader(handle))
    if any(x['evidence_id_1'] not in ids or x['evidence_id_2'] not in ids for x in contradictions):
        raise ValueError('invalid contradiction reference')
    from shapely.geometry import shape
    layers={}; metadata={}
    for path in sorted((root/'boundaries').glob('*.geojson')):
        meta=json.loads(path.with_suffix('.metadata.json').read_text())
        if digest(path)!=meta['geometry_sha256']: raise ValueError('boundary metadata hash mismatch')
        features=[]
        for f in json.loads(path.read_text())['features']:
            geom=shape(f['geometry']); repaired=not geom.is_valid
            if repaired: geom=geom.buffer(0)
            if not geom.is_valid: raise ValueError('unusable boundary geometry')
            features.append({'id':f['properties']['shapeID'],'name':f['properties']['shapeName'],
                             'geometry':geom,'geometry_repaired_for_intersection':repaired})
        layers[path.stem]=features;metadata[path.stem]=meta
    return records,contradictions,layers,metadata,manifest


def load(root=ROOT):
    root=Path(root).resolve()
    return _load(str(root),digest(root/'manifest.json'))


@lru_cache(maxsize=4096)
def parent_chain(root_string, manifest_sha, layer, shape_id):
    layers=load(Path(root_string))[2]
    children=[f for f in layers[layer] if f['id']==shape_id]
    if len(children)!=1:raise ValueError('nonunique administrative shape ID')
    child=children[0]['geometry']; denominator=area(child)
    parents=[]
    for f in layers.get(layer[:3]+'-ADM1',[]):
        if not f['geometry'].intersects(child):continue
        share=area(f['geometry'].intersection(child))/denominator
        if share>=.75:
            parents.append({'shape_id':f['id'],'name':f['name'],'child_coverage_fraction':round(share,4)})
    return {'adm2_shape_id':shape_id,'adm1_candidates':parents,
            'status':'geometric_parent_context' if len(parents)==1 else 'unresolved_or_ambiguous_parent',
            'method':'>=75% geodesic child overlap in retained editions; not a historical/current administrative reform crosswalk'}


def _recent(record, year):
    if year is None:return 'source_date_retained_background_not_year_filtered'
    start,end=record.get('year_start'),record.get('year_end')
    if type(start) is not int or type(end) is not int or start>year: return 'LOW'
    age=max(0,year-end)
    return 'HIGH' if age<=2 else 'MEDIUM' if age<=6 else 'LOW'


def _spatial_context_usable(record,match):
    if match['kind'] in ('administrative_area_intersection','original_point_inside_tile'):return True
    if match['kind']!='geocoded_settlement_inside_tile' or match.get('geocoding_confidence')!='HIGH':return False
    return (record.get('spatial_scale') in ('village','community')
            and record.get('evidence_type')!='research_site'
            and 'island' not in record.get('notes','').lower())


def _usable(record, match, year):
    if year is None:
        return (record.get('source_quality') in ('A','B') and _spatial_context_usable(record,match)
                and record.get('claim_scope')=='named_place_context')
    return (record.get('source_quality') in ('A','B') and _recent(record,year) in ('HIGH','MEDIUM')
            and (record.get('publication_year') is None or record['publication_year']<=year)
            and _spatial_context_usable(record,match)
            and record.get('claim_scope')=='named_place_context')


def _match_geometry(footprint,countries,year,root=ROOT,query_kind='tile_bounds'):
    from shapely.geometry import Point
    countries=sorted(set(COUNTRIES.get(c,c) for c in countries))
    records,contradictions,layers,metadata,manifest=load(root)
    geocode_path=Path(root)/'knowledge/locality_geocodes.jsonl'
    geocodes={eid:g for g in [json.loads(line) for line in geocode_path.read_text().splitlines() if line.strip()]
              if g['status']=='resolved' for eid in g['evidence_ids']} if geocode_path.exists() else {}
    total_area=area(footprint)
    if total_area<=0: raise ValueError('positive query area required')
    intersections={}; admins={}; missing=[]
    for iso in countries:
        for level in ('ADM1','ADM2'):
            key=iso+'-'+level
            if key not in layers:
                missing.append(key);continue
            hit=[]
            for f in layers[key]:
                if not f['geometry'].intersects(footprint):continue
                overlap=f['geometry'].intersection(footprint)
                share=area(overlap)/total_area
                if share<=0:continue
                entry={'shape_id':f['id'],'name':f['name'],'overlap_fraction':min(1.0,share),
                       'evidence_area_fraction_inside_query':min(1.0,area(overlap)/area(f['geometry'])),
                       'overlap_scope':'administrative context, not crop coverage',
                       'boundary_year':metadata[key].get('boundaryYearRepresented'),
                       'geometry_repaired_for_intersection':f['geometry_repaired_for_intersection']}
                hit.append(entry);intersections[(key,f['id'])]=(entry,overlap)
            admins[key]=sorted(hit,key=lambda x:(-x['overlap_fraction'],x['name']))
    matched=[];unresolved=[]; excluded_points=[]
    for r in records:
        if r['country'] not in countries:continue
        record=copy.deepcopy(r);match=None;overlap=None
        if r['evidence_id'] in geocodes:
            geo=geocodes[r['evidence_id']]
            if not footprint.covers(Point(geo['longitude'],geo['latitude'])):
                excluded_points.append(r['evidence_id']);continue
            match={'kind':'geocoded_settlement_inside_tile','level':'settlement',
                   'overlap_fraction':None,'locality_inside_tile':True,
                   'geocoding_confidence':geo['confidence'],
                   'coordinate_is_crop_observation':False,
                   'boundary_layer':None,'boundary_shape_id':None}
            record['supplementary_geocode']=copy.deepcopy(geo)
        elif r['geometry_type']=='point' and r.get('longitude') is not None and r.get('latitude') is not None:
            if footprint.covers(Point(r['longitude'],r['latitude'])):
                match={'kind':'original_point_inside_tile','level':'point','overlap_fraction':None,
                       'locality_inside_tile':True,'boundary_layer':None,'boundary_shape_id':None}
            else:
                excluded_points.append(r['evidence_id']);continue
        elif r.get('boundary_shape_id') and r.get('geocoding_source'):
            key=Path(r['geocoding_source']).stem
            value=intersections.get((key,r['boundary_shape_id']))
            if value:
                entry,overlap=value
                direct=(r['geometry_type'] in ('admin1','admin2') and
                        r.get('geocoding_confidence')=='HIGH' and r.get('claim_scope')=='named_place_context')
                match={'kind':'administrative_area_intersection' if direct else 'parent_context_only',
                       'level':key.split('-')[-1],'boundary_layer':key,'boundary_shape_id':r['boundary_shape_id'],
                       'boundary_name':entry['name'],'overlap_fraction':entry['overlap_fraction'],
                       'evidence_area_fraction_inside_query':entry['evidence_area_fraction_inside_query'],
                       'locality_inside_tile':None,'method':r.get('geocoding_method'),
                       'boundary_year':entry['boundary_year']}
            elif key not in layers: unresolved.append(r['evidence_id'])
            else: continue
        elif r['geometry_type'] in ('admin1','admin2'):
            level=r['geometry_type'].upper();key=r['country']+'-'+level
            name=r.get('admin'+level[-1]+'_normalized') or r.get('place_name_normalized')
            candidates=[f for f in layers.get(key,[]) if norm(f['name'])==norm(name)]
            if len(candidates)==1:
                value=intersections.get((key,candidates[0]['id']))
                if not value:continue
                entry,overlap=value
                match={'kind':'administrative_area_intersection','level':level,'boundary_layer':key,
                       'boundary_shape_id':entry['shape_id'],'boundary_name':entry['name'],
                       'overlap_fraction':entry['overlap_fraction'],'locality_inside_tile':None,
                       'evidence_area_fraction_inside_query':entry['evidence_area_fraction_inside_query'],
                       'method':'unique exact administrative name in retained boundary edition',
                       'boundary_year':entry['boundary_year']}
            else:unresolved.append(r['evidence_id'])
        else:
            if r['geometry_type']!='country':unresolved.append(r['evidence_id'])
            match={'kind':'country_background' if r['geometry_type']=='country' else 'unresolved_locality_background',
                   'level':'country','boundary_layer':None,'boundary_shape_id':None,
                   'overlap_fraction':None,'locality_inside_tile':None}
        if match is None:continue
        record['spatial_match']=match;record['target_year_alignment']={
            'observation_relevance':_recent(r,year),
            'publication':'source_date_retained_background_not_year_filtered' if year is None else 'unknown' if r.get('publication_year') is None else
                          'after_target_year' if r['publication_year']>year else 'by_target_year'}
        record['usable_local_context']=_usable(r,match,year)
        matched.append(record)
    matched_ids={r['evidence_id'] for r in matched}
    links=[x for x in contradictions if x['evidence_id_1'] in matched_ids or x['evidence_id_2'] in matched_ids]
    related_ids={x[k] for x in links for k in ('evidence_id_1','evidence_id_2')}-matched_ids
    related=[{'evidence_id':r['evidence_id'],'summary':r['evidence_summary'],'country':r['country'],
              'source_id':r['source_id'],'scope':'linked counterevidence; not asserted inside query'} for r in records if r['evidence_id'] in related_ids]
    chains={key:[parent_chain(str(Path(root).resolve()),digest(Path(root)/'manifest.json'),key,x['shape_id'])
                 for x in values] for key,values in admins.items() if key.endswith('ADM2')}
    result={'version':VERSION,'summary_language':'en','query_kind':query_kind,'target_year':year,
            'bounds_wgs84':list(footprint.bounds),'countries':countries,'admin_matches':admins,
            'administrative_chains':chains,
            'missing_boundary_layers':missing,'package_sha256':digest(Path(root)/'manifest.json'),
            'evidence':matched,'contradictions':links,'linked_counterevidence':related,
            'unresolved_locality_ids':sorted(set(unresolved)),'excluded_original_points_outside_tile':excluded_points,
            'tile_fraction_estimate':None,'guidance':GUIDANCE}
    result['prior_card']=card(result,layers,footprint)
    from cashew_background_v1 import build as planting_background
    result['planting_background']=planting_background(result)
    result['prior_card']['planting_background']=copy.deepcopy(result['planting_background'])
    return result


def query(bounds,countries,target_year=2025,root=ROOT):
    from shapely.geometry import box
    if len(bounds)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) for x in bounds):
        raise ValueError('finite WGS84 bounds required')
    if not(-180<=bounds[0]<bounds[2]<=180 and -90<=bounds[1]<bounds[3]<=90):
        raise ValueError('invalid WGS84 bounds')
    if target_year is not None and type(target_year) is not int:raise ValueError('integer reference year or None required')
    if not countries:raise ValueError('country context required')
    return _match_geometry(box(*bounds),countries,target_year,root)


def query_admin(country,level,name,target_year=2025,root=ROOT):
    iso=COUNTRIES.get(country,country)
    if level not in ('ADM1','ADM2'):raise ValueError('ADM1 or ADM2 required')
    layers=load(root)[2]
    hits=[f for f in layers.get(iso+'-'+level,[]) if norm(f['name'])==norm(name)]
    if len(hits)!=1:raise ValueError('administrative name unresolved or ambiguous; do not guess')
    result=_match_geometry(hits[0]['geometry'],[iso],target_year,root,query_kind='administrative_polygon')
    result['queried_admin']={'country':iso,'level':level,'name':hits[0]['name'],'shape_id':hits[0]['id']}
    return result


def card(context,layers=None,footprint=None):
    from shapely.ops import unary_union
    records=context['evidence']; usable=[r for r in records if r['usable_local_context']]
    positive=[r for r in usable if r['crop']=='cashew' and r['cashew_signal']=='positive']
    limited=[r for r in usable if r['crop']=='cashew' and r['cashew_signal']=='negative_explicit'
             and r['evidence_type'] in ('production_statistics','production_zone')
             and r.get('claim_strength') not in ('plot_mortality','local_decline','historical_ecological_caution')]
    competitors=[r for r in usable if r['crop']!='cashew' and r['evidence_type']!='project_target_area']
    # Undated/historical named regional evidence remains useful for a
    # background hypothesis, without claiming it measures the reference year.
    older_context=[r for r in records if not r['usable_local_context']
        and r['source_quality'] in ('A','B') and _spatial_context_usable(r,r['spatial_match'])
        and r['claim_scope']=='named_place_context'
        and (context['target_year'] is None or r.get('year_start') is None or r['year_start']<=context['target_year'])
        and (context['target_year'] is None or r.get('publication_year') is None or r['publication_year']<=context['target_year'])]
    older_presence=[r for r in older_context if r['crop']=='cashew' and r['cashew_signal']=='positive']
    older_competitors=[r for r in older_context if r['crop']!='cashew' and r['evidence_type']!='project_target_area']
    low_coverage=0.0
    if footprint is not None and layers is not None:
        polygons=[]
        for r in limited:
            m=r['spatial_match']
            polygons += [f['geometry'].intersection(footprint) for f in layers.get(m['boundary_layer'],[]) if f['id']==m['boundary_shape_id']]
        if polygons:low_coverage=min(1.0,area(unary_union(polygons))/area(footprint))
    dominant=low_coverage>=.5
    signal=('limited_with_presence' if positive else 'documented_limited_context') if dominant else \
           'documented_presence' if positive else 'historical_or_undated_presence' if older_presence else 'unknown'
    categories={'positive_cashew':positive,'explicit_limited_cashew':limited,'competing_crops':competitors,
                'historical_or_undated_presence':older_presence,
                'historical_or_undated_competitors':older_competitors,
                'planned_or_suitability':[r for r in records if r['crop']=='cashew' and r['cashew_signal']=='uncertain'],
                'parent_or_background':[r for r in records if not r['usable_local_context']]}
    # Two per main category, one per secondary category. Full evidence and all
    # counter-links remain in the pinned report; summaries never hide a count.
    snippets=[]; emitted=set()
    for category,rs in categories.items():
        rank=lambda r:(r['source_quality']!='A',r['target_year_alignment']['observation_relevance']!='HIGH',
                       -(r['spatial_match'].get('overlap_fraction') or 0),r['evidence_id'])
        candidates=[r for r in sorted(rs,key=rank) if r['evidence_id'] not in emitted]
        for r in candidates[:2 if category in ('positive_cashew','explicit_limited_cashew','competing_crops') else 1]:
            emitted.add(r['evidence_id'])
            snippet={'id':r['evidence_id'],'category':category,'crop':r['crop'],
                'place':r['place_name_original'],'match':r['spatial_match']['kind'],
                'summary':r['evidence_summary'],'quality':r['source_quality'],
                'observation_relevance':r['target_year_alignment']['observation_relevance'],
                'source_id':r['source_id'],'source_url':r['source_url'],'page':r.get('page'),
                'lines':[r.get('line_start'),r.get('line_end')]}
            if r.get('supplementary_geocode'):
                g=r['supplementary_geocode']
                snippet['geocode']={k:g.get(k) for k in ('latitude','longitude','source_url','confidence','coordinate_scope')}
            snippets.append(snippet)
    return {'version':VERSION,'target_year':context['target_year'],'countries':context['countries'],
            'bounds_wgs84':context.get('bounds_wgs84'),
            'admin_context':{k:[{field:m.get(field) for field in ('shape_id','name','overlap_fraction','boundary_year')}
                                for m in values[:2]] for k,values in context.get('admin_matches',{}).items() if values},
            'cashew_context':signal,'quantitative_cashew_fraction':None,
            'limited_context_covers_majority':dominant,
            'limited_context_footprint_share':round(low_coverage,4),
            'overlap_is_administrative_context_not_cashew_area':True,
            'competing_crop_context':sorted({r['crop'] for r in competitors+older_competitors}),
            'competing_crop_context_strength':'recent_local' if competitors else 'historical_or_undated_region' if older_competitors else 'unknown',
            'evidence_counts':{k:len(v) for k,v in categories.items()},
            'independent_local_positive_families':sorted({r['independence_group'] for r in positive if r['independence_status']=='ORIGINAL_FIELD_OR_STATISTICAL_SOURCE'}),
            'unresolved_localities':len(context['unresolved_locality_ids']),
            'contradiction_ids':[[x['evidence_id_1'],x['evidence_id_2']] for x in context['contradictions']],
            'snippets':snippets,'more_evidence_in_report':len(records)>len(snippets),
            'guidance':GUIDANCE}


def render_card(context):
    c=context['prior_card'];lines=['# Tile prior card','',
        'Reference year: '+str(c['target_year'])+'; countries: '+', '.join(c['countries'])+'.',
        'Cashew context: **'+c['cashew_context']+'**. No measured tile prevalence is supplied.',
        'Competing crops: '+(', '.join(c['competing_crop_context']) or 'unknown')+'.',
        'Matched records by role: '+', '.join(k+'='+str(v) for k,v in c['evidence_counts'].items())+'.',
        'Administrative overlap is context coverage, not crop coverage. Unlocated parent-matched villages are not known to be inside this tile.','']
    identity=('Tile: '+context['tile_id']+'. ') if context.get('tile_id') else ''
    regions='; '.join(k+': '+', '.join(m['name'] for m in values) for k,values in c.get('admin_context',{}).items())
    if identity or regions:lines.insert(2,identity+'Administrative context: '+(regions or 'unknown')+'.')
    for s in c['snippets']:
        locator=(' p. '+str(s['page'])) if s.get('page') is not None else ''
        if s.get('lines') and all(v is not None for v in s['lines']):locator+=' lines '+str(s['lines'][0])+'–'+str(s['lines'][1])
        lines.append('- '+s['id']+' ['+s['category']+'; '+s['match']+'; '+s['quality']+'; '+s['observation_relevance']+']: '+s['summary']+' [Source]('+s['source_url']+')'+locator+'.')
    if c['contradiction_ids']:lines += ['', 'Linked counterevidence: '+ '; '.join('/'.join(x) for x in c['contradiction_ids'])+'. See full report for scope/time differences.']
    lines += ['',GUIDANCE,'']
    text='\n'.join(lines)
    if context.get('planting_background'):
        from cashew_background_v1 import render as render_background
        text=render_background(context,context['planting_background'])+'\n\n'+text
    return text


def prepare(job):
    selection=json.loads(Path(job['supplementary']['selection']['report']).read_text())
    policy=job['geographic_prior_policy']
    if policy['version']!=VERSION:raise ValueError('spatial policy version mismatch')
    from shapely.geometry import box
    footprint=box(*selection['target_bounds'])
    layers=load()[2]
    neighbors=[key[:3] for key,features in layers.items() if key.endswith('ADM1')
               and any(f['geometry'].intersects(footprint) and area(f['geometry'].intersection(footprint))>0 for f in features)]
    countries=sorted(set(neighbors+[COUNTRIES.get(job['country'],job['country'])]))
    context=query(selection['target_bounds'],countries,int(job['target_year']))
    if context['package_sha256']!=policy['package_sha256']:raise ValueError('template/package mismatch')
    path=Path(job['output_root'])/'data/geographic_prior.json'
    path.write_text(json.dumps(context,ensure_ascii=False,indent=2)+'\n')
    card_path=path.with_name('prior_card.md');card_path.write_text(render_card(context))
    return {'version':VERSION,'report':str(path.resolve()),'sha256':digest(path),
            'card':str(card_path.resolve()),'card_sha256':digest(card_path),
            'package_sha256':context['package_sha256']}


def packet(context,pinned):
    result=copy.deepcopy(context['prior_card'])
    background=result.get('planting_background') or {}
    if background:
        background['evidence_details']=background['evidence_details'][:8]
        background['all_support_records_in_context_report']=True
    result['context_report']=pinned['report'];result['context_sha256']=pinned['sha256']
    result['card_path']=pinned.get('card');result['package_sha256']=context['package_sha256']
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bbox',type=float,nargs=4)
    parser.add_argument('--country',action='append',required=True);parser.add_argument('--year',type=int,default=2025)
    parser.add_argument('--admin-level',choices=['ADM1','ADM2']);parser.add_argument('--admin-name')
    parser.add_argument('--output',type=Path);parser.add_argument('--card',type=Path)
    args=parser.parse_args()
    if args.admin_level and args.admin_name:
        if len(args.country)!=1 or args.bbox:parser.error('admin query takes one country and no bbox')
        value=query_admin(args.country[0],args.admin_level,args.admin_name,args.year)
    elif args.bbox:value=query(args.bbox,args.country,args.year)
    else:parser.error('provide --bbox or --admin-level and --admin-name')
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    else:print(text,end='')
    if args.card:args.card.write_text(render_card(value))
