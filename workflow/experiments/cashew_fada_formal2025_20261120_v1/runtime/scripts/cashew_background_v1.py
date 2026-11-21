"""Year-independent planting background and explicit spatial admission evidence."""
from __future__ import annotations

VERSION='cashew_planting_background_v1_20261008'


def presence_support(record, queried_admin=None):
    """Text must describe actual cultivation and belong inside the queried ADM."""
    m=record.get('spatial_match') or {}
    if (record.get('crop')!='cashew' or record.get('cashew_signal')!='positive'
            or record.get('source_quality') not in ('A','B')
            or record.get('claim_scope')!='named_place_context'
            or record.get('claim_strength') in ('suitability','planned_target','planting_material_infrastructure','observed_nursery')
            or record.get('evidence_type')=='project_target_area'):
        return False
    kind=m.get('kind')
    if kind in ('original_point_inside_tile','geocoded_settlement_inside_tile'):
        return (kind=='original_point_inside_tile' or
            (m.get('geocoding_confidence')=='HIGH' and record.get('spatial_scale') in ('village','community')
             and record.get('evidence_type')!='research_site' and 'island' not in record.get('notes','').lower()))
    if kind!='administrative_area_intersection':return False
    if queried_admin is None:
        # Tile context: regional cultivation does not establish an orchard
        # inside the tile; preserve its administrative scope explicitly.
        return True
    from spatial_prior_v2 import norm
    same_name=(m.get('level')=='ADM1' and
               norm(m.get('boundary_name',''))==norm(queried_admin.get('name','')))
    record_inside=(m.get('evidence_area_fraction_inside_query') or 0)>=.75
    equivalent=same_name and (m.get('overlap_fraction') or 0)>=.75
    return bool(record_inside or equivalent)


def build(context, queried_admin=None):
    records=context.get('evidence',[])
    supported=[r for r in records if presence_support(r,queried_admin)]
    details=[]
    for r in supported:
        m=r['spatial_match'];annotation=r.get('background_detail') or {}
        details.append(dict(evidence_id=r['evidence_id'],place=r.get('place_name_original'),
            admin1=r.get('admin1_normalized'),admin2=r.get('admin2_normalized'),
            summary=annotation.get('summary_en',r.get('evidence_summary')),
            cultivation_scale=r.get('claim_strength','presence'),
            reported_value=r.get('reported_value'),reported_unit=r.get('reported_unit'),
            reported_value_scope=r.get('reported_value_scope','named place or group as stated by source; never an implied ADM area fraction'),
            observations=annotation.get('observations',[]),
            documented_localities=annotation.get('documented_localities',[]),
            source_observation_period=[r.get('year_start'),r.get('year_end')],
            source_publication_year=r.get('publication_year'),source_quality=r.get('source_quality'),
            source_id=r.get('source_id'),source_family=r.get('independence_group'),
            source_url=r.get('source_url'),page=r.get('page'),lines=[r.get('line_start'),r.get('line_end')],
            match_kind=m['kind'],query_coverage_fraction=m.get('overlap_fraction'),
            evidence_area_fraction_inside_query=m.get('evidence_area_fraction_inside_query'),
            geographic_scope='named regional context; not a plantation polygon',notes=r.get('notes','')))
    details.sort(key=lambda d:(d['cultivation_scale']!='major_zone',not d['observations'],
                              d['reported_value'] is None,d['source_quality']!='A',str(d['place']),d['evidence_id']))
    return dict(version=VERSION,temporal_policy='shared_planting_background_across_training_years',
        cashew_context='documented_presence' if supported else 'unknown',
        has_cultivation_support=bool(supported),support_evidence_ids=[d['evidence_id'] for d in details],
        supporting_record_count=len(supported),source_families=sorted({r.get('independence_group') or r['source_id'] for r in supported}),
        where_documented=sorted({r['spatial_match'].get('boundary_name') or r['place_name_original'] for r in supported}),
        scale_summary=('source_describes_major_growing_zone' if any(r.get('claim_strength')=='major_zone' for r in supported)
                       else 'source_reports_production_or_orchard_measurements' if any(r.get('reported_value') is not None or (r.get('background_detail') or {}).get('observations') for r in supported)
                       else 'cultivation_documented_extent_not_quantified' if supported else 'unknown'),
        evidence_details=details,measured_cashew_area_fraction=None,
        cautions=['Presence is shared across training years; source dates remain observations, not timeless production totals.',
            'Tonnes, farmer counts, group orchard shares and seedlings are different quantities; do not convert them into pixel prevalence.',
            'Multiple records may share a source family; source families are provenance groups, not guaranteed independent confirmations.',
            'Administrative intersection is regional context. A settlement coordinate is not orchard GPS.',
            'Missing evidence is unknown. Production conflicts, historical boundaries and competitors remain in the full report.'])


def render(context,background,title='Cashew planting background'):
    lines=['# '+title,'','Planting background is shared across training years; source dates are retained below.',
        'Cashew context: **'+background['cashew_context']+'**.',
        'Where documented: '+(', '.join(background['where_documented']) or 'unknown')+'.',
        'Scale: '+background['scale_summary']+'.',
        'Supporting records: '+str(background['supporting_record_count'])+'; source families: '+str(len(background['source_families']))+'.',
        'Measured administrative Cashew area fraction: unknown.','']
    for d in background['evidence_details']:
        locator=(' page '+str(d['page'])) if d['page'] is not None else ''
        if all(v is not None for v in d['lines']):locator+=' lines '+str(d['lines'][0])+'-'+str(d['lines'][1])
        period=d['source_observation_period']
        period_text='undated' if all(v is None for v in period) else str(period[0])+'–'+str(period[1])
        lines += ['## '+str(d['place'])+' — '+d['evidence_id'],
            d['summary'] or 'See source.',
            'Cultivation scale: '+d['cultivation_scale']+'; source observation period: '+period_text+'.']
        if d['reported_value'] is not None:lines.append('Reported amount: '+str(d['reported_value'])+' '+str(d['reported_unit'])+'. Scope: '+d['reported_value_scope']+'.')
        if d['documented_localities']:lines.append('Named local economic/cultivation context: '+', '.join(d['documented_localities'])+'. Exact orchard positions are not supplied.')
        if d['observations']:
            lines += ['','| Source year | Reported amount | Scope | Status |','|---|---:|---|---|']
            for amount in d['observations']:
                value=amount.get('value');value_text=format(value,',') if isinstance(value,(int,float)) else str(value)
                lines.append('| '+str(amount.get('year','unspecified'))+' | '+value_text+' '+str(amount.get('unit',''))+' | '+str(amount.get('scope','as reported'))+' | '+('provisional' if amount.get('provisional') else 'reported')+' |')
            lines.append('')
        fraction=d['evidence_area_fraction_inside_query']
        fraction_text=format(fraction*100,'.2f')+'%' if fraction is not None else 'not applicable to this point/background match'
        lines.append('Spatial attribution: '+d['match_kind']+'; evidence area inside this query: '+fraction_text+'. This is boundary coverage, not crop coverage.')
        if d['notes']:lines.append('Scope / uncertainty: '+d['notes'])
        lines += ['Source: ['+str(d['source_id'])+']('+str(d['source_url'])+')'+locator+'; quality '+str(d['source_quality'])+'; family '+str(d['source_family'])+'.','']
    other=[r for r in context.get('evidence',[]) if r['evidence_id'] not in background['support_evidence_ids']]
    if other:
        lines += ['## Other context and counterevidence','']
        for r in other:
            m=r.get('spatial_match') or {}
            lines.append('- '+r['evidence_id']+' / '+r.get('place_name_original','unknown')+' / '+r.get('crop','unknown')+
                ': '+r.get('evidence_summary','')+' Spatial scope: '+m.get('kind','unknown')+
                '; evidence-area share inside query: '+str(m.get('evidence_area_fraction_inside_query'))+
                '. [Source]('+r['source_url']+').')
    if context.get('contradictions'):
        lines += ['','## Linked disagreements',json_dump(context['contradictions'])]
    lines += ['','## Agent checks after training',
        'Compare the checkpoint-bound map with the named growing places and source-described scale. State: broadly compatible, concern requiring investigation, or insufficient evidence; cite the actual map scope and evidence IDs.',
        'Inspect documented subregions separately when spatial outputs exist. Missing predicted Cashew in a documented locality or high predictions in a competing-crop context motivates inspection, not an automatic error verdict.',
        'Use raw argmax and any requested threshold policy as separate prediction scopes. Do not infer tonnes, exact coverage or accuracy from textual agreement. Keep plausible novel Cashew, source safety and counterevidence visible.',
        'When only a full-region statistic exists, mark within-region location matching unverified. Never invent village or orchard coordinates.','']
    lines += background['cautions']+['']
    return '\n'.join(lines)


def json_dump(value):
    import json
    return json.dumps(value,ensure_ascii=False,indent=2)
