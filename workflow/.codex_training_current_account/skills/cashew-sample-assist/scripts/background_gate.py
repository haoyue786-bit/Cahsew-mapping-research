"""Audit actual local crop research before releasing an annotation for review.

Call record_result AFTER the audited public tool returned, including exceptions
converted to {success:False,error:...}. Global notes, climate and model assertions
never substitute for recorded local public searches/readings. This module does
not make model calls, read human labels or assign crop classes.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import re
import time
import unicodedata
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parent
TOOLS = {'search_regional_evidence', 'read_public_evidence', 'get_climate_context'}
DISCLOSURE = 'LOCAL_BACKGROUND_UNAVAILABLE:'
DEFAULT_CROP_TERMS = ['cashew', 'cashews', 'anacarde', 'anacardier', 'anacardiers',
                      'cajou', 'cocoa', 'cacao', 'cacaoyer', 'cacaoyers', '腰果', '可可']
COUNTRY_ONLY = ['civ', 'gha', 'ghana', 'cote d ivoire', 'ivory coast', '科特迪瓦', '加纳']
PRIMARY_DOMAINS = ['anader.ci', 'gouv.ci', 'gov.gh', 'cocobod.gh', 'tcda.gov.gh',
                   'conseilcotonanacarde.ci', 'fao.org', 'worldagroforestry.org', 'cgiar.org']


def _norm(value):
    text = unicodedata.normalize('NFKD', str(value).casefold())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'[^\w]+', ' ', text, flags=re.UNICODE).strip()


def _contains(text, term):
    text, term = _norm(text), _norm(term)
    return bool(term) and (' ' + term + ' ') in (' ' + text + ' ')


def _items(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if isinstance(x, (str, int, float))]
    return []


def _scope(context):
    detailed = context.get('background_scope') or {}
    names = _items(detailed.get('place_names')) + _items(context.get('scope_terms'))
    excluded = COUNTRY_ONLY + _items(detailed.get('country_names')) + _items(context.get('country'))
    excluded = {_norm(x) for x in excluded}
    names = list(dict.fromkeys(x for x in names if _norm(x) not in excluded and _norm(x)))
    crops = list(dict.fromkeys(_items(context.get('crop_terms')) + DEFAULT_CROP_TERMS))
    return names, crops


def _local_crop(text, context):
    names, crops = _scope(context)
    return any(_contains(text, x) for x in names) and any(_contains(text, x) for x in crops)


def _url(value):
    try:
        p = urlsplit(str(value))
        if p.scheme not in ('https', 'http') or not p.hostname or p.username or p.password:
            return None
        return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip('/'), p.query, ''))
    except ValueError:
        return None


def _primary(url, context):
    address = _url(url)
    if not address:
        return False
    allowed = {_url(u) for u in _items(context.get('known_primary_urls'))}
    if address in allowed:
        return True
    host = urlsplit(address).hostname
    return any(host == d or host.endswith('.' + d) for d in PRIMARY_DOMAINS)


def _root(context):
    cid = context.get('current_candidate_id')
    if not cid or not re.fullmatch(r'[A-Za-z0-9_-]+', str(cid)):
        raise ValueError('A single declared current_candidate_id is required')
    return Path(context.get('background_gate_root') or ROOT / 'background_evidence') / str(cid)


def record_result(context, name, args, result):
    """Record the ACTUAL public tool result; return its public provenance ID.

    Source-learning calls (no current target) and non-background tools are ignored.
    Caller must not synthesize success or use model-written evidence summaries.
    """
    if name not in TOOLS or not context.get('current_candidate_id'):
        return None
    if name == 'get_climate_context' and args.get('candidate_id') != context['current_candidate_id']:
        raise ValueError('Cannot record another AOI climate result')
    metadata = result.get('metadata') or {}
    # These fields come from the real audited implementation, not model claims.
    keys = ['query', 'provider', 'request_url', 'retrieved_utc', 'results', 'source_url',
            'requested_url', 'title', 'text', 'source_sha256', 'extraction',
            'total_extracted_chars', 'cached_public_evidence_replayed']
    selected = {k:metadata[k] for k in keys if k in metadata}
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    event = {'candidate_id':context['current_candidate_id'], 'tool':name,
             'arguments':dict(args), 'success':result.get('success') is True,
             'error':result.get('error'), 'metadata':selected, 'recorded_utc':stamp,
             'scope_terms':_scope(context)[0], 'climate_is_crop_presence_evidence':False}
    identifier = 'background_' + hashlib.sha256(json.dumps(event, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    event['evidence_id'] = identifier
    folder = _root(context) / 'events'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / ('%020d_%s.json' % (time.time_ns(), identifier))
    target.write_text(json.dumps(event, ensure_ascii=False, indent=2) + '\n')
    return identifier


def _failed_actual_attempt(event):
    if event.get('success'):
        return False
    # Argument/schema rejection is not a performed public-network lookup.
    error = str(event.get('error') or '').casefold()
    return any(x in error for x in ['timeout', 'timed out', 'connection', 'network',
        'httperror', 'http ', 'ssl', 'proxy', 'dns', 'certificate', 'unavailable',
        'resolve', 'refused', 'urlerror', '远程', '网络', '超时'])


def _result_text(result):
    bits = []
    for key in ['provider_evidence_summary', 'map_actions_and_findings']:
        bits += [str(v) for v in result.get(key, [])]
    for head in ['fourclass', 'directbinary']:
        value = result.get(head) or {}
        bits += [str(value.get('decision_procedure', ''))]
        bits += [str(v) for v in value.get('limitations', [])]
    return '\n'.join(bits)


def _annotation_flags(result):
    uncertainty, woody_negative = False, False
    for head in ['fourclass', 'directbinary']:
        for row in (result.get(head) or {}).get('region_assignments', []):
            code = row.get('class_id')
            uncertainty |= code == 10
            if head == 'fourclass' and code == 1:
                woody_negative = True
            if head == 'directbinary' and code == 0:
                reason = _norm(row.get('landunit_reason', ''))
                woody_negative |= any(_contains(reason, term) for term in [
                    'tree', 'trees', 'woody', 'cocoa', 'cacao', 'orchard',
                    'plantation', 'agroforestry', 'regrowth', '混合', '树园'])
    return {'contains_uncertainty':bool(uncertainty),
            'contains_woody_noncashew_judgment':bool(woody_negative)}


def audit_result(context, result):
    """Return a bounded-action gate verdict; never raises for absent research.

    Eligible means background policy was met for a REVIEW product, not accuracy
    certification. An unavailable-evidence exception always requires manual review.
    The driver should allow at most two corrective turns, then leave visible review
    status instead of endlessly retrying or treating no evidence as non-Cashew.
    """
    folder = _root(context)
    events = [json.loads(p.read_text()) for p in sorted((folder / 'events').glob('*.json'))]
    events = [e for e in events if e.get('candidate_id') == context['current_candidate_id']]
    searches = [e for e in events if e['tool'] == 'search_regional_evidence'
                and _local_crop(e['arguments'].get('query', ''), context)
                and (e['success'] or _failed_actual_attempt(e))]
    unique = {}
    for e in searches:
        unique[_norm(e['arguments'].get('query', ''))] = e
    searches = list(unique.values())
    reads = [e for e in events if e['tool'] == 'read_public_evidence'
             and (e['success'] or _failed_actual_attempt(e))]
    verified = []
    for e in reads:
        md = e['metadata'];text = str(md.get('text') or '').strip()
        address = md.get('source_url') or md.get('requested_url') or e['arguments'].get('url')
        if not (e['success'] and _url(address) and text):
            continue
        body = str(md.get('title') or '') + '\n' + text
        if _local_crop(body, context):
            verified.append(e)
            continue
        # A successfully opened regional table/document can have its local scope
        # established by the corresponding actual search excerpt/title.
        if not any(_contains(body, x) for x in _scope(context)[1]):
            continue
        for search in searches:
            for hit in search['metadata'].get('results', []):
                excerpt = str(hit.get('title', '')) + ' ' + str(hit.get('snippet', ''))
                if _url(hit.get('url')) == _url(address) and _local_crop(excerpt, context):
                    verified.append(e)
                    break
            if e in verified:
                break
    successful_empty = [e for e in searches if e['success'] and not e['metadata'].get('results')]
    failed_searches = [e for e in searches if _failed_actual_attempt(e)]
    failed_primary_reads = [e for e in reads if _failed_actual_attempt(e)
                            and _primary(e['arguments'].get('url'), context)]
    distinct_read_urls = {_url(e['arguments'].get('url')) for e in reads}
    distinct_read_urls.discard(None)
    # No unbounded retries. These are actual performed fallback paths, not self-report.
    exhausted = (len(searches) >= 2 and
        (len(successful_empty) + len(failed_searches) == len(searches)
         or len(distinct_read_urls) >= 2))
    exhausted |= bool(failed_searches and failed_primary_reads)
    flags = _annotation_flags(result)
    required = bool(context.get('requires_background') or flags['contains_uncertainty']
                    or flags['contains_woody_noncashew_judgment'])
    base = {'candidate_id':context['current_candidate_id'], 'required':required,
        'eligible':False, 'requires_manual_review':False, 'stop_retrying':False,
        'status':'needs_local_crop_search', 'need_action':None,
        'local_search_attempts':len(searches), 'actual_page_read_attempts':len(reads),
        'verified_local_pages':len(verified),
        'evidence_ids':[e['evidence_id'] for e in searches + verified],
        'source_urls':[e['metadata'].get('source_url') or e['arguments'].get('url') for e in verified],
        'climate_calls':sum(e['tool']=='get_climate_context' for e in events),
        'no_evidence_is_not_noncashew_evidence':True,
        'background_does_not_assign_pixel_classes':True,
        'flags':flags, 'scope_terms':_scope(context)[0]}
    if not required:
        base.update(eligible=True, status='not_required_no_unresolved_crop_ambiguity')
    elif searches and verified:
        base.update(eligible=True, status='local_crop_research_verified')
    elif exhausted:
        text = _result_text(result)
        disclosed = DISCLOSURE in text
        base.update(eligible=disclosed, requires_manual_review=True, stop_retrying=True,
            status='local_background_unavailable_review_required' if disclosed else 'must_disclose_unavailable_background',
            limitations='Actual local searches/readings were exhausted without a verified local crop source. This does not establish Cashew absence.')
        if not disclosed:
            base['need_action'] = ('Do not repeat web attempts indefinitely. In limitations add '+DISCLOSURE+
                ' and the actual retrieval failure/empty result. Preserve uncertainty for unresolved Cashew/cocoa/tree-crop areas; '
                'absence of a report is not evidence for Non-cashew. This product requires manual review.')
    elif searches:
        base.update(status='needs_actual_local_page_read', need_action=(
            'Open at least one relevant primary public page for this AOI using read_public_evidence and compare Cashew/cocoa/local tree-crop alternatives. '
            'If the first query gives no usable source, try a different local query or a known primary URL. NASA climate alone is insufficient.'))
    else:
        base['need_action'] = ('Before final classification, perform search_regional_evidence for this actual AOI using a local place/region from '+
            repr(_scope(context)[0])+' together with Cashew/anacarde and/or cocoa/cacao. Then read a relevant public source. '
            'Global background, country-only searches and climate do not satisfy the required local crop investigation.')
    write_target = folder / 'last_audit.json'
    write_target.parent.mkdir(parents=True, exist_ok=True)
    temp = write_target.with_suffix('.json.tmp')
    temp.write_text(json.dumps(base, ensure_ascii=False, indent=2) + '\n')
    temp.replace(write_target)
    return base
