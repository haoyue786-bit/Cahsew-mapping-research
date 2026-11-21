#!/usr/bin/env python3
"""Merge completed visual audits without changing original annotations or human reviews.

This module is also the reference-retrieval API. A selected model proposal is
never gold. User reviews remain a separate, authoritative overlay.
"""
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent
PRESERVED = ('original_label', 'source_annotations', 'geometry', 'source_metadata')
CLASS_NAMES = {0: 'Crop and others', 1: 'Mix trees and wetland/water', 2: 'Cashew', 3: 'Built/city', 10: 'Uncertain'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def original_fingerprint(records):
    return fingerprint({r['id']: {k: r.get(k) for k in PRESERVED} for r in records})


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name + '.', delete=False) as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def source_path(root, value):
    p = Path(value)
    return p if p.is_absolute() else root / p


def attach(root, record, path, role, batch, **metadata):
    path = source_path(root, path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha(path)
    safe = hashlib.sha256(record['id'].encode()).hexdigest()[:20]
    target = root / 'assets' / 'reviewed' / batch / safe / (digest[:20] + path.suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
    if sha(target) != digest:
        raise ValueError('Evidence copy mismatch: ' + str(path))
    image = dict(path=str(target.relative_to(root)), role=role, sha256=digest,
                 source_path=str(path), review_batch=batch, **metadata)
    images = record.setdefault('images', [])
    if not any(x.get('path') == image['path'] and x.get('role') == role for x in images):
        images.append(image)
    return image['path']


def class_conflict(original, proposed):
    old = original.get('class_id')
    if old is None or proposed == 10:
        return None
    space = str(original.get('class_space', original.get('label_space', ''))).lower()
    if 'binary' in space:
        return (old == 2 or original.get('class_name', '').lower() == 'cashew') != (proposed == 2)
    return old != proposed


def human_anchor_label_supported(record, anchor):
    """Accept an explicit human subtype, never invent one from binary negative0."""
    original=record.get('original_label') or {}
    cls=anchor.get('class_id')
    if cls not in (0,1,2,3) or original.get('class_id')!=cls:
        return False
    basis=anchor.get('semantic_basis')
    space=str(original.get('class_space',original.get('label_space',''))).lower()
    if space in ('fourclass','original_fourclass'):
        return basis=='original_human_fourclass'+str(cls) or cls==2 and basis=='original_human_class2'
    # A binary positive explicitly encoded Cashew2 has a known semantic subtype;
    # binary Non-cashew0 does not identify Crop, Mix trees or Built.
    return cls==2 and 'binary' in space and basis=='original_human_class2'


def applicability(result):
    """Separate a supported image-appearance example from historical label claims.

    Mixed/unknown/different capture dates qualify the use; they do not invalidate
    clear visible cover. They never establish a correction to an old human label.
    """
    proposed = result.get('proposed_label', {}).get('class_id')
    sam = result.get('evidence', {}).get('SAM_audit') or {}
    quality = result.get('quality') or {}
    supported = (proposed in (0, 1, 2, 3)
                 and result.get('seedpoint_class') == proposed
                 and result.get('seed_homogeneous') is True
                 and bool(result.get('geometry'))
                 and sam.get('seed_in_selected_mask') is True
                 and sam.get('class_id') == proposed)
    return dict(appearance_reference_eligible=supported and result.get('suitability') == 'keep',
                image_appearance_supported=supported,
                date_qualification=quality.get('date_alignment', 'unknown'),
                historical_label_year_compatible=quality.get('date_alignment') == 'aligned',
                historical_label_correction_status='not_established',
                scope='Visible image appearance at supplied capture-date provenance; original historical label remains unchanged',
                date_note=quality.get('date_note'))


def agent_selections(root):
    """Load independent agent visual choices, bound to frozen results/evidence."""
    path = root / 'AGENT_CURATOR_SELECTIONS.json'
    if not path.exists():
        return {}
    selections = {}
    for item in read(path).get('records', []):
        if item.get('selection_origin') != 'agent_visual_review' or item.get('selected_for_appearance') is not True:
            continue
        if not item.get('observed_images'):
            raise ValueError('Agent visual selection lacks observed image evidence')
        for image in item['observed_images']:
            if sha(source_path(root, image['path'])) != image['sha256']:
                raise ValueError('Agent visual-selection evidence changed: ' + image['path'])
        key = (item['batch_id'], item['anonymous_id'], item['result_sha256'])
        if key in selections:
            raise ValueError('Duplicate agent selection for one frozen result')
        selections[key] = item
    return selections


def merge(root=ROOT, batches=None, dry_run=False):
    root = Path(root).resolve()
    with (root / '.catalog_merge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        catalog_path = root / 'library_catalog.json'
        catalog_sha_before = sha(catalog_path)
        catalog = read(catalog_path)
        before = original_fingerprint(catalog['records'])
        by_id = {r['id']: r for r in catalog['records']}
        selections = agent_selections(root)
        report = dict(status='dry_run' if dry_run else 'merged', batches=[], merged_results=0,
                      merged_catalog_records=0, skipped=[], original_source_fields_sha256=before)
        folders = [root / 'runtime' / 'batches' / b for b in batches] if batches else sorted((root / 'runtime' / 'batches').glob('*'))
        for folder in folders:
            receipt_path = folder / 'BATCH_RECEIPT.json'
            if not receipt_path.is_file() or read(receipt_path).get('status') != 'review_complete':
                report['skipped'].append(dict(batch=folder.name, reason='batch_not_review_complete'))
                continue
            receipt = read(receipt_path)
            mapping = read(folder / 'HOST_MAPPING.json')
            registry = read(folder / 'image_registry.json')
            report['batches'].append(folder.name)
            for result_name in receipt.get('result_files', []):
                result_path = source_path(root, result_name)
                result = read(result_path)
                if result.get('evidence', {}).get('vision_verified') is not True:
                    report['skipped'].append(dict(result=str(result_path), reason='not_vision_verified'))
                    continue
                anonymous = result.get('anonymous_id')
                host = mapping.get(anonymous)
                if host is None:
                    raise ValueError('Result has no HOST_MAPPING: ' + str(anonymous))
                ids = set(host.get('catalog_ids', [])) | {host.get('catalog_id'), host.get('candidate_id'), result['id']}
                ids.discard(None)
                absent = ids - set(by_id)
                if absent:
                    raise ValueError('Unmapped catalog aliases: ' + str(sorted(absent)))
                result_sha = sha(result_path)
                selection = selections.get((folder.name, anonymous, result_sha))
                if selection:
                    if (selection['candidate_id'] != result['id']
                            or selection['class_id'] != result['proposed_label']['class_id']
                            or not applicability(result)['image_appearance_supported']):
                        raise ValueError('Agent selection does not match a supported frozen result')
                evidence = result['evidence']
                # Check actual evidence against the recorded image-delivery hashes.
                for image_id in evidence.get('actual_image_ids', []):
                    item = registry[image_id]
                    if sha(source_path(root, item['path'])) != evidence['image_sha256'][image_id]:
                        raise ValueError('Delivered evidence changed: ' + image_id)
                changed = 0
                for identity in sorted(ids):
                    record = by_id[identity]
                    if (record.get('review_provenance', {}).get('result_sha256') == result_sha
                            and record.get('reference_applicability') is not None
                            and record.get('agent_curator_selection') == selection):
                        continue
                    proposed = copy.deepcopy(result['proposed_label'])
                    proposed.update(class_name=CLASS_NAMES.get(proposed['class_id']),
                                    status='model_proposed_pending_human',
                                    label_conflict=class_conflict(record.get('original_label', {}), proposed['class_id']),
                                    compared_alternative=result.get('alternative_considered'))
                    record['proposed_label'] = proposed
                    record['proposed_geometry'] = copy.deepcopy(result.get('geometry'))
                    for key in ('seedpoint_class', 'seed_homogeneous', 'reference_role', 'quality', 'suitability', 'ten_metre_boundary_review'):
                        record[key] = copy.deepcopy(result.get(key))
                    record['label_conflict'] = proposed['label_conflict']
                    record['label_comparison_host_only'] = dict(original_label=copy.deepcopy(record.get('original_label')),
                        seedpoint_conflict=class_conflict(record.get('original_label', {}), result['seedpoint_class']),
                        original_geometry_modified=False, original_truth_overridden=False,
                        comparison_scope='Image appearance versus original source class; not verification that the historical label was wrong',
                        historical_label_correction_status='not_established',
                        runtime_comparison=copy.deepcopy(result.get('label_comparison_host_only')))
                    record['evidence'] = copy.deepcopy(evidence)
                    record['evidence'].update(reason=result.get('reason'), compared_alternatives=result.get('alternative_considered'),
                        date_consistency=result.get('quality'), scope='Model visual proposal; source labels and original geometry remain separate')
                    record['reference_applicability'] = applicability(result)
                    record['model_original_recommendation'] = result.get('suitability')
                    if selection:
                        record['agent_curator_selection'] = copy.deepcopy(selection)
                        record['selection_origin'] = 'agent_visual_review'
                        record['reference_applicability'].update(appearance_reference_eligible=True,
                            eligibility_origin='agent_visual_review',date_verification_required=False,
                            training_admitted=False)
                    else:
                        record.pop('agent_curator_selection', None)
                        record['selection_origin'] = 'original_model_recommendation'
                    selected = record['reference_applicability']['appearance_reference_eligible']
                    record['curation_status'] = 'selected_proposed_pending_human' if selected else 'reviewed_' + str(result.get('suitability')) + '_pending_human'
                    if result.get('suitability') == 'keep' and not selected:
                        record['curation_status'] = 'requires_support_reconciliation_pending_human'
                    record['review_provenance'] = dict(batch_id=folder.name, anonymous_id=anonymous,
                        result_path=str(result_path), result_sha256=result_sha, batch_receipt=str(receipt_path),
                        batch_status='review_complete', human_approved=False, source_annotations_unchanged=True)
                    if not dry_run:
                        for image_id in evidence.get('actual_image_ids', []):
                            item = registry[image_id]
                            role = 'original_annotation' if item.get('role') == 'candidate_raw_seed' else ('sam_overlay' if 'candidates' in image_id or 'selected' in image_id else 'rgb')
                            attach(root, record, item['path'], role, folder.name, image_id=image_id,
                                   annotation_scope='Seed position only; no source class shown' if role == 'original_annotation' else None,
                                   capture_date=evidence.get('capture_date'), label_year=evidence.get('label_year'))
                    sam = evidence.get('SAM_audit')
                    if sam:
                        sam_record = copy.deepcopy(sam)
                        sam_record.update(polygon=copy.deepcopy(result.get('geometry')), status='model_proposed_pending_human',
                                          alignment_note='UTM10m review only; rasterize on actual AEF affine before training')
                        if not dry_run:
                            sam_record['raw_mask_path'] = attach(root, record, sam['raw_mask_path'], 'sam_mask', folder.name,
                                                               mask_scope='Actual predictor raw mask, not semantic probability')
                            sam_record['selected_mask_path'] = attach(root, record, sam['selected_mask_path'], 'sam_mask', folder.name,
                                                                    mask_scope='Selected region after padding removal')
                            preview = source_path(root, sam['selected_mask_path']).parent / 'selected_region_review.png'
                            if preview.exists():
                                sam_record['preview_path'] = attach(root, record, preview, 'sam_overlay', folder.name)
                        record['sam'] = sam_record
                    else:
                        record.pop('sam', None)
                    changed += 1
                if changed:
                    report['merged_results'] += 1
                    report['merged_catalog_records'] += changed
        after = original_fingerprint(catalog['records'])
        if after != before:
            raise AssertionError('Original annotation fields changed')
        records = catalog['records']
        verified = [r for r in records if r.get('evidence', {}).get('vision_verified') is True and r.get('review_provenance', {}).get('batch_status') == 'review_complete']
        summary = catalog.setdefault('summary', {})
        summary.update(new_visual_reviews_completed=len(verified),
            new_X_expansions_completed=sum(bool(r.get('proposed_geometry')) and bool(r.get('sam')) for r in verified),
            selected_model_proposals=sum(r.get('curation_status') == 'selected_proposed_pending_human' for r in verified),
            unique_visual_review_results=len({r['review_provenance']['result_sha256'] for r in verified}),
            selected_agent_curator_proposals=sum(r.get('curation_status') == 'selected_proposed_pending_human'
                and r.get('selection_origin') == 'agent_visual_review' for r in verified),
            images_available_records=sum(bool(r.get('images')) for r in records))
        report['summary'] = copy.deepcopy(summary)
        report['original_source_fields_unchanged'] = before == after
        report['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if not dry_run:
            # Refuse lost updates from another writer rather than overwriting them.
            if sha(catalog_path) != catalog_sha_before:
                raise RuntimeError('Catalog changed while merging; rerun merge')
            catalog['updated_utc'] = report['completed_utc']
            atomic_write(catalog_path, catalog)
            atomic_write(root / 'MERGE_REVIEW_RECEIPT.json', report)
        return report


def curated_references(root=ROOT, class_id=None, lonlat=None, radius_km=None, limit=None, *, catalog_records=None):
    """Retrieve selected verified proposals, with independent human-overlay semantics.

    Selected, root-reviewed original human anchors are a separate eligible path;
    they do not acquire an app-server visual-audit flag or new SAM geometry.
    Unselected, rejected, flagged and unsupported model proposals are absent.
    Returned ``approval_status`` always distinguishes model proposals from human
    approval. No point correctness, validation score or fitted threshold is used.
    """
    root = Path(root)
    rows = read(root / 'library_catalog.json')['records'] if catalog_records is None else catalog_records
    blocked_results = set()
    blocked_anchor_locations = set()
    reviews = {}
    for row in rows:
        review_file = root / 'human_reviews' / (hashlib.sha256(row['id'].encode()).hexdigest() + '.json')
        review = read(review_file) if review_file.exists() else {}
        reviews[row['id']] = review
        if review.get('action') in ('reject', 'flag'):
            blocked_results.add(row.get('review_provenance', {}).get('result_sha256', row['id']))
            provider_specific = row.get('record_type') in ('google_earth_reference', 'esri_reference')
            if not provider_specific and row.get('longitude') is not None and row.get('latitude') is not None:
                blocked_anchor_locations.add(tuple(round(float(row[k]),8) for k in ('longitude','latitude')))
    output = []
    for row in rows:
        provenance = row.get('review_provenance', {})
        anchor = row.get('source_anchor_selection') or {}
        source_anchor = (anchor.get('selected_for_appearance') is True
                         and human_anchor_label_supported(row,anchor)
                         and bool(anchor.get('source_label_paths')) and bool(anchor.get('source_label_sha256'))
                         and bool(anchor.get('source_rgb_sha256')) and bool(anchor.get('root_review_pages')))
        visual_result = (row.get('evidence', {}).get('vision_verified') is True
                         and provenance.get('batch_status') == 'review_complete')
        if not source_anchor and not visual_result:
            continue
        location = tuple(round(float(row[k]),8) for k in ('longitude','latitude'))
        if (provenance.get('result_sha256', row['id']) in blocked_results
                or source_anchor and location in blocked_anchor_locations):
            continue
        review = reviews[row['id']]
        approved = review.get('action') in ('approve', 'change_class')
        agent_selected = (row.get('selection_origin') == 'agent_visual_review'
                          and row.get('agent_curator_selection', {}).get('selected_for_appearance') is True
                          and row['agent_curator_selection'].get('result_sha256') == provenance.get('result_sha256'))
        if not approved and not source_anchor and ((row.get('suitability') != 'keep' and not agent_selected)
                or row.get('curation_status') != 'selected_proposed_pending_human'
                or row.get('reference_applicability', {}).get('appearance_reference_eligible') is not True):
            continue
        effective_class = review.get('class_id') if approved else (anchor['class_id'] if source_anchor else row.get('proposed_label', {}).get('class_id'))
        if effective_class not in (0, 1, 2, 3) or class_id is not None and effective_class != class_id:
            continue
        result = copy.deepcopy(row)
        result.update(effective_class_id=effective_class, human_review=copy.deepcopy(review),
                      approval_status='human_approved' if approved else ('source_human_labelled_agent_reviewed_pending_human' if source_anchor else 'model_proposed_pending_human'),
                      reference_entry_type='human_labelled_source_anchor' if source_anchor else 'model_visual_proposal',
                      is_gold=False,
                      reference_use_scope=anchor['label_scope'] if source_anchor else 'Image appearance; historical label correction is not established')
        if lonlat is not None:
            lon, lat = lonlat
            ref_lon, ref_lat = anchor.get('anchor_coordinates', [row['longitude'],row['latitude']]) if source_anchor else [row['longitude'],row['latitude']]
            a, b = math.radians(lat), math.radians(ref_lat)
            h = math.sin((b-a)/2)**2 + math.cos(a)*math.cos(b)*math.sin(math.radians(ref_lon-lon)/2)**2
            result['distance_km'] = 6371 * 2 * math.asin(min(1, math.sqrt(h)))
            if radius_km is not None and result['distance_km'] > radius_km:
                continue
        output.append(result)
    output.sort(key=lambda r: (r['approval_status'] != 'human_approved',
        {'primary':0,'variation':1}.get(r.get('source_anchor_selection',{}).get('priority'),2),
        r.get('distance_km', 0), r['id']))
    # Catalog aliases can point to one reviewed region. Keep the preferred human
    # overlay first, while distinct result regions remain distinct even when
    # they use the same RGB. Different effective class decisions stay separate.
    unique = {}
    anchor_locations = {(tuple(round(float(r[k]),8) for k in ('longitude','latitude')),r['effective_class_id'])
                        for r in output if r['reference_entry_type']=='human_labelled_source_anchor'}
    for result in output:
        result_sha = result.get('review_provenance', {}).get('result_sha256')
        location = tuple(round(float(result[k]),8) for k in ('longitude','latitude'))
        key = ('source_location',location,result['effective_class_id']) if (location,result['effective_class_id']) in anchor_locations else (result_sha or result['id'], result['effective_class_id'])
        if key not in unique:
            result['catalog_alias_ids'] = list(dict.fromkeys([result['id']] + result.get('source_anchor_selection',{}).get('catalog_alias_ids',[])))
            unique[key] = result
        else:
            unique[key]['catalog_alias_ids'] = list(dict.fromkeys(unique[key]['catalog_alias_ids'] + [result['id']] + result.get('source_anchor_selection',{}).get('catalog_alias_ids',[])))
    deduplicated = list(unique.values())
    return deduplicated if limit is None else deduplicated[:limit]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--batch', '--batch-id', dest='batch', action='append', help='Completed batch ID; repeatable')
    parser.add_argument('--apply', action='store_true', help='Apply merge (default; accepted for queue callers)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(merge(args.root, args.batch, args.dry_run), ensure_ascii=False, indent=2))
