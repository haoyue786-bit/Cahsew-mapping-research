#!/usr/bin/env python3
"""Screen fresh imagery, annotate the chosen view and write native AEF labels."""

import argparse
from collections import Counter
import re
from pathlib import Path

from _runtime import transport
from _runtime.common import read, write
from annotate import resume_annotation, run_annotation
from imagery.grid import from_aef
from imagery_policy import DATE_FLOOR, distinct_capture_periods
from imagery_preflight import prepare_inventory


def annotation_usage(root):
    turns = 0
    totals = Counter()
    for attempt in sorted((root / 'annotation').glob('[0-9]*')):
        usage = {}
        for folder in (attempt, attempt / 'source_evidence_redelivery', attempt / 'keep_review'):
            status = folder / 'status.json'
            tokens = folder / 'token_usage.json'
            if status.is_file():
                turns += read(status).get('model_calls', 0)
            if tokens.is_file():
                usage = read(tokens)['tokenUsage']['total']
        totals.update(usage)
        separate = attempt / 'woody_review'
        status = separate / 'status.json'
        if status.is_file():
            turns += read(status).get('model_calls', 0)
        tokens = separate / 'token_usage.json'
        if tokens.is_file():
            totals.update(read(tokens)['tokenUsage']['total'])
    return dict(hosted_annotation_turns=turns, token_usage=dict(totals))


def annotate_inventory(root, inventory, *, prefer_dry_season=False, region_matching=False, woody_review=False):
    search_path = root / "imagery" / "SEARCH.json"
    paths = [Path(path) for path in inventory['ranked_candidates']]
    selections = {path: read(path) for path in paths}
    excluded = set()
    pending = list(paths)
    deferred = None
    completed_fallback = None
    auxiliary_checks = 0
    attempts = 0

    while pending:
        dry = [path for path in pending if selections[path]['season'] == 'dry']
        if deferred and not dry:
            break
        if dry and (prefer_dry_season or deferred):
            selected = dry[0]
            pending.remove(selected)
        else:
            selected = pending.pop(0)
        frame = selections[selected]['frame']
        comparisons = [path for path in paths if path != selected and path not in excluded
                       and distinct_capture_periods(frame, selections[path]['frame'])]
        if selections[selected]['season'] != 'dry':
            comparisons.sort(key=lambda path: selections[path]['season'] != 'dry')
        comparison = selections[comparisons[0]]['frame'] if comparisons else None
        available_dry = any(path in pending and selections[path]['season'] == 'dry' for path in comparisons)
        attempts += 1
        attempt = root / 'annotation' / f'{attempts:03d}'
        receipt = run_annotation(selected, attempt, previous_auxiliary_checks=auxiliary_checks,
                                 comparison_frame=comparison, dry_search_exhausted=not available_dry,
                                 defer_for_dry=available_dry, region_matching=region_matching, woody_review=woody_review)
        auxiliary_checks = receipt['auxiliary_checks_used']
        inventory['trials'].append(dict(folder=frame['folder'], provider=frame['provider'], dates=frame['acquisition_dates'],
                                        status=receipt['status'], reason=receipt.get('reason'), attempt=str(attempt)))
        write(search_path, inventory)
        if receipt['status'] == 'completed':
            if not (receipt.get('dry_comparison_requested') and available_dry):
                return dict(receipt, imagery_timing=inventory['timing'], annotation_attempts=attempts)
            completed_fallback = receipt
            deferred = attempt
            continue
        if receipt['status'] == 'needs_dry_image':
            deferred = attempt
        else:
            excluded.add(selected)
    if deferred:
        fallback = completed_fallback or resume_annotation(deferred, allow_deferred=True)
        return dict(fallback, imagery_timing=inventory['timing'], annotation_attempts=attempts,
                    auxiliary_checks_used=auxiliary_checks, dry_fallback=True)
    return dict(status="skipped", reason="no_usable_date_qualified_image", reason_text="No usable imagery",
                acquisition_window=inventory['acquisition_window'], annotations_attempted=attempts,
                auxiliary_checks_used=auxiliary_checks, imagery_timing=inventory['timing'],
                search=str(search_path))


def run_sample(aef, case, country, period, run_dir, *, scene_reviews=(), prefer_dry_season=False,
               region_matching=False, woody_review=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", case):
        raise ValueError("Use a simple unique case ID")
    if not re.fullmatch(r"[A-Z]{3}", country):
        raise ValueError("Use a three-letter country code such as CIV, TGO or GHA")
    grid = from_aef(aef, case, country, period)
    transport.ROOT = run_dir.resolve()
    root = transport.ROOT / case
    root.mkdir(parents=True, exist_ok=False)
    write(root / "GRID.json", grid)
    receipt_path = root / "RECEIPT.json"
    inventory = prepare_inventory(grid, root / "imagery", scene_reviews)
    receipt = annotate_inventory(root, inventory, prefer_dry_season=prefer_dry_season,
                                 region_matching=region_matching, woody_review=woody_review)
    write(receipt_path, dict(receipt, grid=grid, annotation_usage=annotation_usage(root)))
    return receipt_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aef", type=Path, help="The original AEF 128×128 TIF")
    parser.add_argument("--case")
    parser.add_argument("--country", help="ISO 3166-1 alpha-3")
    parser.add_argument("--period", type=int, choices=sorted(DATE_FLOOR),
                        help="Formal label period, independent of the AEF raster year")
    parser.add_argument("--run-dir", type=Path, help="New run directory or a new case within it")
    parser.add_argument("--resume-local", type=Path, help="Resume the local work in a numbered annotation attempt")
    parser.add_argument("--scene-reviews", type=Path, help="Optional JSON list of operator-reviewed same-image points or inspection windows")
    parser.add_argument("--prefer-dry-season", action="store_true", help="Look for local dry-season imagery first on a tile suspected of Cashew")
    parser.add_argument("--region-matching", action="store_true", help="Try detector boxes, batched SAM2, DINOv2 and a group keep review")
    parser.add_argument("--woody-review", action="store_true", help="Independently review woody land units before applying SAM2")
    args = parser.parse_args()
    if args.resume_local:
        attempt = args.resume_local.resolve()
        receipt = resume_annotation(attempt)
        root = attempt.parents[1]
        inventory = read(root / "imagery" / "SEARCH.json")
        target = root / "RECEIPT.json"
        write(target, dict(receipt, imagery_timing=inventory['timing'], annotation_attempts=int(attempt.name),
                           annotation_usage=annotation_usage(root)))
        print(target)
        return
    if not all((args.aef, args.case, args.country, args.period, args.run_dir)):
        parser.error("Provide --aef, --case, --country, --period and --run-dir for a fresh sample")
    if args.region_matching and args.woody_review:
        parser.error("Choose one experimental review mode")
    print(run_sample(args.aef, args.case, args.country, args.period, args.run_dir,
                     scene_reviews=read(args.scene_reviews) if args.scene_reviews else (),
                     prefer_dry_season=args.prefer_dry_season, region_matching=args.region_matching,
                     woody_review=args.woody_review))


if __name__ == "__main__":
    main()
