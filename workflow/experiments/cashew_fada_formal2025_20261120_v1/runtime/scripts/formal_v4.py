"""Explicit production template; historical/ablation jobs keep their own identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "references/formal_training_v4.json"


def load_template(path: Path | None = None) -> dict:
    return json.loads((path or TEMPLATE_PATH).read_text())


def template_identity(path: Path | None = None) -> dict:
    path = path or TEMPLATE_PATH
    return {"profile": load_template(path)["profile"], "template": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def max_epoch_for_job(job: dict) -> int:
    """Resolve the fixed formal budget or an explicitly authorized extension."""
    if job.get("bounded_recovery"):
        if job.get("research_epoch_extension") or job.get("research_epoch_budget"):
            raise ValueError("Do not combine recovery and research budget extensions")
        from recovery_attempt_v4 import ceiling_for_job
        return ceiling_for_job(job)
    extension = job.get("research_epoch_extension")
    if extension:
        if (extension.get("protocol") != "user_requested_019e_epoch100_to_200"
                or int(extension.get("source_epoch", -1)) != 100
                or int(extension.get("maximum_epoch", -1)) != 200):
            raise ValueError("unrecognized research epoch extension")
        return int(extension["maximum_epoch"])

    budget = job.get("research_epoch_budget")
    if budget:
        source = job.get("source", {})
        if (budget.get("protocol") != "user_requested_019e_source2025_from_scratch_to_200"
                or job.get("job_id") != "Ivoire2025_0000000000000000019e"
                or job.get("target_year") != 2025
                or source.get("source_year") != 2025
                or job.get("source_profile") != "source2025_benin_e173"
                or job.get("research_authorization") != "user_requested_019e_sourceyear_comparison_20260924"
                or job.get("resume")
                or int(budget.get("maximum_epoch", -1)) != 200
                or int(job.get("max_epoch", -1)) != 200):
            raise ValueError("unrecognized source-2025 fresh-run epoch budget")
        return int(budget["maximum_epoch"])

    return int(job.get("max_epoch", 200))


def validate_formal_job(job: dict) -> None:
    """Reject experimental methods before loading any raster or allocating CUDA."""
    identity = job.get("formal_training", {})
    pinned = Path(identity.get("template", "")) if identity.get("template") else TEMPLATE_PATH
    # A prepared job pins a byte-for-byte template snapshot in its own root.
    # Never reinterpret an interrupted historical job using today's template.
    if not pinned.is_file():
        raise ValueError("formal template snapshot missing")
    template = load_template(pinned)
    errors = []
    if job.get("formal_training") != template_identity(pinned):
        errors.append("formal template identity/hash mismatch")
    for key, value in template["job_defaults"].items():
        if job.get(key) != value:
            errors.append(f"job.{key} must equal {value!r}")
    if job.get("source_profile") != template["source_profile"] or job.get("source", {}).get("source_year") != template["source_year"]:
        errors.append("formal source must be registered 2019 Benin epoch173")
    if job.get("architecture") != template["architecture"]:
        errors.append(f"architecture must equal formal template {template['architecture']!r}")
    if template['job_defaults'].get('geographic_prior_policy'):
        try:
            from geographic_prior_v1 import packet_evidence
            context = packet_evidence(job)
            if context is None or context['version'] != job['geographic_prior_policy']['version']:
                errors.append('new geographic prior policy requires a pinned context report')
            elif context['version'].startswith('cashew_spatial_prior_v2_'):
                from spatial_prior_v2 import COUNTRIES
                selection_report = json.loads(Path(job['supplementary']['selection']['report']).read_text())
                if (context['target_year'] != job['target_year']
                        or context['bounds_wgs84'] != selection_report['target_bounds']
                        or COUNTRIES.get(job['country'],job['country']) not in context['countries']):
                    errors.append('spatial prior context does not match this tile/year/country')
        except (OSError, KeyError, TypeError, ValueError):
            errors.append('geographic text prior report missing or inconsistent')
    evaluation_policy = template.get("adm1_evaluation_policy")
    if (job.get("target_domain") or {}).get("kind") == "ADM1" and evaluation_policy:
        if not job.get("target_evaluation"):
            errors.append("new ADM1 jobs require the pinned fixed evaluation sample")
    if job.get("target_evaluation"):
        try:
            from evaluation_sample_v1 import names_for_job
            names_for_job(job)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append("invalid fixed target evaluation sample: " + str(exc))
    if job.get("prior"):
        errors.append("historical Planet prior is not part of formal v4")
    if job.get("experimental_fixed_schedule"):
        errors.append("fixed-schedule ablations cannot enter the formal template")
    extension = job.get("research_epoch_extension")
    if extension:
        resume = job.get("resume", {})
        expected_root = str(Path(extension.get("source_run_root", "")).resolve())
        expected_checkpoint = str((Path(expected_root) / "checkpoints/epoch_100.pt").resolve())
        if job.get("job_id") != "Ivoire2025_0000000000000000019e":
            errors.append("this authorized extension is restricted to Ivoire 019e")
        if job.get("max_epoch") != template["job_defaults"]["max_epoch"]:
            errors.append("preserve the pinned formal max_epoch; use extension.maximum_epoch")
        if int(extension.get("source_epoch", -1)) != 100 or int(extension.get("maximum_epoch", -1)) != 200:
            errors.append("authorized extension must continue epoch 100 through max epoch 200")
        if (int(resume.get("epoch", -1)) != 100
                or str(Path(resume.get("source_run_root", "")).resolve()) != expected_root
                or str(Path(resume.get("checkpoint", "")).resolve()) != expected_checkpoint):
            errors.append("extension must resume the declared source run's epoch-100 checkpoint")
        source_job_path = Path(expected_root) / "job.json"
        source_checkpoint = Path(expected_checkpoint)
        if not source_job_path.is_file() or not source_checkpoint.is_file():
            errors.append("extension source job/checkpoint is missing")
        else:
            source_job = json.loads(source_job_path.read_text())
            if (source_job.get("job_id") != job.get("job_id")
                    or source_job.get("formal_training", {}).get("profile") != job.get("formal_training", {}).get("profile")
                    or source_job.get("formal_training", {}).get("sha256") != job.get("formal_training", {}).get("sha256")
                    or source_job.get("max_epoch") != 100
                    or source_job.get("target_year") != 2025
                    or source_job.get("country") != "Ivoire"):
                errors.append("extension source run does not match the pinned 2025 Ivoire 019e formal job")
            if resume.get("checkpoint_sha256") != hashlib.sha256(source_checkpoint.read_bytes()).hexdigest():
                errors.append("extension source checkpoint hash mismatch")
    if template.get("negative_control_policy") != job.get("negative_control_policy"):
        errors.append("negative controls differ from the pinned formal template")
    conditional = template.get("conditional_negative_policy")
    if conditional != job.get("conditional_negative_policy"):
        errors.append("conditional negative policy differs from pinned template")
    if conditional and not job.get("bounded_recovery") and job.get("recovery_policy") != template.get("recovery_policy"):
        errors.append("new jobs require the pinned bounded collapse-recovery policy")
    supplementary = job.get("supplementary", {})
    human_polygons = template.get('polygon_supervision_policy',{}).get('version')=='human_polygon_group_formal_v1'
    count = supplementary.get("count")
    empty_policy = template.get("supplementary_empty_policy", {})
    allow_empty = empty_policy.get("version") == "allow_empty_v1"
    empty = allow_empty and type(count) is int and count == 0
    if allow_empty and (type(count) is not int or count < 0):
        errors.append("supplementary count must be a nonnegative integer")
    for key, value in template["supplementary"].items():
        if empty and key in {"enabled", "loss_weight"}:
            value = empty_policy[key]
        if supplementary.get(key) != value:
            errors.append(f"supplementary.{key} differs from formal template")
    selection = supplementary.get("selection", {})
    if selection.get("method") != template["supplementary_selection"]:
        errors.append("supplementary list must be selected by per-tile feature similarity")
    geographic = bool(template.get("supplementary_geographic_filter", {}).get("enabled"))
    if selection.get("fixed_top_k_used") is not False or selection.get("geographic_distance_used_for_filtering") is not geographic:
        errors.append("supplementary fixed-count/geographic filtering differs from pinned template")
    if geographic and not human_polygons:
        try:
            import math
            report_path = Path(selection["report"])
            report = json.loads(report_path.read_text())
            names_path = Path(supplementary["names_file"])
            if selection.get("report_sha256") != hashlib.sha256(report_path.read_bytes()).hexdigest():
                errors.append("selection report hash mismatch")
            if selection.get("names_file_sha256") != hashlib.sha256(names_path.read_bytes()).hexdigest():
                errors.append("selection list hash mismatch")
            geo = report["geographic_filter"]
            if (geo.get("enabled") is not True or geo.get("radius_km") != template["supplementary_geographic_filter"]["radius_km"]
                    or selection.get("geographic_filter") != geo
                    or report.get("geographic_distance_used_for_filtering") is not True
                    or report.get("fixed_top_k_used") is not False
                    or report.get("selection_rule") != template["supplementary_selection"]
                    or report.get("selected_count") != count):
                errors.append("geographic selection report differs from formal configuration")
            selected = report["selected"]
            names = [x.strip() for x in names_path.read_text().splitlines() if x.strip()]
            if len(selected) != count or names != [x["name"] for x in selected] or len(set(names)) != len(names):
                errors.append("selection names/order/count are inconsistent")
            for item in selected:
                distance = float(item["geo_distance_km"])
                if not math.isfinite(distance) or not 0 <= distance <= geo["radius_km"]:
                    errors.append("selected candidate outside geographic radius")
                score = float(item["feature_distance"])
                if not math.isfinite(score) or not 0 <= score <= float(report["feature_distance_threshold"]):
                    errors.append("selected candidate outside feature threshold")
            coverage = report["source_coverage"]
            if (coverage.get("version") != "heldout_source_feature_coverage_v2"
                    or coverage.get("novelty_assigns_negative_label") is not False
                    or coverage.get("reference_count", 0) <= 0 or coverage.get("calibration_count", 0) <= 0):
                errors.append("source coverage diagnostic missing or invalid")
        except (OSError, ValueError, KeyError, TypeError):
            errors.append("geographic/source coverage evidence missing or invalid")
    if human_polygons:
        try:
            from human_polygon_v1 import validate
            report_path=Path(selection['report']);names_path=Path(supplementary['names_file'])
            if hashlib.sha256(report_path.read_bytes()).hexdigest()!=selection['report_sha256']:raise ValueError('Human selection report changed')
            if hashlib.sha256(names_path.read_bytes()).hexdigest()!=selection['names_file_sha256']:raise ValueError('Human selection list changed')
            report=json.loads(report_path.read_text());names=names_path.read_text().splitlines()
            validate(job,template,report,names)
        except (AssertionError,OSError,ValueError,KeyError,TypeError) as exc:
            errors.append('Invalid human polygon selection: '+str(exc))
    if supplementary.get("count") != selection.get("selected_count"):
        errors.append("selected supplementary count mismatch")
    if not supplementary.get("names_file") or not selection.get("report"):
        errors.append("supplementary list/report provenance missing")
    if empty:
        if supplementary.get("disabled_reason") != empty_policy["disabled_reason"]:
            errors.append("empty supplementary branch requires its recorded disabled reason")
        names = Path(supplementary.get("names_file") or "/missing-empty-list")
        report_path = Path(selection.get("report") or "/missing-empty-report")
        if not names.is_file() or any(line.strip() for line in names.read_text().splitlines()):
            errors.append("empty supplementary branch requires an existing empty selection list")
        elif selection.get("names_file_sha256") != hashlib.sha256(names.read_bytes()).hexdigest():
            errors.append("empty supplementary selection list hash mismatch")
        try:
            report = json.loads(report_path.read_text())
            if selection.get("report_sha256") != hashlib.sha256(report_path.read_bytes()).hexdigest():
                errors.append("empty supplementary selection report hash mismatch")
            if (report.get("selected_count") != 0 or report.get("selected") != []
                    or report.get("selection_rule") != template["supplementary_selection"]
                    or report.get("fixed_top_k_used") is not False
                    or report.get("geographic_distance_used_for_filtering") is not geographic):
                errors.append("empty supplementary selection report is inconsistent")
        except (OSError, ValueError, AttributeError):
            errors.append("empty supplementary selection report missing or invalid")
    elif allow_empty and supplementary.get("disabled_reason"):
        errors.append("nonempty supplementary pool cannot use an empty-branch disabled reason")
    if errors:
        raise ValueError({"formal_v4_contract_errors": errors})
