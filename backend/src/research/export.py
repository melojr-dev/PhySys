from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from typing import Any, Iterable


def _csv_bytes(rows: list[dict[str, Any]], columns: list[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        normalized = dict(row)
        for key, value in list(normalized.items()):
            if isinstance(value, (dict, list)):
                normalized[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        writer.writerow(normalized)
    return output.getvalue().encode("utf-8-sig")


def _checksum_lines(files: Iterable[tuple[str, bytes]]) -> str:
    return "".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}\n"
        for name, content in files
    )


def build_session_bundle(
    *,
    manifest: dict[str, Any],
    measurements: list[dict[str, Any]],
    validation_reviews: list[dict[str, Any]],
    validation_trials: list[dict[str, Any]] | None = None,
    imu_analyses: list[dict[str, Any]] | None = None,
    imu_payloads: dict[str, dict[str, Any] | None] | None = None,
    motor_feature_analyses: list[dict[str, Any]] | None = None,
    longitudinal_analyses: list[dict[str, Any]] | None = None,
) -> bytes:
    manifest_bytes = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")

    measurement_columns = [
        "job_id", "frame", "timestamp_ms", "joint", "movement", "side",
        "angle_raw_deg", "angle_landmark_deg", "angle_plane_deg", "angle_filtered_deg", "angle_clinical_deg",
        "angular_velocity_deg_s", "adm_deg", "source", "measurement_space",
        "measurement_plane", "plane_projection_applied", "body_frame_confidence",
        "view_alignment", "view_deviation_deg", "out_of_plane_deg",
        "direction_rule", "direction_alignment",
        "landmark_confidence_min", "landmark_confidence_mean", "quality",
        "measurement_valid", "quality_flags", "proximal", "vertex", "distal",
        "proximal_3d", "vertex_3d", "distal_3d",
        "inference_ms", "pipeline_ms", "processing_fps",
    ]
    validation_columns = [
        "id", "schema_version", "created_at", "job_id", "patient_id",
        "evaluator_id", "movement", "side", "frame", "timestamp_ms",
        "automatic_angle_deg", "manual_angle_deg", "reference_angle_deg",
        "manual_method", "reference_method", "error_vs_manual_deg",
        "error_vs_reference_deg", "confidence", "source",
        "proximal", "vertex", "distal", "notes",
    ]
    measurements_bytes = _csv_bytes(measurements, measurement_columns)
    validation_bytes = _csv_bytes(validation_reviews, validation_columns)
    validation_trials = validation_trials or []
    trial_columns = [
        "id", "study_id", "created_at", "job_id", "patient_id", "evaluator_id",
        "reference_assessor_id", "movement", "side", "trial_number", "frame",
        "timestamp_ms", "automatic_angle_deg", "reference_angle_deg", "error_deg",
        "absolute_error_deg", "reference_method", "reference_device_model",
        "reference_device_serial", "landmark_confidence_min", "body_frame_confidence",
        "measurement_plane", "view_alignment", "view_deviation_deg", "out_of_plane_deg",
        "direction_rule", "direction_alignment",
        "source", "measurement_valid", "quality", "included", "exclusion_reason", "notes",
    ]
    trial_bytes = _csv_bytes(validation_trials, trial_columns)

    imu_analyses = imu_analyses or []
    imu_payloads = imu_payloads or {}
    imu_index_bytes = json.dumps(imu_analyses, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    motor_feature_analyses = motor_feature_analyses or []
    motor_features_bytes = json.dumps(
        motor_feature_analyses, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    longitudinal_analyses = longitudinal_analyses or []
    longitudinal_bytes = json.dumps(
        longitudinal_analyses, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")

    readme_bytes = (
        "PhysioAI research session bundle\n\n"
        "manifest.json: software/protocol/input provenance and session summary.\n"
        "measurements.csv: one biomechanical measurement per processed frame.\n"
        "validation_reviews.csv: blinded manual image reviews.\n"
        "validation_trials.csv: physical-reference trials linked to frozen study protocols.\n"
        "imu/index.json: summaries of experimental inertial analyses linked to this job.\n"
        "imu/<id>.json: raw inertial payload used to reproduce each linked analysis.\n"
        "motor_features/index.json: experimental non-diagnostic multimodal motor descriptors.\n"
        "longitudinal/index.json: descriptive same-patient longitudinal analyses anchored to comparable sessions.\n"
        "checksums.sha256: SHA-256 of the files above.\n\n"
        "Angles use the clinical convention recorded in manifest.json. "
        "Landmark visibility is a tracking-quality signal, not clinical accuracy.\n"
    ).encode("utf-8")

    files = [
        ("manifest.json", manifest_bytes),
        ("measurements.csv", measurements_bytes),
        ("validation_reviews.csv", validation_bytes),
        ("validation_trials.csv", trial_bytes),
        ("imu/index.json", imu_index_bytes),
        ("motor_features/index.json", motor_features_bytes),
        ("longitudinal/index.json", longitudinal_bytes),
        ("README.txt", readme_bytes),
    ]
    for item in imu_analyses:
        analysis_id = str(item.get("id") or "").strip()
        if not analysis_id:
            continue
        raw = imu_payloads.get(analysis_id)
        content = json.dumps(
            {"record": item, "raw": raw}, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8")
        files.append((f"imu/{analysis_id}.json", content))
    checksum_bytes = _checksum_lines(files).encode("ascii")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files:
            zf.writestr(name, content)
        zf.writestr("checksums.sha256", checksum_bytes)
    return buffer.getvalue()


def build_study_bundle(*, study: dict[str, Any], trials: list[dict[str, Any]], report: dict[str, Any]) -> bytes:
    """Self-contained export for a frozen/closed validation study."""
    manifest = {
        "bundle_schema_version": 1,
        "kind": "validation_study",
        "study": study,
        "trial_count": len(trials),
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    report_bytes = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    trial_columns = [
        "id", "study_id", "created_at", "job_id", "patient_id", "evaluator_id",
        "reference_assessor_id", "movement", "side", "trial_number", "frame",
        "timestamp_ms", "automatic_angle_deg", "reference_angle_deg", "error_deg",
        "absolute_error_deg", "reference_method", "reference_device_model",
        "reference_device_serial", "landmark_confidence_min", "body_frame_confidence",
        "measurement_plane", "view_alignment", "view_deviation_deg", "out_of_plane_deg",
        "direction_rule", "direction_alignment",
        "source", "measurement_valid", "quality", "included", "exclusion_reason", "notes",
    ]
    trials_bytes = _csv_bytes(trials, trial_columns)
    readme = (
        "PhysioAI validation study bundle\n\n"
        "protocol.json: frozen study protocol and SHA-256.\n"
        "trials.csv: all included and excluded method-comparison trials.\n"
        "report.json: descriptive agreement, participant-balanced summaries and cluster-bootstrap intervals.\n"
        "checksums.sha256: SHA-256 for every exported file.\n\n"
        "This is a pre-clinical method-comparison export and does not by itself establish diagnostic validity.\n"
    ).encode("utf-8")
    files = [("protocol.json", manifest_bytes), ("trials.csv", trials_bytes), ("report.json", report_bytes), ("README.txt", readme)]
    checksum_bytes = _checksum_lines(files).encode("ascii")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files:
            zf.writestr(name, content)
        zf.writestr("checksums.sha256", checksum_bytes)
    return buffer.getvalue()


def build_formal_validation_bundle(
    *,
    run: dict[str, Any],
    study: dict[str, Any],
    trials: list[dict[str, Any]],
    report: dict[str, Any],
    statistical_plan: dict[str, Any],
    protocol_lock: dict[str, Any],
    checklist: dict[str, Any],
) -> bytes:
    """Immutable export for one formal collection run.

    The run snapshot identifies exactly which software/protocol/statistical plan
    governed the collection interval. This bundle is method-comparison evidence,
    not a declaration of clinical validity.
    """
    manifest = {
        "bundle_schema_version": 1,
        "kind": "formal_validation_run",
        "run": run,
        "study": study,
        "trial_count": len(trials),
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    report_bytes = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    sap_bytes = json.dumps(statistical_plan, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    lock_bytes = json.dumps(protocol_lock, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    checklist_bytes = json.dumps(checklist, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    trial_columns = [
        "id", "study_id", "created_at", "job_id", "patient_id", "evaluator_id",
        "reference_assessor_id", "movement", "side", "trial_number", "frame",
        "timestamp_ms", "automatic_angle_deg", "reference_angle_deg", "error_deg",
        "absolute_error_deg", "reference_method", "reference_device_model",
        "reference_device_serial", "landmark_confidence_min", "body_frame_confidence",
        "measurement_plane", "view_alignment", "view_deviation_deg", "out_of_plane_deg",
        "direction_rule", "direction_alignment", "source", "measurement_valid", "quality",
        "included", "exclusion_reason", "notes",
    ]
    trials_bytes = _csv_bytes(trials, trial_columns)
    readme = (
        "PhysioAI formal validation run bundle\n\n"
        "manifest.json: immutable run + frozen study snapshot.\n"
        "trials.csv: trials created during this run only.\n"
        "report.json: method-comparison summary for this run.\n"
        "statistical_analysis_plan.json: versioned analysis plan used at run start.\n"
        "protocol_lock.json: frozen clinical protocol contract.\n"
        "formal_validation_checklist.json: operational collection checklist.\n"
        "checksums.sha256: SHA-256 for every exported file.\n\n"
        "This bundle supports reproducibility and audit. It does not establish diagnostic validity or clinical effectiveness.\n"
    ).encode("utf-8")
    files = [
        ("manifest.json", manifest_bytes),
        ("trials.csv", trials_bytes),
        ("report.json", report_bytes),
        ("statistical_analysis_plan.json", sap_bytes),
        ("protocol_lock.json", lock_bytes),
        ("formal_validation_checklist.json", checklist_bytes),
        ("README.txt", readme),
    ]
    checksum_bytes = _checksum_lines(files).encode("ascii")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files:
            zf.writestr(name, content)
        zf.writestr("checksums.sha256", checksum_bytes)
    return buffer.getvalue()
