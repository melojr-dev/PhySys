from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from typing import Any, Iterable


PUBLICATION_PACKAGE_SCHEMA_VERSION = 1


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")


def _csv_bytes(rows: Iterable[dict[str, Any]], columns: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        normalized: dict[str, Any] = {}
        for key in columns:
            value = row.get(key)
            if isinstance(value, (dict, list)):
                normalized[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            elif value is None:
                normalized[key] = ""
            else:
                normalized[key] = value
        writer.writerow(normalized)
    return buffer.getvalue().encode("utf-8-sig")


def _alias_map(values: Iterable[str], prefix: str) -> dict[str, str]:
    unique = sorted({str(v) for v in values if str(v)})
    width = max(3, len(str(max(len(unique), 1))))
    return {raw: f"{prefix}{idx:0{width}d}" for idx, raw in enumerate(unique, 1)}


def anonymize_trials(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a reviewer-facing table without internal identifiers or free text.

    Aliases are scoped to one export bundle. They are intentionally not stable
    across different exports and cannot be reversed without the source database.
    """
    patient_alias = _alias_map((str(r.get("patient_id") or "") for r in rows), "P")
    session_alias = _alias_map((str(r.get("job_id") or "") for r in rows), "S")

    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        patient = str(row.get("patient_id") or "")
        job = str(row.get("job_id") or "")
        auto = row.get("automatic_angle_deg")
        ref = row.get("reference_angle_deg")
        error = None
        abs_error = None
        if auto is not None and ref is not None:
            error = round(float(auto) - float(ref), 4)
            abs_error = round(abs(error), 4)
        out.append({
            "trial_alias": f"T{index:04d}",
            "participant_alias": patient_alias.get(patient, ""),
            "session_alias": session_alias.get(job, ""),
            "movement": row.get("movement"),
            "side": row.get("side"),
            "trial_number": row.get("trial_number"),
            "frame": row.get("frame"),
            "timestamp_ms": row.get("timestamp_ms"),
            "automatic_angle_deg": auto,
            "reference_angle_deg": ref,
            "error_deg": error,
            "absolute_error_deg": abs_error,
            "reference_method": row.get("reference_method"),
            "reference_device_model": row.get("reference_device_model"),
            "landmark_confidence_min": row.get("landmark_confidence_min"),
            "body_frame_confidence": row.get("body_frame_confidence"),
            "measurement_plane": row.get("measurement_plane"),
            "view_alignment": row.get("view_alignment"),
            "view_deviation_deg": row.get("view_deviation_deg"),
            "out_of_plane_deg": row.get("out_of_plane_deg"),
            "direction_rule": row.get("direction_rule"),
            "direction_alignment": row.get("direction_alignment"),
            "source": row.get("source"),
            "measurement_valid": bool(row.get("measurement_valid")),
            "quality": row.get("quality"),
            "included": bool(row.get("included")),
            "exclusion_reason": row.get("exclusion_reason"),
        })
    return out


def _privacy_maps(rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    return (
        _alias_map((str(r.get("patient_id") or "") for r in rows), "P"),
        _alias_map((str(r.get("job_id") or "") for r in rows), "S"),
    )


def sanitize_report(report: dict[str, Any], rows: list[dict[str, Any]], *, study_code: str | None = None) -> dict[str, Any]:
    """Remove internal identifiers embedded in statistical summaries."""
    patient_alias, _ = _privacy_maps(rows)
    clean = json.loads(json.dumps(report, ensure_ascii=False, allow_nan=False))
    clean.pop("study_id", None)
    if study_code:
        clean["study_code"] = study_code
    balanced = clean.get("participant_balanced") or {}
    per = balanced.get("per_participant") or []
    for item in per:
        raw = str(item.pop("patient_id", "") or "")
        item["participant_alias"] = patient_alias.get(raw, "")
    return clean


def publication_readiness(*, run: dict[str, Any], study: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Technical readiness of a closed validation run for external review.

    This is deliberately not called manuscript readiness or clinical validity.
    It only checks whether a reproducible, reviewable package can be built.
    """
    checks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def add(code: str, ok: bool, detail: str) -> None:
        checks.append({"code": code, "ok": bool(ok), "detail": detail})

    def warn(code: str, condition: bool, detail: str) -> None:
        if condition:
            warnings.append({"code": code, "detail": detail})

    counts = report.get("counts") or {}
    add("formal_run_closed", run.get("status") == "closed", "A execução formal deve estar fechada.")
    add("start_snapshot_present", bool(run.get("snapshot_sha256")), "Snapshot inicial imutável ausente.")
    add("close_report_present", bool(run.get("close_report_sha256") or run.get("close_report")), "Relatório de fechamento ausente.")
    add("study_frozen", study.get("status") in {"frozen", "closed"}, "O protocolo do estudo deve permanecer congelado/fechado.")
    add("protocol_hash_present", bool(study.get("protocol_sha256")), "Hash do protocolo do estudo ausente.")
    add("report_scope", report.get("formal_run_scope") == "trials_created_during_run_only", "O relatório deve usar apenas ensaios da execução formal.")
    add("included_trials_present", int(counts.get("trials_included") or 0) > 0, "Nenhum ensaio incluído está disponível para revisão.")
    add("included_participants_present", int(counts.get("participants_included") or 0) > 0, "Nenhum participante incluído está disponível para revisão.")

    target = study.get("target_participants")
    included_participants = int(counts.get("participants_included") or 0)
    if target:
        warn(
            "target_not_reached",
            included_participants < int(target),
            f"A execução inclui {included_participants}/{int(target)} participantes-alvo; o pacote deve ser interpretado como incompleto.",
        )
    warn(
        "small_participant_count",
        0 < included_participants < 10,
        "Número de participantes incluídos é pequeno; estimativas de concordância podem ser instáveis.",
    )
    warn(
        "excluded_trials_present",
        int(counts.get("trials_excluded") or 0) > 0,
        "Há ensaios excluídos; revisar motivos e aderência ao protocolo.",
    )

    return {
        "package_ready": all(item["ok"] for item in checks),
        "checks": checks,
        "warnings": warnings,
        "boundary": "Technical external-review packaging only; does not establish clinical validity, diagnostic performance, or publication acceptance.",
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, str)):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def build_manuscript_summary(*, run: dict[str, Any], study: dict[str, Any], report: dict[str, Any], readiness: dict[str, Any]) -> str:
    counts = report.get("counts") or {}
    agreement = report.get("agreement") or {}
    balanced = report.get("participant_balanced") or {}
    bootstrap = report.get("cluster_bootstrap") or {}
    ci_mae = bootstrap.get("mae_ci95_deg") or [None, None]
    ci_bias = bootstrap.get("bias_ci95_deg") or [None, None]
    lines = [
        "# PhysioAI — resumo técnico para revisão externa",
        "",
        f"- Execução formal: `{run.get('id', '')}`",
        f"- Estudo: `{study.get('code', study.get('id', ''))}` — {study.get('title', '')}",
        f"- Movimento: {study.get('movement', '')}",
        f"- Lado: {study.get('side', '')}",
        f"- Versão do software: `{run.get('software_version', '')}`",
        f"- Protocolo clínico: `{run.get('clinical_protocol_version', '')}`",
        f"- Participantes incluídos: {counts.get('participants_included', 0)}",
        f"- Ensaios incluídos/excluídos: {counts.get('trials_included', 0)}/{counts.get('trials_excluded', 0)}",
        "",
        "## Concordância automática × referência física",
        "",
        f"- MAE: {_fmt(agreement.get('mae_deg'))}°",
        f"- RMSE: {_fmt(agreement.get('rmse_deg'))}°",
        f"- Viés (automático − referência): {_fmt(agreement.get('bias_deg'))}°",
        f"- Limites de concordância de Bland–Altman: {_fmt(agreement.get('loa_lower_deg'))}° a {_fmt(agreement.get('loa_upper_deg'))}°",
        f"- ICC(A,1): {_fmt(agreement.get('icc_a1'), 4)}",
        f"- CCC de Lin: {_fmt(agreement.get('ccc'), 4)}",
        f"- Pearson r: {_fmt(agreement.get('pearson_r'), 4)}",
        f"- Dentro de ±2°: {_fmt(agreement.get('within_2_deg_pct'), 1)}%",
        f"- Dentro de ±5°: {_fmt(agreement.get('within_5_deg_pct'), 1)}%",
        "",
        "## Resumo balanceado por participante",
        "",
        f"- MAE médio por participante: {_fmt(balanced.get('mean_patient_mae_deg'))}°",
        f"- MAE mediano por participante: {_fmt(balanced.get('median_patient_mae_deg'))}°",
        f"- Viés médio por participante: {_fmt(balanced.get('mean_patient_bias_deg'))}°",
        f"- Bootstrap por cluster — IC95% MAE: {_fmt(ci_mae[0])}° a {_fmt(ci_mae[1])}°",
        f"- Bootstrap por cluster — IC95% viés: {_fmt(ci_bias[0])}° a {_fmt(ci_bias[1])}°",
        "",
        "## Limites de interpretação",
        "",
        "Este arquivo resume uma comparação de métodos. Frames/ensaios repetidos do mesmo participante não são tratados como participantes independentes. "
        "Os resultados não constituem diagnóstico, eficácia clínica, MCID nem validação clínica automática do PhysioAI.",
    ]
    if readiness.get("warnings"):
        lines.extend(["", "## Alertas do pacote", ""])
        for item in readiness["warnings"]:
            lines.append(f"- **{item['code']}**: {item['detail']}")
    return "\n".join(lines) + "\n"


def _scale(value: float, min_value: float, max_value: float, start: float, end: float) -> float:
    if max_value <= min_value:
        return (start + end) / 2
    return start + (value - min_value) / (max_value - min_value) * (end - start)


def build_method_comparison_svg(rows: list[dict[str, Any]]) -> str:
    included = [r for r in rows if r.get("included") and r.get("automatic_angle_deg") is not None and r.get("reference_angle_deg") is not None]
    width, height = 760, 520
    left, right, top, bottom = 78, 28, 34, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [float(r["automatic_angle_deg"]) for r in included] + [float(r["reference_angle_deg"]) for r in included]
    max_v = max(values, default=180.0)
    axis_max = max(30.0, math.ceil(max_v / 30.0) * 30.0)
    axis_min = 0.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.ref{stroke:#666;stroke-width:1.5;stroke-dasharray:5 4}.point{fill:#1769aa;fill-opacity:.72}</style>',
    ]
    for tick in range(0, int(axis_max) + 1, 30):
        x = _scale(tick, axis_min, axis_max, left, left + plot_w)
        y = _scale(tick, axis_min, axis_max, top + plot_h, top)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}"/>')
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" font-size="12" text-anchor="middle">{tick}</text>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{tick}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    x0 = _scale(0, axis_min, axis_max, left, left + plot_w)
    y0 = _scale(0, axis_min, axis_max, top + plot_h, top)
    x1 = _scale(axis_max, axis_min, axis_max, left, left + plot_w)
    y1 = _scale(axis_max, axis_min, axis_max, top + plot_h, top)
    parts.append(f'<line class="ref" x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}"/>')
    for row in included:
        x = _scale(float(row["reference_angle_deg"]), axis_min, axis_max, left, left + plot_w)
        y = _scale(float(row["automatic_angle_deg"]), axis_min, axis_max, top + plot_h, top)
        parts.append(f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="4"/>')
    parts.append(f'<text x="{width/2}" y="{height-18}" font-size="14" text-anchor="middle">Referência física (graus)</text>')
    parts.append(f'<text x="18" y="{height/2}" font-size="14" text-anchor="middle" transform="rotate(-90 18 {height/2})">PhysioAI (graus)</text>')
    parts.append(f'<text x="{width/2}" y="20" font-size="16" font-weight="bold" text-anchor="middle">Comparação de métodos</text>')
    parts.append('</svg>')
    return "".join(parts)


def build_bland_altman_svg(report: dict[str, Any]) -> str:
    points = report.get("bland_altman") or []
    agreement = report.get("agreement") or {}
    width, height = 760, 520
    left, right, top, bottom = 78, 28, 34, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    means = [float(p["mean_deg"]) for p in points]
    diffs = [float(p["difference_deg"]) for p in points]
    x_min = min(means, default=0.0)
    x_max = max(means, default=180.0)
    if x_max - x_min < 10:
        x_min = max(0.0, x_min - 5)
        x_max += 5
    lines = [agreement.get("bias_deg"), agreement.get("loa_lower_deg"), agreement.get("loa_upper_deg")]
    y_values = diffs + [float(v) for v in lines if v is not None]
    y_abs = max([abs(v) for v in y_values], default=10.0)
    y_abs = max(2.0, math.ceil((y_abs * 1.15) / 2.0) * 2.0)
    y_min, y_max = -y_abs, y_abs
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.bias{stroke:#1769aa;stroke-width:1.7}.loa{stroke:#b33;stroke-width:1.5;stroke-dasharray:5 4}.point{fill:#1769aa;fill-opacity:.72}</style>',
    ]
    for frac in range(6):
        xv = x_min + (x_max - x_min) * frac / 5
        x = _scale(xv, x_min, x_max, left, left + plot_w)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}"/>')
        parts.append(f'<text x="{x:.2f}" y="{top+plot_h+24}" font-size="12" text-anchor="middle">{xv:.0f}</text>')
    for frac in range(5):
        yv = y_min + (y_max - y_min) * frac / 4
        y = _scale(yv, y_min, y_max, top + plot_h, top)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.2f}" font-size="12" text-anchor="end">{yv:.1f}</text>')
    for row in points:
        x = _scale(float(row["mean_deg"]), x_min, x_max, left, left + plot_w)
        y = _scale(float(row["difference_deg"]), y_min, y_max, top + plot_h, top)
        parts.append(f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="4"/>')
    for key, css, label in (("bias_deg", "bias", "Viés"), ("loa_lower_deg", "loa", "LoA inferior"), ("loa_upper_deg", "loa", "LoA superior")):
        value = agreement.get(key)
        if value is None:
            continue
        y = _scale(float(value), y_min, y_max, top + plot_h, top)
        parts.append(f'<line class="{css}" x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}"/>')
        parts.append(f'<text x="{left+plot_w-4}" y="{y-5:.2f}" font-size="11" text-anchor="end">{label}: {float(value):.2f}°</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    parts.append(f'<text x="{width/2}" y="{height-18}" font-size="14" text-anchor="middle">Média dos métodos (graus)</text>')
    parts.append(f'<text x="18" y="{height/2}" font-size="14" text-anchor="middle" transform="rotate(-90 18 {height/2})">Diferença PhysioAI − referência (graus)</text>')
    parts.append(f'<text x="{width/2}" y="20" font-size="16" font-weight="bold" text-anchor="middle">Bland–Altman</text>')
    parts.append('</svg>')
    return "".join(parts)


def build_traceability_rows(requirements: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in requirements.get("requirements", []):
        rows.append({
            "requirement_id": item.get("id"),
            "category": item.get("category"),
            "requirement": item.get("requirement"),
            "implementation": "; ".join(item.get("implementation") or []),
            "verification": "; ".join(item.get("verification") or []),
            "evidence": item.get("evidence"),
        })
    return rows


def _table_rows(report: dict[str, Any]) -> dict[str, tuple[list[dict[str, Any]], list[str]]]:
    counts = report.get("counts") or {}
    agreement = report.get("agreement") or {}
    balanced = report.get("participant_balanced") or {}
    bootstrap = report.get("cluster_bootstrap") or {}
    count_rows = [{"metric": k, "value": v} for k, v in counts.items()]
    agreement_rows = [{"metric": k, "value": v} for k, v in agreement.items()]
    participant_rows = [{"metric": k, "value": v} for k, v in balanced.items() if k != "per_participant"]
    ci_rows = [{"metric": k, "value": v} for k, v in bootstrap.items() if k != "seed"]
    exclusion_rows = [{"reason": k, "count": v} for k, v in (report.get("exclusion_reasons") or {}).items()]
    return {
        "tables/counts.csv": (count_rows, ["metric", "value"]),
        "tables/agreement.csv": (agreement_rows, ["metric", "value"]),
        "tables/participant_balanced.csv": (participant_rows, ["metric", "value"]),
        "tables/cluster_bootstrap.csv": (ci_rows, ["metric", "value"]),
        "tables/exclusions.csv": (exclusion_rows, ["reason", "count"]),
    }


def _deterministic_zip(files: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, content in files:
            info = zipfile.ZipInfo(filename=name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zf.writestr(info, content)
    return buffer.getvalue()


def build_publication_review_bundle(
    *,
    run: dict[str, Any],
    study: dict[str, Any],
    trials: list[dict[str, Any]],
    report: dict[str, Any],
    statistical_plan: dict[str, Any],
    protocol_lock: dict[str, Any],
    checklist: dict[str, Any],
    requirements: dict[str, Any],
) -> bytes:
    readiness = publication_readiness(run=run, study=study, report=report)
    if not readiness["package_ready"]:
        failed = ", ".join(item["code"] for item in readiness["checks"] if not item["ok"])
        raise ValueError(f"execução não está pronta para pacote de revisão externa: {failed}")

    anon_trials = anonymize_trials(trials)
    external_report = sanitize_report(report, trials, study_code=str(study.get("code") or ""))
    trace_rows = build_traceability_rows(requirements)
    external_manifest = {
        "schema_version": PUBLICATION_PACKAGE_SCHEMA_VERSION,
        "kind": "external_review_package",
        "formal_run_alias": "RUN001",
        "study_code": study.get("code"),
        "study_title": study.get("title"),
        "movement": study.get("movement"),
        "side": study.get("side"),
        "software_version": run.get("software_version"),
        "clinical_protocol_version": run.get("clinical_protocol_version"),
        "study_protocol_sha256": study.get("protocol_sha256"),
        "formal_start_snapshot_sha256": run.get("snapshot_sha256"),
        "formal_close_report_sha256": run.get("close_report_sha256"),
        "counts": report.get("counts"),
        "readiness": readiness,
        "deidentification": {
            "participant_ids": "bundle-local aliases",
            "session_ids": "bundle-local aliases",
            "internal_trial_ids": "omitted",
            "job_ids": "omitted",
            "evaluator_ids": "omitted",
            "reference_device_serials": "omitted",
            "free_text_notes": "omitted",
        },
        "interpretation_boundary": "Method-comparison research package; no diagnostic or clinical-effectiveness claim.",
    }

    files: list[tuple[str, bytes]] = [
        ("manifest.json", _canonical_json_bytes(external_manifest)),
        ("report.json", _canonical_json_bytes(external_report)),
        ("publication_readiness.json", _canonical_json_bytes(readiness)),
        ("manuscript_summary.md", build_manuscript_summary(run={**run, "id": "RUN001"}, study=study, report=external_report, readiness=readiness).encode("utf-8")),
        ("trials_anonymized.csv", _csv_bytes(anon_trials, list(anon_trials[0].keys()) if anon_trials else ["trial_alias"])),
        ("figures/method_comparison.svg", build_method_comparison_svg(anon_trials).encode("utf-8")),
        ("figures/bland_altman.svg", build_bland_altman_svg(external_report).encode("utf-8")),
        ("traceability/requirements.json", _canonical_json_bytes(requirements)),
        ("traceability/matrix.csv", _csv_bytes(trace_rows, ["requirement_id", "category", "requirement", "implementation", "verification", "evidence"])),
        ("governance/statistical_analysis_plan.json", _canonical_json_bytes(statistical_plan)),
        ("governance/protocol_lock.json", _canonical_json_bytes(protocol_lock)),
        ("governance/formal_validation_checklist.json", _canonical_json_bytes(checklist)),
    ]
    for name, (rows, columns) in _table_rows(external_report).items():
        files.append((name, _csv_bytes(rows, columns)))

    readme = (
        "PhysioAI external review package\n\n"
        "This archive is generated from one closed formal validation run.\n"
        "Participant/session identifiers are bundle-local aliases; internal job/evaluator IDs, serial numbers and free-text notes are omitted.\n"
        "The package supports method-comparison review and software traceability. It does not establish clinical validity, diagnostic performance or clinical effectiveness.\n\n"
        "Key files:\n"
        "- manuscript_summary.md: concise source-derived summary.\n"
        "- trials_anonymized.csv: deidentified trial-level data used for analysis.\n"
        "- report.json and tables/: statistical outputs.\n"
        "- figures/: deterministic SVG review figures.\n"
        "- traceability/: requirement-to-implementation-to-test mapping.\n"
        "- governance/: frozen statistical plan, protocol lock and collection checklist.\n"
        "- checksums.sha256: content hashes for every file in this archive.\n"
    ).encode("utf-8")
    files.append(("README.txt", readme))

    checksum = "".join(f"{hashlib.sha256(content).hexdigest()}  {name}\n" for name, content in files).encode("ascii")
    files.append(("checksums.sha256", checksum))
    return _deterministic_zip(files)
