from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
import math
import statistics
from typing import Any, Iterable

LONGITUDINAL_SCHEMA_VERSION = 1


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _linear_slope_per_30d(points: list[tuple[datetime, float]]) -> float | None:
    if len(points) < 2:
        return None
    origin = points[0][0]
    xs = [(ts - origin).total_seconds() / 86400.0 for ts, _ in points]
    ys = [value for _, value in points]
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 1e-12:
        return None
    slope_day = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return slope_day * 30.0


@dataclass(slots=True)
class LongitudinalMetric:
    metric: str
    unit: str
    n: int
    first: float | None
    latest: float | None
    delta: float | None
    delta_pct: float | None
    mean: float | None
    sd: float | None
    cv_pct: float | None
    median_abs_consecutive_change: float | None
    slope_per_30d: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_metric(
    metric: str,
    unit: str,
    observations: Iterable[tuple[str | None, Any]],
) -> LongitudinalMetric:
    clean: list[tuple[datetime | None, float]] = []
    for timestamp, raw in observations:
        value = _finite(raw)
        if value is None:
            continue
        clean.append((_parse_time(timestamp), value))

    if not clean:
        return LongitudinalMetric(metric, unit, 0, None, None, None, None, None, None, None, None, None)

    values = [value for _, value in clean]
    first, latest = values[0], values[-1]
    delta = latest - first if len(values) >= 2 else None
    delta_pct = None
    if delta is not None and abs(first) > 1e-12:
        delta_pct = (delta / abs(first)) * 100.0
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) >= 2 else 0.0
    cv = (sd / abs(mean) * 100.0) if abs(mean) > 1e-12 and len(values) >= 2 else None
    consecutive = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    dated = [(ts, value) for ts, value in clean if ts is not None]
    slope = _linear_slope_per_30d(dated) if len(dated) == len(clean) else None

    def r(value: float | None) -> float | None:
        return None if value is None else round(float(value), 6)

    return LongitudinalMetric(
        metric=metric,
        unit=unit,
        n=len(values),
        first=r(first),
        latest=r(latest),
        delta=r(delta),
        delta_pct=r(delta_pct),
        mean=r(mean),
        sd=r(sd),
        cv_pct=r(cv),
        median_abs_consecutive_change=r(statistics.median(consecutive) if consecutive else None),
        slope_per_30d=r(slope),
    )


def _latest_motor_summary(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return records[-1].get("summary") if isinstance(records[-1].get("summary"), dict) else None


def build_longitudinal_profile(
    sessions: list[dict[str, Any]],
    *,
    anchor_job_id: str,
    motor_features_by_job: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build a descriptive longitudinal profile without clinical interpretation.

    The anchor session defines the comparable protocol/specification cohort. Sessions
    with a different movement, side, protocol version or clinical-spec hash are listed
    as excluded and are never pooled into the primary longitudinal metrics.
    """
    motor_features_by_job = motor_features_by_job or {}
    anchor = next((s for s in sessions if str(s.get("job_id")) == anchor_job_id), None)
    if anchor is None:
        raise ValueError("anchor_job_id não pertence ao conjunto de sessões")

    movement = str(anchor.get("movement") or "")
    side = str(anchor.get("side") or "")
    protocol = str(anchor.get("protocol_version") or "")
    clinical_sha = str(anchor.get("clinical_spec_sha256") or "")

    completed = [s for s in sessions if str(s.get("status")) == "completed"]
    comparable: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for session in completed:
        reasons: list[str] = []
        if str(session.get("movement") or "") != movement:
            reasons.append("movement_mismatch")
        if str(session.get("side") or "") != side:
            reasons.append("side_mismatch")
        if str(session.get("protocol_version") or "") != protocol:
            reasons.append("protocol_mismatch")
        if str(session.get("clinical_spec_sha256") or "") != clinical_sha:
            reasons.append("clinical_spec_mismatch")
        if reasons:
            excluded.append({"job_id": session.get("job_id"), "reasons": reasons})
        else:
            comparable.append(session)

    comparable.sort(key=lambda row: str(row.get("started_at") or ""))
    software_versions = sorted({str(s.get("software_version") or "unknown") for s in comparable})

    def obs(field: str):
        return [(s.get("started_at"), s.get(field)) for s in comparable]

    metrics: dict[str, LongitudinalMetric] = {
        "adm_deg": summarize_metric("adm_deg", "deg", obs("adm_deg")),
        "valid_frame_ratio": summarize_metric(
            "valid_frame_ratio", "ratio",
            [
                (s.get("started_at"), (float(s.get("valid_frames")) / float(s.get("total_frames")))
                 if _finite(s.get("valid_frames")) is not None and _finite(s.get("total_frames")) not in {None, 0.0}
                 else None)
                for s in comparable
            ],
        ),
        "total_reps": summarize_metric("total_reps", "count", obs("total_reps")),
    }

    motor_metric_map = {
        "median_cycle_duration_s": "s",
        "median_cycle_amplitude_deg": "deg",
        "median_cycle_peak_velocity_deg_s": "deg/s",
        "amplitude_decrement_pct": "%",
        "peak_velocity_decrement_pct": "%",
        "fusion_quality_score": "ratio",
    }
    for metric_name, unit in motor_metric_map.items():
        observations: list[tuple[str | None, Any]] = []
        for session in comparable:
            summary = _latest_motor_summary(motor_features_by_job.get(str(session.get("job_id")), []))
            value = None
            if summary:
                if metric_name == "fusion_quality_score":
                    value = summary.get(metric_name)
                else:
                    vision = summary.get("vision") if isinstance(summary.get("vision"), dict) else {}
                    value = vision.get(metric_name)
            observations.append((session.get("started_at"), value))
        metric = summarize_metric(metric_name, unit, observations)
        if metric.n:
            metrics[metric_name] = metric

    flags: list[str] = []
    if len(comparable) < 2:
        flags.append("insufficient_comparable_sessions")
    if len(comparable) == 2:
        flags.append("only_two_sessions")
    if len(software_versions) > 1:
        flags.append("mixed_software_versions_same_protocol")
    if excluded:
        flags.append("noncomparable_sessions_excluded")

    start = comparable[0].get("started_at") if comparable else None
    end = comparable[-1].get("started_at") if comparable else None
    start_dt, end_dt = _parse_time(start), _parse_time(end)
    span_days = None
    if start_dt and end_dt:
        span_days = round((end_dt - start_dt).total_seconds() / 86400.0, 6)

    if len(comparable) >= 3:
        status = "descriptive"
    elif len(comparable) == 2:
        status = "limited"
    else:
        status = "insufficient"

    compact_sessions = [
        {
            "job_id": s.get("job_id"),
            "started_at": s.get("started_at"),
            "completed_at": s.get("completed_at"),
            "software_version": s.get("software_version"),
            "protocol_version": s.get("protocol_version"),
            "clinical_spec_sha256": s.get("clinical_spec_sha256"),
            "adm_deg": s.get("adm_deg"),
            "total_reps": s.get("total_reps"),
            "valid_frames": s.get("valid_frames"),
            "total_frames": s.get("total_frames"),
            "result_sha256": s.get("result_sha256"),
        }
        for s in comparable
    ]

    return {
        "schema_version": LONGITUDINAL_SCHEMA_VERSION,
        "experimental": True,
        "diagnostic": False,
        "intended_use": "descriptive longitudinal research and software-regression review",
        "unsupported_inferences": [
            "clinical_improvement",
            "disease_progression",
            "treatment_response",
            "minimal_clinically_important_difference",
            "diagnosis",
        ],
        "anchor_job_id": anchor_job_id,
        "movement": movement,
        "side": side,
        "protocol_version": protocol,
        "clinical_spec_sha256": clinical_sha,
        "status": status,
        "session_count": len(comparable),
        "excluded_session_count": len(excluded),
        "software_versions": software_versions,
        "mixed_software_versions": len(software_versions) > 1,
        "span_days": span_days,
        "quality_flags": flags,
        "sessions": compact_sessions,
        "excluded_sessions": excluded,
        "metrics": {name: value.to_dict() for name, value in metrics.items()},
        "methodology": {
            "pooling_rule": "same patient + movement + side + protocol_version + clinical_spec_sha256",
            "trend": "ordinary least-squares slope per 30 days; descriptive only",
            "cv": "sample SD / absolute mean; descriptive repeatability signal, not reliability coefficient",
            "version_policy": "software versions may coexist only when protocol/spec hashes remain identical and are flagged",
        },
    }
