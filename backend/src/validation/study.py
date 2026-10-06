from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import statistics
import threading
import uuid
from typing import Any, Iterable

from .agreement import agreement_metrics, bland_altman_points


STUDY_SCHEMA_VERSION = 2
_ALLOWED_STATUS = {"draft", "frozen", "closed"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def participant_balanced_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Error summaries giving every participant equal weight.

    This does not turn repeated measurements into independent observations. It
    summarizes the distribution of participant-level errors and is deliberately
    reported separately from frame/trial-level agreement metrics.
    """
    by_patient: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        patient = str(row.get("patient_id") or "")
        auto = row.get("automatic_angle_deg")
        ref = row.get("reference_angle_deg")
        if not patient or auto is None or ref is None:
            continue
        a, r = float(auto), float(ref)
        if math.isfinite(a) and math.isfinite(r):
            by_patient[patient].append(a - r)

    per_patient: list[dict[str, Any]] = []
    for patient, errors in sorted(by_patient.items()):
        abs_errors = [abs(v) for v in errors]
        per_patient.append({
            "patient_id": patient,
            "n": len(errors),
            "mae_deg": statistics.fmean(abs_errors),
            "bias_deg": statistics.fmean(errors),
            "rmse_deg": math.sqrt(statistics.fmean(v * v for v in errors)),
        })

    if not per_patient:
        return {
            "participants": 0,
            "mean_patient_mae_deg": None,
            "median_patient_mae_deg": None,
            "mean_patient_bias_deg": None,
            "mean_patient_rmse_deg": None,
            "per_participant": [],
        }

    return {
        "participants": len(per_patient),
        "mean_patient_mae_deg": round(statistics.fmean(x["mae_deg"] for x in per_patient), 4),
        "median_patient_mae_deg": round(statistics.median(x["mae_deg"] for x in per_patient), 4),
        "mean_patient_bias_deg": round(statistics.fmean(x["bias_deg"] for x in per_patient), 4),
        "mean_patient_rmse_deg": round(statistics.fmean(x["rmse_deg"] for x in per_patient), 4),
        "per_participant": [
            {
                **x,
                "mae_deg": round(x["mae_deg"], 4),
                "bias_deg": round(x["bias_deg"], 4),
                "rmse_deg": round(x["rmse_deg"], 4),
            }
            for x in per_patient
        ],
    }


def cluster_bootstrap_ci(
    rows: Iterable[dict[str, Any]],
    *,
    iterations: int = 1000,
    seed: int = 20260911,
) -> dict[str, Any]:
    """Deterministic participant-cluster bootstrap CIs for MAE and bias.

    Participants, not frames, are resampled. All trials belonging to a sampled
    participant travel together. This preserves within-participant dependence
    better than naively resampling individual frames.
    """
    by_patient: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        patient = str(row.get("patient_id") or "")
        auto = row.get("automatic_angle_deg")
        ref = row.get("reference_angle_deg")
        if not patient or auto is None or ref is None:
            continue
        a, r = float(auto), float(ref)
        if math.isfinite(a) and math.isfinite(r):
            by_patient[patient].append((a, r))

    patients = sorted(by_patient)
    if len(patients) < 2:
        return {
            "participants": len(patients),
            "iterations": 0,
            "mae_ci95_deg": [None, None],
            "bias_ci95_deg": [None, None],
            "method": "participant_cluster_bootstrap",
        }

    iterations = max(200, min(int(iterations), 10000))
    rng = random.Random(seed)
    maes: list[float] = []
    biases: list[float] = []
    for _ in range(iterations):
        sampled = [rng.choice(patients) for _ in patients]
        pairs: list[tuple[float, float]] = []
        for patient in sampled:
            pairs.extend(by_patient[patient])
        errors = [a - r for a, r in pairs]
        maes.append(statistics.fmean(abs(v) for v in errors))
        biases.append(statistics.fmean(errors))

    return {
        "participants": len(patients),
        "iterations": iterations,
        "mae_ci95_deg": [
            round(float(_percentile(maes, 0.025)), 4),
            round(float(_percentile(maes, 0.975)), 4),
        ],
        "bias_ci95_deg": [
            round(float(_percentile(biases, 0.025)), 4),
            round(float(_percentile(biases, 0.975)), 4),
        ],
        "method": "participant_cluster_bootstrap",
        "seed": seed,
    }


def _angle_bin(angle: float) -> str:
    if angle < 30:
        return "0-<30"
    if angle < 60:
        return "30-<60"
    if angle < 90:
        return "60-<90"
    if angle < 120:
        return "90-<120"
    if angle < 150:
        return "120-<150"
    return "150-180"


def stratified_reference_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        auto = row.get("automatic_angle_deg")
        ref = row.get("reference_angle_deg")
        if auto is None or ref is None:
            continue
        a, r = float(auto), float(ref)
        if math.isfinite(a) and math.isfinite(r):
            groups[_angle_bin(r)].append((a, r))
    return {name: agreement_metrics(pairs) for name, pairs in sorted(groups.items())}


class ValidationStudyStore:
    """Study protocol + trial store for pre-clinical method-comparison studies."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=15.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=15000")
        return con

    def _initialize(self) -> None:
        with self._lock, closing(self._connect()) as con, con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS validation_studies (
                    id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    code TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    protocol_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    status TEXT NOT NULL,
                    frozen_at TEXT,
                    closed_at TEXT,
                    movement TEXT NOT NULL,
                    side TEXT NOT NULL,
                    target_participants INTEGER,
                    repetitions_per_participant INTEGER,
                    min_landmark_confidence REAL NOT NULL,
                    require_observed INTEGER NOT NULL,
                    require_valid_measurement INTEGER NOT NULL,
                    reference_method TEXT NOT NULL,
                    reference_device_model TEXT,
                    reference_device_resolution_deg REAL,
                    camera_setup_json TEXT NOT NULL DEFAULT '{}',
                    protocol_notes TEXT NOT NULL DEFAULT '',
                    protocol_sha256 TEXT
                );

                CREATE TABLE IF NOT EXISTS validation_trials (
                    id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    study_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    evaluator_id TEXT NOT NULL,
                    reference_assessor_id TEXT,
                    movement TEXT NOT NULL,
                    side TEXT NOT NULL,
                    trial_number INTEGER NOT NULL,
                    frame INTEGER NOT NULL,
                    timestamp_ms REAL NOT NULL,
                    automatic_angle_deg REAL NOT NULL,
                    reference_angle_deg REAL NOT NULL,
                    reference_method TEXT NOT NULL,
                    reference_device_model TEXT,
                    reference_device_serial TEXT,
                    landmark_confidence_min REAL,
                    body_frame_confidence REAL,
                    measurement_plane TEXT,
                    view_alignment REAL,
                    view_deviation_deg REAL,
                    out_of_plane_deg REAL,
                    direction_rule TEXT,
                    direction_alignment REAL,
                    source TEXT,
                    measurement_valid INTEGER NOT NULL,
                    quality TEXT,
                    included INTEGER NOT NULL,
                    exclusion_reason TEXT,
                    notes TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(study_id) REFERENCES validation_studies(id) ON DELETE CASCADE,
                    UNIQUE(study_id, job_id, frame)
                );

                CREATE INDEX IF NOT EXISTS idx_validation_studies_status
                    ON validation_studies(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_validation_trials_study
                    ON validation_trials(study_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_validation_trials_patient
                    ON validation_trials(study_id, patient_id);
                CREATE INDEX IF NOT EXISTS idx_validation_trials_session
                    ON validation_trials(study_id, job_id);
                """
            )
            existing = {str(row[1]) for row in con.execute("PRAGMA table_info(validation_trials)").fetchall()}
            additions = {
                "body_frame_confidence": "REAL",
                "measurement_plane": "TEXT",
                "view_alignment": "REAL",
                "view_deviation_deg": "REAL",
                "out_of_plane_deg": "REAL",
                "direction_rule": "TEXT",
                "direction_alignment": "REAL",
            }
            for name, sql_type in additions.items():
                if name not in existing:
                    con.execute(f"ALTER TABLE validation_trials ADD COLUMN {name} {sql_type}")

    @staticmethod
    def _protocol_payload(study: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": int(study.get("schema_version", STUDY_SCHEMA_VERSION)),
            "code": study["code"],
            "title": study["title"],
            "protocol_version": study["protocol_version"],
            "movement": study["movement"],
            "side": study["side"],
            "target_participants": study.get("target_participants"),
            "repetitions_per_participant": study.get("repetitions_per_participant"),
            "min_landmark_confidence": float(study["min_landmark_confidence"]),
            "require_observed": bool(study["require_observed"]),
            "require_valid_measurement": bool(study["require_valid_measurement"]),
            "reference_method": study["reference_method"],
            "reference_device_model": study.get("reference_device_model"),
            "reference_device_resolution_deg": study.get("reference_device_resolution_deg"),
            "camera_setup": study.get("camera_setup") or {},
            "protocol_notes": study.get("protocol_notes") or "",
        }

    def create_study(self, payload: dict[str, Any]) -> dict[str, Any]:
        code = str(payload.get("code") or "").strip().upper()
        title = str(payload.get("title") or "").strip()
        movement = str(payload.get("movement") or "").strip()
        side = str(payload.get("side") or "ambos").strip().lower()
        protocol_version = str(payload.get("protocol_version") or "1.0").strip()
        created_by = str(payload.get("created_by") or "").strip()
        min_conf = float(payload.get("min_landmark_confidence", 0.65))
        if not code or len(code) > 64:
            raise ValueError("code do estudo é obrigatório e deve ter até 64 caracteres")
        if not title or len(title) > 200:
            raise ValueError("title do estudo é obrigatório e deve ter até 200 caracteres")
        if not movement:
            raise ValueError("movement é obrigatório")
        if side not in {"direito", "esquerdo", "ambos"}:
            raise ValueError("side deve ser direito, esquerdo ou ambos")
        if not created_by:
            raise ValueError("created_by é obrigatório")
        if not 0.0 <= min_conf <= 1.0:
            raise ValueError("min_landmark_confidence deve estar entre 0 e 1")

        study_id = str(uuid.uuid4())
        row = {
            "id": study_id,
            "schema_version": STUDY_SCHEMA_VERSION,
            "code": code,
            "title": title,
            "protocol_version": protocol_version,
            "created_at": _utcnow(),
            "created_by": created_by,
            "status": "draft",
            "frozen_at": None,
            "closed_at": None,
            "movement": movement,
            "side": side,
            "target_participants": payload.get("target_participants"),
            "repetitions_per_participant": payload.get("repetitions_per_participant"),
            "min_landmark_confidence": min_conf,
            "require_observed": int(bool(payload.get("require_observed", True))),
            "require_valid_measurement": int(bool(payload.get("require_valid_measurement", True))),
            "reference_method": str(payload.get("reference_method") or "physical_goniometer"),
            "reference_device_model": (str(payload.get("reference_device_model") or "").strip() or None),
            "reference_device_resolution_deg": payload.get("reference_device_resolution_deg"),
            "camera_setup_json": json.dumps(payload.get("camera_setup") or {}, ensure_ascii=False, sort_keys=True),
            "protocol_notes": str(payload.get("protocol_notes") or ""),
            "protocol_sha256": None,
        }
        for key in ("target_participants", "repetitions_per_participant"):
            if row[key] is not None:
                row[key] = int(row[key])
                if row[key] <= 0:
                    raise ValueError(f"{key} deve ser > 0")
        if row["reference_device_resolution_deg"] is not None:
            row["reference_device_resolution_deg"] = float(row["reference_device_resolution_deg"])
            if row["reference_device_resolution_deg"] <= 0:
                raise ValueError("reference_device_resolution_deg deve ser > 0")

        with self._lock, closing(self._connect()) as con, con:
            try:
                con.execute(
                    """
                    INSERT INTO validation_studies (
                        id, schema_version, code, title, protocol_version, created_at,
                        created_by, status, frozen_at, closed_at, movement, side,
                        target_participants, repetitions_per_participant,
                        min_landmark_confidence, require_observed,
                        require_valid_measurement, reference_method,
                        reference_device_model, reference_device_resolution_deg,
                        camera_setup_json, protocol_notes, protocol_sha256
                    ) VALUES (
                        :id, :schema_version, :code, :title, :protocol_version, :created_at,
                        :created_by, :status, :frozen_at, :closed_at, :movement, :side,
                        :target_participants, :repetitions_per_participant,
                        :min_landmark_confidence, :require_observed,
                        :require_valid_measurement, :reference_method,
                        :reference_device_model, :reference_device_resolution_deg,
                        :camera_setup_json, :protocol_notes, :protocol_sha256
                    )
                    """,
                    row,
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"code do estudo já existe: {code}") from exc
        return self.get_study(study_id)

    def get_study(self, study_id: str) -> dict[str, Any]:
        with self._lock, closing(self._connect()) as con:
            row = con.execute("SELECT * FROM validation_studies WHERE id = ?", (study_id,)).fetchone()
        if row is None:
            raise KeyError(study_id)
        return self._study_row(row)

    def list_studies(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        with self._lock, closing(self._connect()) as con:
            rows = con.execute(
                "SELECT * FROM validation_studies ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._study_row(row) for row in rows]

    def freeze(self, study_id: str) -> dict[str, Any]:
        study = self.get_study(study_id)
        if study.get("target_participants") is None:
            raise ValueError("target_participants deve ser definido antes de congelar o protocolo")
        if study.get("repetitions_per_participant") is None:
            raise ValueError("repetitions_per_participant deve ser definido antes de congelar o protocolo")
        if study.get("reference_method") == "physical_goniometer":
            if not study.get("reference_device_model"):
                raise ValueError("reference_device_model é obrigatório para referência por goniômetro físico")
            if study.get("reference_device_resolution_deg") is None:
                raise ValueError("reference_device_resolution_deg é obrigatório para referência por goniômetro físico")
        if study["status"] == "closed":
            raise ValueError("estudo fechado não pode ser congelado novamente")
        if study["status"] == "frozen":
            return study
        protocol = self._protocol_payload(study)
        digest = _canonical_sha256(protocol)
        now = _utcnow()
        with self._lock, closing(self._connect()) as con, con:
            con.execute(
                "UPDATE validation_studies SET status='frozen', frozen_at=?, protocol_sha256=? WHERE id=?",
                (now, digest, study_id),
            )
        return self.get_study(study_id)

    def close(self, study_id: str) -> dict[str, Any]:
        study = self.get_study(study_id)
        if study["status"] == "draft":
            raise ValueError("congele o protocolo antes de fechar o estudo")
        if study["status"] != "closed":
            with self._lock, closing(self._connect()) as con, con:
                con.execute(
                    "UPDATE validation_studies SET status='closed', closed_at=? WHERE id=?",
                    (_utcnow(), study_id),
                )
        return self.get_study(study_id)

    def next_trial_number(self, study_id: str, patient_id: str) -> int:
        with self._lock, closing(self._connect()) as con:
            value = con.execute(
                "SELECT COALESCE(MAX(trial_number), 0) FROM validation_trials WHERE study_id=? AND patient_id=?",
                (study_id, patient_id),
            ).fetchone()[0]
        return int(value) + 1

    def add_trial(self, payload: dict[str, Any]) -> dict[str, Any]:
        study = self.get_study(str(payload["study_id"]))
        if study["status"] != "frozen":
            raise ValueError("o protocolo deve estar congelado para registrar ensaios")
        movement = str(payload["movement"])
        side = str(payload["side"]).lower()
        if movement != study["movement"]:
            raise ValueError("movimento da sessão não corresponde ao protocolo")
        if study["side"] != "ambos" and side != study["side"]:
            raise ValueError("lado da sessão não corresponde ao protocolo")

        auto = float(payload["automatic_angle_deg"])
        ref = float(payload["reference_angle_deg"])
        if not (math.isfinite(auto) and 0.0 <= auto <= 180.0):
            raise ValueError("automatic_angle_deg deve estar entre 0 e 180")
        if not (math.isfinite(ref) and 0.0 <= ref <= 180.0):
            raise ValueError("reference_angle_deg deve estar entre 0 e 180")

        confidence = payload.get("landmark_confidence_min")
        confidence = None if confidence is None else float(confidence)
        source = str(payload.get("source") or "unknown")
        measurement_valid = bool(payload.get("measurement_valid"))
        exclusion: list[str] = []
        if study["require_observed"] and source != "observed":
            exclusion.append("source_not_observed")
        if study["require_valid_measurement"] and not measurement_valid:
            exclusion.append("measurement_invalid")
        if confidence is None or confidence < float(study["min_landmark_confidence"]):
            exclusion.append("landmark_confidence_below_protocol")

        force_exclude = str(payload.get("manual_exclusion_reason") or "").strip()
        if force_exclude:
            exclusion.append(force_exclude)
        included = len(exclusion) == 0

        patient_id = str(payload["patient_id"])
        trial_number = int(payload.get("trial_number") or self.next_trial_number(study["id"], patient_id))
        row = {
            "id": str(uuid.uuid4()),
            "schema_version": STUDY_SCHEMA_VERSION,
            "study_id": study["id"],
            "created_at": _utcnow(),
            "job_id": str(payload["job_id"]),
            "patient_id": patient_id,
            "evaluator_id": str(payload["evaluator_id"]),
            "reference_assessor_id": str(payload.get("reference_assessor_id") or payload["evaluator_id"]),
            "movement": movement,
            "side": side,
            "trial_number": trial_number,
            "frame": int(payload["frame"]),
            "timestamp_ms": float(payload["timestamp_ms"]),
            "automatic_angle_deg": auto,
            "reference_angle_deg": ref,
            "reference_method": str(payload.get("reference_method") or study["reference_method"]),
            "reference_device_model": str(payload.get("reference_device_model") or study.get("reference_device_model") or "") or None,
            "reference_device_serial": str(payload.get("reference_device_serial") or "").strip() or None,
            "landmark_confidence_min": confidence,
            "body_frame_confidence": payload.get("body_frame_confidence"),
            "measurement_plane": payload.get("measurement_plane"),
            "view_alignment": payload.get("view_alignment"),
            "view_deviation_deg": payload.get("view_deviation_deg"),
            "out_of_plane_deg": payload.get("out_of_plane_deg"),
            "direction_rule": payload.get("direction_rule"),
            "direction_alignment": payload.get("direction_alignment"),
            "source": source,
            "measurement_valid": int(measurement_valid),
            "quality": str(payload.get("quality") or "unavailable"),
            "included": int(included),
            "exclusion_reason": ";".join(exclusion) if exclusion else None,
            "notes": str(payload.get("notes") or ""),
        }
        if row["frame"] < 0 or trial_number <= 0:
            raise ValueError("frame e trial_number devem ser válidos")

        with self._lock, closing(self._connect()) as con, con:
            try:
                con.execute(
                    """
                    INSERT INTO validation_trials (
                        id, schema_version, study_id, created_at, job_id, patient_id,
                        evaluator_id, reference_assessor_id, movement, side, trial_number,
                        frame, timestamp_ms, automatic_angle_deg, reference_angle_deg,
                        reference_method, reference_device_model, reference_device_serial,
                        landmark_confidence_min, body_frame_confidence, measurement_plane,
                        view_alignment, view_deviation_deg, out_of_plane_deg,
                        direction_rule, direction_alignment,
                        source, measurement_valid, quality, included, exclusion_reason, notes
                    ) VALUES (
                        :id, :schema_version, :study_id, :created_at, :job_id, :patient_id,
                        :evaluator_id, :reference_assessor_id, :movement, :side, :trial_number,
                        :frame, :timestamp_ms, :automatic_angle_deg, :reference_angle_deg,
                        :reference_method, :reference_device_model, :reference_device_serial,
                        :landmark_confidence_min, :body_frame_confidence, :measurement_plane,
                        :view_alignment, :view_deviation_deg, :out_of_plane_deg,
                        :direction_rule, :direction_alignment,
                        :source, :measurement_valid, :quality, :included, :exclusion_reason, :notes
                    )
                    """,
                    row,
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("este frame já foi registrado neste estudo") from exc
        return self.get_trial(row["id"])

    def get_trial(self, trial_id: str) -> dict[str, Any]:
        with self._lock, closing(self._connect()) as con:
            row = con.execute("SELECT * FROM validation_trials WHERE id=?", (trial_id,)).fetchone()
        if row is None:
            raise KeyError(trial_id)
        return self._trial_row(row)

    def trials(self, study_id: str, *, included_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM validation_trials WHERE study_id=?"
        args: list[Any] = [study_id]
        if included_only:
            sql += " AND included=1"
        sql += " ORDER BY patient_id, trial_number, created_at"
        with self._lock, closing(self._connect()) as con:
            rows = con.execute(sql, args).fetchall()
        return [self._trial_row(row) for row in rows]

    def for_job(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock, closing(self._connect()) as con:
            rows = con.execute(
                "SELECT * FROM validation_trials WHERE job_id=? ORDER BY created_at",
                (job_id,),
            ).fetchall()
        return [self._trial_row(row) for row in rows]

    def report(self, study_id: str, *, bootstrap_iterations: int = 1000) -> dict[str, Any]:
        study = self.get_study(study_id)
        rows = self.trials(study_id)
        included = [row for row in rows if row["included"]]
        pairs = [(float(row["automatic_angle_deg"]), float(row["reference_angle_deg"])) for row in included]
        patients = {row["patient_id"] for row in rows}
        included_patients = {row["patient_id"] for row in included}
        sessions = {row["job_id"] for row in rows}
        reasons = Counter(
            reason
            for row in rows
            if not row["included"] and row.get("exclusion_reason")
            for reason in str(row["exclusion_reason"]).split(";")
            if reason
        )
        by_side: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in included:
            by_side[row["side"]].append((row["automatic_angle_deg"], row["reference_angle_deg"]))

        target = study.get("target_participants")
        completion = None
        if target:
            completion = min(len(included_patients) / int(target) * 100.0, 100.0)

        return {
            "study": study,
            "counts": {
                "participants_total": len(patients),
                "participants_included": len(included_patients),
                "sessions_total": len(sessions),
                "trials_total": len(rows),
                "trials_included": len(included),
                "trials_excluded": len(rows) - len(included),
                "completion_pct": None if completion is None else round(completion, 1),
            },
            "agreement": agreement_metrics(pairs),
            "participant_balanced": participant_balanced_summary(included),
            "cluster_bootstrap": cluster_bootstrap_ci(
                included, iterations=bootstrap_iterations
            ),
            "by_reference_angle": stratified_reference_metrics(included),
            "by_side": {side: agreement_metrics(values) for side, values in sorted(by_side.items())},
            "exclusion_reasons": dict(sorted(reasons.items())),
            "bland_altman": bland_altman_points(pairs),
            "methodology": {
                "primary_error_unit": "degrees",
                "signed_error": "automatic - physical_reference",
                "primary_descriptive_metric": "MAE in included protocol-compliant trials",
                "participant_balance": "participant-level MAE/bias/RMSE reported separately so repeated trials do not masquerade as independent participants",
                "bootstrap": "95% percentile CI with participant-cluster resampling; all trials from a sampled participant stay together",
                "warning": "This report is pre-clinical method-comparison support. Inferential interpretation still depends on the approved study design, recruitment strategy and sample size.",
            },
        }

    def export_trials(self, study_id: str) -> list[dict[str, Any]]:
        return self.trials(study_id)

    def delete_patient(self, patient_id: str) -> int:
        with self._lock, closing(self._connect()) as con, con:
            cur = con.execute("DELETE FROM validation_trials WHERE patient_id=?", (patient_id,))
            return int(cur.rowcount)

    @staticmethod
    def _study_row(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["require_observed"] = bool(out["require_observed"])
        out["require_valid_measurement"] = bool(out["require_valid_measurement"])
        try:
            out["camera_setup"] = json.loads(out.pop("camera_setup_json") or "{}")
        except json.JSONDecodeError:
            out["camera_setup"] = {}
        return out

    @staticmethod
    def _trial_row(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["measurement_valid"] = bool(out["measurement_valid"])
        out["included"] = bool(out["included"])
        out["error_deg"] = round(float(out["automatic_angle_deg"]) - float(out["reference_angle_deg"]), 4)
        out["absolute_error_deg"] = round(abs(out["error_deg"]), 4)
        return out
