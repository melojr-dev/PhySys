from __future__ import annotations

from contextlib import closing
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import uuid
from typing import Any

FORMAL_RUN_SCHEMA_VERSION = 1
_ALLOWED_ETHICS = {"pending", "approved", "not_required"}
_ALLOWED_STATUS = {"open", "closed"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def formal_readiness(
    *,
    study: dict[str, Any],
    ethics_status: str,
    ethics_protocol_id: str | None,
    calibration_date: str | None,
    calibration_due_date: str | None,
    statistical_plan_path: str | os.PathLike[str],
    protocol_lock_path: str | os.PathLike[str],
    software_version: str,
    clinical_protocol_version: str,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(code: str, ok: bool, detail: str) -> None:
        checks.append({"code": code, "ok": bool(ok), "detail": detail})

    add("study_frozen", study.get("status") == "frozen", "O estudo precisa estar congelado antes da coleta formal.")
    add("study_protocol_hash", bool(study.get("protocol_sha256")), "O protocolo congelado precisa possuir SHA-256.")
    add("target_participants", bool(study.get("target_participants")), "Defina o número-alvo de participantes.")
    add("repetitions", bool(study.get("repetitions_per_participant")), "Defina repetições por participante.")
    add("reference_device_model", bool(study.get("reference_device_model")), "Informe o modelo do instrumento de referência.")
    add(
        "reference_device_resolution",
        study.get("reference_device_resolution_deg") is not None,
        "Informe a resolução do instrumento de referência.",
    )
    add("camera_setup", bool(study.get("camera_setup")), "Defina o setup de câmera no protocolo congelado.")
    add(
        "protocol_version_match",
        str(study.get("protocol_version")) == str(clinical_protocol_version),
        f"O estudo formal deve usar o protocolo clínico corrente {clinical_protocol_version}.",
    )

    ethics_status = str(ethics_status or "pending").strip().lower()
    add("ethics_status_valid", ethics_status in _ALLOWED_ETHICS, "Situação ética inválida.")
    ethics_ok = ethics_status == "not_required" or (ethics_status == "approved" and bool(str(ethics_protocol_id or "").strip()))
    add(
        "ethics_gate",
        ethics_ok,
        "Para participantes humanos, registre aprovação ética e identificador do protocolo; 'not_required' só deve ser usado quando realmente aplicável.",
    )

    try:
        calibration = date.fromisoformat(calibration_date) if calibration_date else None
        due = date.fromisoformat(calibration_due_date) if calibration_due_date else None
        calibration_format_ok = True
    except ValueError:
        calibration = due = None
        calibration_format_ok = False
    add("calibration_dates_valid", calibration_format_ok, "Datas de calibração devem usar YYYY-MM-DD.")
    add("reference_calibrated", calibration is not None and due is not None, "Registre data e validade da calibração do instrumento.")
    add(
        "reference_calibration_current",
        due is not None and due >= date.today(),
        "A calibração do instrumento deve estar válida na data da coleta.",
    )

    statistical_plan = Path(statistical_plan_path)
    protocol_lock = Path(protocol_lock_path)
    add("statistical_plan_present", statistical_plan.is_file(), "Plano estatístico versionado ausente.")
    add("protocol_lock_present", protocol_lock.is_file(), "Lock do protocolo clínico ausente.")

    plan = load_json(statistical_plan) if statistical_plan.is_file() else {}
    lock = load_json(protocol_lock) if protocol_lock.is_file() else {}
    add(
        "statistical_plan_protocol_match",
        str(plan.get("clinical_protocol_version")) == str(clinical_protocol_version),
        "Plano estatístico e protocolo clínico precisam referenciar a mesma versão.",
    )
    add(
        "protocol_lock_version_match",
        str(lock.get("protocol_version")) == str(clinical_protocol_version),
        "Protocol lock incompatível com o protocolo clínico corrente.",
    )

    return {
        "ready": all(item["ok"] for item in checks),
        "checks": checks,
        "software_version": software_version,
        "clinical_protocol_version": clinical_protocol_version,
        "study_protocol_sha256": study.get("protocol_sha256"),
        "statistical_plan_sha256": file_sha256(statistical_plan) if statistical_plan.is_file() else None,
        "protocol_lock_sha256": file_sha256(protocol_lock) if protocol_lock.is_file() else None,
    }


class FormalValidationRunStore:
    """Immutable start snapshot + close snapshot for a formal validation collection run."""

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
                CREATE TABLE IF NOT EXISTS formal_validation_runs (
                    id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    study_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    closed_at TEXT,
                    created_by TEXT NOT NULL,
                    status TEXT NOT NULL,
                    software_version TEXT NOT NULL,
                    clinical_protocol_version TEXT NOT NULL,
                    study_protocol_sha256 TEXT NOT NULL,
                    statistical_plan_sha256 TEXT NOT NULL,
                    protocol_lock_sha256 TEXT NOT NULL,
                    ethics_status TEXT NOT NULL,
                    ethics_protocol_id TEXT,
                    reference_calibration_date TEXT NOT NULL,
                    reference_calibration_due_date TEXT NOT NULL,
                    start_trial_count INTEGER NOT NULL,
                    end_trial_count INTEGER,
                    snapshot_json TEXT NOT NULL,
                    snapshot_sha256 TEXT NOT NULL,
                    close_report_json TEXT,
                    close_report_sha256 TEXT,
                    notes TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_formal_runs_study ON formal_validation_runs(study_id, created_at);
                """
            )

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = dict(payload["snapshot"])
        snapshot_sha = canonical_sha256(snapshot)
        run_id = str(uuid.uuid4())
        row = {
            "id": run_id,
            "schema_version": FORMAL_RUN_SCHEMA_VERSION,
            "study_id": str(payload["study_id"]),
            "created_at": _utcnow(),
            "closed_at": None,
            "created_by": str(payload["created_by"]),
            "status": "open",
            "software_version": str(payload["software_version"]),
            "clinical_protocol_version": str(payload["clinical_protocol_version"]),
            "study_protocol_sha256": str(payload["study_protocol_sha256"]),
            "statistical_plan_sha256": str(payload["statistical_plan_sha256"]),
            "protocol_lock_sha256": str(payload["protocol_lock_sha256"]),
            "ethics_status": str(payload["ethics_status"]),
            "ethics_protocol_id": (str(payload.get("ethics_protocol_id") or "").strip() or None),
            "reference_calibration_date": str(payload["reference_calibration_date"]),
            "reference_calibration_due_date": str(payload["reference_calibration_due_date"]),
            "start_trial_count": int(payload.get("start_trial_count", 0)),
            "end_trial_count": None,
            "snapshot_json": json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "snapshot_sha256": snapshot_sha,
            "close_report_json": None,
            "close_report_sha256": None,
            "notes": str(payload.get("notes") or ""),
        }
        with self._lock, closing(self._connect()) as con, con:
            open_run = con.execute(
                "SELECT id FROM formal_validation_runs WHERE study_id=? AND status='open'",
                (row["study_id"],),
            ).fetchone()
            if open_run is not None:
                raise ValueError("Já existe uma execução formal aberta para este estudo")
            con.execute(
                """
                INSERT INTO formal_validation_runs (
                    id,schema_version,study_id,created_at,closed_at,created_by,status,
                    software_version,clinical_protocol_version,study_protocol_sha256,
                    statistical_plan_sha256,protocol_lock_sha256,ethics_status,ethics_protocol_id,
                    reference_calibration_date,reference_calibration_due_date,start_trial_count,end_trial_count,
                    snapshot_json,snapshot_sha256,close_report_json,close_report_sha256,notes
                ) VALUES (
                    :id,:schema_version,:study_id,:created_at,:closed_at,:created_by,:status,
                    :software_version,:clinical_protocol_version,:study_protocol_sha256,
                    :statistical_plan_sha256,:protocol_lock_sha256,:ethics_status,:ethics_protocol_id,
                    :reference_calibration_date,:reference_calibration_due_date,:start_trial_count,:end_trial_count,
                    :snapshot_json,:snapshot_sha256,:close_report_json,:close_report_sha256,:notes
                )
                """,
                row,
            )
        return self.get(run_id)

    def get(self, run_id: str) -> dict[str, Any]:
        with self._lock, closing(self._connect()) as con:
            row = con.execute("SELECT * FROM formal_validation_runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._row(row)

    def list_for_study(self, study_id: str) -> list[dict[str, Any]]:
        with self._lock, closing(self._connect()) as con:
            rows = con.execute(
                "SELECT * FROM formal_validation_runs WHERE study_id=? ORDER BY created_at DESC", (study_id,)
            ).fetchall()
        return [self._row(row) for row in rows]

    def close(self, run_id: str, *, end_trial_count: int, report: dict[str, Any]) -> dict[str, Any]:
        current = self.get(run_id)
        if current["status"] != "open":
            raise ValueError("A execução formal já está encerrada")
        report_json = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._lock, closing(self._connect()) as con, con:
            con.execute(
                """
                UPDATE formal_validation_runs
                   SET status='closed', closed_at=?, end_trial_count=?, close_report_json=?, close_report_sha256=?
                 WHERE id=?
                """,
                (_utcnow(), int(end_trial_count), report_json, hashlib.sha256(report_json.encode()).hexdigest(), run_id),
            )
        return self.get(run_id)

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["snapshot"] = json.loads(out.pop("snapshot_json"))
        report_raw = out.pop("close_report_json")
        out["close_report"] = json.loads(report_raw) if report_raw else None
        out["trials_collected"] = (
            None if out.get("end_trial_count") is None else int(out["end_trial_count"]) - int(out["start_trial_count"])
        )
        return out


def report_for_formal_run(study: dict[str, Any], rows: list[dict[str, Any]], *, bootstrap_iterations: int = 2000) -> dict[str, Any]:
    from collections import Counter, defaultdict
    from .agreement import agreement_metrics, bland_altman_points
    from .study import participant_balanced_summary, cluster_bootstrap_ci, stratified_reference_metrics

    included = [row for row in rows if row.get("included")]
    pairs = [(float(r["automatic_angle_deg"]), float(r["reference_angle_deg"])) for r in included]
    participants = {str(r["patient_id"]) for r in rows}
    included_participants = {str(r["patient_id"]) for r in included}
    sessions = {str(r["job_id"]) for r in rows}
    reasons = Counter(
        reason
        for row in rows
        if not row.get("included") and row.get("exclusion_reason")
        for reason in str(row["exclusion_reason"]).split(";")
        if reason
    )
    by_side: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in included:
        by_side[str(row["side"])].append((float(row["automatic_angle_deg"]), float(row["reference_angle_deg"])))
    target = study.get("target_participants")
    completion = None if not target else min(len(included_participants) / int(target) * 100.0, 100.0)
    return {
        "formal_run_scope": "trials_created_during_run_only",
        "study_id": study["id"],
        "study_protocol_sha256": study.get("protocol_sha256"),
        "counts": {
            "participants_total": len(participants),
            "participants_included": len(included_participants),
            "sessions_total": len(sessions),
            "trials_total": len(rows),
            "trials_included": len(included),
            "trials_excluded": len(rows) - len(included),
            "completion_pct": None if completion is None else round(completion, 1),
        },
        "agreement": agreement_metrics(pairs),
        "participant_balanced": participant_balanced_summary(included),
        "cluster_bootstrap": cluster_bootstrap_ci(included, iterations=bootstrap_iterations),
        "by_reference_angle": stratified_reference_metrics(included),
        "by_side": {side: agreement_metrics(values) for side, values in sorted(by_side.items())},
        "exclusion_reasons": dict(sorted(reasons.items())),
        "bland_altman": bland_altman_points(pairs),
        "interpretation_boundary": "Method-comparison output only; no diagnostic or clinical-effectiveness claim is generated.",
    }
