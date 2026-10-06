from __future__ import annotations

from contextlib import closing

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ClinicalSessionStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=15.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=15000")
        return con

    @staticmethod
    def _columns(con: sqlite3.Connection, table: str) -> set[str]:
        return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})").fetchall()}

    def _initialize(self) -> None:
        with self._lock, closing(self._connect()) as con, con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS clinical_sessions (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL UNIQUE,
                    patient_pseudonym TEXT NOT NULL,
                    evaluator_id TEXT NOT NULL,
                    movement TEXT NOT NULL,
                    side TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    total_frames INTEGER,
                    valid_frames INTEGER,
                    fps REAL,
                    adm_deg REAL,
                    total_reps INTEGER,
                    software_version TEXT,
                    protocol_version TEXT,
                    clinical_spec_sha256 TEXT,
                    result_sha256 TEXT,
                    provenance_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_patient ON clinical_sessions(patient_pseudonym);
                CREATE INDEX IF NOT EXISTS idx_sessions_started ON clinical_sessions(started_at);
                CREATE INDEX IF NOT EXISTS idx_sessions_evaluator ON clinical_sessions(evaluator_id);
                """
            )
            # Migração aditiva para bancos criados pelas Fases 6–8.
            existing = self._columns(con, "clinical_sessions")
            additions = {
                "valid_frames": "INTEGER",
                "fps": "REAL",
                "total_reps": "INTEGER",
                "software_version": "TEXT",
                "protocol_version": "TEXT",
                "clinical_spec_sha256": "TEXT",
                "result_sha256": "TEXT",
                "provenance_json": "TEXT NOT NULL DEFAULT '{}'",
            }
            for name, sql_type in additions.items():
                if name not in existing:
                    con.execute(f"ALTER TABLE clinical_sessions ADD COLUMN {name} {sql_type}")

    def start(
        self,
        *,
        job_id: str,
        patient_pseudonym: str,
        evaluator_id: str,
        movement: str,
        side: str,
        software_version: str | None = None,
        protocol_version: str | None = None,
        clinical_spec_sha256: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "id": str(uuid.uuid4()),
            "job_id": job_id,
            "patient_pseudonym": patient_pseudonym,
            "evaluator_id": evaluator_id,
            "movement": movement,
            "side": side,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "processing",
            "software_version": software_version,
            "protocol_version": protocol_version,
            "clinical_spec_sha256": clinical_spec_sha256,
            "provenance_json": json.dumps(
                provenance or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        }
        with self._lock, closing(self._connect()) as con, con:
            con.execute(
                """
                INSERT INTO clinical_sessions (
                    id, job_id, patient_pseudonym, evaluator_id, movement, side,
                    started_at, status, software_version, protocol_version,
                    clinical_spec_sha256, provenance_json
                ) VALUES (
                    :id, :job_id, :patient_pseudonym, :evaluator_id, :movement, :side,
                    :started_at, :status, :software_version, :protocol_version,
                    :clinical_spec_sha256, :provenance_json
                )
                """,
                payload,
            )
        return self._public(payload)

    def finish(
        self,
        job_id: str,
        *,
        status: str,
        total_frames: int | None = None,
        valid_frames: int | None = None,
        fps: float | None = None,
        adm_deg: float | None = None,
        total_reps: int | None = None,
        result_sha256: str | None = None,
    ) -> None:
        with self._lock, closing(self._connect()) as con, con:
            con.execute(
                """
                UPDATE clinical_sessions
                SET completed_at = ?, status = ?, total_frames = ?, valid_frames = ?,
                    fps = ?, adm_deg = ?, total_reps = ?, result_sha256 = ?
                WHERE job_id = ?
                """,
                (
                    datetime.now(timezone.utc).isoformat(), status, total_frames,
                    valid_frames, fps, adm_deg, total_reps, result_sha256, job_id,
                ),
            )

    def get_by_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as con:
            row = con.execute(
                "SELECT * FROM clinical_sessions WHERE job_id = ?", (job_id,)
            ).fetchone()
        return None if row is None else self._row(row)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        with self._lock, closing(self._connect()) as con:
            rows = con.execute(
                "SELECT * FROM clinical_sessions ORDER BY started_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def recent_for_evaluator(self, evaluator_id: str, limit: int = 50) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        with self._lock, closing(self._connect()) as con:
            rows = con.execute(
                "SELECT * FROM clinical_sessions WHERE evaluator_id = ? ORDER BY started_at DESC LIMIT ?",
                (evaluator_id, safe_limit),
            ).fetchall()
        return [self._row(row) for row in rows]

    def for_patient(self, patient_pseudonym: str, limit: int = 500) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 5000))
        with self._lock, closing(self._connect()) as con:
            rows = con.execute(
                "SELECT * FROM clinical_sessions WHERE patient_pseudonym = ? ORDER BY started_at LIMIT ?",
                (patient_pseudonym, safe_limit),
            ).fetchall()
        return [self._row(row) for row in rows]

    def jobs_for_patient(self, patient_pseudonym: str) -> list[str]:
        with self._lock, closing(self._connect()) as con:
            rows = con.execute(
                "SELECT job_id FROM clinical_sessions WHERE patient_pseudonym = ?",
                (patient_pseudonym,),
            ).fetchall()
        return [str(row["job_id"]) for row in rows]

    def delete_patient(self, patient_pseudonym: str) -> int:
        with self._lock, closing(self._connect()) as con, con:
            cur = con.execute(
                "DELETE FROM clinical_sessions WHERE patient_pseudonym = ?",
                (patient_pseudonym,),
            )
            return int(cur.rowcount)

    @staticmethod
    def _public(payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        raw = result.pop("provenance_json", None)
        if raw is not None:
            try:
                result["provenance"] = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                result["provenance"] = {}
        return result

    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        return self._public(dict(row))
