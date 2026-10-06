from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any
import uuid

LONGITUDINAL_STORE_SCHEMA_VERSION = 1


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class LongitudinalAnalysisStore:
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

    def _initialize(self) -> None:
        with self._lock, closing(self._connect()) as con, con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS longitudinal_analyses (
                    id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    patient_pseudonym TEXT NOT NULL,
                    anchor_job_id TEXT NOT NULL,
                    movement TEXT NOT NULL,
                    side TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_longitudinal_patient
                    ON longitudinal_analyses(patient_pseudonym, movement, side, created_at);
                CREATE INDEX IF NOT EXISTS idx_longitudinal_anchor
                    ON longitudinal_analyses(anchor_job_id, created_at);
                """
            )

    def save(self, *, owner: str, patient_pseudonym: str, anchor_job_id: str,
             movement: str, side: str, source_payload: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
        record = {
            "id": str(uuid.uuid4()),
            "schema_version": LONGITUDINAL_STORE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "owner": owner,
            "patient_pseudonym": patient_pseudonym,
            "anchor_job_id": anchor_job_id,
            "movement": movement,
            "side": side,
            "source_sha256": hashlib.sha256(_canonical_bytes(source_payload)).hexdigest(),
            "source_json": json.dumps(source_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False),
            "summary_json": json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False),
        }
        with self._lock, closing(self._connect()) as con, con:
            con.execute(
                """INSERT INTO longitudinal_analyses (
                    id, schema_version, created_at, owner, patient_pseudonym,
                    anchor_job_id, movement, side, source_sha256, source_json, summary_json
                ) VALUES (
                    :id, :schema_version, :created_at, :owner, :patient_pseudonym,
                    :anchor_job_id, :movement, :side, :source_sha256, :source_json, :summary_json
                )""",
                record,
            )
        return self._public(record)

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as con:
            row = con.execute("SELECT * FROM longitudinal_analyses WHERE id = ?", (analysis_id,)).fetchone()
        return None if row is None else self._row(row)

    def list_for_job(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock, closing(self._connect()) as con:
            rows = con.execute(
                "SELECT * FROM longitudinal_analyses WHERE anchor_job_id = ? ORDER BY created_at", (job_id,)
            ).fetchall()
        return [self._row(row) for row in rows]

    def delete_patient(self, patient_pseudonym: str) -> int:
        with self._lock, closing(self._connect()) as con, con:
            cur = con.execute("DELETE FROM longitudinal_analyses WHERE patient_pseudonym = ?", (patient_pseudonym,))
            return int(cur.rowcount)

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        item = dict(record)
        for raw_key, out_key in (("source_json", "source"), ("summary_json", "summary")):
            raw = item.pop(raw_key, None)
            if isinstance(raw, str):
                item[out_key] = json.loads(raw)
        return item

    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        return self._public(dict(row))
