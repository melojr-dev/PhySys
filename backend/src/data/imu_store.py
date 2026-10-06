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
import zlib


IMU_STORE_SCHEMA_VERSION = 1


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


class IMUAnalysisStore:
    """Persistência experimental do sinal inercial e do resumo analítico.

    O sinal bruto é mantido comprimido no SQLite para que o resultado espectral
    possa ser reproduzido. A API pública retorna apenas metadados/resumo por
    padrão; o payload bruto exige endpoint de exportação autenticado.
    """

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
                CREATE TABLE IF NOT EXISTS imu_analyses (
                    id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    job_id TEXT,
                    device_label TEXT,
                    sample_count INTEGER NOT NULL,
                    duration_s REAL NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    payload_zlib BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_imu_job ON imu_analyses(job_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_imu_owner ON imu_analyses(owner, created_at);
                """
            )

    def save(
        self,
        *,
        owner: str,
        job_id: str | None,
        device_label: str | None,
        raw_payload: dict[str, Any],
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        payload_bytes = _canonical_bytes(raw_payload)
        digest = hashlib.sha256(payload_bytes).hexdigest()
        record = {
            "id": str(uuid.uuid4()),
            "schema_version": IMU_STORE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "owner": owner,
            "job_id": job_id,
            "device_label": (device_label or "").strip() or None,
            "sample_count": int(summary.get("sample_count") or 0),
            "duration_s": float(summary.get("duration_s") or 0.0),
            "input_sha256": digest,
            "summary_json": json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False),
            "payload_zlib": zlib.compress(payload_bytes, level=9),
        }
        with self._lock, closing(self._connect()) as con, con:
            con.execute(
                """
                INSERT INTO imu_analyses (
                    id, schema_version, created_at, owner, job_id, device_label,
                    sample_count, duration_s, input_sha256, summary_json, payload_zlib
                ) VALUES (
                    :id, :schema_version, :created_at, :owner, :job_id, :device_label,
                    :sample_count, :duration_s, :input_sha256, :summary_json, :payload_zlib
                )
                """,
                record,
            )
        return self._public(record)

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as con:
            row = con.execute("SELECT * FROM imu_analyses WHERE id = ?", (analysis_id,)).fetchone()
        return None if row is None else self._row(row)

    def list_for_job(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock, closing(self._connect()) as con:
            rows = con.execute(
                "SELECT * FROM imu_analyses WHERE job_id = ? ORDER BY created_at", (job_id,)
            ).fetchall()
        return [self._row(row) for row in rows]

    def raw_payload(self, analysis_id: str) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as con:
            row = con.execute(
                "SELECT payload_zlib FROM imu_analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
        if row is None:
            return None
        return json.loads(zlib.decompress(row["payload_zlib"]).decode("utf-8"))

    def delete_job(self, job_id: str) -> int:
        with self._lock, closing(self._connect()) as con, con:
            cur = con.execute("DELETE FROM imu_analyses WHERE job_id = ?", (job_id,))
            return int(cur.rowcount)

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        item = dict(record)
        item.pop("payload_zlib", None)
        raw_summary = item.pop("summary_json", None)
        if isinstance(raw_summary, str):
            item["summary"] = json.loads(raw_summary)
        return item

    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        return self._public(dict(row))
