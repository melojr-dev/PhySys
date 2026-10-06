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


class AuditStore:
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
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT,
                    resource_id TEXT,
                    status TEXT NOT NULL,
                    remote_ip TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_events(created_at);
                CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_events(actor);
                CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_events(action);
                """
            )

    def log(
        self,
        *,
        actor: str,
        role: str,
        action: str,
        status: str = "ok",
        resource_type: str | None = None,
        resource_id: str | None = None,
        remote_ip: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # Metadata deve ser operacional; não inserir nomes, vídeo ou texto clínico livre.
        payload = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "role": role,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "status": status,
            "remote_ip": remote_ip,
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":")),
        }
        with self._lock, closing(self._connect()) as con, con:
            con.execute(
                """
                INSERT INTO audit_events (
                    id, created_at, actor, role, action, resource_type,
                    resource_id, status, remote_ip, metadata_json
                ) VALUES (
                    :id, :created_at, :actor, :role, :action, :resource_type,
                    :resource_id, :status, :remote_ip, :metadata_json
                )
                """,
                payload,
            )

    @staticmethod
    def _redact_value(value: Any, target: str) -> Any:
        if isinstance(value, dict):
            return {k: AuditStore._redact_value(v, target) for k, v in value.items()}
        if isinstance(value, list):
            return [AuditStore._redact_value(v, target) for v in value]
        if isinstance(value, str) and value == target:
            return "[REDACTED]"
        return value

    def redact_patient(self, patient_pseudonym: str) -> int:
        """Remove referência reversível ao pseudônimo preservando a trilha operacional."""
        changed = 0
        with self._lock, closing(self._connect()) as con, con:
            rows = con.execute(
                "SELECT id, resource_type, resource_id, metadata_json FROM audit_events"
            ).fetchall()
            for row in rows:
                resource_id = row["resource_id"]
                try:
                    metadata = json.loads(row["metadata_json"] or "{}")
                except json.JSONDecodeError:
                    metadata = {}
                redacted = self._redact_value(metadata, patient_pseudonym)
                new_resource_id = (
                    "[REDACTED]"
                    if row["resource_type"] == "patient" and resource_id == patient_pseudonym
                    else resource_id
                )
                if redacted != metadata or new_resource_id != resource_id:
                    con.execute(
                        "UPDATE audit_events SET resource_id = ?, metadata_json = ? WHERE id = ?",
                        (
                            new_resource_id,
                            json.dumps(redacted, ensure_ascii=False, separators=(",", ":")),
                            row["id"],
                        ),
                    )
                    changed += 1
        return changed

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        with self._lock, closing(self._connect()) as con, con:
            rows = con.execute(
                "SELECT * FROM audit_events ORDER BY created_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["metadata"] = json.loads(item.pop("metadata_json"))
            except json.JSONDecodeError:
                item["metadata"] = {}
            out.append(item)
        return out
