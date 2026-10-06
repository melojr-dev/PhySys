from __future__ import annotations

from contextlib import closing

import json
import math
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agreement import agreement_metrics, bland_altman_points


SCHEMA_VERSION = 1


class ValidationStore:
    """Persistência local e determinística das revisões goniométricas.

    SQLite substitui CSV para os dados de validação porque garante schema,
    transações e consultas estatísticas sem descartar linhas silenciosamente.
    O arquivo fica em ``src/utils/physioai_clinical.sqlite3`` e já é persistido
    pelo volume Docker usado pelo projeto.
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
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=15000")
        return con

    def _initialize(self) -> None:
        with self._lock, closing(self._connect()) as con, con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS validation_reviews (
                    id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    job_id TEXT,
                    patient_id TEXT NOT NULL,
                    evaluator_id TEXT,
                    movement TEXT NOT NULL,
                    side TEXT NOT NULL,
                    frame INTEGER NOT NULL,
                    timestamp_ms REAL NOT NULL,
                    automatic_angle_deg REAL NOT NULL,
                    manual_angle_deg REAL,
                    reference_angle_deg REAL,
                    manual_method TEXT,
                    reference_method TEXT,
                    confidence REAL,
                    source TEXT,
                    proximal_json TEXT,
                    vertex_json TEXT,
                    distal_json TEXT,
                    notes TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_validation_job_frame
                    ON validation_reviews(job_id, frame);
                CREATE INDEX IF NOT EXISTS idx_validation_patient
                    ON validation_reviews(patient_id);
                CREATE INDEX IF NOT EXISTS idx_validation_movement
                    ON validation_reviews(movement, side);
                """
            )
            existing = {str(row[1]) for row in con.execute("PRAGMA table_info(validation_reviews)").fetchall()}
            if "manual_method" not in existing:
                con.execute("ALTER TABLE validation_reviews ADD COLUMN manual_method TEXT")
            if "reference_method" not in existing:
                con.execute("ALTER TABLE validation_reviews ADD COLUMN reference_method TEXT")

    @staticmethod
    def _point_json(value: dict[str, Any] | None) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _point_from_json(value: str | None) -> dict[str, Any] | None:
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    def save_review(self, review: dict[str, Any]) -> dict[str, Any]:
        review_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "id": review_id,
            "schema_version": SCHEMA_VERSION,
            "created_at": created_at,
            "job_id": review.get("job_id"),
            "patient_id": str(review["patient_id"]).strip(),
            "evaluator_id": (str(review.get("evaluator_id") or "").strip() or None),
            "movement": str(review["movement"]),
            "side": str(review["side"]),
            "frame": int(review["frame"]),
            "timestamp_ms": float(review["timestamp_ms"]),
            "automatic_angle_deg": float(review["automatic_angle_deg"]),
            "manual_angle_deg": (
                None if review.get("manual_angle_deg") is None else float(review["manual_angle_deg"])
            ),
            "reference_angle_deg": (
                None if review.get("reference_angle_deg") is None else float(review["reference_angle_deg"])
            ),
            "manual_method": review.get("manual_method"),
            "reference_method": review.get("reference_method"),
            "confidence": None if review.get("confidence") is None else float(review["confidence"]),
            "source": review.get("source"),
            "proximal_json": self._point_json(review.get("proximal")),
            "vertex_json": self._point_json(review.get("vertex")),
            "distal_json": self._point_json(review.get("distal")),
            "notes": str(review.get("notes") or ""),
        }

        if not payload["patient_id"]:
            raise ValueError("patient_id não pode ser vazio")
        if payload["frame"] < 0:
            raise ValueError("frame deve ser >= 0")
        for key in ("automatic_angle_deg", "manual_angle_deg", "reference_angle_deg"):
            value = payload[key]
            if value is not None and (not math.isfinite(value) or value < 0.0 or value > 180.0):
                raise ValueError(f"{key} deve estar entre 0 e 180 graus")
        conf = payload["confidence"]
        if conf is not None and (not math.isfinite(conf) or conf < 0.0 or conf > 1.0):
            raise ValueError("confidence deve estar entre 0 e 1")

        with self._lock, closing(self._connect()) as con, con:
            con.execute(
                """
                INSERT INTO validation_reviews (
                    id, schema_version, created_at, job_id, patient_id,
                    evaluator_id, movement, side, frame, timestamp_ms,
                    automatic_angle_deg, manual_angle_deg, reference_angle_deg,
                    manual_method, reference_method, confidence, source,
                    proximal_json, vertex_json, distal_json, notes
                ) VALUES (
                    :id, :schema_version, :created_at, :job_id, :patient_id,
                    :evaluator_id, :movement, :side, :frame, :timestamp_ms,
                    :automatic_angle_deg, :manual_angle_deg, :reference_angle_deg,
                    :manual_method, :reference_method, :confidence, :source,
                    :proximal_json, :vertex_json, :distal_json, :notes
                )
                """,
                payload,
            )
        return self.get_review(review_id)

    def get_review(self, review_id: str) -> dict[str, Any]:
        with self._lock, closing(self._connect()) as con, con:
            row = con.execute(
                "SELECT * FROM validation_reviews WHERE id = ?", (review_id,)
            ).fetchone()
        if row is None:
            raise KeyError(review_id)
        return self._row_to_dict(row)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 200))
        with self._lock, closing(self._connect()) as con, con:
            rows = con.execute(
                "SELECT * FROM validation_reviews ORDER BY created_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def _select_rows(self, movement: str | None = None, side: str | None = None) -> list[sqlite3.Row]:
        where: list[str] = []
        args: list[Any] = []
        if movement:
            where.append("movement = ?")
            args.append(movement)
        if side:
            where.append("side = ?")
            args.append(side)
        sql = "SELECT * FROM validation_reviews"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at ASC"
        with self._lock, closing(self._connect()) as con, con:
            return con.execute(sql, args).fetchall()

    @staticmethod
    def _pairs(rows: list[sqlite3.Row], comparator: str) -> list[tuple[float, float]]:
        if comparator not in {"manual_angle_deg", "reference_angle_deg"}:
            raise ValueError("comparador inválido")
        pairs: list[tuple[float, float]] = []
        for row in rows:
            value = row[comparator]
            if value is not None:
                pairs.append((float(row["automatic_angle_deg"]), float(value)))
        return pairs

    def stats(self, movement: str | None = None, side: str | None = None) -> dict[str, Any]:
        rows = self._select_rows(movement, side)
        manual_pairs = self._pairs(rows, "manual_angle_deg")
        reference_pairs = self._pairs(rows, "reference_angle_deg")
        patients = {str(row["patient_id"]) for row in rows}
        jobs = {str(row["job_id"]) for row in rows if row["job_id"]}
        evaluators = {str(row["evaluator_id"]) for row in rows if row["evaluator_id"]}
        return {
            "total_reviews": len(rows),
            "unique_patients": len(patients),
            "unique_sessions": len(jobs),
            "unique_evaluators": len(evaluators),
            "movement": movement,
            "side": side,
            "automatic_vs_manual": agreement_metrics(manual_pairs),
            "automatic_vs_reference": agreement_metrics(reference_pairs),
        }

    def report(self, movement: str | None = None, side: str | None = None, max_points: int = 2000) -> dict[str, Any]:
        rows = self._select_rows(movement, side)
        manual_pairs = self._pairs(rows, "manual_angle_deg")
        reference_pairs = self._pairs(rows, "reference_angle_deg")
        stats = self.stats(movement, side)
        max_points = max(1, min(int(max_points), 10000))
        return {
            **stats,
            "bland_altman_manual": bland_altman_points(manual_pairs)[-max_points:],
            "bland_altman_reference": bland_altman_points(reference_pairs)[-max_points:],
            "methodology": {
                "signed_error": "automatic - comparator",
                "icc": "ICC(A,1), two-way absolute agreement, single measurement",
                "ccc": "Lin concordance correlation coefficient",
                "loa": "bias ± 1.96 × sample SD of signed differences",
                "warning": "Frames from the same patient/session are repeated observations and must not be treated as independent participants in inferential analysis.",
            },
        }

    def export_rows(self, movement: str | None = None, side: str | None = None) -> list[dict[str, Any]]:
        return [self._row_to_dict(row) for row in self._select_rows(movement, side)]

    def for_job(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock, closing(self._connect()) as con:
            rows = con.execute(
                "SELECT * FROM validation_reviews WHERE job_id = ? ORDER BY frame, created_at",
                (job_id,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for name in ("proximal", "vertex", "distal"):
            result[name] = self._point_from_json(result.pop(f"{name}_json", None))
        auto = float(result["automatic_angle_deg"])
        manual = result.get("manual_angle_deg")
        reference = result.get("reference_angle_deg")
        result["error_vs_manual_deg"] = (
            None if manual is None else round(auto - float(manual), 4)
        )
        result["error_vs_reference_deg"] = (
            None if reference is None else round(auto - float(reference), 4)
        )
        return result

    def delete_patient(self, patient_id: str) -> int:
        with self._lock, closing(self._connect()) as con, con:
            cur = con.execute(
                "DELETE FROM validation_reviews WHERE patient_id = ?",
                (patient_id,),
            )
            return int(cur.rowcount)
