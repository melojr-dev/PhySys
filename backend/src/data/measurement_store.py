from __future__ import annotations

from contextlib import closing

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable


MEASUREMENT_SCHEMA_VERSION = 3


class MeasurementStore:
    """Persistência frame-a-frame para auditoria e reprodução da análise."""

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
                CREATE TABLE IF NOT EXISTS frame_measurements (
                    job_id TEXT NOT NULL,
                    frame INTEGER NOT NULL,
                    schema_version INTEGER NOT NULL,
                    timestamp_ms REAL NOT NULL,
                    joint TEXT NOT NULL,
                    movement TEXT NOT NULL,
                    side TEXT NOT NULL,
                    angle_raw_deg REAL,
                    angle_landmark_deg REAL,
                    angle_plane_deg REAL,
                    angle_filtered_deg REAL,
                    angle_clinical_deg REAL,
                    angular_velocity_deg_s REAL,
                    adm_deg REAL,
                    confidence REAL,
                    source TEXT NOT NULL,
                    measurement_space TEXT NOT NULL,
                    measurement_plane TEXT,
                    plane_projection_applied INTEGER NOT NULL DEFAULT 0,
                    body_frame_confidence REAL,
                    view_alignment REAL,
                    view_deviation_deg REAL,
                    out_of_plane_deg REAL,
                    direction_rule TEXT,
                    direction_alignment REAL,
                    landmark_confidence_min REAL,
                    landmark_confidence_mean REAL,
                    quality TEXT NOT NULL,
                    measurement_valid INTEGER NOT NULL,
                    quality_flags_json TEXT NOT NULL,
                    proximal_json TEXT,
                    vertex_json TEXT,
                    distal_json TEXT,
                    proximal_3d_json TEXT,
                    vertex_3d_json TEXT,
                    distal_3d_json TEXT,
                    inference_ms REAL,
                    pipeline_ms REAL,
                    processing_fps REAL,
                    PRIMARY KEY (job_id, frame)
                );
                CREATE INDEX IF NOT EXISTS idx_measurements_job_valid
                    ON frame_measurements(job_id, measurement_valid);
                """
            )
            existing = {str(row[1]) for row in con.execute("PRAGMA table_info(frame_measurements)").fetchall()}
            additions = {
                "proximal_3d_json": "TEXT",
                "vertex_3d_json": "TEXT",
                "distal_3d_json": "TEXT",
                "angle_landmark_deg": "REAL",
                "angle_plane_deg": "REAL",
                "measurement_plane": "TEXT",
                "plane_projection_applied": "INTEGER NOT NULL DEFAULT 0",
                "body_frame_confidence": "REAL",
                "view_alignment": "REAL",
                "view_deviation_deg": "REAL",
                "out_of_plane_deg": "REAL",
                "direction_rule": "TEXT",
                "direction_alignment": "REAL",
            }
            for name, sql_type in additions.items():
                if name not in existing:
                    con.execute(f"ALTER TABLE frame_measurements ADD COLUMN {name} {sql_type}")

    @staticmethod
    def _json(value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _load_json(value: str | None, default: Any = None) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    def save_many(self, job_id: str, measurements: Iterable[dict[str, Any]]) -> int:
        rows: list[dict[str, Any]] = []
        for item in measurements:
            rows.append({
                "job_id": job_id,
                "frame": int(item["frame"]),
                "schema_version": MEASUREMENT_SCHEMA_VERSION,
                "timestamp_ms": float(item["timestamp_ms"]),
                "joint": str(item["joint"]),
                "movement": str(item["movement"]),
                "side": str(item["side"]),
                "angle_raw_deg": item.get("angle_raw_deg"),
                "angle_landmark_deg": item.get("angle_landmark_deg"),
                "angle_plane_deg": item.get("angle_plane_deg"),
                "angle_filtered_deg": item.get("angle_filtered_deg"),
                "angle_clinical_deg": item.get("angle_clinical_deg"),
                "angular_velocity_deg_s": item.get("angular_velocity_deg_s"),
                "adm_deg": item.get("adm_deg"),
                "confidence": item.get("confidence"),
                "source": str(item.get("source") or "unknown"),
                "measurement_space": str(item.get("measurement_space") or "unknown"),
                "measurement_plane": item.get("measurement_plane"),
                "plane_projection_applied": int(bool(item.get("plane_projection_applied"))),
                "body_frame_confidence": item.get("body_frame_confidence"),
                "view_alignment": item.get("view_alignment"),
                "view_deviation_deg": item.get("view_deviation_deg"),
                "out_of_plane_deg": item.get("out_of_plane_deg"),
                "direction_rule": item.get("direction_rule"),
                "direction_alignment": item.get("direction_alignment"),
                "landmark_confidence_min": item.get("landmark_confidence_min"),
                "landmark_confidence_mean": item.get("landmark_confidence_mean"),
                "quality": str(item.get("quality") or "unavailable"),
                "measurement_valid": int(bool(item.get("measurement_valid"))),
                "quality_flags_json": self._json(item.get("quality_flags") or []),
                "proximal_json": self._json(item.get("proximal")),
                "vertex_json": self._json(item.get("vertex")),
                "distal_json": self._json(item.get("distal")),
                "proximal_3d_json": self._json(item.get("proximal_3d")),
                "vertex_3d_json": self._json(item.get("vertex_3d")),
                "distal_3d_json": self._json(item.get("distal_3d")),
                "inference_ms": item.get("inference_ms"),
                "pipeline_ms": item.get("pipeline_ms"),
                "processing_fps": item.get("processing_fps"),
            })
        if not rows:
            return 0
        with self._lock, closing(self._connect()) as con, con:
            con.executemany(
                """
                INSERT OR REPLACE INTO frame_measurements (
                    job_id, frame, schema_version, timestamp_ms, joint, movement, side,
                    angle_raw_deg, angle_landmark_deg, angle_plane_deg, angle_filtered_deg, angle_clinical_deg,
                    angular_velocity_deg_s, adm_deg, confidence, source, measurement_space,
                    measurement_plane, plane_projection_applied, body_frame_confidence,
                    view_alignment, view_deviation_deg, out_of_plane_deg,
                    direction_rule, direction_alignment,
                    landmark_confidence_min, landmark_confidence_mean, quality,
                    measurement_valid, quality_flags_json, proximal_json, vertex_json,
                    distal_json, proximal_3d_json, vertex_3d_json, distal_3d_json,
                    inference_ms, pipeline_ms, processing_fps
                ) VALUES (
                    :job_id, :frame, :schema_version, :timestamp_ms, :joint, :movement, :side,
                    :angle_raw_deg, :angle_landmark_deg, :angle_plane_deg, :angle_filtered_deg, :angle_clinical_deg,
                    :angular_velocity_deg_s, :adm_deg, :confidence, :source, :measurement_space,
                    :measurement_plane, :plane_projection_applied, :body_frame_confidence,
                    :view_alignment, :view_deviation_deg, :out_of_plane_deg,
                    :direction_rule, :direction_alignment,
                    :landmark_confidence_min, :landmark_confidence_mean, :quality,
                    :measurement_valid, :quality_flags_json, :proximal_json, :vertex_json,
                    :distal_json, :proximal_3d_json, :vertex_3d_json, :distal_3d_json,
                    :inference_ms, :pipeline_ms, :processing_fps
                )
                """,
                rows,
            )
        return len(rows)

    def get(self, job_id: str, frame: int) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as con:
            row = con.execute(
                "SELECT * FROM frame_measurements WHERE job_id = ? AND frame = ?",
                (job_id, int(frame)),
            ).fetchone()
        return None if row is None else self._row(row)

    def list_for_job(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock, closing(self._connect()) as con:
            rows = con.execute(
                "SELECT * FROM frame_measurements WHERE job_id = ? ORDER BY frame",
                (job_id,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def delete_job(self, job_id: str) -> int:
        with self._lock, closing(self._connect()) as con, con:
            cur = con.execute("DELETE FROM frame_measurements WHERE job_id = ?", (job_id,))
            return int(cur.rowcount)

    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["measurement_valid"] = bool(item["measurement_valid"])
        item["plane_projection_applied"] = bool(item.get("plane_projection_applied"))
        item["quality_flags"] = self._load_json(item.pop("quality_flags_json"), [])
        for name in ("proximal", "vertex", "distal"):
            item[name] = self._load_json(item.pop(f"{name}_json", None))
        for name in ("proximal_3d", "vertex_3d", "distal_3d"):
            item[name] = self._load_json(item.pop(f"{name}_json", None))
        return item
