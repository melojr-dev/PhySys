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


DATASET_SCHEMA_VERSION = 2


class DatasetStore:
    """Dataset estruturado do PhysioAI.

    O CSV legado permanece apenas como artefato histórico. Novas amostras são
    persistidas em SQLite com um único schema para impedir corrupção por append
    de colunas incompatíveis.
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
                CREATE TABLE IF NOT EXISTS analysis_samples (
                    id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    job_id TEXT,
                    patient_id TEXT NOT NULL,
                    movement TEXT NOT NULL,
                    side TEXT NOT NULL,
                    label TEXT NOT NULL,
                    label_marcha TEXT NOT NULL,
                    adm_max REAL NOT NULL,
                    adm_media REAL NOT NULL,
                    adm_ref REAL NOT NULL,
                    pct_adm_ref REAL,
                    vel_pico REAL NOT NULL,
                    total_reps INTEGER NOT NULL,
                    fluidez_ia REAL NOT NULL,
                    stride_length_m REAL,
                    step_width_m REAL,
                    cadencia_passos_min REAL,
                    offset_cameras_ms INTEGER NOT NULL,
                    tipo_sensor TEXT NOT NULL,
                    has_lidar INTEGER NOT NULL,
                    has_stereo INTEGER NOT NULL,
                    hardware TEXT NOT NULL,
                    sequence_json TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    valid_frames INTEGER,
                    result_sha256 TEXT,
                    software_version TEXT,
                    protocol_version TEXT,
                    clinical_spec_sha256 TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_samples_patient
                    ON analysis_samples(patient_id);
                CREATE INDEX IF NOT EXISTS idx_samples_label
                    ON analysis_samples(label);
                CREATE INDEX IF NOT EXISTS idx_samples_movement
                    ON analysis_samples(movement, side);
                """
            )
            existing = {
                str(row[1])
                for row in con.execute("PRAGMA table_info(analysis_samples)").fetchall()
            }
            additions = {
                "job_id": "TEXT",
                "valid_frames": "INTEGER",
                "result_sha256": "TEXT",
                "software_version": "TEXT",
                "protocol_version": "TEXT",
                "clinical_spec_sha256": "TEXT",
            }
            for name, sql_type in additions.items():
                if name not in existing:
                    con.execute(f"ALTER TABLE analysis_samples ADD COLUMN {name} {sql_type}")
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_job ON analysis_samples(job_id)"
            )
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_samples_job_unique "
                "ON analysis_samples(job_id) WHERE job_id IS NOT NULL"
            )

    def save(self, sample: dict[str, Any]) -> dict[str, Any]:
        normalized_sequence = sample.get("sequence") or []
        if not isinstance(normalized_sequence, list) or not normalized_sequence:
            raise ValueError("sequence deve conter ao menos um valor")
        payload = {
            "id": str(uuid.uuid4()),
            "schema_version": DATASET_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "job_id": str(sample.get("job_id") or "").strip() or None,
            "patient_id": str(sample["patient_id"]).strip(),
            "movement": str(sample["movement"]),
            "side": str(sample["side"]),
            "label": str(sample["label"]),
            "label_marcha": str(sample.get("label_marcha") or "Não aplicável"),
            "adm_max": float(sample.get("adm_max") or 0.0),
            "adm_media": float(sample.get("adm_media") or 0.0),
            "adm_ref": float(sample.get("adm_ref") or 0.0),
            "pct_adm_ref": sample.get("pct_adm_ref"),
            "vel_pico": float(sample.get("vel_pico") or 0.0),
            "total_reps": int(sample.get("total_reps") or 0),
            "fluidez_ia": float(sample.get("fluidez_ia") or 0.0),
            "stride_length_m": sample.get("stride_length_m"),
            "step_width_m": sample.get("step_width_m"),
            "cadencia_passos_min": sample.get("cadencia_passos_min"),
            "offset_cameras_ms": int(sample.get("offset_cameras_ms") or 0),
            "tipo_sensor": str(sample.get("tipo_sensor") or "Nenhum"),
            "has_lidar": int(bool(sample.get("has_lidar"))),
            "has_stereo": int(bool(sample.get("has_stereo"))),
            "hardware": str(sample.get("hardware") or "CPU"),
            "sequence_json": json.dumps(normalized_sequence, ensure_ascii=False, separators=(",", ":")),
            "notes": str(sample.get("notes") or ""),
            "valid_frames": (
                int(sample["valid_frames"])
                if sample.get("valid_frames") is not None
                else None
            ),
            "result_sha256": sample.get("result_sha256"),
            "software_version": sample.get("software_version"),
            "protocol_version": sample.get("protocol_version"),
            "clinical_spec_sha256": sample.get("clinical_spec_sha256"),
        }
        if not payload["patient_id"]:
            raise ValueError("patient_id não pode ser vazio")
        if not payload["label"]:
            raise ValueError("label não pode ser vazio")

        with self._lock, closing(self._connect()) as con, con:
            try:
                con.execute(
                    """
                INSERT INTO analysis_samples (
                    id, schema_version, created_at, job_id, patient_id, movement, side,
                    label, label_marcha, adm_max, adm_media, adm_ref, pct_adm_ref,
                    vel_pico, total_reps, fluidez_ia, stride_length_m, step_width_m,
                    cadencia_passos_min, offset_cameras_ms, tipo_sensor, has_lidar,
                    has_stereo, hardware, sequence_json, notes, valid_frames,
                    result_sha256, software_version, protocol_version,
                    clinical_spec_sha256
                ) VALUES (
                    :id, :schema_version, :created_at, :job_id, :patient_id, :movement, :side,
                    :label, :label_marcha, :adm_max, :adm_media, :adm_ref, :pct_adm_ref,
                    :vel_pico, :total_reps, :fluidez_ia, :stride_length_m, :step_width_m,
                    :cadencia_passos_min, :offset_cameras_ms, :tipo_sensor, :has_lidar,
                    :has_stereo, :hardware, :sequence_json, :notes, :valid_frames,
                    :result_sha256, :software_version, :protocol_version,
                    :clinical_spec_sha256
                )
                """,
                    payload,
                )
            except sqlite3.IntegrityError as exc:
                if payload["job_id"]:
                    raise ValueError("Esta sessão já foi catalogada no dataset.") from exc
                raise
        return {"id": payload["id"], "created_at": payload["created_at"]}

    def stats(self) -> dict[str, Any]:
        with self._lock, closing(self._connect()) as con, con:
            total = int(con.execute("SELECT COUNT(*) FROM analysis_samples").fetchone()[0])
            labels = int(con.execute("SELECT COUNT(DISTINCT label) FROM analysis_samples").fetchone()[0])
            patients = int(con.execute("SELECT COUNT(DISTINCT patient_id) FROM analysis_samples").fetchone()[0])
            distribution = {
                row["label"]: int(row["n"])
                for row in con.execute(
                    "SELECT label, COUNT(*) AS n FROM analysis_samples GROUP BY label ORDER BY n DESC"
                ).fetchall()
            }
            rows = con.execute(
                """
                SELECT id, created_at AS timestamp, patient_id AS id_paciente,
                       job_id, movement AS movimento, side AS lado, label, label_marcha,
                       adm_max, adm_media, adm_ref, pct_adm_ref, vel_pico,
                       total_reps, fluidez_ia, stride_length_m, step_width_m,
                       cadencia_passos_min, tipo_sensor, has_lidar, has_stereo,
                       hardware, notes AS notas, valid_frames, result_sha256,
                       software_version, protocol_version, clinical_spec_sha256
                FROM analysis_samples
                ORDER BY created_at DESC LIMIT 10
                """
            ).fetchall()
        return {
            "total_amostras": total,
            "labels_distintos": labels,
            "pacientes_unicos": patients,
            "distribuicao_labels": distribution,
            "ultimas_10": [dict(row) for row in rows],
            "storage": "sqlite",
            "schema_version": DATASET_SCHEMA_VERSION,
        }

    def delete_patient(self, patient_id: str) -> int:
        with self._lock, closing(self._connect()) as con, con:
            cur = con.execute(
                "DELETE FROM analysis_samples WHERE patient_id = ?",
                (patient_id,),
            )
            return int(cur.rowcount)
