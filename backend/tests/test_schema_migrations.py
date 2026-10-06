import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.data.measurement_store import MeasurementStore
from src.data.session_store import ClinicalSessionStore
from src.data.store import DatasetStore
from src.data.imu_store import IMUAnalysisStore
from src.data.motor_feature_store import MotorFeatureStore
from src.data.longitudinal_store import LongitudinalAnalysisStore
from src.validation.store import ValidationStore
from src.validation.study import ValidationStudyStore


class SchemaMigrationTests(unittest.TestCase):
    def test_session_store_migrates_phase8_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clinical.sqlite3"
            con = sqlite3.connect(path)
            con.execute(
                """
                CREATE TABLE clinical_sessions (
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
                    adm_deg REAL
                )
                """
            )
            con.commit(); con.close()
            ClinicalSessionStore(path)
            con = sqlite3.connect(path)
            cols = {row[1] for row in con.execute("PRAGMA table_info(clinical_sessions)")}
            con.close()
            for expected in {
                "valid_frames", "fps", "software_version", "protocol_version",
                "clinical_spec_sha256", "result_sha256", "provenance_json",
                "total_reps",
            }:
                self.assertIn(expected, cols)

    def test_measurement_store_migrates_3d_landmarks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clinical.sqlite3"
            con = sqlite3.connect(path)
            con.execute(
                """
                CREATE TABLE frame_measurements (
                    job_id TEXT NOT NULL,
                    frame INTEGER NOT NULL,
                    schema_version INTEGER NOT NULL,
                    timestamp_ms REAL NOT NULL,
                    joint TEXT NOT NULL,
                    movement TEXT NOT NULL,
                    side TEXT NOT NULL,
                    angle_raw_deg REAL,
                    angle_filtered_deg REAL,
                    angle_clinical_deg REAL,
                    angular_velocity_deg_s REAL,
                    adm_deg REAL,
                    confidence REAL,
                    source TEXT NOT NULL,
                    measurement_space TEXT NOT NULL,
                    landmark_confidence_min REAL,
                    landmark_confidence_mean REAL,
                    quality TEXT NOT NULL,
                    measurement_valid INTEGER NOT NULL,
                    quality_flags_json TEXT NOT NULL,
                    proximal_json TEXT,
                    vertex_json TEXT,
                    distal_json TEXT,
                    inference_ms REAL,
                    pipeline_ms REAL,
                    processing_fps REAL,
                    PRIMARY KEY (job_id, frame)
                )
                """
            )
            con.commit(); con.close()
            MeasurementStore(path)
            con = sqlite3.connect(path)
            cols = {row[1] for row in con.execute("PRAGMA table_info(frame_measurements)")}
            con.close()
            for expected in {
                "proximal_3d_json", "vertex_3d_json", "distal_3d_json",
                "angle_landmark_deg", "angle_plane_deg", "measurement_plane",
                "plane_projection_applied", "body_frame_confidence",
                "view_alignment", "view_deviation_deg", "out_of_plane_deg",
                "direction_rule", "direction_alignment",
            }:
                self.assertIn(expected, cols)

    def test_validation_study_store_migrates_plane_geometry_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clinical.sqlite3"
            con = sqlite3.connect(path)
            con.executescript(
                """
                CREATE TABLE validation_studies (
                    id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL,
                    code TEXT NOT NULL UNIQUE, title TEXT NOT NULL,
                    protocol_version TEXT NOT NULL, created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL, status TEXT NOT NULL,
                    frozen_at TEXT, closed_at TEXT, movement TEXT NOT NULL, side TEXT NOT NULL,
                    target_participants INTEGER, repetitions_per_participant INTEGER,
                    min_landmark_confidence REAL NOT NULL, require_observed INTEGER NOT NULL,
                    require_valid_measurement INTEGER NOT NULL, reference_method TEXT NOT NULL,
                    reference_device_model TEXT, reference_device_resolution_deg REAL,
                    camera_setup_json TEXT NOT NULL DEFAULT '{}', protocol_notes TEXT NOT NULL DEFAULT '',
                    protocol_sha256 TEXT
                );
                CREATE TABLE validation_trials (
                    id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL, study_id TEXT NOT NULL,
                    created_at TEXT NOT NULL, job_id TEXT NOT NULL, patient_id TEXT NOT NULL,
                    evaluator_id TEXT NOT NULL, reference_assessor_id TEXT, movement TEXT NOT NULL,
                    side TEXT NOT NULL, trial_number INTEGER NOT NULL, frame INTEGER NOT NULL,
                    timestamp_ms REAL NOT NULL, automatic_angle_deg REAL NOT NULL,
                    reference_angle_deg REAL NOT NULL, reference_method TEXT NOT NULL,
                    reference_device_model TEXT, reference_device_serial TEXT,
                    landmark_confidence_min REAL, source TEXT, measurement_valid INTEGER NOT NULL,
                    quality TEXT, included INTEGER NOT NULL, exclusion_reason TEXT, notes TEXT NOT NULL DEFAULT '',
                    UNIQUE(study_id, job_id, frame)
                );
                """
            )
            con.commit(); con.close()
            ValidationStudyStore(path)
            con = sqlite3.connect(path)
            cols = {row[1] for row in con.execute("PRAGMA table_info(validation_trials)")}
            con.close()
            for expected in {
                "body_frame_confidence", "measurement_plane", "view_alignment",
                "view_deviation_deg", "out_of_plane_deg",
                "direction_rule", "direction_alignment",
            }:
                self.assertIn(expected, cols)

    def test_dataset_store_migrates_phase8_provenance_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clinical.sqlite3"
            con = sqlite3.connect(path)
            con.execute(
                """
                CREATE TABLE analysis_samples (
                    id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
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
                    notes TEXT NOT NULL DEFAULT ''
                )
                """
            )
            con.commit(); con.close()
            DatasetStore(path)
            con = sqlite3.connect(path)
            cols = {row[1] for row in con.execute("PRAGMA table_info(analysis_samples)")}
            con.close()
            for expected in {
                "job_id", "valid_frames", "result_sha256", "software_version",
                "protocol_version", "clinical_spec_sha256",
            }:
                self.assertIn(expected, cols)

    def test_imu_store_adds_table_to_existing_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clinical.sqlite3"
            con = sqlite3.connect(path)
            con.execute("CREATE TABLE legacy_marker (id INTEGER PRIMARY KEY)")
            con.commit(); con.close()
            IMUAnalysisStore(path)
            con = sqlite3.connect(path)
            tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            cols = {row[1] for row in con.execute("PRAGMA table_info(imu_analyses)")}
            con.close()
            self.assertIn("imu_analyses", tables)
            for expected in {"id", "owner", "job_id", "input_sha256", "summary_json", "payload_zlib"}:
                self.assertIn(expected, cols)

    def test_validation_store_migrates_comparator_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clinical.sqlite3"
            con = sqlite3.connect(path)
            con.execute(
                """
                CREATE TABLE validation_reviews (
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
                    confidence REAL,
                    source TEXT,
                    proximal_json TEXT,
                    vertex_json TEXT,
                    distal_json TEXT,
                    notes TEXT NOT NULL DEFAULT ''
                )
                """
            )
            con.commit(); con.close()
            ValidationStore(path)
            con = sqlite3.connect(path)
            cols = {row[1] for row in con.execute("PRAGMA table_info(validation_reviews)")}
            con.close()
            self.assertIn("manual_method", cols)
            self.assertIn("reference_method", cols)

    def test_motor_feature_store_adds_table_to_existing_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clinical.sqlite3"
            con = sqlite3.connect(path)
            con.execute("CREATE TABLE legacy_marker (id INTEGER PRIMARY KEY)")
            con.commit(); con.close()
            MotorFeatureStore(path)
            con = sqlite3.connect(path)
            tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            cols = {row[1] for row in con.execute("PRAGMA table_info(motor_feature_analyses)")}
            con.close()
            self.assertIn("motor_feature_analyses", tables)
            for expected in {"id", "owner", "job_id", "imu_analysis_id", "source_sha256", "source_json", "summary_json"}:
                self.assertIn(expected, cols)

    def test_longitudinal_store_adds_table_to_existing_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.sqlite3"
            sqlite3.connect(path).close()
            LongitudinalAnalysisStore(path)
            with sqlite3.connect(path) as con:
                tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertIn("longitudinal_analyses", tables)



if __name__ == "__main__":
    unittest.main()
