import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.research.export import build_session_bundle, build_study_bundle
from src.research.provenance import (
    PROTOCOL_VERSION,
    canonical_json_sha256,
    clinical_spec_sha256,
    clinical_spec_snapshot,
    file_sha256,
)


class ProvenanceTests(unittest.TestCase):
    def test_canonical_hash_ignores_dict_insertion_order(self):
        self.assertEqual(
            canonical_json_sha256({"b": 2, "a": 1}),
            canonical_json_sha256({"a": 1, "b": 2}),
        )

    def test_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.bin"
            path.write_bytes(b"physioai")
            self.assertEqual(file_sha256(path), hashlib.sha256(b"physioai").hexdigest())

    def test_clinical_spec_is_versioned_and_hashed(self):
        snapshot = clinical_spec_snapshot()
        self.assertEqual(snapshot["protocol_version"], PROTOCOL_VERSION)
        self.assertIn("capture_preflight", snapshot)
        self.assertEqual(snapshot["capture_preflight"]["version"], "1.0")
        self.assertGreaterEqual(snapshot["capture_preflight"]["thresholds"]["fps_minimum"], 15)
        self.assertEqual(clinical_spec_sha256(), canonical_json_sha256(snapshot))

    def test_session_bundle_checksums_are_valid(self):
        bundle = build_session_bundle(
            manifest={"session": {"job_id": "job-1"}},
            measurements=[{"job_id": "job-1", "frame": 0, "timestamp_ms": 0, "joint": "elbow", "movement": "Flexão de Cotovelo", "side": "direito", "source": "observed", "measurement_space": "world_3d", "quality": "good", "measurement_valid": True}],
            validation_reviews=[],
        )
        with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
            names = set(zf.namelist())
            self.assertIn("manifest.json", names)
            self.assertIn("measurements.csv", names)
            self.assertIn("checksums.sha256", names)
            manifest = json.loads(zf.read("manifest.json"))
            self.assertEqual(manifest["session"]["job_id"], "job-1")
            checksums = zf.read("checksums.sha256").decode("ascii").splitlines()
            for line in checksums:
                digest, name = line.split("  ", 1)
                self.assertEqual(digest, hashlib.sha256(zf.read(name)).hexdigest())


    def test_session_bundle_carries_reproducible_imu_payload(self):
        imu_record = {
            "id": "imu-1", "input_sha256": "a" * 64, "summary": {"sample_count": 3},
        }
        raw = {
            "timestamps_ms": [0.0, 10.0, 20.0],
            "gyro_rad_s": [[0.0, 0.0, 0.0]] * 3,
            "accel_m_s2": None,
        }
        bundle = build_session_bundle(
            manifest={"session": {"job_id": "job-imu"}},
            measurements=[], validation_reviews=[],
            imu_analyses=[imu_record], imu_payloads={"imu-1": raw},
        )
        with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
            self.assertIn("imu/index.json", zf.namelist())
            self.assertIn("imu/imu-1.json", zf.namelist())
            payload = json.loads(zf.read("imu/imu-1.json"))
            self.assertEqual(payload["raw"]["timestamps_ms"], raw["timestamps_ms"])
            self.assertEqual(payload["record"]["id"], "imu-1")
            for line in zf.read("checksums.sha256").decode("ascii").splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(digest, hashlib.sha256(zf.read(name)).hexdigest())


    def test_session_bundle_carries_motor_feature_summary(self):
        motor = {
            "id": "motor-1", "job_id": "job-1", "source_sha256": "b" * 64,
            "summary": {"experimental": True, "diagnostic": False, "status": "usable"},
        }
        bundle = build_session_bundle(
            manifest={"session": {"job_id": "job-1"}},
            measurements=[], validation_reviews=[],
            motor_feature_analyses=[motor],
        )
        with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
            self.assertIn("motor_features/index.json", zf.namelist())
            payload = json.loads(zf.read("motor_features/index.json"))
            self.assertEqual(payload[0]["id"], "motor-1")
            self.assertFalse(payload[0]["summary"]["diagnostic"])
            for line in zf.read("checksums.sha256").decode("ascii").splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(digest, hashlib.sha256(zf.read(name)).hexdigest())

    def test_study_bundle_checksums_are_valid(self):
        bundle = build_study_bundle(
            study={"id": "study-1", "code": "ELBOW-001", "protocol_sha256": "a" * 64},
            trials=[{
                "id": "trial-1", "study_id": "study-1", "job_id": "job-1",
                "patient_id": "P-1", "evaluator_id": "eval", "movement": "Flexão de Cotovelo",
                "side": "direito", "trial_number": 1, "frame": 4, "timestamp_ms": 133.2,
                "automatic_angle_deg": 90.0, "reference_angle_deg": 91.0,
                "error_deg": -1.0, "absolute_error_deg": 1.0, "included": True,
            }],
            report={"counts": {"trials_included": 1}, "agreement": {"mae_deg": 1.0}},
        )
        with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
            self.assertIn("protocol.json", zf.namelist())
            self.assertIn("trials.csv", zf.namelist())
            self.assertIn("report.json", zf.namelist())
            checksums = zf.read("checksums.sha256").decode("ascii").splitlines()
            for line in checksums:
                digest, name = line.split("  ", 1)
                self.assertEqual(digest, hashlib.sha256(zf.read(name)).hexdigest())
    def test_session_bundle_carries_longitudinal_summary(self):
        payload = build_session_bundle(
            manifest={"session": {"job_id": "j1"}},
            measurements=[],
            validation_reviews=[],
            longitudinal_analyses=[{
                "id": "long-1", "anchor_job_id": "j1",
                "source_sha256": "a" * 64,
                "summary": {"experimental": True, "diagnostic": False, "session_count": 3},
            }],
        )
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            exported = json.loads(zf.read("longitudinal/index.json"))
            self.assertEqual(exported[0]["id"], "long-1")
            self.assertFalse(exported[0]["summary"]["diagnostic"])



if __name__ == "__main__":
    unittest.main()
