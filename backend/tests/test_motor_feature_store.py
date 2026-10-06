from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.motor_feature_store import MotorFeatureStore


class MotorFeatureStoreTests(unittest.TestCase):
    def test_roundtrip_and_job_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MotorFeatureStore(Path(tmp) / "clinical.sqlite3")
            saved = store.save(
                owner="alice",
                job_id="job-1",
                imu_analysis_id="imu-1",
                source_payload={"job_id": "job-1", "measurement_sha256": "a" * 64},
                summary={"experimental": True, "diagnostic": False, "status": "usable"},
            )
            self.assertEqual(len(saved["source_sha256"]), 64)
            self.assertEqual(saved["source"]["job_id"], "job-1")
            self.assertFalse(saved["summary"]["diagnostic"])
            self.assertEqual(store.get(saved["id"])["job_id"], "job-1")
            self.assertEqual(len(store.list_for_job("job-1")), 1)
            self.assertEqual(store.delete_job("job-1"), 1)
            self.assertEqual(store.list_for_job("job-1"), [])


if __name__ == "__main__":
    unittest.main()
