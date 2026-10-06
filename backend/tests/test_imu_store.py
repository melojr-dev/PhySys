from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.imu_store import IMUAnalysisStore


class IMUStoreTests(unittest.TestCase):
    def test_roundtrip_and_job_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IMUAnalysisStore(Path(tmp) / "imu.sqlite3")
            raw = {
                "timestamps_ms": [0.0, 10.0, 20.0],
                "gyro_rad_s": [[0.0, 0.0, 0.0]] * 3,
                "accel_m_s2": None,
            }
            summary = {"sample_count": 3, "duration_s": 0.02, "status": "limited"}
            saved = store.save(
                owner="alice", job_id="job-1", device_label="phone", raw_payload=raw, summary=summary
            )
            self.assertEqual(len(saved["input_sha256"]), 64)
            self.assertEqual(saved["summary"]["sample_count"], 3)
            fetched = store.get(saved["id"])
            self.assertIsNotNone(fetched)
            self.assertEqual(store.raw_payload(saved["id"]), raw)
            self.assertEqual(len(store.list_for_job("job-1")), 1)
            self.assertEqual(store.delete_job("job-1"), 1)
            self.assertEqual(store.list_for_job("job-1"), [])


if __name__ == "__main__":
    unittest.main()
