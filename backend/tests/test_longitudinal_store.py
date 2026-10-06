from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.longitudinal_store import LongitudinalAnalysisStore


class LongitudinalStoreTests(unittest.TestCase):
    def test_roundtrip_job_listing_and_patient_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LongitudinalAnalysisStore(Path(tmp) / "clinical.sqlite3")
            saved = store.save(
                owner="alice",
                patient_pseudonym="pseud-1",
                anchor_job_id="job-2",
                movement="Flexão de Cotovelo",
                side="direito",
                source_payload={"jobs": ["job-1", "job-2"]},
                summary={"experimental": True, "diagnostic": False, "session_count": 2},
            )
            self.assertEqual(len(saved["source_sha256"]), 64)
            self.assertEqual(saved["summary"]["session_count"], 2)
            self.assertEqual(store.get(saved["id"])["anchor_job_id"], "job-2")
            self.assertEqual(len(store.list_for_job("job-2")), 1)
            self.assertEqual(store.delete_patient("pseud-1"), 1)
            self.assertEqual(store.list_for_job("job-2"), [])


if __name__ == "__main__":
    unittest.main()
