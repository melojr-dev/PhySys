import tempfile
import unittest
from pathlib import Path

from src.data import ClinicalSessionStore, DatasetStore
from src.validation import ValidationStore


class ClinicalSessionStoreTests(unittest.TestCase):
    def test_session_lifecycle_and_patient_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clinical.sqlite3"
            sessions = ClinicalSessionStore(path)
            sessions.start(
                job_id="job-1",
                patient_pseudonym="P-1234567890ABCDEF",
                evaluator_id="fisioterapeuta",
                movement="Flexão de Cotovelo",
                side="direito",
            )
            sessions.finish("job-1", status="completed", total_frames=120, adm_deg=87.5)
            row = sessions.recent(1)[0]
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["total_frames"], 120)
            self.assertEqual(sessions.delete_patient("P-1234567890ABCDEF"), 1)
            self.assertEqual(sessions.recent(1), [])

    def test_dataset_delete_patient(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clinical.sqlite3"
            store = DatasetStore(path)
            store.save({
                "patient_id": "P-1234567890ABCDEF",
                "movement": "Flexão de Cotovelo",
                "side": "direito",
                "label": "Normal",
                "sequence": [1.0, 2.0],
            })
            self.assertEqual(store.delete_patient("P-1234567890ABCDEF"), 1)
            self.assertEqual(store.stats()["total_amostras"], 0)

    def test_validation_delete_patient(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clinical.sqlite3"
            store = ValidationStore(path)
            store.save_review({
                "patient_id": "P-1234567890ABCDEF",
                "movement": "Flexão de Cotovelo",
                "side": "direito",
                "frame": 1,
                "timestamp_ms": 33.3,
                "automatic_angle_deg": 90.0,
            })
            self.assertEqual(store.delete_patient("P-1234567890ABCDEF"), 1)
            self.assertEqual(store.stats()["total_reviews"], 0)

    def test_session_provenance_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clinical.sqlite3"
            sessions = ClinicalSessionStore(path)
            sessions.start(
                job_id="job-prov", patient_pseudonym="P-ABC", evaluator_id="eval",
                movement="Flexão de Cotovelo", side="direito",
                software_version="2.8.0-alpha.1", protocol_version="1.0",
                clinical_spec_sha256="abc123", provenance={"inputs": [{"sha256": "deadbeef"}]},
            )
            sessions.finish(
                "job-prov", status="completed", total_frames=100, valid_frames=90,
                fps=30.0, adm_deg=120.0, total_reps=4, result_sha256="result-hash",
            )
            row = sessions.get_by_job("job-prov")
            self.assertEqual(row["software_version"], "2.8.0-alpha.1")
            self.assertEqual(row["protocol_version"], "1.0")
            self.assertEqual(row["valid_frames"], 90)
            self.assertEqual(row["total_reps"], 4)
            self.assertEqual(row["provenance"]["inputs"][0]["sha256"], "deadbeef")
            self.assertEqual(row["result_sha256"], "result-hash")

    def test_session_for_patient_is_ordered(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ClinicalSessionStore(Path(tmp) / "clinical.sqlite3")
            for job in ("j1", "j2"):
                store.start(
                    job_id=job, patient_pseudonym="pat", evaluator_id="alice",
                    movement="Flexão de Cotovelo", side="direito",
                    software_version="test", protocol_version="1.4",
                    clinical_spec_sha256="a" * 64, provenance={},
                )
            rows = store.for_patient("pat")
            self.assertEqual([row["job_id"] for row in rows], ["j1", "j2"])


    def test_recent_for_evaluator_is_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ClinicalSessionStore(Path(tmp) / "clinical.sqlite3")
            for job, evaluator in (("a", "alice"), ("b", "bob")):
                store.start(
                    job_id=job, patient_pseudonym="pat", evaluator_id=evaluator,
                    movement="Flexão de Cotovelo", side="direito",
                    software_version="test", protocol_version="1.4",
                    clinical_spec_sha256="a" * 64, provenance={},
                )
            self.assertEqual([row["job_id"] for row in store.recent_for_evaluator("alice")], ["a"])
            self.assertEqual([row["job_id"] for row in store.recent_for_evaluator("bob")], ["b"])



if __name__ == "__main__":
    unittest.main()
