from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.store import DatasetStore


class DatasetStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = DatasetStore(Path(self.tmp.name) / "test.sqlite3")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_fixed_schema_dataset(self) -> None:
        self.store.save({
            "patient_id": "PAC-001",
            "movement": "Flexão de Cotovelo",
            "side": "direito",
            "label": "Normal",
            "label_marcha": "Não aplicável",
            "adm_max": 110.0,
            "adm_media": 62.0,
            "adm_ref": 145.0,
            "pct_adm_ref": 73.3,
            "vel_pico": 80.0,
            "total_reps": 3,
            "fluidez_ia": 0.0,
            "offset_cameras_ms": 0,
            "tipo_sensor": "Nenhum",
            "has_lidar": False,
            "has_stereo": False,
            "hardware": "CPU",
            "sequence": [0.0, 20.0, 40.0],
        })
        stats = self.store.stats()
        self.assertEqual(stats["total_amostras"], 1)
        self.assertEqual(stats["labels_distintos"], 1)
        self.assertEqual(stats["pacientes_unicos"], 1)
        self.assertEqual(stats["distribuicao_labels"]["Normal"], 1)
        self.assertEqual(stats["storage"], "sqlite")

    def test_empty_sequence_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.save({
                "patient_id": "PAC-001",
                "movement": "Flexão de Cotovelo",
                "side": "direito",
                "label": "Normal",
                "sequence": [],
            })

    def test_job_provenance_and_duplicate_job_are_enforced(self) -> None:
        sample = {
            "job_id": "job-1",
            "patient_id": "P-ABC",
            "movement": "Flexão de Cotovelo",
            "side": "direito",
            "label": "Normal",
            "sequence": [0.0, 15.0, 30.0],
            "valid_frames": 3,
            "result_sha256": "deadbeef",
            "software_version": "2.8.0-alpha.1",
            "protocol_version": "1.0",
            "clinical_spec_sha256": "abc123",
        }
        self.store.save(sample)
        stats = self.store.stats()
        self.assertEqual(stats["ultimas_10"][0]["job_id"], "job-1")
        self.assertEqual(stats["ultimas_10"][0]["result_sha256"], "deadbeef")
        with self.assertRaises(ValueError):
            self.store.save(sample)


if __name__ == "__main__":
    unittest.main()
