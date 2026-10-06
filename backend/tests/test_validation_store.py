from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.validation.store import ValidationStore


class ValidationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ValidationStore(Path(self.tmp.name) / "test.sqlite3")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_save_and_metrics(self) -> None:
        base = {
            "job_id": "job-1",
            "patient_id": "PAC-001",
            "movement": "Flexão de Cotovelo",
            "side": "direito",
            "frame": 10,
            "timestamp_ms": 333.3,
            "automatic_angle_deg": 91.0,
            "manual_angle_deg": 90.0,
            "reference_angle_deg": 89.0,
            "confidence": 0.95,
            "source": "observed",
            "proximal": {"x": 0.2, "y": 0.2},
            "vertex": {"x": 0.4, "y": 0.4},
            "distal": {"x": 0.6, "y": 0.5},
            "notes": "teste",
        }
        saved = self.store.save_review(base)
        self.assertEqual(saved["error_vs_manual_deg"], 1.0)
        self.assertEqual(saved["error_vs_reference_deg"], 2.0)

        base["frame"] = 11
        base["automatic_angle_deg"] = 86.0
        base["manual_angle_deg"] = 90.0
        base["reference_angle_deg"] = 90.0
        self.store.save_review(base)

        stats = self.store.stats("Flexão de Cotovelo", "direito")
        self.assertEqual(stats["total_reviews"], 2)
        self.assertAlmostEqual(stats["automatic_vs_manual"]["mae_deg"], 2.5)
        self.assertAlmostEqual(stats["automatic_vs_reference"]["mae_deg"], 3.0)
        self.assertAlmostEqual(stats["automatic_vs_manual"]["bias_deg"], -1.5)
        self.assertEqual(stats["automatic_vs_manual"]["within_5_deg_pct"], 100.0)


    def test_report_contains_agreement_and_counts(self) -> None:
        for frame, auto, ref in [(0, 10.0, 10.0), (1, 40.0, 39.0), (2, 80.0, 81.0)]:
            self.store.save_review({
                "job_id": "session-1",
                "patient_id": "PAC-001",
                "evaluator_id": "FISIO-01",
                "movement": "Flexão de Cotovelo",
                "side": "direito",
                "frame": frame,
                "timestamp_ms": frame * 33.3,
                "automatic_angle_deg": auto,
                "reference_angle_deg": ref,
            })
        report = self.store.report("Flexão de Cotovelo", "direito")
        self.assertEqual(report["unique_patients"], 1)
        self.assertEqual(report["unique_sessions"], 1)
        self.assertEqual(report["unique_evaluators"], 1)
        self.assertEqual(report["automatic_vs_reference"]["n"], 3)
        self.assertEqual(len(report["bland_altman_reference"]), 3)
        self.assertIn("ICC(A,1)", report["methodology"]["icc"])

    def test_angle_bounds_are_enforced(self) -> None:
        with self.assertRaises(ValueError):
            self.store.save_review({
                "patient_id": "PAC-001",
                "movement": "Flexão de Cotovelo",
                "side": "direito",
                "frame": 0,
                "timestamp_ms": 0,
                "automatic_angle_deg": 181,
            })

    def test_comparator_methods_roundtrip(self):
        self.store.save_review({
            "job_id": "job-method",
            "patient_id": "P-123",
            "movement": "Flexão de Cotovelo",
            "side": "direito",
            "frame": 3,
            "timestamp_ms": 100.0,
            "automatic_angle_deg": 90.0,
            "manual_angle_deg": 91.0,
            "reference_angle_deg": 92.0,
            "manual_method": "image_2d_three_point",
            "reference_method": "physical_goniometer",
        })
        row = self.store.for_job("job-method")[0]
        self.assertEqual(row["manual_method"], "image_2d_three_point")
        self.assertEqual(row["reference_method"], "physical_goniometer")


if __name__ == "__main__":
    unittest.main()
