from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.validation.study import (
    ValidationStudyStore,
    participant_balanced_summary,
    cluster_bootstrap_ci,
)


class ValidationStudyStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ValidationStudyStore(Path(self.tmp.name) / "study.sqlite3")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def create_frozen(self):
        study = self.store.create_study({
            "code": "ELBOW-001",
            "title": "Validação do cotovelo",
            "protocol_version": "1.0",
            "created_by": "investigator",
            "movement": "Flexão de Cotovelo",
            "side": "ambos",
            "target_participants": 20,
            "repetitions_per_participant": 3,
            "min_landmark_confidence": 0.70,
            "require_observed": True,
            "require_valid_measurement": True,
            "reference_method": "physical_goniometer",
            "reference_device_model": "Gonio Test",
            "reference_device_resolution_deg": 1.0,
            "camera_setup": {"view": "sagittal"},
        })
        return self.store.freeze(study["id"])

    def add(self, study_id: str, *, patient="P-1", job="J-1", frame=1,
            auto=90.0, ref=89.0, conf=0.95, source="observed", valid=True):
        return self.store.add_trial({
            "study_id": study_id,
            "job_id": job,
            "patient_id": patient,
            "evaluator_id": "eval",
            "movement": "Flexão de Cotovelo",
            "side": "direito",
            "frame": frame,
            "timestamp_ms": frame * 33.3,
            "automatic_angle_deg": auto,
            "reference_angle_deg": ref,
            "landmark_confidence_min": conf,
            "body_frame_confidence": 0.94,
            "measurement_plane": "sagittal",
            "view_alignment": 0.96,
            "view_deviation_deg": 16.3,
            "out_of_plane_deg": 8.2,
            "direction_rule": "anterior_positive",
            "direction_alignment": 0.84,
            "source": source,
            "measurement_valid": valid,
            "quality": "excellent",
        })

    def test_protocol_freeze_hash_is_stable_and_trials_require_freeze(self):
        study = self.store.create_study({
            "code": "E-1", "title": "Teste", "created_by": "admin",
            "movement": "Flexão de Cotovelo", "side": "direito",
            "target_participants": 10, "repetitions_per_participant": 3,
            "reference_device_model": "Goniômetro Teste", "reference_device_resolution_deg": 1.0,
        })
        with self.assertRaises(ValueError):
            self.add(study["id"])
        frozen = self.store.freeze(study["id"])
        self.assertEqual(frozen["status"], "frozen")
        self.assertEqual(len(frozen["protocol_sha256"]), 64)
        self.assertEqual(self.store.freeze(study["id"])["protocol_sha256"], frozen["protocol_sha256"])

    def test_protocol_qc_excludes_low_confidence_and_predicted(self):
        study = self.create_frozen()
        good = self.add(study["id"], frame=1)
        low = self.add(study["id"], frame=2, conf=0.40)
        pred = self.add(study["id"], frame=3, source="predicted")
        self.assertTrue(good["included"])
        self.assertFalse(low["included"])
        self.assertIn("landmark_confidence_below_protocol", low["exclusion_reason"])
        self.assertFalse(pred["included"])
        self.assertIn("source_not_observed", pred["exclusion_reason"])

    def test_duplicate_frame_in_same_study_rejected(self):
        study = self.create_frozen()
        self.add(study["id"], frame=4)
        with self.assertRaises(ValueError):
            self.add(study["id"], frame=4)

    def test_report_counts_and_participant_balanced_metrics(self):
        study = self.create_frozen()
        self.add(study["id"], patient="P-1", job="J-1", frame=1, auto=90, ref=90)
        self.add(study["id"], patient="P-1", job="J-1", frame=2, auto=100, ref=90)
        self.add(study["id"], patient="P-2", job="J-2", frame=1, auto=80, ref=82)
        report = self.store.report(study["id"], bootstrap_iterations=200)
        self.assertEqual(report["counts"]["participants_included"], 2)
        self.assertEqual(report["counts"]["trials_included"], 3)
        self.assertAlmostEqual(report["agreement"]["mae_deg"], 4.0)
        # P1 MAE=5; P2 MAE=2 => equal participant mean=3.5
        self.assertAlmostEqual(report["participant_balanced"]["mean_patient_mae_deg"], 3.5)
        self.assertEqual(report["cluster_bootstrap"]["participants"], 2)
        self.assertEqual(report["cluster_bootstrap"]["iterations"], 200)
        self.assertIn("60-<90", report["by_reference_angle"])
        self.assertIn("90-<120", report["by_reference_angle"])

    def test_patient_deletion_removes_trials_only(self):
        study = self.create_frozen()
        self.add(study["id"], patient="P-1", frame=1)
        self.add(study["id"], patient="P-2", job="J-2", frame=1)
        self.assertEqual(self.store.delete_patient("P-1"), 1)
        rows = self.store.trials(study["id"])
        self.assertEqual([x["patient_id"] for x in rows], ["P-2"])
        self.assertEqual(self.store.get_study(study["id"])["code"], "ELBOW-001")

    def test_trial_preserves_plane_geometry_metadata(self):
        study = self.create_frozen()
        trial = self.add(study["id"], frame=7)
        self.assertEqual(trial["measurement_plane"], "sagittal")
        self.assertAlmostEqual(trial["body_frame_confidence"], 0.94)
        self.assertAlmostEqual(trial["view_alignment"], 0.96)
        self.assertAlmostEqual(trial["view_deviation_deg"], 16.3)
        self.assertAlmostEqual(trial["out_of_plane_deg"], 8.2)
        self.assertEqual(trial["direction_rule"], "anterior_positive")
        self.assertAlmostEqual(trial["direction_alignment"], 0.84)

    def test_participant_balanced_summary_does_not_weight_more_frames_more(self):
        rows = [
            {"patient_id": "P1", "automatic_angle_deg": 10, "reference_angle_deg": 0},
            {"patient_id": "P1", "automatic_angle_deg": 10, "reference_angle_deg": 0},
            {"patient_id": "P1", "automatic_angle_deg": 10, "reference_angle_deg": 0},
            {"patient_id": "P2", "automatic_angle_deg": 0, "reference_angle_deg": 0},
        ]
        summary = participant_balanced_summary(rows)
        self.assertEqual(summary["participants"], 2)
        self.assertEqual(summary["mean_patient_mae_deg"], 5.0)

    def test_cluster_bootstrap_requires_two_participants(self):
        ci = cluster_bootstrap_ci([
            {"patient_id": "P1", "automatic_angle_deg": 10, "reference_angle_deg": 9}
        ])
        self.assertEqual(ci["participants"], 1)
        self.assertEqual(ci["iterations"], 0)
        self.assertEqual(ci["mae_ci95_deg"], [None, None])


if __name__ == "__main__":
    unittest.main()
