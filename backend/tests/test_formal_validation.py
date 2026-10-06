from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from src.validation.formal import (
    FormalValidationRunStore,
    formal_readiness,
    report_for_formal_run,
)
from src.research.export import build_formal_validation_bundle


class FormalValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sap = self.root / "sap.json"
        self.lock = self.root / "lock.json"
        self.sap.write_text(json.dumps({"clinical_protocol_version": "1.4"}), encoding="utf-8")
        self.lock.write_text(json.dumps({"protocol_version": "1.4"}), encoding="utf-8")
        self.study = {
            "id": "study-1",
            "status": "frozen",
            "protocol_sha256": "a" * 64,
            "protocol_version": "1.4",
            "target_participants": 20,
            "repetitions_per_participant": 3,
            "reference_device_model": "Goniometro",
            "reference_device_resolution_deg": 1.0,
            "camera_setup": {"view": "sagittal"},
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def readiness(self, **overrides):
        args = dict(
            study=self.study,
            ethics_status="approved",
            ethics_protocol_id="CEP-123",
            calibration_date=date.today().isoformat(),
            calibration_due_date=(date.today() + timedelta(days=180)).isoformat(),
            statistical_plan_path=self.sap,
            protocol_lock_path=self.lock,
            software_version="2.19.0-rc.2",
            clinical_protocol_version="1.4",
        )
        args.update(overrides)
        return formal_readiness(**args)

    def test_readiness_passes_with_complete_metadata(self):
        report = self.readiness()
        self.assertTrue(report["ready"])
        self.assertTrue(all(x["ok"] for x in report["checks"]))

    def test_readiness_blocks_pending_ethics(self):
        report = self.readiness(ethics_status="pending", ethics_protocol_id=None)
        self.assertFalse(report["ready"])
        failed = {x["code"] for x in report["checks"] if not x["ok"]}
        self.assertIn("ethics_gate", failed)

    def test_readiness_blocks_expired_calibration(self):
        report = self.readiness(calibration_due_date=(date.today() - timedelta(days=1)).isoformat())
        self.assertFalse(report["ready"])
        self.assertIn("reference_calibration_current", {x["code"] for x in report["checks"] if not x["ok"]})

    def test_readiness_requires_current_protocol(self):
        bad = dict(self.study)
        bad["protocol_version"] = "1.3"
        report = self.readiness(study=bad)
        self.assertFalse(report["ready"])
        self.assertIn("protocol_version_match", {x["code"] for x in report["checks"] if not x["ok"]})

    def test_store_snapshot_is_immutable_and_single_open_run(self):
        store = FormalValidationRunStore(self.root / "db.sqlite3")
        payload = {
            "study_id": "study-1", "created_by": "admin", "software_version": "2.19.0-rc.2",
            "clinical_protocol_version": "1.4", "study_protocol_sha256": "a" * 64,
            "statistical_plan_sha256": "b" * 64, "protocol_lock_sha256": "c" * 64,
            "ethics_status": "approved", "ethics_protocol_id": "CEP-123",
            "reference_calibration_date": "2026-01-01", "reference_calibration_due_date": "2099-01-01",
            "start_trial_count": 2, "snapshot": {"x": 1}, "notes": "",
        }
        run = store.create(payload)
        self.assertEqual(run["snapshot"], {"x": 1})
        self.assertEqual(len(run["snapshot_sha256"]), 64)
        with self.assertRaises(ValueError):
            store.create(payload)
        closed = store.close(run["id"], end_trial_count=5, report={"ok": True})
        self.assertEqual(closed["status"], "closed")
        self.assertEqual(closed["trials_collected"], 3)
        self.assertEqual(closed["close_report"], {"ok": True})

    def test_formal_report_uses_only_given_rows(self):
        rows = [
            {"patient_id":"P1","job_id":"J1","side":"direito","included":True,"automatic_angle_deg":90.0,"reference_angle_deg":88.0,"exclusion_reason":None},
            {"patient_id":"P2","job_id":"J2","side":"direito","included":True,"automatic_angle_deg":100.0,"reference_angle_deg":101.0,"exclusion_reason":None},
        ]
        report = report_for_formal_run({"id":"S1","protocol_sha256":"a"*64,"target_participants":2}, rows, bootstrap_iterations=200)
        self.assertEqual(report["counts"]["trials_total"], 2)
        self.assertAlmostEqual(report["agreement"]["mae_deg"], 1.5)
        self.assertEqual(report["formal_run_scope"], "trials_created_during_run_only")

    def test_formal_bundle_contains_governance_files(self):
        blob = build_formal_validation_bundle(
            run={"id":"R1"}, study={"id":"S1"}, trials=[], report={},
            statistical_plan={"clinical_protocol_version":"1.4"},
            protocol_lock={"protocol_version":"1.4"}, checklist={"schema_version":1},
        )
        import io, zipfile
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            names = set(zf.namelist())
        self.assertTrue({"manifest.json","trials.csv","report.json","statistical_analysis_plan.json","protocol_lock.json","formal_validation_checklist.json","checksums.sha256"}.issubset(names))


if __name__ == "__main__":
    unittest.main()
