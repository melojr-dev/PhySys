from __future__ import annotations

import csv
import hashlib
import io
import json
import unittest
import zipfile
import sys
import types

# The publication module itself does not use MediaPipe, but importing the
# research package also loads provenance/capture modules. Install a tiny stub
# so this pure packaging test remains runnable outside the Docker image.
if "mediapipe.python.solutions.pose" not in sys.modules:
    pose_module = types.ModuleType("mediapipe.python.solutions.pose")
    solutions_module = types.ModuleType("mediapipe.python.solutions")
    solutions_module.pose = pose_module
    python_module = types.ModuleType("mediapipe.python")
    python_module.solutions = solutions_module
    mediapipe_module = types.ModuleType("mediapipe")
    mediapipe_module.python = python_module
    sys.modules["mediapipe"] = mediapipe_module
    sys.modules["mediapipe.python"] = python_module
    sys.modules["mediapipe.python.solutions"] = solutions_module
    sys.modules["mediapipe.python.solutions.pose"] = pose_module

from src.research.publication import (
    anonymize_trials,
    build_bland_altman_svg,
    build_method_comparison_svg,
    build_publication_review_bundle,
    publication_readiness,
)
from src.validation.formal import report_for_formal_run


class PublicationPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = {
            "id": "study-internal-42",
            "code": "ELBOW-VAL",
            "title": "Comparação de método do cotovelo",
            "status": "frozen",
            "protocol_sha256": "a" * 64,
            "protocol_version": "1.4",
            "movement": "Flexão de Cotovelo",
            "side": "ambos",
            "target_participants": 2,
            "reference_device_model": "Goniometer X",
        }
        self.rows = [
            {
                "id": "internal-trial-1", "study_id": self.study["id"], "created_at": "2026-09-12T00:00:01+00:00",
                "job_id": "secret-job-a", "patient_id": "patient-hmac-A", "evaluator_id": "alice",
                "reference_assessor_id": "clinician-1", "movement": "Flexão de Cotovelo", "side": "direito",
                "trial_number": 1, "frame": 20, "timestamp_ms": 666.7,
                "automatic_angle_deg": 90.0, "reference_angle_deg": 88.0,
                "reference_method": "goniometer", "reference_device_model": "Goniometer X",
                "reference_device_serial": "SERIAL-SECRET-1", "landmark_confidence_min": 0.94,
                "body_frame_confidence": 0.91, "measurement_plane": "sagittal", "view_alignment": 0.96,
                "view_deviation_deg": 3.0, "out_of_plane_deg": 2.0, "direction_rule": "flexion_positive",
                "direction_alignment": 0.95, "source": "observed", "measurement_valid": True,
                "quality": "high", "included": True, "exclusion_reason": None,
                "notes": "Patient mentioned a private detail",
            },
            {
                "id": "internal-trial-2", "study_id": self.study["id"], "created_at": "2026-09-12T00:00:02+00:00",
                "job_id": "secret-job-b", "patient_id": "patient-hmac-B", "evaluator_id": "bob",
                "reference_assessor_id": "clinician-2", "movement": "Flexão de Cotovelo", "side": "esquerdo",
                "trial_number": 1, "frame": 25, "timestamp_ms": 833.3,
                "automatic_angle_deg": 100.0, "reference_angle_deg": 101.0,
                "reference_method": "goniometer", "reference_device_model": "Goniometer X",
                "reference_device_serial": "SERIAL-SECRET-2", "landmark_confidence_min": 0.90,
                "body_frame_confidence": 0.90, "measurement_plane": "sagittal", "view_alignment": 0.94,
                "view_deviation_deg": 4.0, "out_of_plane_deg": 3.0, "direction_rule": "flexion_positive",
                "direction_alignment": 0.94, "source": "observed", "measurement_valid": True,
                "quality": "high", "included": True, "exclusion_reason": None,
                "notes": "another private note",
            },
        ]
        self.report = report_for_formal_run(self.study, self.rows, bootstrap_iterations=200)
        self.run = {
            "id": "formal-run-secret",
            "status": "closed",
            "software_version": "2.21.0-rc.4",
            "clinical_protocol_version": "1.4",
            "snapshot_sha256": "b" * 64,
            "close_report_sha256": "c" * 64,
            "close_report": self.report,
        }
        self.requirements = {
            "schema_version": 1,
            "requirements": [{
                "id": "PRIV-001", "category": "privacy", "requirement": "deidentify",
                "implementation": ["backend/src/research/publication.py"],
                "verification": ["backend/tests/test_publication_package.py"],
                "evidence": "aliases",
            }],
        }

    def test_readiness_requires_closed_formal_run(self):
        ok = publication_readiness(run=self.run, study=self.study, report=self.report)
        self.assertTrue(ok["package_ready"])
        open_run = dict(self.run, status="open", close_report_sha256=None)
        bad = publication_readiness(run=open_run, study=self.study, report=self.report)
        self.assertFalse(bad["package_ready"])
        failed = {x["code"] for x in bad["checks"] if not x["ok"]}
        self.assertIn("formal_run_closed", failed)

    def test_anonymization_removes_internal_identifiers_and_free_text(self):
        rows = anonymize_trials(self.rows)
        dumped = json.dumps(rows, ensure_ascii=False)
        for forbidden in [
            "patient-hmac-A", "patient-hmac-B", "secret-job-a", "secret-job-b",
            "alice", "bob", "clinician-1", "SERIAL-SECRET-1", "private detail",
        ]:
            self.assertNotIn(forbidden, dumped)
        self.assertEqual(rows[0]["participant_alias"], "P001")
        self.assertEqual(rows[1]["participant_alias"], "P002")
        self.assertIn("error_deg", rows[0])

    def test_svg_figures_are_generated_without_external_runtime(self):
        anon = anonymize_trials(self.rows)
        comparison = build_method_comparison_svg(anon)
        bland = build_bland_altman_svg(self.report)
        self.assertIn("<svg", comparison)
        self.assertIn("Comparação de métodos", comparison)
        self.assertIn("<svg", bland)
        self.assertIn("Bland", bland)

    def test_external_bundle_is_deidentified_traceable_and_deterministic(self):
        kwargs = dict(
            run=self.run,
            study=self.study,
            trials=self.rows,
            report=self.report,
            statistical_plan={"clinical_protocol_version": "1.4", "primary": "MAE"},
            protocol_lock={"protocol_version": "1.4"},
            checklist={"schema_version": 1},
            requirements=self.requirements,
        )
        blob1 = build_publication_review_bundle(**kwargs)
        blob2 = build_publication_review_bundle(**kwargs)
        self.assertEqual(hashlib.sha256(blob1).hexdigest(), hashlib.sha256(blob2).hexdigest())

        with zipfile.ZipFile(io.BytesIO(blob1)) as zf:
            names = set(zf.namelist())
            required = {
                "manifest.json", "report.json", "manuscript_summary.md", "trials_anonymized.csv",
                "figures/method_comparison.svg", "figures/bland_altman.svg",
                "traceability/matrix.csv", "governance/protocol_lock.json", "checksums.sha256",
            }
            self.assertTrue(required.issubset(names))
            aggregate = b"\n".join(zf.read(name) for name in names if name != "checksums.sha256")
            for forbidden in [b"patient-hmac-A", b"secret-job-a", b"SERIAL-SECRET-1", b"private detail", b"alice"]:
                self.assertNotIn(forbidden, aggregate)

            checksum_lines = zf.read("checksums.sha256").decode("ascii").strip().splitlines()
            for line in checksum_lines:
                expected, name = line.split("  ", 1)
                self.assertEqual(hashlib.sha256(zf.read(name)).hexdigest(), expected)

            trial_text = zf.read("trials_anonymized.csv").decode("utf-8-sig")
            rows = list(csv.DictReader(io.StringIO(trial_text)))
            self.assertEqual(rows[0]["participant_alias"], "P001")
            self.assertEqual(rows[1]["session_alias"], "S002")

    def test_target_shortfall_is_warning_not_data_fabrication(self):
        study = dict(self.study, target_participants=20)
        ready = publication_readiness(run=self.run, study=study, report=self.report)
        self.assertTrue(ready["package_ready"])
        self.assertIn("target_not_reached", {x["code"] for x in ready["warnings"]})


if __name__ == "__main__":
    unittest.main()
