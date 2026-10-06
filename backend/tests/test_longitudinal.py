from __future__ import annotations

import unittest

from src.core.longitudinal import build_longitudinal_profile, summarize_metric


class LongitudinalTests(unittest.TestCase):
    def test_metric_summary_delta_cv_and_slope(self):
        metric = summarize_metric(
            "adm_deg", "deg",
            [
                ("2026-01-01T00:00:00+00:00", 100.0),
                ("2026-01-31T00:00:00+00:00", 110.0),
                ("2026-03-02T00:00:00+00:00", 120.0),
            ],
        )
        self.assertEqual(metric.n, 3)
        self.assertEqual(metric.delta, 20.0)
        self.assertEqual(metric.delta_pct, 20.0)
        self.assertAlmostEqual(metric.slope_per_30d or 0.0, 10.0, places=5)
        self.assertAlmostEqual(metric.median_abs_consecutive_change or 0.0, 10.0)

    def test_profile_excludes_incompatible_protocol(self):
        sessions = [
            self._session("j1", "2026-01-01T00:00:00+00:00", 90, protocol="1.4"),
            self._session("j2", "2026-02-01T00:00:00+00:00", 95, protocol="1.4"),
            self._session("j3", "2026-03-01T00:00:00+00:00", 110, protocol="1.5"),
        ]
        result = build_longitudinal_profile(sessions, anchor_job_id="j2")
        self.assertEqual(result["session_count"], 2)
        self.assertEqual(result["excluded_session_count"], 1)
        self.assertIn("noncomparable_sessions_excluded", result["quality_flags"])
        self.assertEqual(result["metrics"]["adm_deg"]["latest"], 95.0)

    def test_mixed_software_same_protocol_is_flagged_not_excluded(self):
        sessions = [
            self._session("j1", "2026-01-01T00:00:00+00:00", 90, software="2.15"),
            self._session("j2", "2026-02-01T00:00:00+00:00", 91, software="2.16"),
            self._session("j3", "2026-03-01T00:00:00+00:00", 92, software="2.16"),
        ]
        result = build_longitudinal_profile(sessions, anchor_job_id="j3")
        self.assertEqual(result["session_count"], 3)
        self.assertTrue(result["mixed_software_versions"])
        self.assertIn("mixed_software_versions_same_protocol", result["quality_flags"])

    def test_motor_feature_metrics_are_optional(self):
        sessions = [
            self._session("j1", "2026-01-01T00:00:00+00:00", 90),
            self._session("j2", "2026-02-01T00:00:00+00:00", 95),
        ]
        motor = {
            "j1": [{"summary": {"vision": {"median_cycle_amplitude_deg": 40.0}, "fusion_quality_score": 0.8}}],
            "j2": [{"summary": {"vision": {"median_cycle_amplitude_deg": 44.0}, "fusion_quality_score": 0.9}}],
        }
        result = build_longitudinal_profile(sessions, anchor_job_id="j2", motor_features_by_job=motor)
        self.assertEqual(result["metrics"]["median_cycle_amplitude_deg"]["delta"], 4.0)
        self.assertAlmostEqual(result["metrics"]["fusion_quality_score"]["latest"], 0.9)

    def test_single_session_is_insufficient_and_non_diagnostic(self):
        result = build_longitudinal_profile(
            [self._session("j1", "2026-01-01T00:00:00+00:00", 90)], anchor_job_id="j1"
        )
        self.assertEqual(result["status"], "insufficient")
        self.assertFalse(result["diagnostic"])
        self.assertIn("clinical_improvement", result["unsupported_inferences"])

    def test_missing_anchor_raises(self):
        with self.assertRaises(ValueError):
            build_longitudinal_profile([self._session("j1", "2026-01-01T00:00:00+00:00", 90)], anchor_job_id="missing")

    @staticmethod
    def _session(job: str, started: str, adm: float, *, protocol: str = "1.4", software: str = "2.16"):
        return {
            "job_id": job,
            "patient_pseudonym": "pat",
            "evaluator_id": "alice",
            "movement": "Flexão de Cotovelo",
            "side": "direito",
            "started_at": started,
            "completed_at": started,
            "status": "completed",
            "total_frames": 100,
            "valid_frames": 90,
            "fps": 30.0,
            "adm_deg": adm,
            "total_reps": 5,
            "software_version": software,
            "protocol_version": protocol,
            "clinical_spec_sha256": "a" * 64,
            "result_sha256": job * 8,
        }


if __name__ == "__main__":
    unittest.main()
