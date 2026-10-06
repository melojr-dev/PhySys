from __future__ import annotations

import unittest

import numpy as np

from src.core.gait import analyze_gait_temporal


def synthetic_gait(fps: float = 30.0, seconds: float = 10.0) -> np.ndarray:
    n = int(fps * seconds)
    t = np.arange(n) / fps
    skel = np.zeros((n, 33, 3), dtype=np.float64)
    skel[:, 23, :] = (-0.1, 0.0, 0.0)
    skel[:, 24, :] = (0.1, 0.0, 0.0)
    skel[:, 29, 1] = 0.25 + 0.04 * np.cos(2 * np.pi * 1.0 * t)
    skel[:, 30, 1] = 0.25 + 0.04 * np.cos(2 * np.pi * 1.0 * t + np.pi)
    skel[:, 29, 0] = -0.12
    skel[:, 30, 0] = 0.12
    skel[:, 29, 2] = 0.1 * t
    skel[:, 30, 2] = 0.1 * t
    return skel


class GaitTemporalTests(unittest.TestCase):
    def test_temporal_gait_detects_cadence_and_events(self):
        result = analyze_gait_temporal(synthetic_gait(), 30.0)
        self.assertTrue(result.experimental)
        self.assertGreaterEqual(result.event_count, 16)
        self.assertGreaterEqual(result.left_event_count, 8)
        self.assertGreaterEqual(result.right_event_count, 8)
        self.assertIsNotNone(result.cadence_steps_min)
        self.assertAlmostEqual(result.cadence_steps_min, 120.0, delta=10.0)
        self.assertIsNotNone(result.step_time_symmetry_ratio)
        self.assertGreater(result.step_time_symmetry_ratio, 0.9)
        self.assertIsNone(result.stride_length_m)
        self.assertIn(
            "spatial_metrics_disabled_non_global_coordinates",
            result.quality_flags,
        )

    def test_metric_global_enables_stride_length(self):
        result = analyze_gait_temporal(
            synthetic_gait(), 30.0, coordinate_space="metric_global"
        )
        self.assertIsNotNone(result.stride_length_m)
        self.assertGreater(result.stride_length_m, 0.0)
        self.assertNotIn(
            "spatial_metrics_disabled_non_global_coordinates",
            result.quality_flags,
        )

    def test_invalid_shape_rejected(self):
        with self.assertRaises(ValueError):
            analyze_gait_temporal(np.zeros((10, 10, 3)), 30.0)

    def test_short_recording_flagged(self):
        result = analyze_gait_temporal(synthetic_gait(seconds=2.0), 30.0)
        self.assertIn("recording_too_short", result.quality_flags)
        self.assertEqual(result.status, "limited")


if __name__ == "__main__":
    unittest.main()
