from __future__ import annotations

import unittest

import numpy as np

from src.core.motor_features import analyze_repetitive_motion, build_multimodal_motor_profile


class MotorFeatureTests(unittest.TestCase):
    @staticmethod
    def _measurements(*, fs: float = 30.0, seconds: float = 12.0, freq: float = 0.8,
                      amplitude_start: float = 30.0, amplitude_end: float | None = None):
        n = int(fs * seconds)
        t = np.arange(n, dtype=np.float64) / fs
        if amplitude_end is None:
            amp = np.full(n, amplitude_start)
        else:
            amp = np.linspace(amplitude_start, amplitude_end, n)
        angle = 60.0 + amp * np.sin(2.0 * np.pi * freq * t)
        return [
            {
                "frame": i,
                "timestamp_ms": float(tt * 1000.0),
                "angle_clinical_deg": float(value),
                "measurement_valid": True,
            }
            for i, (tt, value) in enumerate(zip(t, angle))
        ]

    def test_repetitive_motion_extracts_cycles_without_diagnostic_claim(self):
        result = analyze_repetitive_motion(self._measurements())
        self.assertEqual(result.status, "usable")
        self.assertGreaterEqual(result.cycle_count, 7)
        self.assertAlmostEqual(result.median_cycle_duration_s or 0.0, 1.25, delta=0.15)
        self.assertGreater(result.median_cycle_amplitude_deg or 0.0, 50.0)
        self.assertIsNone(result.tremor_dominant_frequency_hz)

    def test_amplitude_decrement_is_descriptive(self):
        result = analyze_repetitive_motion(
            self._measurements(amplitude_start=35.0, amplitude_end=18.0)
        )
        self.assertIsNotNone(result.amplitude_decrement_pct)
        self.assertGreater(result.amplitude_decrement_pct or 0.0, 20.0)

    def test_invalid_frames_reduce_quality(self):
        rows = self._measurements()
        for row in rows[:160]:
            row["measurement_valid"] = False
        result = analyze_repetitive_motion(rows)
        self.assertIn("low_valid_frame_ratio", result.quality_flags)
        self.assertEqual(result.status, "limited")

    def test_multimodal_profile_keeps_non_diagnostic_contract(self):
        rows = self._measurements()
        imu_record = {
            "id": "imu-1",
            "summary": {
                "status": "usable",
                "quality_score": 0.9,
                "dominant_frequency_hz": 5.1,
                "configured_band_power_ratio": 0.72,
                "periodicity_index": 0.81,
                "synchronization": {"overlap_ratio": 0.95},
            },
        }
        result = build_multimodal_motor_profile(rows, imu_record=imu_record)
        self.assertTrue(result.experimental)
        self.assertFalse(result.diagnostic)
        self.assertEqual(result.fusion_mode, "video_imu")
        self.assertIn("parkinson_diagnosis", result.unsupported_inferences)
        self.assertEqual(result.imu_analysis_id, "imu-1")

    def test_vision_only_mode_is_explicit(self):
        result = build_multimodal_motor_profile(self._measurements())
        self.assertEqual(result.fusion_mode, "vision_only")
        self.assertEqual(result.status, "vision_only")
        self.assertIn("imu_not_linked", result.quality_flags)


if __name__ == "__main__":
    unittest.main()
