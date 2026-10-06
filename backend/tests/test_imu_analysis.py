from __future__ import annotations

import unittest

import numpy as np

from src.core.imu import analyze_imu_signal, synchronize_imu_to_video


class IMUAnalysisTests(unittest.TestCase):
    def _signal(self, *, fs: float = 100.0, seconds: float = 10.0, freq: float = 5.0):
        n = int(fs * seconds)
        t_s = np.arange(n, dtype=np.float64) / fs
        timestamps_ms = t_s * 1000.0
        wave = 0.35 * np.sin(2.0 * np.pi * freq * t_s)
        gyro = np.column_stack((wave, 0.05 * wave, np.zeros(n)))
        accel = np.column_stack((
            0.04 * np.sin(2.0 * np.pi * 1.0 * t_s),
            np.zeros(n),
            np.full(n, 9.81),
        ))
        return timestamps_ms, gyro, accel

    def test_detects_dominant_frequency_without_diagnostic_claim(self):
        t, gyro, accel = self._signal()
        result = analyze_imu_signal(t, gyro, accel_m_s2=accel, band_hz=(4.0, 6.0))
        self.assertTrue(result.experimental)
        self.assertFalse(result.diagnostic)
        self.assertEqual(result.status, "usable")
        self.assertAlmostEqual(result.sampling_rate_hz, 100.0, delta=0.5)
        self.assertIsNotNone(result.dominant_frequency_hz)
        self.assertAlmostEqual(result.dominant_frequency_hz, 5.0, delta=0.35)
        self.assertIsNotNone(result.configured_band_power_ratio)
        self.assertGreater(result.configured_band_power_ratio, 0.75)
        self.assertIsNotNone(result.periodicity_index)

    def test_sampling_jitter_is_flagged(self):
        t, gyro, _ = self._signal(seconds=8.0)
        # jitter alternado suficientemente grande para exceder CV de 5%
        t = t + np.where(np.arange(t.size) % 2 == 0, -1.5, 1.5)
        t = np.maximum.accumulate(t)
        result = analyze_imu_signal(t, gyro)
        self.assertIn("high_sampling_jitter", result.quality_flags)

    def test_video_sync_respects_offset_convention(self):
        imu = np.arange(0.0, 1000.0, 10.0)
        video = np.arange(100.0, 900.0, 40.0)
        sync = synchronize_imu_to_video(imu, video, offset_ms=100.0, linked_job_id="job-1")
        self.assertEqual(sync.linked_job_id, "job-1")
        self.assertGreater(sync.overlap_ratio, 0.95)
        self.assertAlmostEqual(sync.median_alignment_error_ms or 0.0, 0.0, delta=0.01)

    def test_invalid_shape_rejected(self):
        with self.assertRaises(ValueError):
            analyze_imu_signal(np.arange(100), np.zeros((100, 2)))

    def test_short_signal_rejected(self):
        with self.assertRaises(ValueError):
            analyze_imu_signal(np.arange(10), np.zeros((10, 3)))


if __name__ == "__main__":
    unittest.main()
