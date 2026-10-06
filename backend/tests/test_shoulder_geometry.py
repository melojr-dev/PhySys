from __future__ import annotations

import unittest

from src.core.clinical import clinical_angle_from_points_2d, direct_internal_angle


class ShoulderClinicalConventionTests(unittest.TestCase):
    def test_direct_converter_preserves_projected_angle(self) -> None:
        self.assertAlmostEqual(direct_internal_angle(0.0), 0.0)
        self.assertAlmostEqual(direct_internal_angle(90.0), 90.0)
        self.assertAlmostEqual(direct_internal_angle(180.0), 180.0)

    def test_neutral_arm_is_zero_degrees(self) -> None:
        hip = (0.0, -1.0)
        shoulder = (0.0, 0.0)
        elbow = (0.0, -2.0)
        self.assertAlmostEqual(
            clinical_angle_from_points_2d("Flexão de Ombro", hip, shoulder, elbow),
            0.0,
        )

    def test_ninety_degree_shoulder_elevation_is_ninety(self) -> None:
        hip = (0.0, -1.0)
        shoulder = (0.0, 0.0)
        elbow = (1.0, 0.0)
        self.assertAlmostEqual(
            clinical_angle_from_points_2d("Flexão de Ombro", hip, shoulder, elbow),
            90.0,
        )
        self.assertAlmostEqual(
            clinical_angle_from_points_2d("Abdução de Ombro", hip, shoulder, elbow),
            90.0,
        )

    def test_overhead_elevation_is_180(self) -> None:
        hip = (0.0, -1.0)
        shoulder = (0.0, 0.0)
        elbow = (0.0, 1.0)
        self.assertAlmostEqual(
            clinical_angle_from_points_2d("Flexão de Ombro", hip, shoulder, elbow),
            180.0,
        )


if __name__ == "__main__":
    unittest.main()
