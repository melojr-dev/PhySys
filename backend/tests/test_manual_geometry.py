import unittest

from src.core.clinical import (
    clinical_angle_from_points_2d,
    internal_angle_from_points_2d,
)


class ManualGeometryTests(unittest.TestCase):
    def test_straight_line_is_180_internal_and_zero_flexion(self):
        p = {"x": 0.0, "y": 0.0}
        v = {"x": 1.0, "y": 0.0}
        d = {"x": 2.0, "y": 0.0}
        self.assertAlmostEqual(internal_angle_from_points_2d(p, v, d), 180.0, places=7)
        self.assertAlmostEqual(clinical_angle_from_points_2d("Flexão de Cotovelo", p, v, d), 0.0, places=7)

    def test_right_angle_is_90_flexion(self):
        p = (0.0, 0.0)
        v = (1.0, 0.0)
        d = (1.0, 1.0)
        self.assertAlmostEqual(internal_angle_from_points_2d(p, v, d), 90.0, places=7)
        self.assertAlmostEqual(clinical_angle_from_points_2d("Flexão de Joelho", p, v, d), 90.0, places=7)


    def test_hip_flexion_uses_same_internal_angle_convention(self):
        p = (0.0, -1.0)
        v = (0.0, 0.0)
        d = (1.0, 0.0)
        self.assertAlmostEqual(internal_angle_from_points_2d(p, v, d), 90.0, places=7)
        self.assertAlmostEqual(clinical_angle_from_points_2d("Flexão de Quadril", p, v, d), 90.0, places=7)

    def test_ankle_dorsiflexion_uses_ninety_degree_neutral(self):
        # 90° tíbia-pé = neutro; 70° = 20° de dorsiflexão.
        neutral = clinical_angle_from_points_2d("Dorsiflexão de Tornozelo", (0, -1), (0, 0), (1, 0))
        self.assertAlmostEqual(neutral, 0.0, places=7)
        angle70 = clinical_angle_from_points_2d(
            "Dorsiflexão de Tornozelo",
            (0, -1),
            (0, 0),
            (0.9396926208, -0.3420201433),
        )
        self.assertAlmostEqual(angle70, 20.0, places=5)

    def test_uniform_scale_does_not_change_angle(self):
        a = clinical_angle_from_points_2d("Flexão de Cotovelo", (0, 0), (1, 0), (1, 1))
        b = clinical_angle_from_points_2d("Flexão de Cotovelo", (0, 0), (100, 0), (100, 100))
        self.assertAlmostEqual(a, b, places=7)

    def test_coincident_points_are_rejected(self):
        with self.assertRaises(ValueError):
            internal_angle_from_points_2d((0, 0), (0, 0), (1, 1))


if __name__ == "__main__":
    unittest.main()
