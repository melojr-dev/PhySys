from __future__ import annotations

import unittest

from src.core.biomechanics import elbow_flexion_from_internal_angle


class ElbowClinicalConventionTests(unittest.TestCase):
    def test_extension_is_zero(self) -> None:
        self.assertAlmostEqual(elbow_flexion_from_internal_angle(180.0), 0.0)

    def test_ninety_internal_is_ninety_flexion(self) -> None:
        self.assertAlmostEqual(elbow_flexion_from_internal_angle(90.0), 90.0)

    def test_deep_flexion(self) -> None:
        self.assertAlmostEqual(elbow_flexion_from_internal_angle(35.0), 145.0)


if __name__ == "__main__":
    unittest.main()
