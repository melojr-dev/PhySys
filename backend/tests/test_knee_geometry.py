from __future__ import annotations

import unittest

from src.core.biomechanics import knee_flexion_from_internal_angle


class KneeClinicalConventionTests(unittest.TestCase):
    def test_extension_is_zero(self) -> None:
        self.assertAlmostEqual(knee_flexion_from_internal_angle(180.0), 0.0)

    def test_ninety_internal_is_ninety_flexion(self) -> None:
        self.assertAlmostEqual(knee_flexion_from_internal_angle(90.0), 90.0)

    def test_flexion_135(self) -> None:
        self.assertAlmostEqual(knee_flexion_from_internal_angle(45.0), 135.0)


if __name__ == "__main__":
    unittest.main()
