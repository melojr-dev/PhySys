import unittest

from src.validation.agreement import agreement_metrics, bland_altman_points, icc_a1


class AgreementMetricsTests(unittest.TestCase):
    def test_perfect_agreement(self) -> None:
        pairs = [(10, 10), (30, 30), (60, 60), (100, 100), (140, 140)]
        metrics = agreement_metrics(pairs)
        self.assertEqual(metrics["mae_deg"], 0.0)
        self.assertEqual(metrics["bias_deg"], 0.0)
        self.assertAlmostEqual(metrics["ccc"], 1.0, places=6)
        self.assertAlmostEqual(metrics["icc_a1"], 1.0, places=6)
        self.assertAlmostEqual(metrics["pearson_r"], 1.0, places=6)

    def test_constant_positive_bias(self) -> None:
        pairs = [(15, 10), (35, 30), (65, 60), (105, 100), (145, 140)]
        metrics = agreement_metrics(pairs)
        self.assertEqual(metrics["bias_deg"], 5.0)
        self.assertEqual(metrics["mae_deg"], 5.0)
        self.assertEqual(metrics["within_5_deg_pct"], 100.0)
        self.assertLess(metrics["ccc"], 1.0)
        self.assertLess(metrics["icc_a1"], 1.0)
        self.assertAlmostEqual(metrics["pearson_r"], 1.0, places=6)

    def test_bland_altman_convention(self) -> None:
        points = bland_altman_points([(100, 96), (80, 82)])
        self.assertEqual(points[0], {"mean_deg": 98.0, "difference_deg": 4.0})
        self.assertEqual(points[1], {"mean_deg": 81.0, "difference_deg": -2.0})

    def test_icc_requires_two_targets(self) -> None:
        self.assertIsNone(icc_a1([(10, 10)]))


if __name__ == "__main__":
    unittest.main()
