import tempfile
import unittest
from pathlib import Path

from src.data.measurement_store import MeasurementStore


class MeasurementStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MeasurementStore(Path(self.tmp.name) / "clinical.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def _row(self, frame=0):
        return {
            "frame": frame,
            "timestamp_ms": frame * 33.333,
            "joint": "direito_elbow",
            "movement": "Flexão de Cotovelo",
            "side": "direito",
            "angle_raw_deg": 80.0,
            "angle_landmark_deg": 81.0,
            "angle_plane_deg": 80.5,
            "angle_filtered_deg": 82.0,
            "angle_clinical_deg": 98.0,
            "angular_velocity_deg_s": 12.0,
            "adm_deg": 40.0,
            "confidence": 0.9,
            "source": "observed",
            "measurement_space": "world_3d",
            "measurement_plane": "sagittal",
            "plane_projection_applied": True,
            "body_frame_confidence": 0.92,
            "view_alignment": 0.97,
            "view_deviation_deg": 14.1,
            "out_of_plane_deg": 8.0,
            "direction_rule": "anterior_positive",
            "direction_alignment": 0.82,
            "landmark_confidence_min": 0.9,
            "landmark_confidence_mean": 0.94,
            "quality": "excellent",
            "measurement_valid": True,
            "quality_flags": [],
            "proximal": {"x": 0.1, "y": 0.2, "confidence": 0.95},
            "vertex": {"x": 0.2, "y": 0.3, "confidence": 0.93},
            "distal": {"x": 0.3, "y": 0.2, "confidence": 0.9},
            "proximal_3d": {"x": 0.1, "y": 0.2, "z": -0.1},
            "vertex_3d": {"x": 0.2, "y": 0.3, "z": -0.2},
            "distal_3d": {"x": 0.3, "y": 0.2, "z": -0.3},
            "inference_ms": 10.2,
            "pipeline_ms": 14.8,
            "processing_fps": 67.5,
        }

    def test_roundtrip_frame(self):
        self.assertEqual(self.store.save_many("job-1", [self._row()]), 1)
        row = self.store.get("job-1", 0)
        self.assertIsNotNone(row)
        self.assertTrue(row["measurement_valid"])
        self.assertEqual(row["quality_flags"], [])
        self.assertAlmostEqual(row["angle_landmark_deg"], 81.0)
        self.assertAlmostEqual(row["angle_plane_deg"], 80.5)
        self.assertEqual(row["measurement_plane"], "sagittal")
        self.assertTrue(row["plane_projection_applied"])
        self.assertAlmostEqual(row["view_alignment"], 0.97)
        self.assertEqual(row["direction_rule"], "anterior_positive")
        self.assertAlmostEqual(row["direction_alignment"], 0.82)
        self.assertAlmostEqual(row["angle_clinical_deg"], 98.0)
        self.assertAlmostEqual(row["vertex"]["x"], 0.2)
        self.assertAlmostEqual(row["vertex_3d"]["z"], -0.2)

    def test_order_and_delete(self):
        self.store.save_many("job-1", [self._row(2), self._row(0), self._row(1)])
        rows = self.store.list_for_job("job-1")
        self.assertEqual([r["frame"] for r in rows], [0, 1, 2])
        self.assertEqual(self.store.delete_job("job-1"), 3)
        self.assertEqual(self.store.list_for_job("job-1"), [])


if __name__ == "__main__":
    unittest.main()
