from __future__ import annotations

import unittest

from src.core.clinical import (
    ankle_dorsiflexion_from_internal_angle,
    flexion_from_internal_angle,
    get_movement_spec,
    public_movement_specs,
    tracking_quality,
    apply_plane_quality,
)


class ClinicalSpecsTests(unittest.TestCase):
    def test_flexion_convention(self) -> None:
        self.assertAlmostEqual(flexion_from_internal_angle(180.0), 0.0)
        self.assertAlmostEqual(flexion_from_internal_angle(90.0), 90.0)
        self.assertAlmostEqual(flexion_from_internal_angle(45.0), 135.0)

    def test_elbow_and_knee_are_explicit_specs(self) -> None:
        elbow = get_movement_spec("Flexão de Cotovelo")
        knee = get_movement_spec("Flexão de Joelho")
        self.assertIsNotNone(elbow)
        self.assertIsNotNone(knee)
        self.assertEqual(elbow.vertex_node, "cotovelo")
        self.assertEqual(knee.vertex_node, "joelho")
        self.assertEqual(knee.joint_code, "knee")


    def test_hip_and_ankle_are_explicit_preclinical_specs(self) -> None:
        hip = get_movement_spec("Flexão de Quadril")
        ankle = get_movement_spec("Dorsiflexão de Tornozelo")
        self.assertIsNotNone(hip)
        self.assertIsNotNone(ankle)
        self.assertEqual(hip.joint_code, "hip")
        self.assertEqual((hip.proximal_node, hip.vertex_node, hip.distal_node), ("ombro", "quadril", "joelho"))
        self.assertEqual(ankle.joint_code, "ankle")
        self.assertEqual((ankle.proximal_node, ankle.vertex_node, ankle.distal_node), ("joelho", "tornozelo", "pe"))
        self.assertEqual(hip.reference_rom_deg, 120.0)
        self.assertEqual(ankle.reference_rom_deg, 20.0)
        self.assertFalse(hip.validated)
        self.assertFalse(ankle.validated)

    def test_ankle_dorsiflexion_convention(self) -> None:
        self.assertAlmostEqual(ankle_dorsiflexion_from_internal_angle(90.0), 0.0)
        self.assertAlmostEqual(ankle_dorsiflexion_from_internal_angle(80.0), 10.0)
        self.assertAlmostEqual(ankle_dorsiflexion_from_internal_angle(70.0), 20.0)
        self.assertAlmostEqual(ankle_dorsiflexion_from_internal_angle(100.0), 0.0)

    def test_public_specs_expose_capture_protocol_and_converter(self) -> None:
        specs = {item["name"]: item for item in public_movement_specs()}
        self.assertEqual(set(specs), {
            "Flexão de Cotovelo",
            "Flexão de Joelho",
            "Flexão de Quadril",
            "Dorsiflexão de Tornozelo",
            "Flexão de Ombro",
            "Abdução de Ombro",
        })
        ankle = specs["Dorsiflexão de Tornozelo"]
        self.assertEqual(ankle["convention"], "ankle_dorsiflexion_from_internal_angle")
        self.assertEqual(ankle["measurementPlane"], "sagittal")
        self.assertEqual(ankle["recommendedView"], "lateral")
        self.assertEqual(ankle["landmarks"]["vertex"], "tornozelo")
        self.assertGreaterEqual(len(ankle["captureInstructions"]), 2)

    def test_tracking_quality_is_conservative(self) -> None:
        q = tracking_quality([0.95, 0.88, 0.91], source="observed")
        self.assertTrue(q["measurement_valid"])
        self.assertEqual(q["quality"], "good")
        self.assertAlmostEqual(q["landmark_confidence_min"], 0.88)

    def test_one_low_landmark_invalidates_measurement(self) -> None:
        q = tracking_quality([0.95, 0.40, 0.91], source="observed")
        self.assertFalse(q["measurement_valid"])
        self.assertIn("low_landmark_visibility", q["quality_flags"])

    def test_predicted_never_becomes_valid_measurement(self) -> None:
        q = tracking_quality([0.99, 0.99, 0.99], source="predicted")
        self.assertFalse(q["measurement_valid"])
        self.assertEqual(q["quality"], "predicted")

    def test_preclinical_status_and_reference_values(self) -> None:
        elbow = get_movement_spec("Flexão de Cotovelo")
        knee = get_movement_spec("Flexão de Joelho")
        self.assertFalse(elbow.validated)
        self.assertFalse(knee.validated)
        self.assertEqual(elbow.reference_rom_deg, 145.0)
        self.assertEqual(knee.reference_rom_deg, 135.0)

    def test_public_specs_expose_plane_quality_limits(self) -> None:
        specs = {item["name"]: item for item in public_movement_specs()}
        elbow = specs["Flexão de Cotovelo"]
        self.assertGreaterEqual(elbow["minViewAlignment"], 0.8)
        self.assertGreater(elbow["maxOutOfPlaneDeg"], 0.0)
        self.assertTrue(elbow["requirePlaneProjection"])

    def test_plane_quality_rejects_misaligned_capture(self) -> None:
        base = tracking_quality([0.98, 0.97, 0.96], source="observed")
        result = apply_plane_quality(
            base, body_frame_available=True, body_frame_confidence=0.95,
            view_alignment=0.3, min_view_alignment=0.8,
            out_of_plane_deg=5.0, max_out_of_plane_deg=35.0,
            projection_applied=True,
        )
        self.assertFalse(result["measurement_valid"])
        self.assertIn("camera_view_misaligned", result["quality_flags"])

    def test_plane_quality_accepts_good_geometry(self) -> None:
        base = tracking_quality([0.98, 0.97, 0.96], source="observed")
        result = apply_plane_quality(
            base, body_frame_available=True, body_frame_confidence=0.95,
            view_alignment=0.96, min_view_alignment=0.8,
            out_of_plane_deg=8.0, max_out_of_plane_deg=35.0,
            projection_applied=True,
        )
        self.assertTrue(result["measurement_valid"])
        self.assertEqual(result["quality_flags"], [])

    def test_shoulder_specs_are_multiplanar_and_directional(self) -> None:
        flex = get_movement_spec("Flexão de Ombro")
        abd = get_movement_spec("Abdução de Ombro")
        self.assertIsNotNone(flex)
        self.assertIsNotNone(abd)
        self.assertEqual(flex.measurement_plane, "sagittal")
        self.assertEqual(abd.measurement_plane, "frontal")
        self.assertEqual(flex.converter_id, "direct_internal_angle")
        self.assertEqual(abd.converter_id, "direct_internal_angle")
        self.assertEqual(flex.direction_rule, "anterior_positive")
        self.assertEqual(abd.direction_rule, "lateral_outward")
        self.assertEqual((flex.proximal_node, flex.vertex_node, flex.distal_node), ("quadril", "ombro", "cotovelo"))
        self.assertFalse(flex.validated)
        self.assertFalse(abd.validated)

    def test_direction_gate_rejects_opposite_motion(self) -> None:
        base = tracking_quality([0.98, 0.97, 0.96], source="observed")
        result = apply_plane_quality(
            base, body_frame_available=True, body_frame_confidence=0.95,
            view_alignment=0.96, min_view_alignment=0.8,
            out_of_plane_deg=8.0, max_out_of_plane_deg=35.0,
            projection_applied=True, direction_rule="anterior_positive",
            direction_alignment=-0.8, min_direction_alignment=-0.1,
        )
        self.assertFalse(result["measurement_valid"])
        self.assertIn("movement_direction_mismatch", result["quality_flags"])

    def test_direction_gate_accepts_neutral_or_correct_direction(self) -> None:
        for alignment in (0.0, 0.85):
            base = tracking_quality([0.98, 0.97, 0.96], source="observed")
            result = apply_plane_quality(
                base, body_frame_available=True, body_frame_confidence=0.95,
                view_alignment=0.96, min_view_alignment=0.8,
                out_of_plane_deg=8.0, max_out_of_plane_deg=35.0,
                projection_applied=True, direction_rule="anterior_positive",
                direction_alignment=alignment, min_direction_alignment=-0.1,
            )
            self.assertTrue(result["measurement_valid"])


if __name__ == "__main__":
    unittest.main()
