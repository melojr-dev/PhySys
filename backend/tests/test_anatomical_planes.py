from __future__ import annotations

import unittest

import numpy as np

from src.core.anatomy import (
    alignment_deviation_deg,
    build_body_frame,
    camera_view_alignment,
    max_out_of_plane_deg,
    movement_direction_alignment,
    projected_internal_angle_deg,
)


class AnatomicalPlaneTests(unittest.TestCase):
    def _skeleton_lateral_view(self) -> np.ndarray:
        skel = np.zeros((33, 3), dtype=np.float64)
        # Eixo mediolateral alinhado ao eixo Z da câmera -> vista lateral ideal.
        skel[11] = [0.0, 1.0, -0.2]  # left shoulder
        skel[12] = [0.0, 1.0, 0.2]   # right shoulder
        skel[23] = [0.0, 0.0, -0.16] # left hip
        skel[24] = [0.0, 0.0, 0.16]  # right hip
        return skel

    def test_build_body_frame_and_lateral_alignment(self) -> None:
        frame = build_body_frame(self._skeleton_lateral_view())
        self.assertIsNotNone(frame)
        alignment = camera_view_alignment(frame, "sagittal")
        self.assertAlmostEqual(alignment, 1.0, places=6)
        self.assertAlmostEqual(alignment_deviation_deg(alignment), 0.0, places=6)

    def test_projected_angle_ignores_normal_component(self) -> None:
        frame = build_body_frame(self._skeleton_lateral_view())
        self.assertIsNotNone(frame)
        # Sagital neste arranjo = XY. Componentes Z devem ser removidos.
        vertex = np.array([0.0, 0.0, 0.0])
        proximal = np.array([0.0, 1.0, 0.7])
        distal = np.array([1.0, 0.0, -0.4])
        angle = projected_internal_angle_deg(proximal, vertex, distal, frame=frame, plane="sagittal")
        self.assertAlmostEqual(angle, 90.0, places=5)

    def test_out_of_plane_detects_large_component(self) -> None:
        frame = build_body_frame(self._skeleton_lateral_view())
        self.assertIsNotNone(frame)
        vertex = np.array([0.0, 0.0, 0.0])
        proximal = np.array([0.0, 1.0, 1.0])
        distal = np.array([1.0, 0.0, 0.0])
        value = max_out_of_plane_deg(proximal, vertex, distal, frame=frame, plane="sagittal")
        self.assertAlmostEqual(value, 45.0, places=5)

    def test_frontal_camera_is_invalid_for_sagittal_capture(self) -> None:
        skel = np.zeros((33, 3), dtype=np.float64)
        skel[11] = [-0.2, 1.0, 0.0]
        skel[12] = [0.2, 1.0, 0.0]
        skel[23] = [-0.16, 0.0, 0.0]
        skel[24] = [0.16, 0.0, 0.0]
        frame = build_body_frame(skel)
        self.assertIsNotNone(frame)
        self.assertAlmostEqual(camera_view_alignment(frame, "sagittal"), 0.0, places=6)
        self.assertAlmostEqual(alignment_deviation_deg(0.0), 90.0, places=6)

    def test_degenerate_body_frame_returns_none(self) -> None:
        self.assertIsNone(build_body_frame(np.zeros((33, 3), dtype=np.float64)))

    def test_shoulder_flexion_direction_in_sagittal_plane(self) -> None:
        frame = build_body_frame(self._skeleton_lateral_view())
        self.assertIsNotNone(frame)
        # Neste referencial sintético, anterior aponta para -X.
        flexion = movement_direction_alignment(
            np.array([-1.0, 0.0, 0.0]), frame=frame, plane="sagittal",
            direction_rule="anterior_positive", side="direito",
        )
        extension = movement_direction_alignment(
            np.array([1.0, 0.0, 0.0]), frame=frame, plane="sagittal",
            direction_rule="anterior_positive", side="direito",
        )
        self.assertAlmostEqual(flexion, 1.0, places=6)
        self.assertAlmostEqual(extension, -1.0, places=6)

    def test_shoulder_abduction_direction_respects_side(self) -> None:
        skel = np.zeros((33, 3), dtype=np.float64)
        skel[11] = [-0.2, 1.0, 0.0]
        skel[12] = [0.2, 1.0, 0.0]
        skel[23] = [-0.2, 0.0, 0.0]
        skel[24] = [0.2, 0.0, 0.0]
        frame = build_body_frame(skel)
        self.assertIsNotNone(frame)
        right = movement_direction_alignment(
            np.array([1.0, 0.0, 0.0]), frame=frame, plane="frontal",
            direction_rule="lateral_outward", side="direito",
        )
        left = movement_direction_alignment(
            np.array([-1.0, 0.0, 0.0]), frame=frame, plane="frontal",
            direction_rule="lateral_outward", side="esquerdo",
        )
        crossed = movement_direction_alignment(
            np.array([-1.0, 0.0, 0.0]), frame=frame, plane="frontal",
            direction_rule="lateral_outward", side="direito",
        )
        self.assertAlmostEqual(right, 1.0, places=6)
        self.assertAlmostEqual(left, 1.0, places=6)
        self.assertAlmostEqual(crossed, -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
