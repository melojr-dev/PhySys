from __future__ import annotations

import enum
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


class _PoseLandmark(enum.IntEnum):
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


def _install_fake_mediapipe() -> None:
    if "mediapipe.python.solutions.pose" in sys.modules:
        return
    pose_module = types.ModuleType("mediapipe.python.solutions.pose")
    pose_module.PoseLandmark = _PoseLandmark
    pose_module.Pose = object
    solutions_module = types.ModuleType("mediapipe.python.solutions")
    solutions_module.pose = pose_module
    python_module = types.ModuleType("mediapipe.python")
    python_module.solutions = solutions_module
    mediapipe_module = types.ModuleType("mediapipe")
    mediapipe_module.python = python_module
    sys.modules["mediapipe"] = mediapipe_module
    sys.modules["mediapipe.python"] = python_module
    sys.modules["mediapipe.python.solutions"] = solutions_module
    sys.modules["mediapipe.python.solutions.pose"] = pose_module


_install_fake_mediapipe()
from src.core import capture  # noqa: E402


class _FakePose:
    def __init__(self, result):
        self.result = result

    def process(self, _image):
        return self.result

    def close(self):
        pass


def _landmark(x=0.5, y=0.5, z=0.0, visibility=0.97):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def _good_result():
    two_d = [_landmark() for _ in range(33)]
    # Todos os landmarks usados no enquadramento ficam dentro do frame.
    coords_2d = {
        11: (0.46, 0.25), 12: (0.54, 0.25),
        23: (0.46, 0.56), 24: (0.54, 0.56),
        14: (0.62, 0.42), 16: (0.70, 0.58),
    }
    for idx, (x, y) in coords_2d.items():
        two_d[idx] = _landmark(x, y, 0.0, 0.97)

    world = [_landmark(0.0, 0.0, 0.0, 0.97) for _ in range(33)]
    # lateral = eixo Z, superior = eixo Y, logo a vista lateral (normal sagital)
    # fica alinhada ao eixo óptico +Z usado pelo protocolo.
    world[11] = _landmark(0.0, 1.0, -0.20)
    world[12] = _landmark(0.0, 1.0, 0.20)
    world[23] = _landmark(0.0, 0.0, -0.20)
    world[24] = _landmark(0.0, 0.0, 0.20)
    world[14] = _landmark(0.30, 0.62, 0.20)
    world[16] = _landmark(0.55, 0.28, 0.20)
    return SimpleNamespace(
        pose_landmarks=SimpleNamespace(landmark=two_d),
        pose_world_landmarks=SimpleNamespace(landmark=world),
    )


def _make_video(path: Path, *, width=1280, height=720, fps=30.0, frames=5):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter indisponível")
    tile = 40
    yy, xx = np.indices((height, width))
    checker = (((xx // tile + yy // tile) % 2) * 170 + 40).astype(np.uint8)
    base = np.dstack([checker, checker, checker])
    for i in range(frames):
        frame = base.copy()
        cv2.circle(frame, (120 + i * 20, 120), 18, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


class CapturePreflightTests(unittest.TestCase):
    def test_good_capture_passes_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "good.mp4"
            _make_video(video)
            fake = _FakePose(_good_result())
            with patch.object(capture.mp_pose, "Pose", return_value=fake):
                report = capture.evaluate_capture_preflight(
                    str(video), "Flexão de Cotovelo", "direito", max_samples=5
                )
            self.assertTrue(report.can_analyze)
            self.assertEqual(report.status, "pass")
            self.assertGreaterEqual(report.score, 90)
            by_code = {item.code: item for item in report.checks}
            self.assertEqual(by_code["pose_detection"].status, "pass")
            self.assertEqual(by_code["view_alignment"].status, "pass")
            self.assertEqual(by_code["movement_plane"].status, "pass")

    def test_missing_pose_blocks_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "no_pose.mp4"
            _make_video(video)
            result = SimpleNamespace(pose_landmarks=None, pose_world_landmarks=None)
            with patch.object(capture.mp_pose, "Pose", return_value=_FakePose(result)):
                report = capture.evaluate_capture_preflight(
                    str(video), "Flexão de Cotovelo", "direito", max_samples=5
                )
            self.assertFalse(report.can_analyze)
            self.assertEqual(report.status, "fail")
            by_code = {item.code: item for item in report.checks}
            self.assertEqual(by_code["pose_detection"].status, "fail")
            self.assertEqual(by_code["landmark_visibility"].status, "fail")

    def test_invalid_video_returns_failed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.mp4"
            path.write_bytes(b"not-a-video")
            report = capture.evaluate_capture_preflight(
                str(path), "Flexão de Cotovelo", "direito", max_samples=5
            )
            self.assertFalse(report.can_analyze)
            self.assertEqual(report.status, "fail")
            self.assertEqual(report.checks[0].code, "decode")


if __name__ == "__main__":
    unittest.main()
