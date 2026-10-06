from __future__ import annotations

import enum
import os
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient


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


class _Pose:
    def __init__(self, **_kwargs):
        pass

    def process(self, _image):
        # Exercita deliberadamente o caminho sem corpo detectado.
        return SimpleNamespace(pose_landmarks=None, pose_world_landmarks=None)

    def close(self):
        pass


def _install_fake_mediapipe() -> None:
    pose_module = types.ModuleType("mediapipe.python.solutions.pose")
    pose_module.Pose = _Pose
    pose_module.PoseLandmark = _PoseLandmark

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


def _make_video(path: Path, frames: int = 6) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
    )
    if not writer.isOpened():
        raise RuntimeError("VideoWriter indisponível para teste E2E")
    for i in range(frames):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.circle(image, (20 + i * 15, 60), 8, (255, 255, 255), -1)
        writer.write(image)
    writer.release()


class APIEndToEndMockedPoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        os.environ["PHYSIOAI_ENV"] = "development"
        os.environ["PHYSIOAI_DATA_DIR"] = cls.tmp.name
        os.environ["PHYSIOAI_MIN_FREE_DISK_MB"] = "64"
        os.environ["PHYSIOAI_MAX_INFLIGHT_JOBS"] = "1"
        os.environ["PHYSIOAI_WS_QUEUE_MAX"] = "64"
        for key in (
            "PHYSIOAI_AUTH_SECRET",
            "PHYSIOAI_ADMIN_PASSWORD",
            "PHYSIOAI_PSEUDONYM_SECRET",
        ):
            os.environ.pop(key, None)
        _install_fake_mediapipe()
        # test_capture_preflight pode ter importado src.core.capture antes deste
        # módulo; atualiza também a referência já materializada do pose module.
        import src.core.capture as capture  # noqa: PLC0415
        capture.mp_pose.Pose = _Pose
        capture.mp_pose.PoseLandmark = _PoseLandmark
        # Import only after fake module + runtime directory are configured.
        import server  # noqa: PLC0415

        cls.server = server

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _login(self, client: TestClient) -> str:
        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "physioai-local"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["access_token"]


    def test_movement_config_exposes_phase15_protocol(self):
        with TestClient(self.server.app) as client:
            token = self._login(client)
            response = client.get(
                "/api/config/movimentos",
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["protocol_version"], "1.4")
            self.assertEqual(len(payload["clinical_spec_sha256"]), 64)
            by_name = {item["name"]: item for item in payload["items"]}
            self.assertIn("Flexão de Quadril", by_name)
            self.assertIn("Dorsiflexão de Tornozelo", by_name)
            self.assertIn("Flexão de Ombro", by_name)
            self.assertIn("Abdução de Ombro", by_name)
            self.assertEqual(by_name["Flexão de Ombro"]["directionRule"], "anterior_positive")
            self.assertEqual(by_name["Abdução de Ombro"]["measurementPlane"], "frontal")
            self.assertEqual(
                by_name["Dorsiflexão de Tornozelo"]["convention"],
                "ankle_dorsiflexion_from_internal_angle",
            )
            self.assertTrue(by_name["Flexão de Cotovelo"]["requirePlaneProjection"])
            self.assertGreaterEqual(by_name["Flexão de Cotovelo"]["minViewAlignment"], 0.8)

    def test_health_auth_upload_websocket_result_and_session(self):
        video = Path(self.tmp.name) / "smoke.mp4"
        _make_video(video)
        job_id = str(uuid.uuid4())

        with TestClient(self.server.app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            ready = client.get("/ready")
            self.assertEqual(ready.status_code, 200, ready.text)
            self.assertEqual(ready.json()["checks"]["database"]["quick_check"], "ok")
            self.assertEqual(client.get("/api/auth/me").status_code, 401)

            token = self._login(client)
            headers = {"Authorization": f"Bearer {token}"}
            self.assertEqual(client.get("/api/auth/me", headers=headers).status_code, 200)
            self.assertEqual(client.get("/version").status_code, 200)

            with video.open("rb") as handle:
                preflight = client.post(
                    "/api/capture/preflight",
                    headers=headers,
                    data={
                        "id_paciente": "E2E-LOCAL",
                        "movimento": "Flexão de Cotovelo",
                        "lado": "direito",
                    },
                    files={"video_a": ("smoke.mp4", handle, "video/mp4")},
                )
            self.assertEqual(preflight.status_code, 200, preflight.text)
            preflight_payload = preflight.json()
            self.assertEqual(preflight_payload["status"], "fail")
            self.assertFalse(preflight_payload["can_analyze"])

            response = client.post(
                "/api/analise/processar",
                headers=headers,
                data={
                    "id_paciente": "E2E-LOCAL",
                    "movimento": "Flexão de Cotovelo",
                    "lado": "direito",
                    "modo_gpu": "false",
                    "retornar_frames": "false",
                    "gerar_graficos": "false",
                    "retornar_skeleton": "false",
                    "job_id": job_id,
                    "preflight_id": preflight_payload["preflight_id"],
                    "preflight_override_reason": "Teste E2E deliberado sem pose detectável.",
                },
            )
            self.assertEqual(response.status_code, 202, response.text)
            self.assertEqual(response.json()["job_id"], job_id)

            frames = []
            terminal = None
            with client.websocket_connect(
                f"/ws/analise/{job_id}", subprotocols=["physioai", token]
            ) as ws:
                for _ in range(64):
                    event = ws.receive_json()
                    if event.get("type") == "analysis.frame":
                        frames.append(event)
                    if event.get("type") in {"done", "error", "cancelled"}:
                        terminal = event
                        break

            self.assertIsNotNone(terminal)
            self.assertEqual(terminal["type"], "done", terminal)
            self.assertGreater(len(frames), 0)
            self.assertTrue(all(not f["measurement"]["measurementValid"] for f in frames))
            self.assertTrue(all(f["measurement"]["angleLandmarkDeg"] is None for f in frames))

            result = client.get(f"/api/analise/resultado/{job_id}", headers=headers)
            self.assertEqual(result.status_code, 200, result.text)
            payload = result.json()
            self.assertEqual(payload["status"], "ok")
            self.assertTrue(payload["result_sha256"])
            self.assertEqual(payload["total_frames"], len(frames))

            session = client.get(f"/api/sessoes/{job_id}", headers=headers)
            self.assertEqual(session.status_code, 200, session.text)
            self.assertEqual(session.json()["session"]["status"], "completed")
            self.assertEqual(session.json()["measurement_count"], len(frames))


    def test_experimental_imu_analysis_is_persisted_and_exportable(self):
        fs = 100.0
        seconds = 6.0
        n = int(fs * seconds)
        timestamps = [i * 1000.0 / fs for i in range(n)]
        gyro = [[0.25 * float(np.sin(2 * np.pi * 5.0 * (i / fs))), 0.0, 0.0] for i in range(n)]
        accel = [[0.0, 0.0, 9.81] for _ in range(n)]
        with TestClient(self.server.app) as client:
            token = self._login(client)
            headers = {"Authorization": f"Bearer {token}"}
            response = client.post(
                "/api/experimental/imu/analyze",
                headers=headers,
                json={
                    "timestamps_ms": timestamps,
                    "gyro_rad_s": gyro,
                    "accel_m_s2": accel,
                    "band_low_hz": 4.0,
                    "band_high_hz": 6.0,
                    "device_label": "synthetic-phone",
                },
            )
            self.assertEqual(response.status_code, 201, response.text)
            record = response.json()
            self.assertEqual(record["owner"], "admin")
            self.assertFalse(record["summary"]["diagnostic"])
            self.assertAlmostEqual(record["summary"]["dominant_frequency_hz"], 5.0, delta=0.4)
            analysis_id = record["id"]

            fetched = client.get(f"/api/experimental/imu/{analysis_id}", headers=headers)
            self.assertEqual(fetched.status_code, 200, fetched.text)
            exported = client.get(f"/api/experimental/imu/{analysis_id}/export.json", headers=headers)
            self.assertEqual(exported.status_code, 200, exported.text)
            payload = exported.json()
            self.assertEqual(len(payload["raw"]["timestamps_ms"]), n)
            self.assertEqual(payload["record"]["input_sha256"], record["input_sha256"])



    def test_multimodal_motor_features_use_persisted_video_and_imu(self):
        job_id = str(uuid.uuid4())
        patient = "P-MOTORFEATURE0001"
        self.server.session_store.start(
            job_id=job_id, patient_pseudonym=patient, evaluator_id="admin",
            movement="Flexão de Cotovelo", side="direito",
            software_version="test", protocol_version="test",
            clinical_spec_sha256="0" * 64, provenance={},
        )
        fs_video = 30.0
        seconds = 12.0
        frame_count = int(fs_video * seconds)
        t = np.arange(frame_count, dtype=np.float64) / fs_video
        angle = 60.0 + 30.0 * np.sin(2.0 * np.pi * 0.8 * t)
        self.server.measurement_store.save_many(job_id, [
            {
                "frame": i, "timestamp_ms": float(tt * 1000.0),
                "joint": "elbow", "movement": "Flexão de Cotovelo", "side": "direito",
                "angle_clinical_deg": float(a), "measurement_valid": True,
                "source": "observed", "measurement_space": "relative_body_3d",
                "quality": "good",
            }
            for i, (tt, a) in enumerate(zip(t, angle))
        ])

        fs_imu = 100.0
        n = int(fs_imu * seconds)
        timestamps = [i * 1000.0 / fs_imu for i in range(n)]
        gyro = [[0.3 * float(np.sin(2 * np.pi * 5.0 * (i / fs_imu))), 0.0, 0.0] for i in range(n)]
        with TestClient(self.server.app) as client:
            token = self._login(client)
            headers = {"Authorization": f"Bearer {token}"}
            imu = client.post(
                "/api/experimental/imu/analyze", headers=headers,
                json={
                    "timestamps_ms": timestamps, "gyro_rad_s": gyro,
                    "job_id": job_id, "band_low_hz": 4.0, "band_high_hz": 6.0,
                    "device_label": "synthetic-phone",
                },
            )
            self.assertEqual(imu.status_code, 201, imu.text)
            imu_id = imu.json()["id"]
            response = client.post(
                "/api/experimental/motor-features/analyze", headers=headers,
                json={"job_id": job_id, "imu_analysis_id": imu_id},
            )
            self.assertEqual(response.status_code, 201, response.text)
            record = response.json()
            self.assertFalse(record["summary"]["diagnostic"])
            self.assertEqual(record["summary"]["fusion_mode"], "video_imu")
            self.assertGreaterEqual(record["summary"]["vision"]["cycle_count"], 7)
            self.assertIn("parkinson_diagnosis", record["summary"]["unsupported_inferences"])

            fetched = client.get(f"/api/experimental/motor-features/{record['id']}", headers=headers)
            self.assertEqual(fetched.status_code, 200, fetched.text)
            exported = client.get(f"/api/experimental/motor-features/{record['id']}/export.json", headers=headers)
            self.assertEqual(exported.status_code, 200, exported.text)
            session = client.get(f"/api/sessoes/{job_id}", headers=headers)
            self.assertEqual(session.status_code, 200, session.text)
            self.assertEqual(session.json()["motor_feature_analysis_count"], 1)

    def test_analysis_requires_preflight_and_rejects_mismatch(self):
        video = Path(self.tmp.name) / "preflight_required.mp4"
        _make_video(video)
        with TestClient(self.server.app) as client:
            token = self._login(client)
            headers = {"Authorization": f"Bearer {token}"}

            with video.open("rb") as handle:
                direct = client.post(
                    "/api/analise/processar",
                    headers=headers,
                    data={
                        "id_paciente": "PREFLIGHT-E2E",
                        "movimento": "Flexão de Cotovelo",
                        "lado": "direito",
                    },
                    files={"video_a": ("preflight_required.mp4", handle, "video/mp4")},
                )
            self.assertEqual(direct.status_code, 428, direct.text)

            with video.open("rb") as handle:
                checked = client.post(
                    "/api/capture/preflight",
                    headers=headers,
                    data={
                        "id_paciente": "PREFLIGHT-E2E",
                        "movimento": "Flexão de Cotovelo",
                        "lado": "direito",
                    },
                    files={"video_a": ("preflight_required.mp4", handle, "video/mp4")},
                )
            self.assertEqual(checked.status_code, 200, checked.text)
            preflight_id = checked.json()["preflight_id"]

            mismatch = client.post(
                "/api/analise/processar",
                headers=headers,
                data={
                    "id_paciente": "PREFLIGHT-E2E",
                    "movimento": "Flexão de Joelho",
                    "lado": "direito",
                    "preflight_id": preflight_id,
                    "preflight_override_reason": "Override de teste com justificativa suficiente.",
                },
            )
            self.assertEqual(mismatch.status_code, 409, mismatch.text)

            discarded = client.delete(f"/api/capture/preflight/{preflight_id}", headers=headers)
            self.assertEqual(discarded.status_code, 200, discarded.text)
            self.assertEqual(discarded.json()["status"], "discarded")

    def test_validation_study_api_uses_server_authoritative_measurement(self):
        job_id = str(uuid.uuid4())
        patient = "P-ABCDEF0123456789"
        self.server.session_store.start(
            job_id=job_id, patient_pseudonym=patient, evaluator_id="admin",
            movement="Flexão de Cotovelo", side="direito",
            software_version="test", protocol_version="test",
            clinical_spec_sha256="0" * 64, provenance={},
        )
        self.server.measurement_store.save_many(job_id, [{
            "frame": 12, "timestamp_ms": 400.0, "joint": "elbow",
            "movement": "Flexão de Cotovelo", "side": "direito",
            "angle_raw_deg": 89.0, "angle_landmark_deg": 89.5,
            "angle_filtered_deg": 90.0, "angle_clinical_deg": 90.0,
            "angular_velocity_deg_s": 0.0, "adm_deg": 0.0,
            "confidence": 0.96, "source": "observed",
            "measurement_space": "pose_world_landmarks",
            "landmark_confidence_min": 0.94, "landmark_confidence_mean": 0.96,
            "quality": "excellent", "measurement_valid": True,
            "quality_flags": [],
        }])

        with TestClient(self.server.app) as client:
            token = self._login(client)
            headers = {"Authorization": f"Bearer {token}"}
            created = client.post("/api/estudos", headers=headers, json={
                "code": "E2E-STUDY",
                "title": "Estudo E2E",
                "movement": "Flexão de Cotovelo",
                "side": "direito",
                "target_participants": 10,
                "min_landmark_confidence": 0.70,
                "reference_device_model": "Goniômetro E2E",
                "reference_device_resolution_deg": 1.0,
                "repetitions_per_participant": 3,
            })
            self.assertEqual(created.status_code, 201, created.text)
            study_id = created.json()["id"]
            frozen = client.post(f"/api/estudos/{study_id}/freeze", headers=headers)
            self.assertEqual(frozen.status_code, 200, frozen.text)
            self.assertEqual(len(frozen.json()["protocol_sha256"]), 64)

            trial = client.post(f"/api/estudos/{study_id}/ensaios", headers=headers, json={
                "job_id": job_id, "frame": 12, "reference_angle_deg": 92.0,
                # Qualquer tentativa de fornecer ângulo automático pelo cliente não existe no contrato.
                "notes": "ensaio E2E",
            })
            self.assertEqual(trial.status_code, 201, trial.text)
            body = trial.json()
            self.assertEqual(body["automatic_angle_deg"], 90.0)
            self.assertEqual(body["reference_angle_deg"], 92.0)
            self.assertTrue(body["included"])

            report = client.get(f"/api/estudos/{study_id}/relatorio?bootstrap_iterations=200", headers=headers)
            self.assertEqual(report.status_code, 200, report.text)
            self.assertEqual(report.json()["counts"]["trials_included"], 1)
            self.assertEqual(report.json()["agreement"]["mae_deg"], 2.0)


    def test_longitudinal_endpoint_uses_comparable_server_sessions(self):
        patient = "P-LONG-E2E"
        jobs = []
        for index, adm in enumerate((80.0, 86.0, 91.0), start=1):
            job_id = f"long-e2e-{index}-{uuid.uuid4()}"
            jobs.append(job_id)
            self.server.session_store.start(
                job_id=job_id, patient_pseudonym=patient, evaluator_id="admin",
                movement="Flexão de Cotovelo", side="direito",
                software_version="2.16-test", protocol_version="1.4",
                clinical_spec_sha256="a" * 64, provenance={},
            )
            self.server.session_store.finish(
                job_id, status="completed", total_frames=100, valid_frames=90,
                fps=30.0, adm_deg=adm, total_reps=5, result_sha256=(str(index) * 64),
            )

        with TestClient(self.server.app) as client:
            token = self._login(client)
            headers = {"Authorization": f"Bearer {token}"}
            response = client.post(
                "/api/experimental/longitudinal/analyze", headers=headers,
                json={"job_id": jobs[-1], "include_motor_features": False},
            )
            self.assertEqual(response.status_code, 201, response.text)
            body = response.json()
            self.assertFalse(body["summary"]["diagnostic"])
            self.assertEqual(body["summary"]["session_count"], 3)
            self.assertEqual(body["summary"]["metrics"]["adm_deg"]["delta"], 11.0)
            self.assertEqual(len(body["source_sha256"]), 64)

            fetched = client.get(
                f"/api/experimental/longitudinal/{body['id']}", headers=headers
            )
            self.assertEqual(fetched.status_code, 200, fetched.text)
            exported = client.get(
                f"/api/experimental/longitudinal/{body['id']}/export.json", headers=headers
            )
            self.assertEqual(exported.status_code, 200, exported.text)



    def test_experimental_gait_endpoint_is_authenticated_and_temporal_only(self):
        fps = 30.0
        n = 300
        t = np.arange(n) / fps
        skel = np.zeros((n, 33, 3), dtype=float)
        skel[:, 23, :] = (-0.1, 0.0, 0.0)
        skel[:, 24, :] = (0.1, 0.0, 0.0)
        skel[:, 29, 0] = -0.12
        skel[:, 30, 0] = 0.12
        skel[:, 29, 1] = 0.25 + 0.04 * np.cos(2 * np.pi * t)
        skel[:, 30, 1] = 0.25 + 0.04 * np.cos(2 * np.pi * t + np.pi)
        skel[:, 29, 2] = 0.1
        skel[:, 30, 2] = 0.1

        with TestClient(self.server.app) as client:
            unauth = client.post(
                "/api/experimental/gait/analyze",
                json={
                    "fps": fps,
                    "coordinate_space": "relative_body_3d",
                    "skeleton_frames": skel.tolist(),
                },
            )
            self.assertEqual(unauth.status_code, 401)
            token = self._login(client)
            response = client.post(
                "/api/experimental/gait/analyze",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "fps": fps,
                    "coordinate_space": "relative_body_3d",
                    "skeleton_frames": skel.tolist(),
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertTrue(body["experimental"])
            self.assertGreater(body["event_count"], 10)
            self.assertIsNone(body["stride_length_m"])
            self.assertIn(
                "spatial_metrics_disabled_non_global_coordinates",
                body["quality_flags"],
            )


if __name__ == "__main__":
    unittest.main()
