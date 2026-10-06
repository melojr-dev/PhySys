from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import median
from typing import Any

import cv2
import numpy as np
from mediapipe.python.solutions import pose as mp_pose

from src.core.anatomy import (
    alignment_deviation_deg,
    build_body_frame,
    camera_view_alignment,
    max_out_of_plane_deg,
)
from src.core.clinical import get_movement_spec

PREFLIGHT_SPEC_VERSION = "1.0"
PREFLIGHT_THRESHOLDS: dict[str, float] = {
    "framing_margin": 0.045,
    "resolution_recommended_min_dimension": 720.0,
    "resolution_minimum_min_dimension": 480.0,
    "fps_recommended": 24.0,
    "fps_minimum": 15.0,
    "pose_detection_recommended": 0.85,
    "pose_detection_minimum": 0.65,
    "landmark_visibility_recommended": 0.80,
    "landmark_visibility_minimum": 0.65,
    "framing_recommended": 0.95,
    "framing_minimum": 0.80,
    "body_frame_recommended": 0.80,
    "body_frame_minimum": 0.60,
    "movement_plane_recommended": 0.80,
    "movement_plane_minimum": 0.60,
    "brightness_recommended_low": 45.0,
    "brightness_recommended_high": 210.0,
    "brightness_tolerable_low": 30.0,
    "brightness_tolerable_high": 230.0,
    "contrast_recommended": 25.0,
    "contrast_tolerable": 15.0,
    "sharpness_recommended": 60.0,
    "sharpness_tolerable": 25.0,
}


def capture_preflight_spec_snapshot() -> dict[str, Any]:
    return {
        "version": PREFLIGHT_SPEC_VERSION,
        "thresholds": dict(PREFLIGHT_THRESHOLDS),
        "critical_checks": [
            "decode", "resolution_below_minimum", "fps", "pose_detection",
            "landmark_visibility", "framing", "body_frame",
            "view_alignment", "movement_plane",
        ],
        "noncritical_checks": ["lighting", "contrast", "sharpness"],
        "sampling": "uniform temporal sampling over the uploaded clip",
        "note": "preflight is a capture gate; frame-level validity remains authoritative during analysis",
    }


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    code: str
    label: str
    status: str  # pass | warn | fail | info
    message: str
    value: float | int | str | None = None
    threshold: str | None = None
    critical: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CapturePreflightReport:
    status: str  # pass | warn | fail
    can_analyze: bool
    score: float
    movement: str
    side: str
    sampled_frames: int
    detected_frames: int
    best_frame_index: int | None
    best_frame_timestamp_ms: float | None
    video: dict[str, Any]
    summary: dict[str, Any]
    checks: tuple[PreflightCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "can_analyze": self.can_analyze,
            "score": self.score,
            "movement": self.movement,
            "side": self.side,
            "sampled_frames": self.sampled_frames,
            "detected_frames": self.detected_frames,
            "best_frame_index": self.best_frame_index,
            "best_frame_timestamp_ms": self.best_frame_timestamp_ms,
            "video": self.video,
            "summary": self.summary,
            "checks": [item.to_dict() for item in self.checks],
        }


def _finite_median(values: list[float]) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(median(finite)) if finite else None


def _node_indexes(side: str) -> dict[str, int]:
    left = side.strip().lower() == "esquerdo"
    p = mp_pose.PoseLandmark
    return {
        "ombro": int((p.LEFT_SHOULDER if left else p.RIGHT_SHOULDER).value),
        "cotovelo": int((p.LEFT_ELBOW if left else p.RIGHT_ELBOW).value),
        "pulso": int((p.LEFT_WRIST if left else p.RIGHT_WRIST).value),
        "quadril": int((p.LEFT_HIP if left else p.RIGHT_HIP).value),
        "joelho": int((p.LEFT_KNEE if left else p.RIGHT_KNEE).value),
        "tornozelo": int((p.LEFT_ANKLE if left else p.RIGHT_ANKLE).value),
        "pe": int((p.LEFT_FOOT_INDEX if left else p.RIGHT_FOOT_INDEX).value),
    }


def _check(code: str, label: str, status: str, message: str, *, value: Any = None,
           threshold: str | None = None, critical: bool = False) -> PreflightCheck:
    return PreflightCheck(
        code=code,
        label=label,
        status=status,
        message=message,
        value=value,
        threshold=threshold,
        critical=critical,
    )


def evaluate_capture_preflight(
    video_path: str,
    movement: str,
    side: str,
    *,
    max_samples: int = 12,
) -> CapturePreflightReport:
    """Avalia rapidamente se um vídeo é adequado ao protocolo biomecânico.

    O preflight não substitui a validação frame a frame do pipeline. Ele usa uma
    amostra temporal do vídeo para detectar falhas de captura antes de iniciar a
    análise completa e evitar consumo desnecessário de CPU/GPU.
    """
    spec = get_movement_spec(movement)
    if spec is None:
        raise ValueError(f"Movimento clínico não suportado: {movement}")
    if side not in {"direito", "esquerdo"}:
        raise ValueError("side deve ser 'direito' ou 'esquerdo'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return CapturePreflightReport(
            status="fail", can_analyze=False, score=0.0, movement=movement, side=side,
            sampled_frames=0, detected_frames=0, best_frame_index=None,
            best_frame_timestamp_ms=None,
            video={}, summary={},
            checks=(_check("decode", "Leitura do vídeo", "fail", "O arquivo não pôde ser decodificado.", critical=True),),
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = (total_frames / fps) if fps > 0 and total_frames > 0 else None

    if total_frames <= 0:
        sample_indices = [0]
    else:
        count = min(max(3, max_samples), total_frames)
        sample_indices = sorted(set(np.linspace(0, total_frames - 1, count, dtype=int).tolist()))

    nodes = _node_indexes(side)
    required = [nodes[spec.proximal_node], nodes[spec.vertex_node], nodes[spec.distal_node]]

    brightness: list[float] = []
    contrast: list[float] = []
    sharpness: list[float] = []
    min_visibility: list[float] = []
    mean_visibility: list[float] = []
    framing_ok: list[float] = []
    frame_confidence: list[float] = []
    body_frame_available: list[float] = []
    body_frame_confidence: list[float] = []
    view_alignments: list[float] = []
    view_deviations: list[float] = []
    out_of_plane: list[float] = []
    plane_ok: list[float] = []
    detected = 0
    sampled = 0
    best_score = -1.0
    best_index: int | None = None

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,
        enable_segmentation=False,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    try:
        for index in sample_indices:
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            sampled += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(float(np.mean(gray)))
            contrast.append(float(np.std(gray)))
            sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks is None:
                frame_confidence.append(0.0)
                body_frame_available.append(0.0)
                continue

            detected += 1
            lms = result.pose_landmarks.landmark
            vis = [float(lms[i].visibility) for i in required]
            vmin = min(vis)
            vmean = sum(vis) / len(vis)
            min_visibility.append(vmin)
            mean_visibility.append(vmean)
            frame_confidence.append(vmin)

            # Enquadramento conservador: pontos clínicos + cintura escapular/pélvica
            framing_indexes = sorted(set(required + [11, 12, 23, 24]))
            margin = PREFLIGHT_THRESHOLDS["framing_margin"]
            in_frame = [
                margin <= float(lms[i].x) <= 1.0 - margin
                and margin <= float(lms[i].y) <= 1.0 - margin
                for i in framing_indexes
            ]
            framing = sum(1.0 for value in in_frame if value) / max(len(in_frame), 1)
            framing_ok.append(framing)

            frame_score = vmin * 0.7 + framing * 0.3

            if result.pose_world_landmarks is not None:
                world = np.array(
                    [[float(lm.x), float(lm.y), float(lm.z)] for lm in result.pose_world_landmarks.landmark],
                    dtype=np.float64,
                )
                body = build_body_frame(world)
                if body is not None:
                    body_frame_available.append(1.0)
                    body_frame_confidence.append(float(body.axis_consistency))
                    alignment = camera_view_alignment(body, spec.measurement_plane)
                    deviation = alignment_deviation_deg(alignment)
                    view_alignments.append(float(alignment))
                    view_deviations.append(float(deviation))
                    oop = max_out_of_plane_deg(
                        world[required[0]], world[required[1]], world[required[2]],
                        frame=body, plane=spec.measurement_plane,
                    )
                    if oop is not None:
                        out_of_plane.append(float(oop))
                        plane_valid = float(oop) <= float(spec.max_out_of_plane_deg)
                        plane_ok.append(1.0 if plane_valid else 0.0)
                    view_valid = alignment >= float(spec.min_view_alignment)
                    frame_score += (0.15 if view_valid else 0.0)
                else:
                    body_frame_available.append(0.0)
            else:
                body_frame_available.append(0.0)

            if frame_score > best_score:
                best_score = frame_score
                best_index = int(index)
    finally:
        pose.close()
        cap.release()

    checks: list[PreflightCheck] = []
    checks.append(_check(
        "decode", "Leitura do vídeo", "pass" if sampled > 0 else "fail",
        "Vídeo decodificado com sucesso." if sampled > 0 else "Nenhum frame pôde ser lido.",
        value=sampled, threshold="> 0 frames", critical=True,
    ))

    min_dimension = min(width, height) if width and height else 0
    if min_dimension >= PREFLIGHT_THRESHOLDS["resolution_recommended_min_dimension"]:
        resolution_status = "pass"
    elif min_dimension >= PREFLIGHT_THRESHOLDS["resolution_minimum_min_dimension"]:
        resolution_status = "warn"
    else:
        resolution_status = "fail"
    checks.append(_check(
        "resolution", "Resolução", resolution_status,
        f"Vídeo {width}×{height}px.", value=f"{width}x{height}",
        threshold="mín. recomendado 720px no menor eixo; aceitável ≥480px", critical=min_dimension < 480,
    ))

    if fps >= PREFLIGHT_THRESHOLDS["fps_recommended"]:
        fps_status = "pass"
    elif fps >= PREFLIGHT_THRESHOLDS["fps_minimum"]:
        fps_status = "warn"
    else:
        fps_status = "fail"
    checks.append(_check(
        "fps", "Taxa de quadros", fps_status,
        f"Captura em {fps:.1f} FPS." if fps > 0 else "FPS não identificado.",
        value=round(fps, 2), threshold="recomendado ≥24 FPS; mínimo 15 FPS", critical=fps < PREFLIGHT_THRESHOLDS["fps_minimum"],
    ))

    detection_ratio = detected / sampled if sampled else 0.0
    if detection_ratio >= PREFLIGHT_THRESHOLDS["pose_detection_recommended"]:
        status = "pass"
    elif detection_ratio >= PREFLIGHT_THRESHOLDS["pose_detection_minimum"]:
        status = "warn"
    else:
        status = "fail"
    checks.append(_check(
        "pose_detection", "Detecção corporal", status,
        f"Pose detectada em {detection_ratio:.0%} dos frames amostrados.",
        value=round(detection_ratio, 4), threshold="≥85% recomendado; ≥65% mínimo", critical=status == "fail",
    ))

    visibility = _finite_median(min_visibility)
    if visibility is None:
        status = "fail"
    elif visibility >= PREFLIGHT_THRESHOLDS["landmark_visibility_recommended"]:
        status = "pass"
    elif visibility >= PREFLIGHT_THRESHOLDS["landmark_visibility_minimum"]:
        status = "warn"
    else:
        status = "fail"
    checks.append(_check(
        "landmark_visibility", "Visibilidade dos pontos clínicos", status,
        "Visibilidade mediana do pior landmark clínico."
        if visibility is not None else "Landmarks clínicos não ficaram disponíveis.",
        value=round(visibility, 4) if visibility is not None else None,
        threshold="≥0,80 recomendado; ≥0,65 mínimo", critical=status == "fail",
    ))

    framing = _finite_median(framing_ok)
    if framing is None:
        status = "fail"
    elif framing >= PREFLIGHT_THRESHOLDS["framing_recommended"]:
        status = "pass"
    elif framing >= PREFLIGHT_THRESHOLDS["framing_minimum"]:
        status = "warn"
    else:
        status = "fail"
    checks.append(_check(
        "framing", "Enquadramento anatômico", status,
        "Proporção mediana de landmarks essenciais dentro da margem segura do quadro.",
        value=round(framing, 4) if framing is not None else None,
        threshold="≥95% recomendado; ≥80% mínimo", critical=status == "fail",
    ))

    body_ratio = sum(body_frame_available) / len(body_frame_available) if body_frame_available else 0.0
    if body_ratio >= PREFLIGHT_THRESHOLDS["body_frame_recommended"]:
        status = "pass"
    elif body_ratio >= PREFLIGHT_THRESHOLDS["body_frame_minimum"]:
        status = "warn"
    else:
        status = "fail"
    checks.append(_check(
        "body_frame", "Referencial corporal", status,
        f"Referencial anatômico bilateral disponível em {body_ratio:.0%} dos frames com amostragem.",
        value=round(body_ratio, 4), threshold="≥80% recomendado; ≥60% mínimo", critical=status == "fail",
    ))

    alignment = _finite_median(view_alignments)
    if alignment is None:
        status = "fail"
    elif alignment >= spec.min_view_alignment:
        status = "pass"
    elif alignment >= max(0.0, spec.min_view_alignment - 0.10):
        status = "warn"
    else:
        status = "fail"
    checks.append(_check(
        "view_alignment", "Orientação da câmera", status,
        f"Alinhamento mediano com a vista {spec.recommended_view}."
        if alignment is not None else "Não foi possível estimar o alinhamento da câmera.",
        value=round(alignment, 4) if alignment is not None else None,
        threshold=f"≥{spec.min_view_alignment:.2f}", critical=status == "fail",
    ))

    plane_ratio = sum(plane_ok) / len(plane_ok) if plane_ok else 0.0
    if plane_ratio >= PREFLIGHT_THRESHOLDS["movement_plane_recommended"]:
        status = "pass"
    elif plane_ratio >= PREFLIGHT_THRESHOLDS["movement_plane_minimum"]:
        status = "warn"
    else:
        status = "fail"
    checks.append(_check(
        "movement_plane", "Movimento no plano esperado", status,
        f"{plane_ratio:.0%} dos frames com geometria suficiente permaneceram dentro do limite fora do plano.",
        value=round(plane_ratio, 4), threshold=f"≤{spec.max_out_of_plane_deg:.0f}° fora do plano", critical=status == "fail",
    ))

    light = _finite_median(brightness)
    if light is None:
        light_status = "fail"
    elif PREFLIGHT_THRESHOLDS["brightness_recommended_low"] <= light <= PREFLIGHT_THRESHOLDS["brightness_recommended_high"]:
        light_status = "pass"
    elif PREFLIGHT_THRESHOLDS["brightness_tolerable_low"] <= light <= PREFLIGHT_THRESHOLDS["brightness_tolerable_high"]:
        light_status = "warn"
    else:
        light_status = "fail"
    checks.append(_check(
        "lighting", "Iluminação", light_status,
        "Luminância mediana dos frames amostrados.",
        value=round(light, 1) if light is not None else None,
        threshold="45–210 recomendado; 30–230 tolerável", critical=False,
    ))

    contrast_value = _finite_median(contrast)
    if contrast_value is None:
        contrast_status = "fail"
    elif contrast_value >= PREFLIGHT_THRESHOLDS["contrast_recommended"]:
        contrast_status = "pass"
    elif contrast_value >= PREFLIGHT_THRESHOLDS["contrast_tolerable"]:
        contrast_status = "warn"
    else:
        contrast_status = "fail"
    checks.append(_check(
        "contrast", "Contraste", contrast_status,
        "Desvio-padrão de intensidade em escala de cinza.",
        value=round(contrast_value, 1) if contrast_value is not None else None,
        threshold="≥25 recomendado; ≥15 tolerável", critical=False,
    ))

    sharp = _finite_median(sharpness)
    if sharp is None:
        sharp_status = "fail"
    elif sharp >= PREFLIGHT_THRESHOLDS["sharpness_recommended"]:
        sharp_status = "pass"
    elif sharp >= PREFLIGHT_THRESHOLDS["sharpness_tolerable"]:
        sharp_status = "warn"
    else:
        sharp_status = "fail"
    checks.append(_check(
        "sharpness", "Nitidez", sharp_status,
        "Variância mediana do Laplaciano; usada somente como alerta operacional.",
        value=round(sharp, 1) if sharp is not None else None,
        threshold="≥60 recomendado; ≥25 tolerável", critical=False,
    ))

    critical_fail = any(item.critical and item.status == "fail" for item in checks)
    any_warn_or_fail = any(item.status in {"warn", "fail"} for item in checks)
    overall = "fail" if critical_fail else ("warn" if any_warn_or_fail else "pass")
    can_analyze = not critical_fail

    weights = {"pass": 1.0, "warn": 0.65, "fail": 0.0, "info": 1.0}
    score = 100.0 * sum(weights.get(item.status, 0.0) for item in checks) / max(len(checks), 1)

    best_timestamp_ms = None
    if best_index is not None and fps > 0:
        best_timestamp_ms = (best_index / fps) * 1000.0

    summary = {
        "pose_detection_ratio": round(detection_ratio, 4),
        "landmark_visibility_median": round(visibility, 4) if visibility is not None else None,
        "framing_ratio_median": round(framing, 4) if framing is not None else None,
        "body_frame_ratio": round(body_ratio, 4),
        "body_frame_confidence_median": round(_finite_median(body_frame_confidence), 4) if _finite_median(body_frame_confidence) is not None else None,
        "view_alignment_median": round(alignment, 4) if alignment is not None else None,
        "view_deviation_deg_median": round(_finite_median(view_deviations), 2) if _finite_median(view_deviations) is not None else None,
        "out_of_plane_deg_median": round(_finite_median(out_of_plane), 2) if _finite_median(out_of_plane) is not None else None,
        "movement_plane_pass_ratio": round(plane_ratio, 4),
        "brightness_median": round(light, 1) if light is not None else None,
        "contrast_median": round(contrast_value, 1) if contrast_value is not None else None,
        "sharpness_median": round(sharp, 1) if sharp is not None else None,
    }
    video = {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_s": round(duration_s, 3) if duration_s is not None else None,
    }
    return CapturePreflightReport(
        status=overall,
        can_analyze=can_analyze,
        score=round(score, 1),
        movement=movement,
        side=side,
        sampled_frames=sampled,
        detected_frames=detected,
        best_frame_index=best_index,
        best_frame_timestamp_ms=round(best_timestamp_ms, 1) if best_timestamp_ms is not None else None,
        video=video,
        summary=summary,
        checks=tuple(checks),
    )
