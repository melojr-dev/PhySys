from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


_EPS = 1e-8


@dataclass(frozen=True, slots=True)
class BodyFrame:
    """Referencial anatômico aproximado derivado de ombros e quadris.

    Os eixos são unitários e expressos no mesmo espaço 3D do esqueleto:
      - lateral: esquerda -> direita;
      - superior: quadril -> ombro;
      - anterior: produto vetorial lateral × superior.

    O sinal do eixo anterior não é usado para validade clínica nesta fase; para
    projeções por plano importa o subespaço/normal, não o sentido absoluto.
    """

    origin: np.ndarray
    lateral: np.ndarray
    superior: np.ndarray
    anterior: np.ndarray
    axis_consistency: float

    def plane_normal(self, plane: str) -> np.ndarray:
        key = plane.strip().lower()
        if key == "sagittal":
            return self.lateral
        if key == "frontal":
            return self.anterior
        if key == "transverse":
            return self.superior
        raise ValueError(f"Plano anatômico não suportado: {plane}")


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm < _EPS:
        return None
    return arr / norm


def build_body_frame(skeleton: np.ndarray) -> BodyFrame | None:
    """Constrói referencial corporal a partir dos landmarks MediaPipe Pose.

    Índices MediaPipe usados: ombros 11/12 e quadris 23/24.
    Retorna ``None`` quando a geometria bilateral é degenerada ou não finita.
    """
    skel = np.asarray(skeleton, dtype=np.float64)
    if skel.ndim != 2 or skel.shape[0] < 25 or skel.shape[1] < 3:
        return None

    pts = skel[[11, 12, 23, 24], :3]
    if not np.all(np.isfinite(pts)):
        return None

    left_shoulder, right_shoulder, left_hip, right_hip = pts
    shoulder_axis = _normalize(right_shoulder - left_shoulder)
    hip_axis = _normalize(right_hip - left_hip)
    if shoulder_axis is None or hip_axis is None:
        return None

    # Evita cancelamento quando uma estimativa inverte a orientação de um par.
    if float(np.dot(shoulder_axis, hip_axis)) < 0.0:
        hip_axis = -hip_axis

    lateral = _normalize(shoulder_axis + hip_axis)
    if lateral is None:
        return None

    shoulder_mid = (left_shoulder + right_shoulder) * 0.5
    hip_mid = (left_hip + right_hip) * 0.5
    vertical_raw = shoulder_mid - hip_mid
    # Gram-Schmidt: eixo superior deve ser ortogonal ao mediolateral.
    vertical_ortho = vertical_raw - float(np.dot(vertical_raw, lateral)) * lateral
    superior = _normalize(vertical_ortho)
    if superior is None:
        return None

    anterior = _normalize(np.cross(lateral, superior))
    if anterior is None:
        return None
    # Reortogonaliza o eixo superior para estabilidade numérica.
    superior = _normalize(np.cross(anterior, lateral))
    if superior is None:
        return None

    axis_consistency = abs(float(np.dot(shoulder_axis, hip_axis)))
    origin = (shoulder_mid + hip_mid) * 0.5
    return BodyFrame(
        origin=origin.astype(np.float64),
        lateral=lateral.astype(np.float64),
        superior=superior.astype(np.float64),
        anterior=anterior.astype(np.float64),
        axis_consistency=max(0.0, min(1.0, axis_consistency)),
    )


def project_vector_to_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    vec = np.asarray(vector, dtype=np.float64).reshape(3)
    n = _normalize(np.asarray(normal, dtype=np.float64))
    if n is None or not np.all(np.isfinite(vec)):
        return None
    projected = vec - float(np.dot(vec, n)) * n
    if float(np.linalg.norm(projected)) < _EPS:
        return None
    return projected


def internal_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float | None:
    a = _normalize(v1)
    b = _normalize(v2)
    if a is None or b is None:
        return None
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def projected_internal_angle_deg(
    proximal: np.ndarray,
    vertex: np.ndarray,
    distal: np.ndarray,
    *,
    frame: BodyFrame,
    plane: str,
) -> float | None:
    normal = frame.plane_normal(plane)
    v1 = np.asarray(proximal, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    v2 = np.asarray(distal, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    p1 = project_vector_to_plane(v1, normal)
    p2 = project_vector_to_plane(v2, normal)
    if p1 is None or p2 is None:
        return None
    return internal_angle_deg(p1, p2)


def segment_out_of_plane_deg(vector: np.ndarray, normal: np.ndarray) -> float | None:
    """Ângulo do segmento para fora do plano (0° = totalmente no plano)."""
    v = _normalize(vector)
    n = _normalize(normal)
    if v is None or n is None:
        return None
    ratio = abs(float(np.clip(np.dot(v, n), -1.0, 1.0)))
    return float(np.degrees(np.arcsin(ratio)))


def max_out_of_plane_deg(
    proximal: np.ndarray,
    vertex: np.ndarray,
    distal: np.ndarray,
    *,
    frame: BodyFrame,
    plane: str,
) -> float | None:
    normal = frame.plane_normal(plane)
    values = [
        segment_out_of_plane_deg(np.asarray(proximal) - np.asarray(vertex), normal),
        segment_out_of_plane_deg(np.asarray(distal) - np.asarray(vertex), normal),
    ]
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return max(finite) if finite else None



def movement_direction_alignment(
    vector: np.ndarray,
    *,
    frame: BodyFrame,
    plane: str,
    direction_rule: str | None,
    side: str,
) -> float | None:
    """Retorna alinhamento direcional (-1..1) do segmento dentro do plano.

    Regras atualmente suportadas:
      - ``anterior_positive``: segmento deve apontar para o eixo anterior do corpo;
      - ``lateral_outward``: segmento deve apontar lateralmente para fora do corpo,
        respeitando o lado avaliado.

    O valor 0 é esperado em posições neutras/verticais e permanece válido.
    Valores negativos relevantes indicam movimento no sentido oposto ao protocolo.
    """
    if not direction_rule:
        return None
    projected = project_vector_to_plane(np.asarray(vector, dtype=np.float64), frame.plane_normal(plane))
    unit = _normalize(projected) if projected is not None else None
    if unit is None:
        return None

    rule = direction_rule.strip().lower()
    if rule == "anterior_positive":
        target = frame.anterior
    elif rule == "lateral_outward":
        normalized_side = side.strip().lower()
        if normalized_side in {"direito", "right", "d"}:
            target = frame.lateral
        elif normalized_side in {"esquerdo", "left", "e"}:
            target = -frame.lateral
        else:
            raise ValueError(f"Lado não suportado para regra direcional: {side}")
    else:
        raise ValueError(f"Regra direcional não suportada: {direction_rule}")

    value = float(np.dot(unit, target))
    return max(-1.0, min(1.0, value))

def camera_view_alignment(frame: BodyFrame, plane: str, camera_forward: Iterable[float] = (0.0, 0.0, 1.0)) -> float:
    """Alinhamento 0..1 entre eixo óptico e normal do plano de medição.

    1.0 representa câmera aproximadamente perpendicular ao plano anatômico
    solicitado (por exemplo, vista lateral para plano sagital).
    """
    normal = frame.plane_normal(plane)
    camera = _normalize(np.asarray(tuple(camera_forward), dtype=np.float64))
    if camera is None:
        raise ValueError("Eixo óptico inválido")
    return max(0.0, min(1.0, abs(float(np.dot(normal, camera)))))


def alignment_deviation_deg(alignment: float) -> float:
    value = max(0.0, min(1.0, float(alignment)))
    return float(np.degrees(np.arccos(value)))
