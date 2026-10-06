from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable


def clamp_angle_deg(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Ângulo deve ser finito")
    return max(0.0, min(180.0, float(value)))


def flexion_from_internal_angle(internal_angle_deg: float) -> float:
    """Converte ângulo interno segmentar para flexão clínica.

    Convenção PhysioAI:
      180° internos -> 0° de flexão
       90° internos -> 90° de flexão
    """
    return clamp_angle_deg(180.0 - float(internal_angle_deg))


def direct_internal_angle(internal_angle_deg: float) -> float:
    """Usa diretamente o ângulo projetado como grandeza clínica (0..180°)."""
    return clamp_angle_deg(float(internal_angle_deg))


def ankle_dorsiflexion_from_internal_angle(internal_angle_deg: float) -> float:
    """Converte o ângulo tíbia–tornozelo–antepé em dorsiflexão clínica.

    Convenção operacional PhysioAI para vista sagital:
      90° internos -> 0° de dorsiflexão (posição neutra aproximada)
      70° internos -> 20° de dorsiflexão

    Ângulos acima de 90° representam componente de flexão plantar e, nesta
    medição específica de dorsiflexão, são truncados em 0° em vez de receberem
    sinal negativo. Flexão plantar deve ser tratada como movimento separado.
    """
    if not math.isfinite(float(internal_angle_deg)):
        raise ValueError("Ângulo deve ser finito")
    return max(0.0, min(90.0, 90.0 - float(internal_angle_deg)))


def _xy(point: object) -> tuple[float, float]:
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    if isinstance(point, (tuple, list)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    x = getattr(point, "x", None)
    y = getattr(point, "y", None)
    if x is None or y is None:
        raise ValueError("Ponto 2D inválido")
    return float(x), float(y)


def internal_angle_from_points_2d(proximal: object, vertex: object, distal: object) -> float:
    """Ângulo interno no vértice, em graus, calculado de três pontos 2D."""
    ax, ay = _xy(proximal)
    bx, by = _xy(vertex)
    cx, cy = _xy(distal)
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-12 or n2 < 1e-12:
        raise ValueError("Pontos coincidentes não definem um ângulo")
    cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def clinical_angle_from_points_2d(
    movement: str, proximal: object, vertex: object, distal: object
) -> float:
    spec = get_movement_spec(movement)
    if spec is None:
        raise ValueError(f"Movimento clínico não suportado: {movement}")
    return spec.clinical_angle(internal_angle_from_points_2d(proximal, vertex, distal))


@dataclass(frozen=True, slots=True)
class MovementSpec:
    name: str
    joint_code: str
    proximal_node: str
    vertex_node: str
    distal_node: str
    reference_rom_deg: float
    enter_flexion_deg: float
    complete_extension_deg: float
    converter: Callable[[float], float]
    converter_id: str
    measurement_plane: str
    recommended_view: str
    capture_instructions: tuple[str, ...]
    min_view_alignment: float = 0.80
    max_out_of_plane_deg: float = 35.0
    require_plane_projection: bool = True
    direction_rule: str | None = None
    min_direction_alignment: float = -0.10
    validated: bool = False

    def clinical_angle(self, internal_angle_deg: float) -> float:
        return self.converter(internal_angle_deg)


MOVEMENT_SPECS: dict[str, MovementSpec] = {
    "Flexão de Cotovelo": MovementSpec(
        name="Flexão de Cotovelo",
        joint_code="elbow",
        proximal_node="ombro",
        vertex_node="cotovelo",
        distal_node="pulso",
        reference_rom_deg=145.0,
        enter_flexion_deg=90.0,
        complete_extension_deg=30.0,
        converter=flexion_from_internal_angle,
        converter_id="flexion_from_internal_angle",
        measurement_plane="sagittal",
        recommended_view="lateral",
        capture_instructions=(
            "Manter ombro, cotovelo e punho do lado avaliado visíveis durante todo o movimento.",
            "Posicionar a câmera aproximadamente perpendicular ao plano sagital do membro.",
            "Evitar rotação excessiva do tronco e deslocamento da câmera durante a série.",
        ),
    ),
    "Flexão de Joelho": MovementSpec(
        name="Flexão de Joelho",
        joint_code="knee",
        proximal_node="quadril",
        vertex_node="joelho",
        distal_node="tornozelo",
        reference_rom_deg=135.0,
        enter_flexion_deg=70.0,
        complete_extension_deg=20.0,
        converter=flexion_from_internal_angle,
        converter_id="flexion_from_internal_angle",
        measurement_plane="sagittal",
        recommended_view="lateral",
        capture_instructions=(
            "Manter quadril, joelho e tornozelo do lado avaliado visíveis.",
            "Usar vista lateral com a câmera perpendicular ao plano sagital.",
            "Evitar rotação pélvica e oclusão do joelho pelo membro contralateral.",
        ),
    ),
    "Flexão de Quadril": MovementSpec(
        name="Flexão de Quadril",
        joint_code="hip",
        proximal_node="ombro",
        vertex_node="quadril",
        distal_node="joelho",
        reference_rom_deg=120.0,
        enter_flexion_deg=60.0,
        complete_extension_deg=20.0,
        converter=flexion_from_internal_angle,
        converter_id="flexion_from_internal_angle",
        measurement_plane="sagittal",
        recommended_view="lateral",
        capture_instructions=(
            "Manter ombro, quadril e joelho do lado avaliado completamente visíveis.",
            "Usar vista lateral; o plano sagital do corpo deve permanecer aproximadamente perpendicular ao eixo óptico.",
            "Minimizar compensação por flexão lombar ou inclinação do tronco, pois ela altera a referência proximal.",
        ),
    ),
    "Flexão de Ombro": MovementSpec(
        name="Flexão de Ombro",
        joint_code="shoulder",
        proximal_node="quadril",
        vertex_node="ombro",
        distal_node="cotovelo",
        reference_rom_deg=180.0,
        enter_flexion_deg=90.0,
        complete_extension_deg=20.0,
        converter=direct_internal_angle,
        converter_id="direct_internal_angle",
        measurement_plane="sagittal",
        recommended_view="lateral",
        capture_instructions=(
            "Manter quadril, ombro e cotovelo do lado avaliado visíveis durante toda a elevação anterior.",
            "Usar vista lateral com a câmera aproximadamente perpendicular ao plano sagital.",
            "Executar flexão anterior do ombro; extensão posterior é rejeitada pelo gate direcional.",
            "Evitar extensão lombar e rotação do tronco, que alteram a referência proximal.",
        ),
        direction_rule="anterior_positive",
        min_direction_alignment=-0.10,
    ),
    "Abdução de Ombro": MovementSpec(
        name="Abdução de Ombro",
        joint_code="shoulder",
        proximal_node="quadril",
        vertex_node="ombro",
        distal_node="cotovelo",
        reference_rom_deg=180.0,
        enter_flexion_deg=90.0,
        complete_extension_deg=20.0,
        converter=direct_internal_angle,
        converter_id="direct_internal_angle",
        measurement_plane="frontal",
        recommended_view="anterior_or_posterior",
        capture_instructions=(
            "Manter quadril, ombro e cotovelo do lado avaliado visíveis durante toda a elevação lateral.",
            "Usar vista anterior ou posterior com a câmera aproximadamente perpendicular ao plano frontal.",
            "Executar abdução para fora do corpo; adução cruzando a linha média é rejeitada pelo gate direcional.",
            "Evitar inclinação lateral do tronco e elevação compensatória do ombro.",
        ),
        direction_rule="lateral_outward",
        min_direction_alignment=-0.10,
    ),
    "Dorsiflexão de Tornozelo": MovementSpec(
        name="Dorsiflexão de Tornozelo",
        joint_code="ankle",
        proximal_node="joelho",
        vertex_node="tornozelo",
        distal_node="pe",
        reference_rom_deg=20.0,
        enter_flexion_deg=10.0,
        complete_extension_deg=3.0,
        converter=ankle_dorsiflexion_from_internal_angle,
        converter_id="ankle_dorsiflexion_from_internal_angle",
        measurement_plane="sagittal",
        recommended_view="lateral",
        capture_instructions=(
            "Manter joelho, tornozelo e antepé do lado avaliado visíveis.",
            "Usar vista lateral e evitar rotação do pé em relação à câmera.",
            "A convenção usa aproximadamente 90° tíbia–pé como neutro; flexão plantar não é quantificada por este movimento.",
        ),
    ),
}


def get_movement_spec(name: str) -> MovementSpec | None:
    return MOVEMENT_SPECS.get(name)


def public_movement_specs() -> list[dict[str, object]]:
    return [
        {
            "name": spec.name,
            "jointCode": spec.joint_code,
            "referenceRomDeg": spec.reference_rom_deg,
            "validated": spec.validated,
            "validationStatus": "preclinical",
            "convention": spec.converter_id,
            "measurementPlane": spec.measurement_plane,
            "recommendedView": spec.recommended_view,
            "landmarks": {
                "proximal": spec.proximal_node,
                "vertex": spec.vertex_node,
                "distal": spec.distal_node,
            },
            "captureInstructions": list(spec.capture_instructions),
            "minViewAlignment": spec.min_view_alignment,
            "maxOutOfPlaneDeg": spec.max_out_of_plane_deg,
            "requirePlaneProjection": spec.require_plane_projection,
            "directionRule": spec.direction_rule,
            "minDirectionAlignment": spec.min_direction_alignment,
        }
        for spec in MOVEMENT_SPECS.values()
    ]


def tracking_quality(
    confidences: list[float | None],
    *,
    source: str,
    valid_threshold: float = 0.65,
) -> dict[str, object]:
    """Resume qualidade de rastreamento sem tratá-la como acurácia clínica."""
    finite = [float(v) for v in confidences if v is not None and math.isfinite(float(v))]
    if not finite:
        return {
            "landmark_confidence_min": None,
            "landmark_confidence_mean": None,
            "quality": "unavailable",
            "measurement_valid": False,
            "quality_flags": ["landmarks_unavailable"],
        }

    minimum = min(finite)
    mean = sum(finite) / len(finite)
    flags: list[str] = []
    if source != "observed":
        flags.append("predicted_landmarks")
    if minimum < valid_threshold:
        flags.append("low_landmark_visibility")

    valid = source == "observed" and len(finite) == 3 and minimum >= valid_threshold
    if source != "observed":
        quality = "predicted"
    elif minimum >= 0.90:
        quality = "excellent"
    elif minimum >= 0.80:
        quality = "good"
    elif minimum >= valid_threshold:
        quality = "acceptable"
    else:
        quality = "low"

    return {
        "landmark_confidence_min": round(minimum, 4),
        "landmark_confidence_mean": round(mean, 4),
        "quality": quality,
        "measurement_valid": valid,
        "quality_flags": flags,
    }


def apply_plane_quality(
    base: dict[str, object],
    *,
    body_frame_available: bool,
    body_frame_confidence: float | None,
    view_alignment: float | None,
    min_view_alignment: float,
    out_of_plane_deg: float | None,
    max_out_of_plane_deg: float,
    projection_applied: bool,
    direction_rule: str | None = None,
    direction_alignment: float | None = None,
    min_direction_alignment: float = -1.0,
) -> dict[str, object]:
    """Combina rastreamento e geometria de captura sem inferir acurácia clínica.

    Esta função decide apenas se o frame satisfaz o protocolo geométrico para
    entrar nas métricas clínicas. Ela não transforma visibilidade/alinhamento em
    probabilidade de acerto do ângulo.
    """
    result = dict(base)
    flags = list(result.get("quality_flags") or [])
    valid = bool(result.get("measurement_valid"))

    if not body_frame_available:
        flags.append("body_frame_unavailable")
        valid = False
    if body_frame_confidence is None or body_frame_confidence < 0.65:
        flags.append("low_body_frame_visibility")
        valid = False
    if not projection_applied:
        flags.append("anatomical_plane_projection_unavailable")
        valid = False
    if view_alignment is None:
        flags.append("camera_alignment_unavailable")
        valid = False
    elif view_alignment < float(min_view_alignment):
        flags.append("camera_view_misaligned")
        valid = False
    if out_of_plane_deg is None:
        flags.append("out_of_plane_unavailable")
        valid = False
    elif out_of_plane_deg > float(max_out_of_plane_deg):
        flags.append("out_of_plane_motion")
        valid = False

    if direction_rule:
        if direction_alignment is None:
            flags.append("movement_direction_unavailable")
            valid = False
        elif direction_alignment < float(min_direction_alignment):
            flags.append("movement_direction_mismatch")
            valid = False

    # Remove duplicatas preservando ordem para telemetria estável.
    result["quality_flags"] = list(dict.fromkeys(flags))
    result["measurement_valid"] = valid
    if not valid and result.get("quality") not in ("predicted", "unavailable"):
        result["quality"] = "low"
    return result
