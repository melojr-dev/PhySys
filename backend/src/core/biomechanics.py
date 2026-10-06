from __future__ import annotations

# Compatibilidade pública com versões anteriores.
from src.core.clinical import (
    ankle_dorsiflexion_from_internal_angle,
    clamp_angle_deg,
    flexion_from_internal_angle,
)


def elbow_flexion_from_internal_angle(internal_angle_deg: float) -> float:
    return flexion_from_internal_angle(internal_angle_deg)


def knee_flexion_from_internal_angle(internal_angle_deg: float) -> float:
    return flexion_from_internal_angle(internal_angle_deg)


def hip_flexion_from_internal_angle(internal_angle_deg: float) -> float:
    return flexion_from_internal_angle(internal_angle_deg)


def ankle_dorsiflexion_from_internal(internal_angle_deg: float) -> float:
    return ankle_dorsiflexion_from_internal_angle(internal_angle_deg)
