from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np
from scipy.signal import find_peaks, savgol_filter

LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_HIP = 23
RIGHT_HIP = 24


@dataclass(frozen=True, slots=True)
class GaitEvent:
    side: str
    kind: str
    frame: int
    timestamp_s: float
    prominence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GaitTemporalAnalysis:
    schema_version: int
    experimental: bool
    coordinate_space: str
    duration_s: float
    fps: float
    total_frames: int
    valid_frame_ratio: float
    event_count: int
    left_event_count: int
    right_event_count: int
    cadence_steps_min: float | None
    mean_step_time_s: float | None
    step_time_cv_pct: float | None
    left_step_time_s: float | None
    right_step_time_s: float | None
    step_time_symmetry_ratio: float | None
    left_stride_time_s: float | None
    right_stride_time_s: float | None
    stride_time_symmetry_ratio: float | None
    stride_length_m: float | None
    quality_score: float
    status: str
    quality_flags: tuple[str, ...]
    events: tuple[GaitEvent, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["events"] = [event.to_dict() for event in self.events]
        payload["quality_flags"] = list(self.quality_flags)
        return payload


def _safe_mean(values: Iterable[float]) -> float | None:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def _symmetry_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return float(min(a, b) / max(a, b))


def _smooth(signal: np.ndarray, fps: float) -> np.ndarray:
    if signal.size < 5:
        return signal.astype(np.float64, copy=True)
    window = max(5, int(round(fps * 0.15)))
    if window % 2 == 0:
        window += 1
    if window >= signal.size:
        window = signal.size - 1 if signal.size % 2 == 0 else signal.size
    if window < 5:
        return signal.astype(np.float64, copy=True)
    return savgol_filter(signal, window_length=window, polyorder=2, mode="interp")


def _heel_events(signal: np.ndarray, side: str, fps: float) -> list[GaitEvent]:
    smooth = _smooth(signal, fps)
    finite = np.isfinite(smooth)
    if finite.sum() < max(8, int(fps)):
        return []
    if not finite.all():
        idx = np.arange(smooth.size)
        smooth = np.interp(idx, idx[finite], smooth[finite])

    spread = float(np.nanstd(smooth))
    prominence = max(1e-4, spread * 0.18)
    distance = max(2, int(round(fps * 0.30)))
    peaks, props = find_peaks(-smooth, distance=distance, prominence=prominence)
    prominences = props.get("prominences", np.zeros(len(peaks), dtype=np.float64))
    return [
        GaitEvent(
            side=side,
            kind="estimated_heel_strike",
            frame=int(frame),
            timestamp_s=float(frame / fps),
            prominence=float(prom),
        )
        for frame, prom in zip(peaks, prominences)
    ]


def _alternating_step_intervals(events: list[GaitEvent]) -> tuple[list[float], list[float], list[float]]:
    all_steps: list[float] = []
    left_steps: list[float] = []
    right_steps: list[float] = []
    for prev, cur in zip(events, events[1:]):
        if prev.side == cur.side:
            continue
        dt = cur.timestamp_s - prev.timestamp_s
        if not (0.20 <= dt <= 2.0):
            continue
        all_steps.append(dt)
        if cur.side == "left":
            left_steps.append(dt)
        else:
            right_steps.append(dt)
    return all_steps, left_steps, right_steps


def _stride_intervals(events: list[GaitEvent], side: str) -> list[float]:
    times = [event.timestamp_s for event in events if event.side == side]
    return [b - a for a, b in zip(times, times[1:]) if 0.40 <= (b - a) <= 3.0]


def _stride_lengths_metric_global(skel: np.ndarray, events: list[GaitEvent]) -> float | None:
    values: list[float] = []
    for side, idx in (("left", LEFT_HEEL), ("right", RIGHT_HEEL)):
        frames = [event.frame for event in events if event.side == side]
        for a, b in zip(frames, frames[1:]):
            p0 = skel[a, idx]
            p1 = skel[b, idx]
            if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
                values.append(float(np.linalg.norm(p1 - p0)))
    return _safe_mean(values)


def analyze_gait_temporal(
    skeleton_frames: np.ndarray | list,
    fps: float,
    *,
    coordinate_space: str = "relative_body_3d",
) -> GaitTemporalAnalysis:
    """Analisa temporalidade da marcha sem promover métricas experimentais a clínica.

    Espera skeleton no formato ``(N, 33, 3)``. Eventos de contato são estimados
    por mínimos locais da trajetória vertical dos calcanhares e devem ser
    tratados como *candidatos* até validação específica. Métricas espaciais só
    são retornadas quando ``coordinate_space == 'metric_global'``.
    """
    skel = np.asarray(skeleton_frames, dtype=np.float64)
    if skel.ndim != 3 or skel.shape[1:] != (33, 3):
        raise ValueError("skeleton_frames deve ter formato (N, 33, 3)")
    if not math.isfinite(float(fps)) or not (5.0 <= float(fps) <= 240.0):
        raise ValueError("fps deve estar entre 5 e 240")
    n = int(skel.shape[0])
    if n < 2:
        raise ValueError("São necessários pelo menos 2 frames")

    fps = float(fps)
    duration = (n - 1) / fps
    heel_points = skel[:, [LEFT_HEEL, RIGHT_HEEL], :]
    hips = skel[:, [LEFT_HIP, RIGHT_HIP], :]
    valid_mask = np.isfinite(heel_points).all(axis=(1, 2)) & np.isfinite(hips).all(axis=(1, 2))
    nonzero_mask = (np.linalg.norm(heel_points, axis=2) > 1e-8).all(axis=1)
    valid_mask &= nonzero_mask
    valid_ratio = float(valid_mask.mean())

    left_y = skel[:, LEFT_HEEL, 1].copy()
    right_y = skel[:, RIGHT_HEEL, 1].copy()
    left_y[~valid_mask] = np.nan
    right_y[~valid_mask] = np.nan

    events = _heel_events(left_y, "left", fps) + _heel_events(right_y, "right", fps)
    events.sort(key=lambda event: (event.frame, event.side))
    all_steps, left_steps, right_steps = _alternating_step_intervals(events)
    left_strides = _stride_intervals(events, "left")
    right_strides = _stride_intervals(events, "right")

    mean_step = _safe_mean(all_steps)
    cadence = (60.0 / mean_step) if mean_step and mean_step > 0 else None
    step_cv = None
    if len(all_steps) >= 2 and mean_step and mean_step > 0:
        step_cv = float(np.std(np.asarray(all_steps), ddof=1) / mean_step * 100.0)

    left_step = _safe_mean(left_steps)
    right_step = _safe_mean(right_steps)
    left_stride = _safe_mean(left_strides)
    right_stride = _safe_mean(right_strides)

    flags: list[str] = []
    if duration < 4.0:
        flags.append("recording_too_short")
    if valid_ratio < 0.80:
        flags.append("low_valid_frame_ratio")
    if len(events) < 6:
        flags.append("too_few_gait_events")
    same_side_adjacent = sum(1 for a, b in zip(events, events[1:]) if a.side == b.side)
    if events and same_side_adjacent / max(1, len(events) - 1) > 0.25:
        flags.append("poor_event_alternation")
    if coordinate_space != "metric_global":
        flags.append("spatial_metrics_disabled_non_global_coordinates")

    quality = 1.0
    quality -= min(0.45, max(0.0, 0.90 - valid_ratio))
    if "recording_too_short" in flags:
        quality -= 0.15
    if "too_few_gait_events" in flags:
        quality -= 0.20
    if "poor_event_alternation" in flags:
        quality -= 0.15
    quality = float(np.clip(quality, 0.0, 1.0))
    status = "usable" if quality >= 0.75 and len(events) >= 6 else "limited"

    stride_length = (
        _stride_lengths_metric_global(skel, events)
        if coordinate_space == "metric_global" and len(events) >= 4
        else None
    )

    return GaitTemporalAnalysis(
        schema_version=1,
        experimental=True,
        coordinate_space=coordinate_space,
        duration_s=round(duration, 4),
        fps=round(fps, 4),
        total_frames=n,
        valid_frame_ratio=round(valid_ratio, 4),
        event_count=len(events),
        left_event_count=sum(1 for event in events if event.side == "left"),
        right_event_count=sum(1 for event in events if event.side == "right"),
        cadence_steps_min=round(cadence, 4) if cadence is not None else None,
        mean_step_time_s=round(mean_step, 4) if mean_step is not None else None,
        step_time_cv_pct=round(step_cv, 4) if step_cv is not None else None,
        left_step_time_s=round(left_step, 4) if left_step is not None else None,
        right_step_time_s=round(right_step, 4) if right_step is not None else None,
        step_time_symmetry_ratio=(
            round(_symmetry_ratio(left_step, right_step), 4)
            if _symmetry_ratio(left_step, right_step) is not None else None
        ),
        left_stride_time_s=round(left_stride, 4) if left_stride is not None else None,
        right_stride_time_s=round(right_stride, 4) if right_stride is not None else None,
        stride_time_symmetry_ratio=(
            round(_symmetry_ratio(left_stride, right_stride), 4)
            if _symmetry_ratio(left_stride, right_stride) is not None else None
        ),
        stride_length_m=round(stride_length, 4) if stride_length is not None else None,
        quality_score=round(quality, 4),
        status=status,
        quality_flags=tuple(flags),
        events=tuple(events),
    )
