from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np
from scipy.signal import detrend, find_peaks, savgol_filter, welch


MOTOR_FEATURE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class VisionMotorFeatures:
    valid_frame_count: int
    total_frame_count: int
    valid_frame_ratio: float
    duration_s: float
    sampling_rate_hz: float | None
    angular_range_deg: float | None
    median_abs_velocity_deg_s: float | None
    p95_abs_velocity_deg_s: float | None
    cycle_count: int
    median_cycle_duration_s: float | None
    cycle_duration_cv_pct: float | None
    median_cycle_amplitude_deg: float | None
    cycle_amplitude_cv_pct: float | None
    median_cycle_peak_velocity_deg_s: float | None
    cycle_peak_velocity_cv_pct: float | None
    amplitude_decrement_pct: float | None
    peak_velocity_decrement_pct: float | None
    tremor_band_hz: tuple[float, float]
    tremor_dominant_frequency_hz: float | None
    tremor_band_power_ratio: float | None
    tremor_periodicity_index: float | None
    quality_score: float
    status: str
    quality_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quality_flags"] = list(self.quality_flags)
        payload["tremor_band_hz"] = list(self.tremor_band_hz)
        return payload


@dataclass(frozen=True, slots=True)
class MultimodalMotorProfile:
    schema_version: int
    experimental: bool
    diagnostic: bool
    intended_use: str
    min_cycle_prominence_deg: float
    unsupported_inferences: tuple[str, ...]
    vision: VisionMotorFeatures
    fusion_mode: str
    imu_analysis_id: str | None
    imu_status: str | None
    imu_quality_score: float | None
    imu_dominant_frequency_hz: float | None
    imu_band_power_ratio: float | None
    imu_periodicity_index: float | None
    video_imu_overlap_ratio: float | None
    cross_modal_frequency_delta_hz: float | None
    fusion_quality_score: float
    status: str
    quality_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["unsupported_inferences"] = list(self.unsupported_inferences)
        payload["quality_flags"] = list(self.quality_flags)
        payload["vision"] = self.vision.to_dict()
        return payload


def _cv_pct(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    mean = float(np.mean(finite))
    if abs(mean) < 1e-9:
        return None
    return float(np.std(finite, ddof=1) / abs(mean) * 100.0)


def _decrement_pct(values: np.ndarray) -> float | None:
    """Percentual de redução entre início e fim; positivo = redução.

    É um descritor longitudinal do ensaio, não uma classificação de
    bradicinesia/fatigabilidade. Requer ao menos quatro ciclos.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        return None
    split = max(2, finite.size // 3)
    first = float(np.mean(finite[:split]))
    last = float(np.mean(finite[-split:]))
    if abs(first) < 1e-9:
        return None
    return float((first - last) / abs(first) * 100.0)


def _spectral_angle_metrics(
    angles_deg: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
) -> tuple[float | None, float | None, float | None]:
    if angles_deg.size < 64 or fs < 20.0:
        return None, None, None
    clean = detrend(np.asarray(angles_deg, dtype=np.float64), type="linear")
    if float(np.std(clean)) < 1e-6:
        return None, 0.0, 0.0
    nperseg = min(clean.size, max(64, int(round(fs * 4.0))))
    freqs, psd = welch(clean, fs=fs, nperseg=nperseg, detrend="linear", scaling="density")
    finite = np.isfinite(freqs) & np.isfinite(psd) & (psd >= 0)
    freqs, psd = freqs[finite], psd[finite]
    if freqs.size < 2:
        return None, None, None
    analysis = (freqs >= 0.5) & (freqs <= min(20.0, fs * 0.45))
    if np.count_nonzero(analysis) < 2:
        return None, None, None
    af, ap = freqs[analysis], psd[analysis]
    total = float(np.trapezoid(ap, af))
    if total <= 1e-12:
        return None, 0.0, 0.0
    low, high = band_hz
    band = (freqs >= low) & (freqs <= high)
    band_power = float(np.trapezoid(psd[band], freqs[band])) if np.count_nonzero(band) > 1 else 0.0
    ratio = float(np.clip(band_power / total, 0.0, 1.0))
    if ratio < 1e-4 or np.count_nonzero(band) == 0 or float(np.max(psd[band])) <= 0:
        return None, ratio, 0.0
    band_freqs = freqs[band]
    band_psd = psd[band]
    dominant = float(band_freqs[int(np.argmax(band_psd))])
    prob = band_psd / max(float(np.sum(band_psd)), 1e-12)
    nonzero = prob[prob > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum() / math.log(max(len(prob), 2)))
    periodicity = float(np.clip(1.0 - entropy, 0.0, 1.0))
    return dominant, ratio, periodicity


def analyze_repetitive_motion(
    measurements: Iterable[dict[str, Any]],
    *,
    min_cycle_prominence_deg: float = 8.0,
    tremor_band_hz: tuple[float, float] = (3.0, 8.0),
) -> VisionMotorFeatures:
    """Extrai descritores de uma série angular clínica frame-a-frame.

    Somente frames que passaram pelo protocolo de validade biomecânica entram
    na série. As métricas de decremento, ritmo e espectro são experimentais e
    não equivalem a diagnóstico de bradicinesia ou tremor.
    """
    rows = list(measurements)
    valid_rows = [
        row for row in rows
        if bool(row.get("measurement_valid"))
        and row.get("angle_clinical_deg") is not None
        and row.get("timestamp_ms") is not None
        and np.isfinite(float(row["angle_clinical_deg"]))
        and np.isfinite(float(row["timestamp_ms"]))
    ]
    valid_rows.sort(key=lambda row: float(row["timestamp_ms"]))
    total = len(rows)
    valid_n = len(valid_rows)
    ratio = valid_n / max(1, total)
    flags: list[str] = []

    if valid_n < 12:
        flags.append("insufficient_valid_video_samples")
        return VisionMotorFeatures(
            valid_frame_count=valid_n, total_frame_count=total,
            valid_frame_ratio=round(ratio, 4), duration_s=0.0,
            sampling_rate_hz=None, angular_range_deg=None,
            median_abs_velocity_deg_s=None, p95_abs_velocity_deg_s=None,
            cycle_count=0, median_cycle_duration_s=None, cycle_duration_cv_pct=None,
            median_cycle_amplitude_deg=None, cycle_amplitude_cv_pct=None,
            median_cycle_peak_velocity_deg_s=None, cycle_peak_velocity_cv_pct=None,
            amplitude_decrement_pct=None, peak_velocity_decrement_pct=None,
            tremor_band_hz=tremor_band_hz, tremor_dominant_frequency_hz=None,
            tremor_band_power_ratio=None, tremor_periodicity_index=None,
            quality_score=0.0, status="insufficient", quality_flags=tuple(flags),
        )

    t = np.array([float(row["timestamp_ms"]) for row in valid_rows], dtype=np.float64)
    a = np.array([float(row["angle_clinical_deg"]) for row in valid_rows], dtype=np.float64)
    order = np.argsort(t)
    t, a = t[order], a[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, a = t[keep], a[keep]
    valid_n = int(t.size)
    if t.size < 12:
        flags.append("insufficient_unique_timestamps")
        return analyze_repetitive_motion([], min_cycle_prominence_deg=min_cycle_prominence_deg, tremor_band_hz=tremor_band_hz)

    dt = np.diff(t)
    median_dt_ms = float(np.median(dt))
    fs = 1000.0 / median_dt_ms if median_dt_ms > 0 else 0.0
    duration_s = float((t[-1] - t[0]) / 1000.0)

    # Série uniformizada: evita que jitter do processamento altere derivadas e PSD.
    uniform_t = np.arange(t[0], t[-1] + median_dt_ms * 0.25, median_dt_ms)
    uniform_a = np.interp(uniform_t, t, a)
    if uniform_a.size >= 7:
        desired = max(5, int(round(fs * 0.15)) | 1)
        window = min(desired, uniform_a.size if uniform_a.size % 2 == 1 else uniform_a.size - 1)
        if window >= 5:
            smooth = savgol_filter(uniform_a, window_length=window, polyorder=2, mode="interp")
        else:
            smooth = uniform_a
    else:
        smooth = uniform_a

    velocity = np.gradient(smooth, median_dt_ms / 1000.0)
    abs_v = np.abs(velocity)
    angular_range = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))

    min_distance = max(1, int(round(fs * 0.30)))
    prominence = max(2.0, float(min_cycle_prominence_deg))
    peaks, _ = find_peaks(smooth, prominence=prominence, distance=min_distance)
    troughs, _ = find_peaks(-smooth, prominence=prominence, distance=min_distance)
    anchors = peaks if len(peaks) >= len(troughs) else troughs

    cycle_durations: list[float] = []
    cycle_amplitudes: list[float] = []
    cycle_peak_velocities: list[float] = []
    for left, right in zip(anchors[:-1], anchors[1:]):
        if right <= left + 1:
            continue
        duration = float((uniform_t[right] - uniform_t[left]) / 1000.0)
        if duration < 0.20 or duration > 10.0:
            continue
        segment = smooth[left:right + 1]
        segment_v = abs_v[left:right + 1]
        amplitude = float(np.max(segment) - np.min(segment))
        if amplitude < prominence:
            continue
        cycle_durations.append(duration)
        cycle_amplitudes.append(amplitude)
        cycle_peak_velocities.append(float(np.percentile(segment_v, 95)))

    cd = np.asarray(cycle_durations, dtype=np.float64)
    ca = np.asarray(cycle_amplitudes, dtype=np.float64)
    cv = np.asarray(cycle_peak_velocities, dtype=np.float64)
    dom, band_ratio, periodicity = _spectral_angle_metrics(smooth, fs, tremor_band_hz)

    if ratio < 0.70:
        flags.append("low_valid_frame_ratio")
    if duration_s < 5.0:
        flags.append("video_recording_too_short")
    if fs < 20.0:
        flags.append("video_sampling_rate_low_for_tremor_band")
    if len(cycle_durations) < 3:
        flags.append("few_repetition_cycles")

    quality = 1.0
    quality -= max(0.0, 0.75 - ratio) * 0.8
    if duration_s < 5.0:
        quality -= 0.20
    if fs < 20.0:
        quality -= 0.10
    if len(cycle_durations) < 3:
        quality -= 0.20
    quality = float(np.clip(quality, 0.0, 1.0))
    status = "usable" if quality >= 0.75 and valid_n >= 32 and ratio >= 0.70 else "limited"

    return VisionMotorFeatures(
        valid_frame_count=valid_n,
        total_frame_count=total,
        valid_frame_ratio=round(ratio, 4),
        duration_s=round(duration_s, 4),
        sampling_rate_hz=round(fs, 4) if fs > 0 else None,
        angular_range_deg=round(angular_range, 4),
        median_abs_velocity_deg_s=round(float(np.median(abs_v)), 4),
        p95_abs_velocity_deg_s=round(float(np.percentile(abs_v, 95)), 4),
        cycle_count=len(cycle_durations),
        median_cycle_duration_s=round(float(np.median(cd)), 4) if cd.size else None,
        cycle_duration_cv_pct=round(_cv_pct(cd), 4) if _cv_pct(cd) is not None else None,
        median_cycle_amplitude_deg=round(float(np.median(ca)), 4) if ca.size else None,
        cycle_amplitude_cv_pct=round(_cv_pct(ca), 4) if _cv_pct(ca) is not None else None,
        median_cycle_peak_velocity_deg_s=round(float(np.median(cv)), 4) if cv.size else None,
        cycle_peak_velocity_cv_pct=round(_cv_pct(cv), 4) if _cv_pct(cv) is not None else None,
        amplitude_decrement_pct=round(_decrement_pct(ca), 4) if _decrement_pct(ca) is not None else None,
        peak_velocity_decrement_pct=round(_decrement_pct(cv), 4) if _decrement_pct(cv) is not None else None,
        tremor_band_hz=tremor_band_hz,
        tremor_dominant_frequency_hz=round(dom, 4) if dom is not None else None,
        tremor_band_power_ratio=round(band_ratio, 6) if band_ratio is not None else None,
        tremor_periodicity_index=round(periodicity, 6) if periodicity is not None else None,
        quality_score=round(quality, 4),
        status=status,
        quality_flags=tuple(dict.fromkeys(flags)),
    )


def build_multimodal_motor_profile(
    measurements: Iterable[dict[str, Any]],
    *,
    imu_record: dict[str, Any] | None = None,
    min_cycle_prominence_deg: float = 8.0,
    tremor_band_hz: tuple[float, float] = (3.0, 8.0),
) -> MultimodalMotorProfile:
    vision = analyze_repetitive_motion(
        measurements,
        min_cycle_prominence_deg=min_cycle_prominence_deg,
        tremor_band_hz=tremor_band_hz,
    )
    imu_summary = (imu_record or {}).get("summary") if imu_record else None
    imu_summary = imu_summary if isinstance(imu_summary, dict) else None
    imu_id = str((imu_record or {}).get("id") or "").strip() or None

    imu_quality = None
    imu_status = None
    imu_freq = None
    imu_band = None
    imu_periodicity = None
    overlap = None
    flags = list(vision.quality_flags)

    if imu_summary is not None:
        imu_quality = float(imu_summary.get("quality_score")) if imu_summary.get("quality_score") is not None else None
        imu_status = str(imu_summary.get("status") or "") or None
        imu_freq = float(imu_summary["dominant_frequency_hz"]) if imu_summary.get("dominant_frequency_hz") is not None else None
        imu_band = float(imu_summary["configured_band_power_ratio"]) if imu_summary.get("configured_band_power_ratio") is not None else None
        imu_periodicity = float(imu_summary["periodicity_index"]) if imu_summary.get("periodicity_index") is not None else None
        sync = imu_summary.get("synchronization")
        if isinstance(sync, dict) and sync.get("overlap_ratio") is not None:
            overlap = float(sync["overlap_ratio"])
            if overlap < 0.80:
                flags.append("low_video_imu_overlap")
        if imu_status != "usable":
            flags.append("imu_quality_limited")
    else:
        flags.append("imu_not_linked")

    delta = None
    if vision.tremor_dominant_frequency_hz is not None and imu_freq is not None:
        delta = abs(float(vision.tremor_dominant_frequency_hz) - float(imu_freq))

    components = [vision.quality_score]
    if imu_quality is not None:
        components.append(imu_quality)
    if overlap is not None:
        components.append(overlap)
    fusion_quality = float(np.mean(components)) if components else 0.0
    if imu_summary is None:
        status = "vision_only" if vision.status == "usable" else "limited"
    else:
        status = "usable" if vision.status == "usable" and imu_status == "usable" and fusion_quality >= 0.75 else "limited"

    return MultimodalMotorProfile(
        schema_version=MOTOR_FEATURE_SCHEMA_VERSION,
        experimental=True,
        diagnostic=False,
        intended_use="research_feature_extraction",
        min_cycle_prominence_deg=round(float(min_cycle_prominence_deg), 4),
        unsupported_inferences=(
            "parkinson_diagnosis",
            "rigidity_or_cogwheel_classification",
            "clinical_bradykinesia_classification",
        ),
        vision=vision,
        fusion_mode="video_imu" if imu_summary is not None else "vision_only",
        imu_analysis_id=imu_id,
        imu_status=imu_status,
        imu_quality_score=round(imu_quality, 4) if imu_quality is not None else None,
        imu_dominant_frequency_hz=round(imu_freq, 4) if imu_freq is not None else None,
        imu_band_power_ratio=round(imu_band, 6) if imu_band is not None else None,
        imu_periodicity_index=round(imu_periodicity, 6) if imu_periodicity is not None else None,
        video_imu_overlap_ratio=round(overlap, 4) if overlap is not None else None,
        cross_modal_frequency_delta_hz=round(delta, 4) if delta is not None else None,
        fusion_quality_score=round(fusion_quality, 4),
        status=status,
        quality_flags=tuple(dict.fromkeys(flags)),
    )
