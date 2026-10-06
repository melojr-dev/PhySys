from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np
from scipy.signal import detrend, welch


@dataclass(frozen=True, slots=True)
class IMUSynchronization:
    linked_job_id: str | None
    offset_ms: float
    video_frame_count: int
    aligned_frame_count: int
    overlap_ratio: float
    median_alignment_error_ms: float | None
    p95_alignment_error_ms: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IMUAnalysis:
    schema_version: int
    experimental: bool
    diagnostic: bool
    sample_count: int
    duration_s: float
    sampling_rate_hz: float
    sampling_jitter_cv_pct: float
    invalid_sample_ratio: float
    gyro_rms_rad_s: float
    gyro_p95_rad_s: float
    accel_dynamic_rms_m_s2: float | None
    dominant_frequency_hz: float | None
    dominant_frequency_power_ratio: float | None
    configured_band_hz: tuple[float, float]
    configured_band_power_ratio: float | None
    spectral_entropy: float | None
    periodicity_index: float | None
    quality_score: float
    status: str
    quality_flags: tuple[str, ...]
    synchronization: IMUSynchronization | None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["quality_flags"] = list(self.quality_flags)
        payload["configured_band_hz"] = list(self.configured_band_hz)
        if self.synchronization is not None:
            payload["synchronization"] = self.synchronization.to_dict()
        return payload


def _safe_percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, q)) if finite.size else 0.0


def _principal_gyro_signal(gyro: np.ndarray) -> np.ndarray:
    centered = gyro - np.nanmedian(gyro, axis=0, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    if centered.shape[0] < 3 or np.allclose(centered, 0.0):
        return np.zeros(centered.shape[0], dtype=np.float64)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    signal = centered @ axis
    return np.asarray(signal, dtype=np.float64)


def _spectral_metrics(
    signal: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    if signal.size < 32 or fs <= 0:
        return None, None, None, None, None
    clean = detrend(np.asarray(signal, dtype=np.float64), type="constant")
    if float(np.std(clean)) < 1e-8:
        return None, None, 0.0, 1.0, 0.0
    nperseg = min(signal.size, max(64, int(round(fs * 4.0))))
    freqs, psd = welch(clean, fs=fs, nperseg=nperseg, detrend="linear", scaling="density")
    finite = np.isfinite(freqs) & np.isfinite(psd) & (psd >= 0)
    freqs, psd = freqs[finite], psd[finite]
    analysis_mask = (freqs >= 0.5) & (freqs <= min(20.0, fs * 0.45))
    if not np.any(analysis_mask):
        return None, None, None, None, None
    af, ap = freqs[analysis_mask], psd[analysis_mask]
    total_power = float(np.trapezoid(ap, af)) if af.size > 1 else float(ap.sum())
    if total_power <= 1e-12:
        return None, None, 0.0, 1.0, 0.0
    peak_idx = int(np.argmax(ap))
    dominant_frequency = float(af[peak_idx])
    bin_power_ratio = float(ap[peak_idx] / max(float(ap.sum()), 1e-12))
    low, high = band_hz
    band_mask = (freqs >= low) & (freqs <= high)
    band_power = float(np.trapezoid(psd[band_mask], freqs[band_mask])) if np.count_nonzero(band_mask) > 1 else 0.0
    band_ratio = float(np.clip(band_power / total_power, 0.0, 1.0))

    prob = ap / max(float(ap.sum()), 1e-12)
    nonzero = prob[prob > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum() / math.log(max(len(prob), 2)))
    entropy = float(np.clip(entropy, 0.0, 1.0))
    periodicity = float(np.clip(1.0 - entropy, 0.0, 1.0))
    return dominant_frequency, bin_power_ratio, band_ratio, entropy, periodicity


def synchronize_imu_to_video(
    imu_timestamps_ms: np.ndarray | list[float],
    video_timestamps_ms: np.ndarray | list[float],
    *,
    offset_ms: float = 0.0,
    linked_job_id: str | None = None,
) -> IMUSynchronization:
    """Resume o alinhamento temporal entre IMU e vídeo.

    Convenção: ``video_time_ms ~= imu_time_ms + offset_ms``. Portanto, offset
    positivo desloca a série IMU para frente no eixo temporal do vídeo.
    """
    imu = np.asarray(imu_timestamps_ms, dtype=np.float64)
    video = np.asarray(video_timestamps_ms, dtype=np.float64)
    if imu.ndim != 1 or video.ndim != 1:
        raise ValueError("timestamps devem ser vetores 1D")
    if imu.size < 2 or video.size < 1:
        return IMUSynchronization(linked_job_id, float(offset_ms), int(video.size), 0, 0.0, None, None)
    aligned_imu = imu + float(offset_ms)
    valid_imu = aligned_imu[np.isfinite(aligned_imu)]
    valid_video = video[np.isfinite(video)]
    if valid_imu.size < 2 or valid_video.size == 0:
        return IMUSynchronization(linked_job_id, float(offset_ms), int(video.size), 0, 0.0, None, None)

    median_dt = float(np.median(np.diff(valid_imu)))
    max_error = max(20.0, median_dt * 1.5)
    errors: list[float] = []
    for t in valid_video:
        idx = int(np.searchsorted(valid_imu, t))
        candidates = []
        if idx < valid_imu.size:
            candidates.append(abs(float(valid_imu[idx] - t)))
        if idx > 0:
            candidates.append(abs(float(valid_imu[idx - 1] - t)))
        if candidates:
            err = min(candidates)
            if err <= max_error:
                errors.append(err)
    aligned = len(errors)
    ratio = aligned / max(1, valid_video.size)
    return IMUSynchronization(
        linked_job_id=linked_job_id,
        offset_ms=round(float(offset_ms), 4),
        video_frame_count=int(valid_video.size),
        aligned_frame_count=aligned,
        overlap_ratio=round(float(ratio), 4),
        median_alignment_error_ms=round(float(np.median(errors)), 4) if errors else None,
        p95_alignment_error_ms=round(float(np.percentile(errors, 95)), 4) if errors else None,
    )


def analyze_imu_signal(
    timestamps_ms: np.ndarray | list[float],
    gyro_rad_s: np.ndarray | list[list[float]],
    *,
    accel_m_s2: np.ndarray | list[list[float]] | None = None,
    band_hz: tuple[float, float] = (3.0, 8.0),
    video_timestamps_ms: np.ndarray | list[float] | None = None,
    video_offset_ms: float = 0.0,
    linked_job_id: str | None = None,
) -> IMUAnalysis:
    """Analisa telemetria inercial sem emitir classificação diagnóstica.

    O sinal giroscópico é projetado no primeiro componente principal para
    reduzir dependência da orientação física do aparelho. A análise espectral é
    descritiva: frequência dominante, razão de potência em uma banda configurada
    e entropia espectral. Esses campos não equivalem a diagnóstico de tremor,
    bradicinesia ou rigidez.
    """
    t = np.asarray(timestamps_ms, dtype=np.float64)
    gyro = np.asarray(gyro_rad_s, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("timestamps_ms deve ser vetor 1D")
    if gyro.ndim != 2 or gyro.shape[1] != 3:
        raise ValueError("gyro_rad_s deve ter formato (N,3)")
    if t.size != gyro.shape[0]:
        raise ValueError("timestamps e giroscópio devem ter o mesmo N")
    if t.size < 32:
        raise ValueError("São necessárias ao menos 32 amostras IMU")
    if t.size > 60000:
        raise ValueError("Máximo de 60000 amostras IMU por análise")

    accel = None
    if accel_m_s2 is not None:
        accel = np.asarray(accel_m_s2, dtype=np.float64)
        if accel.shape != gyro.shape:
            raise ValueError("accel_m_s2 deve ter formato (N,3) e mesmo N do giroscópio")

    low, high = float(band_hz[0]), float(band_hz[1])
    if not (0.2 <= low < high <= 20.0):
        raise ValueError("band_hz deve satisfazer 0.2 <= low < high <= 20 Hz")

    finite_t = np.isfinite(t)
    finite_gyro = np.isfinite(gyro).all(axis=1)
    finite_accel = np.ones_like(finite_t, dtype=bool) if accel is None else np.isfinite(accel).all(axis=1)
    valid = finite_t & finite_gyro & finite_accel
    invalid_ratio = 1.0 - float(valid.mean())
    if valid.sum() < 32:
        raise ValueError("Amostras IMU válidas insuficientes")

    t_valid = t[valid]
    gyro_valid = gyro[valid]
    accel_valid = accel[valid] if accel is not None else None
    order = np.argsort(t_valid)
    t_valid = t_valid[order]
    gyro_valid = gyro_valid[order]
    if accel_valid is not None:
        accel_valid = accel_valid[order]

    dt = np.diff(t_valid)
    if np.any(dt <= 0):
        unique_mask = np.concatenate(([True], dt > 0))
        t_valid = t_valid[unique_mask]
        gyro_valid = gyro_valid[unique_mask]
        if accel_valid is not None:
            accel_valid = accel_valid[unique_mask]
        dt = np.diff(t_valid)
    if dt.size < 2:
        raise ValueError("timestamps IMU não possuem progressão temporal suficiente")

    median_dt_ms = float(np.median(dt))
    if median_dt_ms <= 0:
        raise ValueError("intervalo temporal IMU inválido")
    fs = 1000.0 / median_dt_ms
    jitter_cv = float(np.std(dt, ddof=1) / median_dt_ms * 100.0) if dt.size >= 2 else 0.0
    duration_s = float((t_valid[-1] - t_valid[0]) / 1000.0)

    # Reamostragem uniforme para PSD. Evita interpretar jitter do clock como
    # conteúdo espectral do movimento.
    uniform_t = np.arange(t_valid[0], t_valid[-1] + median_dt_ms * 0.25, median_dt_ms)
    if uniform_t.size > 60000:
        uniform_t = uniform_t[:60000]
    gyro_uniform = np.column_stack([
        np.interp(uniform_t, t_valid, gyro_valid[:, axis]) for axis in range(3)
    ])
    principal = _principal_gyro_signal(gyro_uniform)

    gyro_mag = np.linalg.norm(gyro_uniform, axis=1)
    gyro_rms = float(np.sqrt(np.mean(np.square(gyro_mag))))
    gyro_p95 = _safe_percentile(gyro_mag, 95)

    accel_rms = None
    if accel_valid is not None:
        accel_uniform = np.column_stack([
            np.interp(uniform_t, t_valid, accel_valid[:, axis]) for axis in range(3)
        ])
        # Remove componente quase-estática (gravidade + bias do aparelho).
        accel_dynamic = accel_uniform - np.median(accel_uniform, axis=0, keepdims=True)
        accel_rms = float(np.sqrt(np.mean(np.sum(np.square(accel_dynamic), axis=1))))

    dom, dom_ratio, band_ratio, entropy, periodicity = _spectral_metrics(principal, fs, (low, high))

    flags: list[str] = []
    if duration_s < 5.0:
        flags.append("recording_too_short")
    if not (20.0 <= fs <= 250.0):
        flags.append("sampling_rate_outside_recommended_range")
    if jitter_cv > 5.0:
        flags.append("high_sampling_jitter")
    if invalid_ratio > 0.02:
        flags.append("invalid_samples_present")
    if dt.size and float(np.percentile(dt, 95)) > median_dt_ms * 1.8:
        flags.append("sampling_gaps_detected")
    if gyro_rms < 1e-4:
        flags.append("near_static_gyro_signal")

    sync = None
    if video_timestamps_ms is not None:
        sync = synchronize_imu_to_video(
            t_valid,
            video_timestamps_ms,
            offset_ms=video_offset_ms,
            linked_job_id=linked_job_id,
        )
        if sync.overlap_ratio < 0.80:
            flags.append("low_video_imu_overlap")

    quality = 1.0
    if duration_s < 5.0:
        quality -= 0.20
    if not (20.0 <= fs <= 250.0):
        quality -= 0.15
    quality -= min(0.25, jitter_cv / 100.0)
    quality -= min(0.25, invalid_ratio * 2.0)
    if "sampling_gaps_detected" in flags:
        quality -= 0.10
    if sync is not None and sync.overlap_ratio < 0.80:
        quality -= min(0.25, (0.80 - sync.overlap_ratio) * 0.5)
    quality = float(np.clip(quality, 0.0, 1.0))
    status = "usable" if quality >= 0.75 and duration_s >= 5.0 else "limited"

    return IMUAnalysis(
        schema_version=1,
        experimental=True,
        diagnostic=False,
        sample_count=int(t_valid.size),
        duration_s=round(duration_s, 4),
        sampling_rate_hz=round(fs, 4),
        sampling_jitter_cv_pct=round(jitter_cv, 4),
        invalid_sample_ratio=round(invalid_ratio, 6),
        gyro_rms_rad_s=round(gyro_rms, 6),
        gyro_p95_rad_s=round(gyro_p95, 6),
        accel_dynamic_rms_m_s2=round(accel_rms, 6) if accel_rms is not None else None,
        dominant_frequency_hz=round(dom, 4) if dom is not None else None,
        dominant_frequency_power_ratio=round(dom_ratio, 6) if dom_ratio is not None else None,
        configured_band_hz=(round(low, 4), round(high, 4)),
        configured_band_power_ratio=round(band_ratio, 6) if band_ratio is not None else None,
        spectral_entropy=round(entropy, 6) if entropy is not None else None,
        periodicity_index=round(periodicity, 6) if periodicity is not None else None,
        quality_score=round(quality, 4),
        status=status,
        quality_flags=tuple(flags),
        synchronization=sync,
    )
