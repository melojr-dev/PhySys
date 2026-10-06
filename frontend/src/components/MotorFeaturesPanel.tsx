import React, { useState } from "react";
import { api } from "../api";
import type { MotorFeatureRecord } from "../types";

function fmt(value: number | null | undefined, digits = 2) {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

export function MotorFeaturesPanel({
  jobId,
  imuAnalysisId,
}: {
  jobId: string | null;
  imuAnalysisId: string | null;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [record, setRecord] = useState<MotorFeatureRecord | null>(null);

  async function analyze() {
    if (!jobId || busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await api.analyzeExperimentalMotorFeatures({
        job_id: jobId,
        imu_analysis_id: imuAnalysisId,
      });
      setRecord(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao extrair descritores motores");
    } finally {
      setBusy(false);
    }
  }

  const summary = record?.summary;
  const vision = summary?.vision;

  return (
    <section className="motor-panel panel">
      <div className="section-title-row">
        <div><span className="eyebrow">Experimental</span><h2>Perfil motor multimodal</h2></div>
        <span className="data-count">pesquisa · não diagnóstico</span>
      </div>
      <p className="methodology-warning">
        Resume ritmo, amplitude, velocidade, decremento ao longo de movimentos repetitivos e descritores espectrais de vídeo/IMU. Nenhum valor classifica Parkinson, bradicinesia clínica, rigidez ou roda dentada.
      </p>
      <div className="sensor-actions">
        <button className="secondary-button" disabled={!jobId || busy} onClick={analyze}>
          {busy ? "Extraindo…" : "Extrair descritores motores"}
        </button>
        <span>{imuAnalysisId ? "Vídeo + IMU vinculados" : "Somente vídeo; importe IMU acima para fusão"}</span>
      </div>
      {error && <div className="review-status error">{error}</div>}
      {summary && vision && (
        <>
          <div className="sensor-grid">
            <div><span>Modo</span><strong>{summary.fusion_mode === "video_imu" ? "Vídeo + IMU" : "Somente vídeo"}</strong></div>
            <div><span>Ciclos detectados</span><strong>{vision.cycle_count}</strong></div>
            <div><span>Duração mediana ciclo</span><strong>{fmt(vision.median_cycle_duration_s, 3)} s</strong></div>
            <div><span>Amplitude mediana</span><strong>{fmt(vision.median_cycle_amplitude_deg, 1)}°</strong></div>
            <div><span>CV amplitude</span><strong>{fmt(vision.cycle_amplitude_cv_pct, 1)}%</strong></div>
            <div><span>CV duração</span><strong>{fmt(vision.cycle_duration_cv_pct, 1)}%</strong></div>
            <div><span>Vel. pico mediana</span><strong>{fmt(vision.median_cycle_peak_velocity_deg_s, 1)} °/s</strong></div>
            <div><span>Decremento amplitude</span><strong>{fmt(vision.amplitude_decrement_pct, 1)}%</strong></div>
            <div><span>Decremento vel. pico</span><strong>{fmt(vision.peak_velocity_decrement_pct, 1)}%</strong></div>
            <div><span>Freq. vídeo na banda</span><strong>{fmt(vision.tremor_dominant_frequency_hz, 2)} Hz</strong></div>
            <div><span>Freq. IMU</span><strong>{fmt(summary.imu_dominant_frequency_hz, 2)} Hz</strong></div>
            <div><span>Δ frequência vídeo↔IMU</span><strong>{fmt(summary.cross_modal_frequency_delta_hz, 2)} Hz</strong></div>
            <div><span>Overlap vídeo↔IMU</span><strong>{summary.video_imu_overlap_ratio == null ? "—" : `${fmt(summary.video_imu_overlap_ratio * 100, 1)}%`}</strong></div>
            <div><span>Qualidade de fusão</span><strong>{fmt(summary.fusion_quality_score * 100, 0)}%</strong></div>
            <div><span>Status</span><strong>{summary.status}</strong></div>
            <div><span>SHA-256 fonte</span><strong className="hash-value">{record.source_sha256.slice(0, 16)}</strong></div>
          </div>
          {summary.quality_flags.length > 0 && <p className="review-help">Flags: {summary.quality_flags.join(", ")}</p>}
        </>
      )}
      <p className="review-help">Decrementos positivos significam redução entre o início e o fim do ensaio. São descritores longitudinais e não critérios diagnósticos.</p>
    </section>
  );
}
