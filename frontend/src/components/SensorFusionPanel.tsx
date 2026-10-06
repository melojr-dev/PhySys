import React, { useMemo, useState } from "react";
import { api } from "../api";
import type { IMUAnalysisRecord } from "../types";

function fmt(value: number | null | undefined, digits = 2) {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

function parseCsv(text: string) {
  const rows = text.trim().split(/\r?\n/).filter(Boolean);
  if (rows.length < 2) throw new Error("CSV sem amostras suficientes.");
  const header = rows[0].split(",").map((v) => v.trim().toLowerCase());
  const idx = (name: string) => header.indexOf(name);
  const required = ["timestamp_ms", "gx", "gy", "gz"];
  for (const name of required) if (idx(name) < 0) throw new Error(`Coluna obrigatória ausente: ${name}`);
  const hasAccel = ["ax", "ay", "az"].every((name) => idx(name) >= 0);
  const timestamps_ms: number[] = [];
  const gyro_rad_s: number[][] = [];
  const accel_m_s2: number[][] = [];
  for (const line of rows.slice(1)) {
    const cells = line.split(",").map((v) => v.trim());
    const values = required.map((name) => Number(cells[idx(name)]));
    if (values.some((value) => !Number.isFinite(value))) continue;
    let accel: number[] | null = null;
    if (hasAccel) {
      accel = ["ax", "ay", "az"].map((name) => Number(cells[idx(name)]));
      if (!accel.every(Number.isFinite)) continue;
    }
    timestamps_ms.push(values[0]);
    gyro_rad_s.push(values.slice(1));
    if (accel) accel_m_s2.push(accel);
  }
  if (timestamps_ms.length < 32) throw new Error("São necessárias ao menos 32 amostras válidas.");
  return { timestamps_ms, gyro_rad_s, accel_m_s2: hasAccel ? accel_m_s2 : undefined };
}

export function SensorFusionPanel({ jobId, onAnalysis }: { jobId: string | null; onAnalysis?: (record: IMUAnalysisRecord) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [offsetMs, setOffsetMs] = useState(0);
  const [bandLow, setBandLow] = useState(3);
  const [bandHigh, setBandHigh] = useState(8);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [record, setRecord] = useState<IMUAnalysisRecord | null>(null);

  const sync = record?.summary.synchronization ?? null;
  const flags = useMemo(() => record?.summary.quality_flags ?? [], [record]);

  async function analyze() {
    if (!file || busy) return;
    setBusy(true);
    setError("");
    try {
      const parsed = parseCsv(await file.text());
      const result = await api.analyzeExperimentalImu({
        ...parsed,
        band_low_hz: bandLow,
        band_high_hz: bandHigh,
        job_id: jobId,
        video_offset_ms: offsetMs,
        device_label: file.name,
      });
      setRecord(result);
      onAnalysis?.(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao analisar IMU");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="sensor-panel panel">
      <div className="section-title-row">
        <div><span className="eyebrow">Experimental</span><h2>Sensor fusion · IMU</h2></div>
        <span className="data-count">não diagnóstico</span>
      </div>
      <p className="methodology-warning">
        Importa telemetria de smartphone para análise espectral e sincronização temporal. Frequência dominante e potência de banda são descritores do sinal, não diagnóstico de Parkinson, tremor, bradicinesia ou rigidez.
      </p>
      <div className="sensor-controls">
        <label className="field"><span>CSV IMU</span><input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></label>
        <label className="field compact"><span>Offset vídeo↔IMU (ms)</span><input type="number" value={offsetMs} onChange={(e) => setOffsetMs(Number(e.target.value))} /></label>
        <label className="field compact"><span>Banda inicial (Hz)</span><input type="number" step="0.1" min="0.2" max="19" value={bandLow} onChange={(e) => setBandLow(Number(e.target.value))} /></label>
        <label className="field compact"><span>Banda final (Hz)</span><input type="number" step="0.1" min="0.3" max="20" value={bandHigh} onChange={(e) => setBandHigh(Number(e.target.value))} /></label>
      </div>
      <div className="sensor-actions">
        <button className="secondary-button" disabled={!file || busy || bandLow >= bandHigh} onClick={analyze}>{busy ? "Analisando…" : "Analisar IMU"}</button>
        <span>{jobId ? "Vinculada à sessão de vídeo atual" : "Sem sessão de vídeo vinculada"}</span>
      </div>
      {error && <div className="review-status error">{error}</div>}
      {record && (
        <div className="sensor-grid">
          <div><span>Amostragem</span><strong>{fmt(record.summary.sampling_rate_hz, 1)} Hz</strong></div>
          <div><span>Jitter</span><strong>{fmt(record.summary.sampling_jitter_cv_pct, 2)}%</strong></div>
          <div><span>Frequência dominante</span><strong>{fmt(record.summary.dominant_frequency_hz, 2)} Hz</strong></div>
          <div><span>Potência na banda</span><strong>{record.summary.configured_band_power_ratio == null ? "—" : `${fmt(record.summary.configured_band_power_ratio * 100, 1)}%`}</strong></div>
          <div><span>Periodicidade</span><strong>{fmt(record.summary.periodicity_index, 3)}</strong></div>
          <div><span>RMS giroscópio</span><strong>{fmt(record.summary.gyro_rms_rad_s, 4)} rad/s</strong></div>
          <div><span>Qualidade</span><strong>{record.summary.status} · {fmt(record.summary.quality_score * 100, 0)}%</strong></div>
          <div><span>Overlap vídeo</span><strong>{sync ? `${fmt(sync.overlap_ratio * 100, 1)}%` : "—"}</strong></div>
          <div><span>Erro sync mediano</span><strong>{sync ? `${fmt(sync.median_alignment_error_ms, 2)} ms` : "—"}</strong></div>
          <div><span>SHA-256</span><strong className="hash-value">{record.input_sha256.slice(0, 16)}</strong></div>
        </div>
      )}
      {flags.length > 0 && <p className="review-help">Flags: {flags.join(", ")}</p>}
      <p className="review-help">CSV esperado: timestamp_ms,gx,gy,gz e opcionalmente ax,ay,az. Giroscópio em rad/s; aceleração em m/s².</p>
    </section>
  );
}
