import React, { useState } from "react";
import { api } from "../api";
import type { LongitudinalAnalysisRecord, LongitudinalMetricSummary } from "../types";

function fmt(value: number | null | undefined, digits = 2) {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

function MetricCard({ title, metric }: { title: string; metric?: LongitudinalMetricSummary }) {
  if (!metric || metric.n === 0) return null;
  return (
    <div>
      <span>{title}</span>
      <strong>{fmt(metric.latest, metric.unit === "ratio" ? 3 : 2)} {metric.unit}</strong>
      <small>Δ {fmt(metric.delta, 2)} · CV {fmt(metric.cv_pct, 1)}%</small>
    </div>
  );
}

export function LongitudinalPanel({ jobId }: { jobId: string | null }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [record, setRecord] = useState<LongitudinalAnalysisRecord | null>(null);

  async function analyze() {
    if (!jobId || busy) return;
    setBusy(true);
    setError("");
    try {
      setRecord(await api.analyzeExperimentalLongitudinal({ job_id: jobId, include_motor_features: true }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao gerar análise longitudinal");
    } finally {
      setBusy(false);
    }
  }

  const summary = record?.summary;
  return (
    <section className="longitudinal-panel panel">
      <div className="section-title-row">
        <div><span className="eyebrow">Experimental</span><h2>Seguimento longitudinal</h2></div>
        <span className="data-count">descritivo · não diagnóstico</span>
      </div>
      <p className="methodology-warning">
        Compara apenas sessões do mesmo paciente, movimento, lado, protocolo e especificação clínica. Tendências não significam melhora, progressão de doença ou resposta terapêutica.
      </p>
      <div className="sensor-actions">
        <button className="secondary-button" disabled={!jobId || busy} onClick={analyze}>
          {busy ? "Comparando…" : "Comparar sessões compatíveis"}
        </button>
        {summary && <span>{summary.session_count} sessões comparáveis · {summary.span_days == null ? "intervalo —" : `${fmt(summary.span_days, 1)} dias`}</span>}
      </div>
      {error && <div className="review-status error">{error}</div>}
      {summary && (
        <>
          <div className="sensor-grid">
            <div><span>Status</span><strong>{summary.status}</strong></div>
            <div><span>Sessões comparáveis</span><strong>{summary.session_count}</strong></div>
            <div><span>Sessões excluídas</span><strong>{summary.excluded_session_count}</strong></div>
            <div><span>Versões de software</span><strong>{summary.software_versions.join(", ")}</strong></div>
            <MetricCard title="ADM" metric={summary.metrics.adm_deg} />
            <MetricCard title="Frames válidos" metric={summary.metrics.valid_frame_ratio} />
            <MetricCard title="Repetições" metric={summary.metrics.total_reps} />
            <MetricCard title="Amplitude mediana/ciclo" metric={summary.metrics.median_cycle_amplitude_deg} />
            <MetricCard title="Duração mediana/ciclo" metric={summary.metrics.median_cycle_duration_s} />
            <MetricCard title="Vel. pico mediana/ciclo" metric={summary.metrics.median_cycle_peak_velocity_deg_s} />
            <MetricCard title="Decremento amplitude" metric={summary.metrics.amplitude_decrement_pct} />
            <MetricCard title="Qualidade de fusão" metric={summary.metrics.fusion_quality_score} />
          </div>
          {summary.quality_flags.length > 0 && <p className="review-help">Flags: {summary.quality_flags.join(", ")}</p>}
          <p className="review-help">SHA-256 fonte: <span className="hash-value">{record.source_sha256.slice(0, 20)}</span></p>
        </>
      )}
    </section>
  );
}
