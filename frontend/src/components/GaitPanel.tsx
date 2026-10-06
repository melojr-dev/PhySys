import React, { useState } from "react";
import { api } from "../api";
import type { AnalysisResult, GaitTemporalAnalysis } from "../types";

function fmt(value: number | null | undefined, digits = 2) {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

export function GaitPanel({ result }: { result: AnalysisResult | null }) {
  const [record, setRecord] = useState<GaitTemporalAnalysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const canAnalyze = Boolean(result?.skeleton_3d?.length && result?.fps);

  async function analyze() {
    if (!result?.skeleton_3d?.length || !result.fps || busy) return;
    setBusy(true);
    setError("");
    try {
      setRecord(await api.analyzeExperimentalGait({
        fps: result.fps,
        coordinate_space: "relative_body_3d",
        skeleton_frames: result.skeleton_3d,
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha na análise temporal de marcha");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="portal-card">
      <div className="portal-card-heading"><div><span className="eyebrow">Experimental</span><h2>Marcha temporal</h2></div><span className="status-pill neutral">não diagnóstico</span></div>
      <p className="portal-copy">Analisa eventos temporais, cadência e simetria usando o esqueleto da sessão atual. Em coordenadas corporais relativas, distâncias em metros permanecem bloqueadas.</p>
      <div className="portal-actions"><button className="primary-button" disabled={!canAnalyze || busy} onClick={analyze}>{busy ? "Analisando…" : "Analisar sessão atual"}</button><span>{canAnalyze ? `${result?.skeleton_3d?.length ?? 0} frames disponíveis` : "Conclua uma avaliação com esqueleto 3D primeiro"}</span></div>
      {error && <div className="review-status error">{error}</div>}
      {record && (
        <>
          <div className="portal-metric-grid">
            <div><span>Cadência</span><strong>{fmt(record.cadence_steps_min, 1)}</strong><small>passos/min</small></div>
            <div><span>Tempo médio de passo</span><strong>{fmt(record.mean_step_time_s, 3)}</strong><small>s</small></div>
            <div><span>CV do passo</span><strong>{fmt(record.step_time_cv_pct, 1)}%</strong><small>variabilidade</small></div>
            <div><span>Simetria temporal</span><strong>{fmt(record.step_time_symmetry_ratio, 3)}</strong><small>razão</small></div>
            <div><span>Eventos</span><strong>{record.event_count}</strong><small>{record.left_event_count} E · {record.right_event_count} D</small></div>
            <div><span>Qualidade</span><strong>{fmt(record.quality_score * 100, 0)}%</strong><small>{record.status}</small></div>
          </div>
          <div className="portal-table-wrap"><table className="portal-table"><thead><tr><th>Frame</th><th>Lado</th><th>Evento</th><th>Tempo</th><th>Proeminência</th></tr></thead><tbody>{record.events.slice(0, 30).map((event, index) => <tr key={`${event.frame}-${index}`}><td>{event.frame}</td><td>{event.side}</td><td>{event.kind}</td><td>{fmt(event.timestamp_s, 3)} s</td><td>{fmt(event.prominence, 3)}</td></tr>)}</tbody></table></div>
          {record.quality_flags.length > 0 && <div className="methodology-warning">Flags: {record.quality_flags.join(" · ")}</div>}
        </>
      )}
    </section>
  );
}
