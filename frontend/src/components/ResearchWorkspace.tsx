import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { ClinicalSessionDetail, ClinicalSessionSummary, DatasetStats } from "../types";

function fmt(value: number | null | undefined, digits = 1) { return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits); }
function dateLabel(value?: string | null) { return value ? new Date(value).toLocaleString("pt-BR") : "—"; }
function getRecentRows(stats: DatasetStats | null): Array<Record<string, unknown>> { return (stats?.ultimas_10 as Array<Record<string, unknown>> | undefined) ?? []; }

export function ResearchWorkspace() {
  const [tab, setTab] = useState<"sessions" | "dataset">("sessions");
  const [sessions, setSessions] = useState<ClinicalSessionSummary[]>([]);
  const [selectedJob, setSelectedJob] = useState("");
  const [detail, setDetail] = useState<ClinicalSessionDetail | null>(null);
  const [dataset, setDataset] = useState<DatasetStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const selected = useMemo(() => sessions.find((item) => item.job_id === selectedJob) ?? null, [sessions, selectedJob]);

  async function loadSessions() {
    setLoading(true); setError("");
    try {
      const response = await api.sessions(150);
      setSessions(response.items);
      if (!selectedJob && response.items[0]) setSelectedJob(response.items[0].job_id);
    } catch (err) { setError(err instanceof Error ? err.message : "Falha ao carregar sessões"); }
    finally { setLoading(false); }
  }

  async function loadDataset() {
    setLoading(true); setError("");
    try { setDataset(await api.datasetStats()); }
    catch (err) { setError(err instanceof Error ? err.message : "Falha ao carregar dataset"); }
    finally { setLoading(false); }
  }

  useEffect(() => { loadSessions(); }, []);
  useEffect(() => { if (selectedJob) api.session(selectedJob).then(setDetail).catch(() => setDetail(null)); }, [selectedJob]);
  useEffect(() => { if (tab === "dataset" && !dataset) loadDataset(); }, [tab]);

  const recentRows = getRecentRows(dataset);
  return (
    <div className="portal-page">
      <div className="portal-page-header"><div><span className="eyebrow">Pesquisa</span><h1>Sessões e dataset científico</h1><p>Rastreabilidade das avaliações, exportação reproduzível e visão do conjunto catalogado.</p></div><div className="segmented-control"><button className={tab === "sessions" ? "active" : ""} onClick={() => setTab("sessions")}>Sessões</button><button className={tab === "dataset" ? "active" : ""} onClick={() => setTab("dataset")}>Dataset</button></div></div>
      {error && <div className="review-status error">{error}</div>}

      {tab === "sessions" ? (
        <div className="portal-split">
          <section className="portal-card">
            <div className="portal-card-heading"><div><span className="eyebrow">Histórico</span><h2>Sessões clínicas</h2></div><button className="export-link" onClick={loadSessions} disabled={loading}>Atualizar</button></div>
            <div className="portal-table-wrap tall"><table className="portal-table"><thead><tr><th>Data</th><th>Movimento</th><th>Lado</th><th>ADM</th><th>QC</th><th>Status</th></tr></thead><tbody>{sessions.map((session) => <tr key={session.job_id} className={selectedJob === session.job_id ? "selected-row" : ""} onClick={() => setSelectedJob(session.job_id)}><td>{dateLabel(session.started_at)}</td><td>{session.movement}</td><td>{session.side}</td><td>{fmt(session.adm_deg)}°</td><td>{session.total_frames ? `${session.valid_frames ?? 0}/${session.total_frames}` : "—"}</td><td><span className={`status-pill ${session.status === "completed" ? "success" : "neutral"}`}>{session.status}</span></td></tr>)}</tbody></table></div>
          </section>

          <aside className="portal-card sticky-card">
            <div className="portal-card-heading"><div><span className="eyebrow">Proveniência</span><h2>Detalhe da sessão</h2></div></div>
            {!selected || !detail ? <div className="portal-empty">Selecione uma sessão.</div> : <>
              <div className="detail-list"><div><span>Job</span><strong className="mono">{selected.job_id.slice(0, 16)}</strong></div><div><span>Paciente pseudonimizado</span><strong className="mono">{selected.patient_pseudonym.slice(0, 16)}</strong></div><div><span>Avaliador</span><strong>{selected.evaluator_id}</strong></div><div><span>Software</span><strong>{selected.software_version}</strong></div><div><span>Protocolo</span><strong>{selected.protocol_version}</strong></div><div><span>FPS</span><strong>{fmt(selected.fps)}</strong></div><div><span>Frames válidos</span><strong>{detail.valid_measurement_count}/{detail.measurement_count}</strong></div><div><span>Revisões</span><strong>{detail.validation_review_count}</strong></div><div><span>Ensaios físicos</span><strong>{detail.validation_trial_count}</strong></div><div><span>IMU</span><strong>{detail.imu_analysis_count}</strong></div><div><span>Perfis motores</span><strong>{detail.motor_feature_analysis_count}</strong></div><div><span>Longitudinal</span><strong>{detail.longitudinal_analysis_count}</strong></div></div>
              <div className="hash-block"><span>SHA-256 medições</span><code>{detail.measurement_sha256}</code></div>
              <div className="portal-actions"><button className="primary-button" onClick={() => api.downloadSessionBundle(selected.job_id)}>Exportar bundle da sessão</button></div>
            </>}
          </aside>
        </div>
      ) : (
        <div className="portal-stack">
          <section className="portal-card">
            <div className="portal-card-heading"><div><span className="eyebrow">Dataset científico</span><h2>Resumo</h2></div><button className="export-link" onClick={loadDataset} disabled={loading}>Atualizar</button></div>
            <div className="portal-metric-grid">
              <div><span>Amostras</span><strong>{String(dataset?.total_amostras ?? dataset?.total_samples ?? 0)}</strong><small>catalogadas</small></div>
              <div><span>Pacientes</span><strong>{String(dataset?.pacientes_unicos ?? dataset?.patients ?? 0)}</strong><small>pseudonimizados</small></div>
              <div><span>Labels</span><strong>{String(dataset?.labels_distintos ?? 0)}</strong><small>distintos</small></div>
              <div><span>Storage</span><strong>{String(dataset?.storage ?? "—")}</strong><small>schema {String(dataset?.schema_version ?? "—")}</small></div>
            </div>
            {Boolean(dataset?.distribuicao_labels) && (
              <div className="label-distribution">
                {Object.entries(dataset?.distribuicao_labels ?? {}).map(([label, count]) => (
                  <span key={label}>
                    <strong>{count}</strong>
                    {label}
                  </span>
                ))}
              </div>
            )}
          </section>
          <section className="portal-card"><div className="portal-card-heading"><div><span className="eyebrow">Catalogação recente</span><h2>Últimas amostras</h2></div></div>{!recentRows.length ? <div className="portal-empty">Nenhuma amostra catalogada.</div> : <div className="portal-table-wrap"><table className="portal-table"><thead><tr><th>Data</th><th>Movimento</th><th>Lado</th><th>Label</th><th>ADM</th><th>Frames válidos</th><th>Versão</th></tr></thead><tbody>{recentRows.map((row, index) => <tr key={String(row.id ?? index)}><td>{dateLabel(String(row.timestamp ?? ""))}</td><td>{String(row.movimento ?? "—")}</td><td>{String(row.lado ?? "—")}</td><td>{String(row.label ?? "—")}</td><td>{row.adm_max == null ? "—" : `${fmt(Number(row.adm_max))}°`}</td><td>{String(row.valid_frames ?? "—")}</td><td>{String(row.software_version ?? "—")}</td></tr>)}</tbody></table></div>}</section>
        </div>
      )}
    </div>
  );
}
