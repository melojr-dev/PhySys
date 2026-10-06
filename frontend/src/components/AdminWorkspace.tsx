import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { AuditEvent, MetricsSnapshot, SystemHealth } from "../types";

function fmt(value: number | null | undefined, digits = 1) { return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits); }
function dateLabel(value?: string | null) { return value ? new Date(value).toLocaleString("pt-BR") : "—"; }

export function AdminWorkspace() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [audit, setAudit] = useState<AuditEvent[]>([]);
  const [patientId, setPatientId] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function reload() {
    setBusy(true); setMessage("");
    try {
      const [h, m, a] = await Promise.all([api.systemHealth(), api.systemMetrics(), api.audit(200)]);
      setHealth(h); setMetrics(m); setAudit(a.items);
    } catch (err) { setMessage(err instanceof Error ? err.message : "Falha ao carregar administração"); }
    finally { setBusy(false); }
  }
  useEffect(() => { reload(); }, []);

  const gaugeMap = useMemo(() => Object.fromEntries((metrics?.gauges ?? []).map((x) => [x.name, x.value])), [metrics]);
  const counterMap = useMemo(() => Object.fromEntries((metrics?.counters ?? []).map((x) => [x.name, x.value])), [metrics]);

  async function deletePatient() {
    if (!patientId.trim() || confirmation !== "EXCLUIR") return;
    if (!window.confirm("Esta operação remove dados clínicos do paciente pseudonimizado. Continuar?")) return;
    setBusy(true); setMessage("");
    try {
      const result = await api.deletePatient(patientId.trim());
      setMessage(`Exclusão concluída: ${JSON.stringify(result)}`);
      setPatientId(""); setConfirmation("");
      await reload();
    } catch (err) { setMessage(err instanceof Error ? err.message : "Falha na exclusão"); }
    finally { setBusy(false); }
  }

  return <div className="portal-page">
    <div className="portal-page-header"><div><span className="eyebrow">Administração</span><h1>Operação e governança</h1><p>Readiness, métricas, auditoria e controles LGPD.</p></div><button className="secondary-button" disabled={busy} onClick={reload}>Atualizar</button></div>
    {message && <div className="review-status">{message}</div>}
    <div className="portal-dashboard-grid">
      <section className="portal-card"><div className="portal-card-heading"><div><span className="eyebrow">Runtime</span><h2>Saúde do sistema</h2></div><span className={`status-pill ${health?.status === "ok" ? "success" : "warning"}`}>{health?.status ?? "—"}</span></div><div className="detail-list"><div><span>Versão</span><strong>{health?.versao ?? "—"}</strong></div><div><span>Protocolo</span><strong>{health?.protocol_version ?? "—"}</strong></div><div><span>Analisador</span><strong>{health?.analisador_pronto ? "pronto" : "indisponível"}</strong></div><div><span>GPU</span><strong>{health?.gpu_disponivel ? "disponível" : "CPU"}</strong></div><div><span>Jobs máximos</span><strong>{health?.max_inflight_jobs ?? "—"}</strong></div><div><span>Uploads</span><strong>{health?.max_upload_mb ?? "—"} MB</strong></div><div><span>Preflight</span><strong>{health?.preflight_required ? "obrigatório" : "opcional"}</strong></div><div><span>Pendentes</span><strong>{health?.preflights_pending ?? 0}</strong></div></div></section>
      <section className="portal-card"><div className="portal-card-heading"><div><span className="eyebrow">Observabilidade</span><h2>Métricas</h2></div><span className="data-count">uptime {fmt((metrics?.uptime_seconds ?? 0) / 3600, 1)} h</span></div><div className="portal-metric-grid compact"><div><span>Jobs ativos</span><strong>{String(gaugeMap.physioai_jobs_inflight ?? 0)}</strong></div><div><span>Jobs concluídos</span><strong>{String(counterMap.physioai_jobs_completed_total ?? 0)}</strong></div><div><span>Falhas</span><strong>{String(counterMap.physioai_jobs_failed_total ?? 0)}</strong></div><div><span>Cancelamentos</span><strong>{String(counterMap.physioai_jobs_cancelled_total ?? 0)}</strong></div></div><div className="timer-list">{(metrics?.timers ?? []).slice(0, 8).map((timer) => <div key={timer.name}><span>{timer.name}</span><strong>{fmt(timer.avg_seconds * 1000)} ms</strong><small>máx {fmt(timer.max_seconds * 1000)} ms · n={timer.count}</small></div>)}</div></section>
    </div>
    <section className="portal-card"><div className="portal-card-heading"><div><span className="eyebrow">Auditoria</span><h2>Eventos recentes</h2></div><span className="data-count">{audit.length}</span></div><div className="portal-table-wrap tall"><table className="portal-table"><thead><tr><th>Data</th><th>Ator</th><th>Ação</th><th>Recurso</th><th>Status</th><th>IP</th></tr></thead><tbody>{audit.map((event) => <tr key={event.id}><td>{dateLabel(event.created_at)}</td><td>{event.actor}<small className="cell-sub">{event.role}</small></td><td className="mono">{event.action}</td><td>{event.resource_type ?? "—"}<small className="cell-sub mono">{event.resource_id?.slice(0, 18) ?? ""}</small></td><td>{event.status}</td><td className="mono">{event.remote_ip ?? "—"}</td></tr>)}</tbody></table></div></section>
    <section className="portal-card danger-zone"><div className="portal-card-heading"><div><span className="eyebrow">LGPD</span><h2>Exclusão de dados do paciente</h2></div><span className="status-pill danger">irreversível</span></div><p className="portal-copy">Informe o identificador original usado na coleta. O backend calcula o pseudônimo e remove sessões, medições, IMU, dataset, revisões e referências relacionadas.</p><div className="portal-form-grid"><label className="field"><span>ID do paciente</span><input value={patientId} onChange={(e) => setPatientId(e.target.value)} /></label><label className="field"><span>Digite EXCLUIR</span><input value={confirmation} onChange={(e) => setConfirmation(e.target.value)} /></label></div><button className="danger-button" disabled={busy || !patientId.trim() || confirmation !== "EXCLUIR"} onClick={deletePatient}>Excluir dados do paciente</button></section>
  </div>;
}
