import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { FormalReadiness, FormalRun, PublicationReadiness, ValidationStudy } from "../types";

function shortHash(value?: string | null) { return value ? value.slice(0, 12) : "—"; }
function dateLabel(value?: string | null) { return value ? new Date(value).toLocaleString("pt-BR") : "—"; }

export function FormalValidationPanel({ isAdmin }: { isAdmin: boolean }) {
  const [studies, setStudies] = useState<ValidationStudy[]>([]);
  const [studyId, setStudyId] = useState("");
  const [runs, setRuns] = useState<FormalRun[]>([]);
  const [readiness, setReadiness] = useState<FormalReadiness | null>(null);
  const [publication, setPublication] = useState<Record<string, PublicationReadiness>>({});
  const [ethicsStatus, setEthicsStatus] = useState("pending");
  const [ethicsProtocol, setEthicsProtocol] = useState("");
  const [calibrationDate, setCalibrationDate] = useState("");
  const [calibrationDue, setCalibrationDue] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  const selected = useMemo(() => studies.find((item) => item.id === studyId) ?? null, [studies, studyId]);
  const payload = {
    ethics_status: ethicsStatus,
    ethics_protocol_id: ethicsProtocol.trim() || null,
    reference_calibration_date: calibrationDate || null,
    reference_calibration_due_date: calibrationDue || null,
    notes,
  };

  async function loadStudies() {
    const response = await api.studies(200);
    setStudies(response.items);
    const preferred = response.items.find((item) => item.status === "frozen")?.id ?? response.items[0]?.id ?? "";
    setStudyId((current) => current && response.items.some((item) => item.id === current) ? current : preferred);
  }

  async function loadRuns(id: string) {
    if (!id) { setRuns([]); return; }
    const response = await api.formalRuns(id);
    setRuns(response.items);
  }

  useEffect(() => { loadStudies().catch(() => setMessage("Falha ao carregar estudos.")); }, []);
  useEffect(() => { loadRuns(studyId).catch(() => setMessage("Falha ao carregar execuções formais.")); setReadiness(null); }, [studyId]);

  async function checkReadiness() {
    if (!studyId) return;
    setBusy(true); setMessage("");
    try { setReadiness(await api.formalReadiness(studyId, payload)); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Falha no readiness"); }
    finally { setBusy(false); }
  }

  async function createRun() {
    if (!studyId || !isAdmin) return;
    setBusy(true); setMessage("");
    try {
      const result = await api.createFormalRun(studyId, payload);
      setMessage(`Execução formal iniciada: ${result.id.slice(0, 8)}`);
      await loadRuns(studyId);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Falha ao iniciar execução formal"); }
    finally { setBusy(false); }
  }

  async function closeRun(runId: string) {
    if (!isAdmin || !window.confirm("Fechar esta execução formal? O snapshot final será imutável.")) return;
    setBusy(true); setMessage("");
    try { await api.closeFormalRun(runId); await loadRuns(studyId); setMessage("Execução formal fechada."); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Falha ao fechar execução"); }
    finally { setBusy(false); }
  }

  async function checkPublication(runId: string) {
    try {
      const result = await api.publicationReadiness(runId);
      setPublication((prev) => ({ ...prev, [runId]: result }));
    } catch (error) { setMessage(error instanceof Error ? error.message : "Falha no readiness de publicação"); }
  }

  return (
    <div className="portal-stack">
      <section className="portal-card">
        <div className="portal-card-heading"><div><span className="eyebrow">Governança científica</span><h2>Execução formal de validação</h2></div><span className={`status-pill ${selected?.status === "frozen" ? "success" : "warning"}`}>{selected?.status ?? "sem estudo"}</span></div>
        <div className="portal-form-grid">
          <label className="field"><span>Estudo</span><select value={studyId} onChange={(e) => setStudyId(e.target.value)}><option value="">Selecione</option>{studies.map((study) => <option value={study.id} key={study.id}>{study.code} · {study.title} · {study.status}</option>)}</select></label>
          <label className="field"><span>Situação ética declarada</span><select value={ethicsStatus} onChange={(e) => setEthicsStatus(e.target.value)}><option value="pending">Pendente</option><option value="approved">Aprovado</option><option value="not_required">Não aplicável</option></select></label>
          <label className="field"><span>Protocolo CEP / identificador</span><input value={ethicsProtocol} onChange={(e) => setEthicsProtocol(e.target.value)} placeholder="ex.: CAAE / protocolo" /></label>
          <label className="field"><span>Calibração do goniômetro</span><input type="date" value={calibrationDate} onChange={(e) => setCalibrationDate(e.target.value)} /></label>
          <label className="field"><span>Validade da calibração</span><input type="date" value={calibrationDue} onChange={(e) => setCalibrationDue(e.target.value)} /></label>
          <label className="field portal-span-2"><span>Notas da execução</span><input value={notes} onChange={(e) => setNotes(e.target.value)} /></label>
        </div>
        {selected && <div className="protocol-strip"><span>Movimento <strong>{selected.movement}</strong></span><span>Protocolo <strong>{selected.protocol_version}</strong></span><span>Hash <strong>{shortHash(selected.protocol_sha256)}</strong></span><span>Alvo <strong>{selected.target_participants ?? "—"}</strong></span></div>}
        <div className="portal-actions"><button className="secondary-button" disabled={!studyId || busy} onClick={checkReadiness}>Verificar prontidão</button>{isAdmin && <button className="primary-button" disabled={!studyId || busy || readiness?.ready !== true} onClick={createRun}>Iniciar execução formal</button>}</div>
        {readiness && <div className="readiness-grid">{readiness.checks.map((check) => <div key={check.code} className={`readiness-check ${check.ok ? "ok" : "fail"}`}><strong>{check.ok ? "OK" : "PENDENTE"}</strong><span>{check.detail}</span></div>)}</div>}
        {message && <div className="review-status">{message}</div>}
      </section>

      <section className="portal-card">
        <div className="portal-card-heading"><div><span className="eyebrow">Coleta formal</span><h2>Execuções</h2></div><span className="data-count">{runs.length} registro(s)</span></div>
        {!runs.length ? <div className="portal-empty">Nenhuma execução formal para o estudo selecionado.</div> : (
          <div className="portal-table-wrap"><table className="portal-table"><thead><tr><th>Status</th><th>Início</th><th>Versão</th><th>Ética</th><th>Snapshot</th><th>Ações</th></tr></thead><tbody>{runs.map((run) => {
            const pub = publication[run.id];
            return <tr key={run.id}><td><span className={`status-pill ${run.status === "closed" ? "success" : "warning"}`}>{run.status}</span></td><td>{dateLabel(run.created_at)}</td><td>{run.software_version}</td><td>{run.ethics_status}</td><td className="mono">{shortHash(run.snapshot_sha256)}</td><td><div className="table-actions">{run.status === "open" && isAdmin && <button onClick={() => closeRun(run.id)}>Fechar</button>}<button onClick={() => api.downloadFormalRun(run.id)}>Bundle</button><button onClick={() => checkPublication(run.id)}>Publicação</button>{pub?.package_ready && isAdmin && <button onClick={() => api.downloadPublicationPackage(run.id)}>Revisão externa</button>}</div>{pub && <small className={pub.package_ready ? "text-ok" : "text-warn"}>{pub.package_ready ? "pacote pronto" : `${pub.checks.filter((x) => !x.ok).length} pendência(s)`}</small>}</td></tr>;
          })}</tbody></table></div>
        )}
      </section>
    </div>
  );
}
