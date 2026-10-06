import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { MovementConfig, ValidationStudy, ValidationStudyReport } from "../types";

function fmt(value: number | null | undefined, digits = 2) { return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits); }

export function StudyManagementPanel({ isAdmin }: { isAdmin: boolean }) {
  const [studies, setStudies] = useState<ValidationStudy[]>([]);
  const [movements, setMovements] = useState<MovementConfig[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [report, setReport] = useState<ValidationStudyReport | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [form, setForm] = useState({ code: "", title: "", movement: "", side: "ambos", target: "20", reps: "3", device: "", resolution: "1.0" });
  const selected = useMemo(() => studies.find((x) => x.id === selectedId) ?? null, [studies, selectedId]);

  async function reload(preferred?: string) {
    const [studyResponse, movementResponse] = await Promise.all([api.studies(200), api.movements()]);
    setStudies(studyResponse.items); setMovements(movementResponse.items);
    if (!form.movement && movementResponse.items[0]) setForm((prev) => ({ ...prev, movement: movementResponse.items[0].name }));
    setSelectedId((current) => preferred ?? (current && studyResponse.items.some((x) => x.id === current) ? current : studyResponse.items[0]?.id ?? ""));
  }
  useEffect(() => { reload().catch(() => setMessage("Falha ao carregar estudos.")); }, []);
  useEffect(() => { if (selectedId) api.studyReport(selectedId).then(setReport).catch(() => setReport(null)); else setReport(null); }, [selectedId]);

  async function create() {
    if (!isAdmin || !form.code.trim() || !form.title.trim() || !form.movement || !form.device.trim()) return;
    setBusy(true); setMessage("");
    try {
      const movement = movements.find((x) => x.name === form.movement);
      const created = await api.createStudy({
        code: form.code.trim().toUpperCase(), title: form.title.trim(), movement: form.movement, side: form.side,
        target_participants: Number(form.target), repetitions_per_participant: Number(form.reps), min_landmark_confidence: 0.7,
        require_observed: true, require_valid_measurement: true, reference_method: "physical_goniometer",
        reference_device_model: form.device.trim(), reference_device_resolution_deg: Number(form.resolution),
        camera_setup: { recommended_view: movement?.recommendedView, measurement_plane: movement?.measurementPlane, landmarks: movement?.landmarks },
        protocol_notes: "Criado pelo portal de estudos PhysioAI.",
      });
      setShowCreate(false); setForm((prev) => ({ ...prev, code: "", title: "" })); await reload(created.id); setMessage("Estudo criado em rascunho.");
    } catch (err) { setMessage(err instanceof Error ? err.message : "Falha ao criar estudo"); }
    finally { setBusy(false); }
  }

  async function freeze() { if (!selected || !isAdmin) return; setBusy(true); try { await api.freezeStudy(selected.id); await reload(selected.id); setMessage("Protocolo congelado."); } catch (e) { setMessage(e instanceof Error ? e.message : "Falha"); } finally { setBusy(false); } }
  async function close() { if (!selected || !isAdmin || !window.confirm("Encerrar este estudo?")) return; setBusy(true); try { await api.closeStudy(selected.id); await reload(selected.id); setMessage("Estudo encerrado."); } catch (e) { setMessage(e instanceof Error ? e.message : "Falha"); } finally { setBusy(false); } }

  return <section className="portal-card">
    <div className="portal-card-heading"><div><span className="eyebrow">Protocolos</span><h2>Estudos de validação</h2></div><div className="portal-actions no-margin">{isAdmin && <button className="secondary-button" onClick={() => setShowCreate((v) => !v)}>{showCreate ? "Cancelar" : "Novo estudo"}</button>}</div></div>
    {showCreate && <div className="study-create-grid portal-create-box"><label className="field"><span>Código</span><input value={form.code} onChange={(e) => setForm({ ...form, code: e.target.value })} /></label><label className="field"><span>Título</span><input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} /></label><label className="field"><span>Movimento</span><select value={form.movement} onChange={(e) => setForm({ ...form, movement: e.target.value })}>{movements.map((m) => <option key={m.name}>{m.name}</option>)}</select></label><label className="field"><span>Lado</span><select value={form.side} onChange={(e) => setForm({ ...form, side: e.target.value })}><option value="ambos">Ambos</option><option value="direito">Direito</option><option value="esquerdo">Esquerdo</option></select></label><label className="field"><span>Participantes alvo</span><input value={form.target} onChange={(e) => setForm({ ...form, target: e.target.value })} /></label><label className="field"><span>Repetições/participante</span><input value={form.reps} onChange={(e) => setForm({ ...form, reps: e.target.value })} /></label><label className="field"><span>Instrumento de referência</span><input value={form.device} onChange={(e) => setForm({ ...form, device: e.target.value })} /></label><label className="field"><span>Resolução (°)</span><input value={form.resolution} onChange={(e) => setForm({ ...form, resolution: e.target.value })} /></label><button className="primary-button" disabled={busy} onClick={create}>Criar estudo</button></div>}
    <div className="study-management-grid">
      <div className="portal-table-wrap tall"><table className="portal-table"><thead><tr><th>Código</th><th>Movimento</th><th>Status</th><th>Alvo</th></tr></thead><tbody>{studies.map((study) => <tr key={study.id} className={selectedId === study.id ? "selected-row" : ""} onClick={() => setSelectedId(study.id)}><td><strong>{study.code}</strong><small className="cell-sub">{study.title}</small></td><td>{study.movement}<small className="cell-sub">{study.side}</small></td><td><span className={`status-pill ${study.status === "frozen" ? "success" : study.status === "draft" ? "warning" : "neutral"}`}>{study.status}</span></td><td>{study.target_participants ?? "—"}</td></tr>)}</tbody></table></div>
      <div>{!selected ? <div className="portal-empty">Selecione um estudo.</div> : <><div className="detail-list"><div><span>Protocolo</span><strong>{selected.protocol_version}</strong></div><div><span>Instrumento</span><strong>{selected.reference_device_model ?? "—"}</strong></div><div><span>Resolução</span><strong>{fmt(selected.reference_device_resolution_deg)}°</strong></div><div><span>Participantes</span><strong>{report?.counts.participants_included ?? 0}/{selected.target_participants ?? "—"}</strong></div><div><span>Ensaios incluídos</span><strong>{report?.counts.trials_included ?? 0}</strong></div><div><span>MAE</span><strong>{fmt(report?.agreement.mae_deg)}°</strong></div><div><span>ICC(A,1)</span><strong>{fmt(report?.agreement.icc_a1, 3)}</strong></div></div><div className="portal-actions"><button onClick={() => api.downloadStudyCsv(selected.id, selected.code)}>CSV</button><button onClick={() => api.downloadStudyBundle(selected.id, selected.code)}>Bundle</button>{selected.status === "draft" && isAdmin && <button className="primary-button" onClick={freeze}>Congelar</button>}{selected.status === "frozen" && isAdmin && <button className="secondary-button" onClick={close}>Encerrar</button>}</div></>}</div>
    </div>
    {message && <div className="review-status">{message}</div>}
  </section>;
}
