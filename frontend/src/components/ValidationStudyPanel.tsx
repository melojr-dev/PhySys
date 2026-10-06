import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { JointMeasurement, MovementConfig, ValidationStudy, ValidationStudyReport } from "../types";

function fmt(value: number | null | undefined, digits = 1) {
  return value == null || Number.isNaN(value) ? "—" : value.toFixed(digits);
}

export function ValidationStudyPanel({
  movement,
  side,
  jobId,
  frame,
  measurement,
  isAdmin,
  protocolVersion,
  movementConfig,
}: {
  movement: string;
  side: string;
  jobId: string | null;
  frame: number;
  measurement: JointMeasurement | null;
  isAdmin: boolean;
  protocolVersion: string;
  movementConfig: MovementConfig | null;
}) {
  const [studies, setStudies] = useState<ValidationStudy[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [report, setReport] = useState<ValidationStudyReport | null>(null);
  const [referenceText, setReferenceText] = useState("");
  const [deviceSerial, setDeviceSerial] = useState("");
  const [trialNotes, setTrialNotes] = useState("");
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  const [newCode, setNewCode] = useState("");
  const [newTitle, setNewTitle] = useState("");
  const [newTarget, setNewTarget] = useState("20");
  const [newDevice, setNewDevice] = useState("");
  const [newResolution, setNewResolution] = useState("1.0");
  const [newMinConfidence, setNewMinConfidence] = useState("0.70");

  const compatible = useMemo(
    () => studies.filter((study) => study.movement === movement && (study.side === "ambos" || study.side === side)),
    [studies, movement, side],
  );
  const selected = compatible.find((study) => study.id === selectedId) ?? null;

  async function reloadStudies(preferredId?: string) {
    const response = await api.studies();
    setStudies(response.items);
    const options = response.items.filter((study) => study.movement === movement && (study.side === "ambos" || study.side === side));
    const next =
      (preferredId && options.some((x) => x.id === preferredId) ? preferredId : "") ||
      (selectedId && options.some((x) => x.id === selectedId) ? selectedId : "") ||
      options.find((x) => x.status === "frozen")?.id || options[0]?.id || "";
    setSelectedId(next);
  }

  useEffect(() => {
    reloadStudies().catch(() => setStatus("Falha ao carregar estudos."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [movement, side]);

  useEffect(() => {
    setReport(null);
    if (!selectedId) return;
    api.studyReport(selectedId).then(setReport).catch(() => undefined);
  }, [selectedId]);

  async function createStudy() {
    if (!newCode.trim() || !newTitle.trim()) return;
    setBusy(true);
    setStatus("Criando protocolo…");
    try {
      const target = Number(newTarget);
      const minConfidence = Number(newMinConfidence.replace(",", "."));
      const resolution = Number(newResolution.replace(",", "."));
      const created = await api.createStudy({
        code: newCode.trim(),
        title: newTitle.trim(),
        protocol_version: protocolVersion,
        movement,
        side: "ambos",
        target_participants: Number.isFinite(target) && target > 0 ? target : null,
        repetitions_per_participant: 3,
        min_landmark_confidence: Number.isFinite(minConfidence) ? minConfidence : 0.70,
        require_observed: true,
        require_valid_measurement: true,
        reference_method: "physical_goniometer",
        reference_device_model: newDevice.trim() || null,
        reference_device_resolution_deg: Number.isFinite(resolution) && resolution > 0 ? resolution : 1.0,
        camera_setup: {
          orientation: movementConfig?.recommendedView ?? "lateral",
          measurement_plane: movementConfig?.measurementPlane ?? "sagittal",
          landmarks: movementConfig?.landmarks ?? null,
          capture_instructions: movementConfig?.captureInstructions ?? [],
        },
        protocol_notes: "Protocolo pré-clínico criado pela Clinical Motion Station.",
      });
      await reloadStudies(created.id);
      setStatus("Protocolo criado em rascunho. Revise e congele antes da coleta.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Falha ao criar estudo");
    } finally {
      setBusy(false);
    }
  }

  async function freezeStudy() {
    if (!selected) return;
    setBusy(true);
    setStatus("Congelando protocolo…");
    try {
      const frozen = await api.freezeStudy(selected.id);
      await reloadStudies(frozen.id);
      setStatus(`Protocolo congelado · ${frozen.protocol_sha256?.slice(0, 12) ?? "hash indisponível"}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Falha ao congelar protocolo");
    } finally {
      setBusy(false);
    }
  }

  async function savePhysicalReference() {
    if (!selected || !jobId || measurement?.angleClinicalDeg == null) return;
    const reference = Number(referenceText.replace(",", "."));
    if (!Number.isFinite(reference) || reference < 0 || reference > 180) {
      setStatus("Informe uma referência física entre 0° e 180°.");
      return;
    }
    setBusy(true);
    setStatus("Registrando ensaio…");
    try {
      const trial = await api.addStudyTrial(selected.id, {
        job_id: jobId,
        frame,
        reference_angle_deg: reference,
        reference_device_serial: deviceSerial.trim() || null,
        notes: trialNotes,
      });
      const updated = await api.studyReport(selected.id);
      setReport(updated);
      setReferenceText("");
      setTrialNotes("");
      setStatus(trial.included
        ? `Ensaio ${trial.trial_number} incluído · erro ${trial.error_deg >= 0 ? "+" : ""}${trial.error_deg.toFixed(1)}°`
        : `Ensaio registrado, mas excluído pelo protocolo: ${trial.exclusion_reason}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Falha ao registrar ensaio");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="study-panel panel">
      <div className="section-title-row">
        <div><span className="eyebrow">Protocolo experimental</span><h2>Estudo de validação</h2></div>
        <span className="data-count">{report?.counts.participants_included ?? 0} participantes · {report?.counts.trials_included ?? 0} ensaios incluídos</span>
      </div>

      <div className="study-toolbar">
        <label className="field compact"><span>Estudo compatível</span>
          <select value={selectedId} onChange={(e) => setSelectedId(e.target.value)}>
            <option value="">Nenhum selecionado</option>
            {compatible.map((study) => <option key={study.id} value={study.id}>{study.code} · {study.status}</option>)}
          </select>
        </label>
        {selected && <div className="protocol-badge"><span>{selected.status}</span><strong>{selected.protocol_sha256?.slice(0, 12) || "não congelado"}</strong></div>}
        {selected?.status === "draft" && isAdmin && <button className="secondary-button" disabled={busy} onClick={freezeStudy}>Congelar protocolo</button>}
        {selected && <button className="export-link" onClick={() => api.downloadStudyCsv(selected.id, selected.code)}>CSV</button>}
        {selected && <button className="export-link" onClick={() => api.downloadStudyBundle(selected.id, selected.code)}>Pacote científico</button>}
      </div>

      {!selected && isAdmin && (
        <div className="study-create-grid">
          <label className="field compact"><span>Código</span><input value={newCode} onChange={(e) => setNewCode(e.target.value.toUpperCase())} placeholder="ELBOW-001" /></label>
          <label className="field compact"><span>Título</span><input value={newTitle} onChange={(e) => setNewTitle(e.target.value)} placeholder="Validação de flexão do cotovelo" /></label>
          <label className="field compact"><span>Participantes alvo</span><input inputMode="numeric" value={newTarget} onChange={(e) => setNewTarget(e.target.value)} /></label>
          <label className="field compact"><span>Goniômetro / modelo</span><input value={newDevice} onChange={(e) => setNewDevice(e.target.value)} placeholder="modelo / fabricante" /></label>
          <label className="field compact"><span>Resolução (°)</span><input inputMode="decimal" value={newResolution} onChange={(e) => setNewResolution(e.target.value)} /></label>
          <label className="field compact"><span>Visibilidade mínima</span><input inputMode="decimal" value={newMinConfidence} onChange={(e) => setNewMinConfidence(e.target.value)} /></label>
          <button className="primary-button" disabled={busy || !newCode || !newTitle || !newDevice} onClick={createStudy}>Criar protocolo</button>
        </div>
      )}

      {selected?.status === "frozen" && (
        <div className="physical-reference-grid">
          <div className="reference-context">
            <span>Frame atual</span><strong>{jobId ? frame + 1 : "—"}</strong>
            <span>PhysioAI</span><strong>{fmt(measurement?.angleClinicalDeg)}°</strong>
            <span>QC</span><strong>{measurement?.measurementValid ? "válida" : "não válida"}</strong>
          </div>
          <label className="field compact"><span>Referência física (°)</span><input inputMode="decimal" value={referenceText} onChange={(e) => setReferenceText(e.target.value)} placeholder="0–180" /></label>
          <label className="field compact"><span>Serial do instrumento</span><input value={deviceSerial} onChange={(e) => setDeviceSerial(e.target.value)} placeholder="opcional" /></label>
          <label className="field compact"><span>Observação do ensaio</span><input value={trialNotes} onChange={(e) => setTrialNotes(e.target.value)} placeholder="posição, compensação, ocorrência…" /></label>
          <button className="primary-button" disabled={busy || !jobId || measurement?.angleClinicalDeg == null} onClick={savePhysicalReference}>Registrar referência física</button>
        </div>
      )}

      {report && (
        <>
          <div className="study-metrics-grid">
            <div><span>MAE ensaios</span><strong>{fmt(report.agreement.mae_deg)}°</strong></div>
            <div><span>Bias</span><strong>{fmt(report.agreement.bias_deg)}°</strong></div>
            <div><span>ICC(A,1)</span><strong>{fmt(report.agreement.icc_a1, 3)}</strong></div>
            <div><span>MAE balanceado/participante</span><strong>{fmt(report.participant_balanced.mean_patient_mae_deg)}°</strong></div>
            <div><span>IC95% MAE cluster</span><strong>{fmt(report.cluster_bootstrap.mae_ci95_deg[0])}–{fmt(report.cluster_bootstrap.mae_ci95_deg[1])}°</strong></div>
            <div><span>Conclusão amostral</span><strong>{report.counts.completion_pct == null ? "—" : `${fmt(report.counts.completion_pct, 0)}%`}</strong></div>
          </div>
          {report.counts.trials_excluded > 0 && <p className="methodology-warning">{report.counts.trials_excluded} ensaio(s) excluído(s) pelos critérios congelados do protocolo. Motivos: {Object.entries(report.exclusion_reasons).map(([key, value]) => `${key}: ${value}`).join(" · ")}</p>}
          <p className="methodology-warning">{report.methodology.warning}</p>
        </>
      )}
      {status && <div className="review-status">{status}</div>}
    </section>
  );
}
