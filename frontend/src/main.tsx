import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { api } from "./api";
import { ManualGoniometerOverlay, type ManualPoints } from "./components/ManualGoniometerOverlay";
import { AnatomicalOverlay } from "./components/AnatomicalOverlay";
import { MarkerlessHome } from "./components/MarkerlessHome";
import { MotionInsightChart } from "./components/MotionInsightChart";
import { VirtualBodyViewer } from "./components/VirtualBodyViewer";
import { ValidationStudyPanel } from "./components/ValidationStudyPanel";
import { WorkspaceNav, type WorkspaceKey } from "./components/WorkspaceNav";
import { WorkspaceHub } from "./components/WorkspaceHub";
import { clinicalAngleDeg, signedErrorDeg } from "./lib/geometry";
import type {
  AnalysisFrameEvent,
  AnalysisResult,
  JointMeasurement,
  ValidationReport,
  AuthUser,
  MovementConfig,
  CapturePreflightReport,
} from "./types";
import "./styles.css";

const fallbackMovementConfigs: MovementConfig[] = [
  { name: "Flexão de Cotovelo", jointCode: "elbow", referenceRomDeg: 145, validated: false, validationStatus: "preclinical", convention: "flexion_from_internal_angle", measurementPlane: "sagittal", recommendedView: "lateral", landmarks: { proximal: "ombro", vertex: "cotovelo", distal: "pulso" }, captureInstructions: ["Vista lateral com ombro, cotovelo e punho visíveis."] },
  { name: "Flexão de Joelho", jointCode: "knee", referenceRomDeg: 135, validated: false, validationStatus: "preclinical", convention: "flexion_from_internal_angle", measurementPlane: "sagittal", recommendedView: "lateral", landmarks: { proximal: "quadril", vertex: "joelho", distal: "tornozelo" }, captureInstructions: ["Vista lateral com quadril, joelho e tornozelo visíveis."] },
  { name: "Flexão de Quadril", jointCode: "hip", referenceRomDeg: 120, validated: false, validationStatus: "preclinical", convention: "flexion_from_internal_angle", measurementPlane: "sagittal", recommendedView: "lateral", landmarks: { proximal: "ombro", vertex: "quadril", distal: "joelho" }, captureInstructions: ["Vista lateral com ombro, quadril e joelho visíveis; minimizar compensação lombar."] },
  { name: "Dorsiflexão de Tornozelo", jointCode: "ankle", referenceRomDeg: 20, validated: false, validationStatus: "preclinical", convention: "ankle_dorsiflexion_from_internal_angle", measurementPlane: "sagittal", recommendedView: "lateral", landmarks: { proximal: "joelho", vertex: "tornozelo", distal: "pe" }, captureInstructions: ["Vista lateral com joelho, tornozelo e antepé visíveis; evitar rotação do pé."] },
  { name: "Flexão de Ombro", jointCode: "shoulder", referenceRomDeg: 180, validated: false, validationStatus: "preclinical", convention: "direct_internal_angle", measurementPlane: "sagittal", recommendedView: "lateral", landmarks: { proximal: "quadril", vertex: "ombro", distal: "cotovelo" }, captureInstructions: ["Vista lateral; executar elevação anterior sem compensação do tronco."], directionRule: "anterior_positive", minDirectionAlignment: -0.1 },
  { name: "Abdução de Ombro", jointCode: "shoulder", referenceRomDeg: 180, validated: false, validationStatus: "preclinical", convention: "direct_internal_angle", measurementPlane: "frontal", recommendedView: "anterior_or_posterior", landmarks: { proximal: "quadril", vertex: "ombro", distal: "cotovelo" }, captureInstructions: ["Vista anterior/posterior; executar elevação lateral sem inclinação compensatória."], directionRule: "lateral_outward", minDirectionAlignment: -0.1 },
];
const appVersion = import.meta.env.VITE_PHYSIOAI_VERSION || "dev";

function fmt(value: number | null | undefined, digits = 1) {
  return value == null || Number.isNaN(value) ? "—" : value.toFixed(digits);
}

function confidenceLabel(value: number | null | undefined) {
  if (value == null) return "Sem estimativa";
  if (value >= 0.9) return "Excelente";
  if (value >= 0.75) return "Boa";
  if (value >= 0.55) return "Moderada";
  return "Baixa";
}

function qualityClass(value: number | null | undefined) {
  if (value == null) return "neutral";
  if (value >= 0.9) return "good";
  if (value >= 0.75) return "ok";
  return "warn";
}

function MiniAngleChart({
  values,
  current,
  onSelect,
}: {
  values: number[];
  current: number;
  onSelect?: (index: number) => void;
}) {
  const width = 820;
  const height = 220;
  const pad = 22;
  const safeValues = values.length > 1 ? values : [0, 0];
  const min = Math.min(...safeValues);
  const max = Math.max(...safeValues);
  const range = Math.max(max - min, 1);
  const points = safeValues
    .map((v, i) => {
      const x = pad + (i / Math.max(safeValues.length - 1, 1)) * (width - pad * 2);
      const y = height - pad - ((v - min) / range) * (height - pad * 2);
      return `${x},${y}`;
    })
    .join(" ");
  const cursorX =
    pad +
    (Math.min(current, safeValues.length - 1) / Math.max(safeValues.length - 1, 1)) *
      (width - pad * 2);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className={`angle-chart ${onSelect ? "interactive" : ""}`}
      role="img"
      aria-label="Ângulo clínico ao longo do tempo"
      onPointerDown={(event) => {
        if (!onSelect || values.length === 0) return;
        const rect = event.currentTarget.getBoundingClientRect();
        const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
        onSelect(Math.round(fraction * Math.max(values.length - 1, 0)));
      }}
    >
      <line x1={pad} x2={width - pad} y1={height - pad} y2={height - pad} className="chart-axis" />
      <line x1={pad} x2={pad} y1={pad} y2={height - pad} className="chart-axis" />
      <polyline points={points} fill="none" className="chart-line" />
      <line x1={cursorX} x2={cursorX} y1={pad} y2={height - pad} className="chart-cursor" />
      <text x={pad + 4} y={pad + 12} className="chart-caption">{fmt(max)}°</text>
      <text x={pad + 4} y={height - pad - 6} className="chart-caption">{fmt(min)}°</text>
    </svg>
  );
}

function BlandAltmanChart({
  points,
  bias,
  lower,
  upper,
}: {
  points: Array<{ mean_deg: number; difference_deg: number }>;
  bias: number | null | undefined;
  lower: number | null | undefined;
  upper: number | null | undefined;
}) {
  const width = 820;
  const height = 220;
  const pad = 28;
  if (!points.length) return <div className="ba-empty">Sem pares com referência física.</div>;
  const xs = points.map((p) => p.mean_deg);
  const ys = points.map((p) => p.difference_deg);
  const guideY = [bias, lower, upper].filter((v): v is number => v != null && Number.isFinite(v));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys, ...guideY);
  const maxY = Math.max(...ys, ...guideY);
  const rangeX = Math.max(maxX - minX, 1);
  const rangeY = Math.max(maxY - minY, 1);
  const px = (value: number) => pad + ((value - minX) / rangeX) * (width - pad * 2);
  const py = (value: number) => height - pad - ((value - minY) / rangeY) * (height - pad * 2);
  const lines = [
    { value: bias, label: "bias" },
    { value: lower, label: "LoA−" },
    { value: upper, label: "LoA+" },
  ];
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="ba-chart" role="img" aria-label="Bland Altman PhysioAI versus referência física">
      <line x1={pad} x2={width - pad} y1={height - pad} y2={height - pad} className="chart-axis" />
      <line x1={pad} x2={pad} y1={pad} y2={height - pad} className="chart-axis" />
      {lines.map((line) => line.value == null ? null : (
        <g key={line.label}>
          <line x1={pad} x2={width - pad} y1={py(line.value)} y2={py(line.value)} className={`ba-guide ${line.label === "bias" ? "bias" : "loa"}`} />
          <text x={width - pad - 4} y={py(line.value) - 4} textAnchor="end" className="chart-caption">{line.label} {fmt(line.value)}°</text>
        </g>
      ))}
      {points.map((point, index) => <circle key={index} cx={px(point.mean_deg)} cy={py(point.difference_deg)} r="4" className="ba-point" />)}
      <text x={pad + 4} y={pad + 12} className="chart-caption">Auto − referência</text>
      <text x={width - pad} y={height - 7} textAnchor="end" className="chart-caption">média dos métodos (°)</text>
    </svg>
  );
}

function automaticPoints(measurement: JointMeasurement | null): ManualPoints | null {
  if (!measurement?.proximal || !measurement.vertex || !measurement.distal) return null;
  return {
    proximal: { ...measurement.proximal },
    vertex: { ...measurement.vertex },
    distal: { ...measurement.distal },
  };
}


function LoginScreen({ onAuthenticated }: { onAuthenticated: (user: AuthUser) => void }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      const session = await api.login(username, password);
      onAuthenticated(session.user);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha na autenticação");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-shell">
      <form className="login-card" onSubmit={submit}>
        <div className="brand-mark large">P</div>
        <span className="eyebrow">PhysioAI · Markerless Core</span>
        <h1>Human Motion Analysis</h1>
        <p>Análise de movimento por vídeo comum, sem marcadores corporais. Acesso restrito à equipe autorizada; dados de pesquisa são pseudonimizados no backend.</p>
        <label className="field"><span>Usuário</span><input autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} /></label>
        <label className="field"><span>Senha</span><input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} /></label>
        {error && <div className="login-error">{error}</div>}
        <button className="primary-button" disabled={busy || !username || !password}>{busy ? "Entrando…" : "Entrar"}</button>
      </form>
    </div>
  );
}

function App() {
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [activeWorkspace, setActiveWorkspace] = useState<WorkspaceKey>("home");
  const [backendOnline, setBackendOnline] = useState(false);
  const [patientId, setPatientId] = useState("PAC-001");
  const [movementConfigs, setMovementConfigs] = useState<MovementConfig[]>(fallbackMovementConfigs);
  const [protocolVersion, setProtocolVersion] = useState("1.4");
  const [movement, setMovement] = useState("Flexão de Cotovelo");
  const [side, setSide] = useState("direito");
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoSize, setVideoSize] = useState({ width: 1920, height: 1080 });
  const [preflight, setPreflight] = useState<CapturePreflightReport | null>(null);
  const [preflightBusy, setPreflightBusy] = useState(false);
  const [preflightError, setPreflightError] = useState("");
  const [preflightOverrideReason, setPreflightOverrideReason] = useState("");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState("Pronto para nova avaliação");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [analysisJobId, setAnalysisJobId] = useState<string | null>(null);
  const [imuAnalysisId, setImuAnalysisId] = useState<string | null>(null);
  const [liveFrames, setLiveFrames] = useState<Array<AnalysisFrameEvent | undefined>>([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [engineeringOpen, setEngineeringOpen] = useState(false);
  const [anatomicalOverlayEnabled, setAnatomicalOverlayEnabled] = useState(true);
  const [presentationMode, setPresentationMode] = useState(true);
  const [visualMode, setVisualMode] = useState<"video" | "split" | "3d">("video");

  const [reviewMode, setReviewMode] = useState(false);
  const [manualPoints, setManualPoints] = useState<ManualPoints | null>(null);
  const [reviewNotes, setReviewNotes] = useState("");
  const [reviewStatus, setReviewStatus] = useState("");
  const [validationReport, setValidationReport] = useState<ValidationReport | null>(null);
  const [savingReview, setSavingReview] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);
  const jobIdRef = useRef<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  // Live analysis is slower than video decoding. Keep the overlay frozen on the
  // last frame actually presented by the browser and only advance it after the
  // corresponding seek completes. This avoids drawing frame N landmarks over
  // an older video frame while analysis is running.
  const livePreviewPendingRef = useRef<AnalysisFrameEvent | null>(null);
  const livePreviewActiveRef = useRef<AnalysisFrameEvent | null>(null);
  const livePreviewSeekingRef = useRef(false);

  useEffect(() => {
    if (!api.getToken()) {
      setAuthReady(true);
      return;
    }
    api.me()
      .then(setAuthUser)
      .catch(() => api.logout())
      .finally(() => setAuthReady(true));
  }, []);

  useEffect(() => {
    if (!authUser) return;
    const health = () => api.systemHealth().then(() => setBackendOnline(true)).catch(() => setBackendOnline(false));
    health();
    const interval = window.setInterval(health, 15000);
    return () => window.clearInterval(interval);
  }, [authUser]);

  useEffect(() => {
    if (!authUser) return;
    api.movements()
      .then((payload) => {
        if (payload.protocol_version) setProtocolVersion(payload.protocol_version);
        if (payload.items?.length) {
          setMovementConfigs(payload.items);
          if (!payload.items.some((item) => item.name === movement)) setMovement(payload.items[0].name);
        }
      })
      .catch(() => undefined);
  }, [authUser]);

  useEffect(() => {
    if (!authUser) return;
    api.validationReport(movement, side).then(setValidationReport).catch(() => undefined);
  }, [movement, side, authUser]);

  useEffect(() => {
    if (!videoFile) {
      setVideoUrl(null);
      return;
    }
    const url = URL.createObjectURL(videoFile);
    setVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  useEffect(() => {
    if (preflight?.preflight_id) api.discardPreflight(preflight.preflight_id).catch(() => undefined);
    setPreflight(null);
    setPreflightError("");
    setPreflightOverrideReason("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoFile, movement, side, patientId]);

  useEffect(() => {
    if (visualMode === "3d" || !videoRef.current) return;
    const event = result?.frame_measurements?.[frameIndex];
    if (event) videoRef.current.currentTime = event.timestampMs / 1000;
  }, [visualMode]);

  const resultFrames = result?.frame_measurements ?? [];
  const selectedEvent = resultFrames[frameIndex] ?? liveFrames[frameIndex] ?? null;
  const selectedMeasurement = selectedEvent?.measurement ?? null;
  const selectedSystem = selectedEvent?.system;

  const liveAngles = useMemo(
    () => liveFrames.filter(Boolean).map((event) => event!.measurement.angleClinicalDeg ?? 0),
    [liveFrames],
  );
  const angles = result?.angulos_player?.length ? result.angulos_player : liveAngles;
  const currentAngle = selectedMeasurement?.angleClinicalDeg ?? result?.angulo_clinico_medio ?? null;
  const confidence = selectedMeasurement?.confidence ?? null;
  const velocity = selectedMeasurement?.angularVelocityDegS ?? result?.vel_media ?? null;
  const adm = selectedMeasurement?.admDeg ?? result?.adm_observada ?? result?.adm_max ?? null;
  const movementConfig = movementConfigs.find((item) => item.name === movement) ?? movementConfigs[0] ?? null;

  const manualAngle = useMemo(() => {
    if (!manualPoints) return null;
    return clinicalAngleDeg(
      movementConfig?.convention,
      manualPoints.proximal,
      manualPoints.vertex,
      manualPoints.distal,
    );
  }, [manualPoints, movementConfig]);

  const autoVsManual = signedErrorDeg(selectedMeasurement?.angleClinicalDeg, manualAngle);

  function resetReview() {
    setReviewMode(false);
    setManualPoints(null);
    setReviewNotes("");
    setReviewStatus("");
  }

  async function runPreflight() {
    if (!videoFile || preflightBusy || running) return;
    setPreflightBusy(true);
    setPreflightError("");
    setStatusText("Verificando qualidade da captura");
    try {
      const fd = new FormData();
      fd.append("id_paciente", patientId.trim() || "PAC-001");
      fd.append("movimento", movement);
      fd.append("lado", side);
      fd.append("video_a", videoFile);
      const report = await api.capturePreflight(fd);
      setPreflight(report);
      if (videoRef.current && report.best_frame_timestamp_ms != null) {
        videoRef.current.currentTime = report.best_frame_timestamp_ms / 1000;
      }
      setStatusText(report.status === "pass" ? "Captura aprovada" : report.status === "warn" ? "Captura com ressalvas" : "Captura reprovada");
    } catch (error) {
      setPreflightError(error instanceof Error ? error.message : "Falha no preflight");
      setStatusText("Falha ao verificar captura");
    } finally {
      setPreflightBusy(false);
    }
  }

  async function recoverAnalysisWithoutSocket(jobId: string) {
    if (jobIdRef.current !== jobId) return;
    setStatusText("Canal em tempo real indisponível · acompanhando pelo servidor");
    for (let attempt = 0; attempt < 300; attempt += 1) {
      if (jobIdRef.current !== jobId) return;
      try {
        const data = await api.tryGetResult(jobId);
        if (data) {
          setResult(data);
          setAnalysisJobId(jobId);
          setProgress(100);
          setFrameIndex(0);
          if (videoRef.current) {
            videoRef.current.pause();
            const firstTimestampMs = data.frame_measurements?.[0]?.timestampMs ?? 0;
            videoRef.current.currentTime = firstTimestampMs / 1000;
          }
          setStatusText("Análise concluída · modo de recuperação HTTP");
          setRunning(false);
          jobIdRef.current = null;
          return;
        }
      } catch (error) {
        setStatusText(error instanceof Error ? error.message : "Falha ao recuperar resultado");
        setRunning(false);
        jobIdRef.current = null;
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 2000));
    }
    if (jobIdRef.current === jobId) {
      setStatusText("O processamento excedeu o tempo de acompanhamento. Consulte os logs do servidor.");
      setRunning(false);
      jobIdRef.current = null;
    }
  }

  async function startAnalysis() {
    if (!videoFile || !preflight || running) return;
    if (!preflight.can_analyze && preflightOverrideReason.trim().length < 12) {
      setPreflightError("Captura reprovada: informe justificativa de override com pelo menos 12 caracteres.");
      return;
    }

    const jobId = crypto.randomUUID();
    jobIdRef.current = jobId;
    setAnalysisJobId(jobId);
    setImuAnalysisId(null);
    const fd = new FormData();
    fd.append("id_paciente", patientId.trim() || "PAC-001");
    fd.append("movimento", movement);
    fd.append("lado", side);
    fd.append("offset_ms", "0");
    fd.append("tipo_sensor", "Nenhum");
    fd.append("job_id", jobId);
    fd.append("preflight_id", preflight.preflight_id);
    if (!preflight.can_analyze) fd.append("preflight_override_reason", preflightOverrideReason.trim());
    fd.append("retornar_frames", "false");
    fd.append("gerar_graficos", "false");
    fd.append("retornar_skeleton", "true");

    setRunning(true);
    setProgress(0);
    setResult(null);
    setLiveFrames([]);
    setFrameIndex(0);
    resetLivePreviewSync();
    resetReview();
    setStatusText("Enviando avaliação");
    socketRef.current?.close();

    try {
      await api.startAnalysis(fd);
      setPreflight(null);
      setPreflightOverrideReason("");

      const socket = api.openSocket(jobId, async (event) => {
        if (event.type === "progress") {
          setProgress(Math.round(event.pct * 100));
          setStatusText("Processando vídeo");
          return;
        }

        if (event.type === "analysis.frame") {
          setLiveFrames((previous) => {
            const next = [...previous];
            next[event.frame] = event;
            return next;
          });
          setStatusText(event.measurement.source === "observed" ? "Telemetria ao vivo" : "Landmark predito");
          queueLivePreview(event);
          return;
        }

        if (event.type === "done") {
          setStatusText("Consolidando resultados");
          try {
            const data = await api.getResult(event.job_id);
            setResult(data);
            setAnalysisJobId(event.job_id);
            setProgress(100);
            setFrameIndex(0);
            if (videoRef.current) {
              videoRef.current.pause();
              const firstTimestampMs = data.frame_measurements?.[0]?.timestampMs ?? 0;
              videoRef.current.currentTime = firstTimestampMs / 1000;
            }
            setStatusText("Análise concluída");
          } catch (error) {
            setStatusText(error instanceof Error ? error.message : "Falha ao obter resultado");
          } finally {
            resetLivePreviewSync();
            setRunning(false);
            jobIdRef.current = null;
            socket.close();
          }
          return;
        }

        if (event.type === "cancelled") {
          resetLivePreviewSync();
          setRunning(false);
          setStatusText("Análise cancelada");
          jobIdRef.current = null;
          socket.close();
          return;
        }

        if (event.type === "error") {
          setStatusText(event.detail);
          resetLivePreviewSync();
          setRunning(false);
          jobIdRef.current = null;
          socket.close();
        }
      }, () => {
        if (jobIdRef.current === jobId) void recoverAnalysisWithoutSocket(jobId);
      });
      socketRef.current = socket;
      setStatusText("Análise em execução");
    } catch (error) {
      resetLivePreviewSync();
      setRunning(false);
      jobIdRef.current = null;
      setStatusText(error instanceof Error ? error.message : "Falha ao iniciar análise");
    }
  }

  async function cancelAnalysis() {
    const jobId = jobIdRef.current;
    if (!jobId) return;
    setStatusText("Cancelando análise…");
    try {
      await api.cancelAnalysis(jobId);
    } catch (error) {
      setStatusText(error instanceof Error ? error.message : "Falha ao cancelar análise");
    }
  }

  function resetLivePreviewSync() {
    livePreviewPendingRef.current = null;
    livePreviewActiveRef.current = null;
    livePreviewSeekingRef.current = false;
  }

  function pumpLivePreview() {
    if (livePreviewSeekingRef.current) return;
    const next = livePreviewPendingRef.current;
    if (!next) return;

    livePreviewPendingRef.current = null;
    const video = videoRef.current;
    if (!video) {
      setFrameIndex(next.frame);
      return;
    }

    livePreviewActiveRef.current = next;
    livePreviewSeekingRef.current = true;
    video.pause();
    const targetTime = next.timestampMs / 1000;

    // Avoid waiting for a seeked event when the browser is already on this frame.
    if (Math.abs(video.currentTime - targetTime) <= 0.008) {
      setFrameIndex(next.frame);
      livePreviewActiveRef.current = null;
      livePreviewSeekingRef.current = false;
      window.queueMicrotask(pumpLivePreview);
      return;
    }

    video.currentTime = targetTime;
  }

  function queueLivePreview(event: AnalysisFrameEvent) {
    // Latest-wins queue: do not accumulate seeks when backend processing emits
    // telemetry faster than the browser can decode/paint requested frames.
    livePreviewPendingRef.current = event;
    pumpLivePreview();
  }

  function handleVideoSeeked(video: HTMLVideoElement) {
    const active = livePreviewActiveRef.current;
    if (running && active) {
      // The displayed video frame is now the one represented by `active`; only
      // now may the anatomical overlay advance to the same measurement.
      setFrameIndex(active.frame);
      livePreviewActiveRef.current = null;
      livePreviewSeekingRef.current = false;
      window.queueMicrotask(pumpLivePreview);
      return;
    }
    syncFrameFromVideo(video.currentTime);
  }

  function syncFrameFromVideo(currentTime: number) {
    if (running || reviewMode || !resultFrames.length) return;
    const events = resultFrames;
    const targetMs = currentTime * 1000;
    let bestIndex = 0;
    let bestDistance = Math.abs(events[0].timestampMs - targetMs);
    for (let i = 1; i < events.length; i += 1) {
      const distance = Math.abs(events[i].timestampMs - targetMs);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = i;
      } else if (events[i].timestampMs > targetMs && distance > bestDistance) {
        break;
      }
    }
    setFrameIndex(bestIndex);
  }

  function seekFrame(index: number) {
    const maxIndex = Math.max((resultFrames.length || angles.length) - 1, 0);
    const bounded = Math.max(0, Math.min(index, maxIndex));
    setFrameIndex(bounded);
    resetReview();
    const event = resultFrames[bounded] ?? liveFrames[bounded];
    if (videoRef.current) {
      if (event) videoRef.current.currentTime = event.timestampMs / 1000;
      else if (result?.fps) videoRef.current.currentTime = bounded / result.fps;
      videoRef.current.pause();
    }
  }

  function beginReview() {
    const points = automaticPoints(selectedMeasurement);
    if (!points || selectedMeasurement?.angleClinicalDeg == null) return;
    videoRef.current?.pause();
    setManualPoints(points);
    setReviewMode(true);
    setReviewStatus(`Arraste os três pontos. PROX = ${movementConfig?.landmarks.proximal ?? "proximal"}, EIXO = ${movementConfig?.landmarks.vertex ?? "articulação"}, DIST = ${movementConfig?.landmarks.distal ?? "distal"}.`);
  }

  function restoreAutomaticPoints() {
    const points = automaticPoints(selectedMeasurement);
    if (points) setManualPoints(points);
  }

  async function saveReview() {
    if (!analysisJobId || !selectedEvent || selectedMeasurement?.angleClinicalDeg == null || !manualPoints || manualAngle == null) return;
    setSavingReview(true);
    setReviewStatus("Salvando revisão…");
    try {
      await api.saveValidation({
        job_id: analysisJobId,
        frame: selectedEvent.frame,
        proximal: manualPoints.proximal,
        vertex: manualPoints.vertex,
        distal: manualPoints.distal,
        notes: reviewNotes,
      });
      const report = await api.validationReport(movement, side);
      setValidationReport(report);
      setReviewStatus("Revisão registrada.");
    } catch (error) {
      setReviewStatus(error instanceof Error ? error.message : "Falha ao salvar revisão");
    } finally {
      setSavingReview(false);
    }
  }

  const totalSelectable = resultFrames.length || angles.length;

  if (!authReady) return <div className="login-shell"><div className="login-card">Carregando sessão…</div></div>;
  if (!authUser) return <LoginScreen onAuthenticated={(user) => { setAuthUser(user); setBackendOnline(true); }} />;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark">P</div>
          <div>
            <div className="brand-name">PhysioAI</div>
            <div className="brand-subtitle">Markerless Human Motion Analysis · {appVersion}</div>
          </div>
        </div>
        <div className="session-status">
          <span className={`status-dot ${backendOnline ? "online" : "offline"}`} />
          <div>
            <strong>{backendOnline ? "Backend conectado" : "Backend indisponível"}</strong>
            <span>{statusText}</span>
          </div>
        </div>
        <div className="topbar-mode">
          <span className="markerless-core-badge"><i /> MARKERLESS</span>
          <button className={`mode-switch ${presentationMode ? "active" : ""}`} onClick={() => setPresentationMode((value) => !value)} title="Alternar entre apresentação simplificada e visão de pesquisa">
            {presentationMode ? "Apresentação" : "Pesquisa"}
          </button>
        </div>
        <div className="user-session">
          <div><strong>{authUser.username}</strong><span>{authUser.role}</span></div>
          <button className="secondary-button compact-button" onClick={() => { api.logout(); setAuthUser(null); }}>Sair</button>
        </div>
      </header>

      <WorkspaceNav active={activeWorkspace} onChange={setActiveWorkspace} isAdmin={authUser.role === "admin"} />

      {activeWorkspace === "home" ? (
        <MarkerlessHome
          result={result}
          onStart={() => setActiveWorkspace("assessment")}
          onOpenResearch={() => setActiveWorkspace("research")}
        />
      ) : activeWorkspace === "assessment" ? (
      <main className={`workspace ${presentationMode ? "presentation-mode" : "research-mode"}`}>
        <aside className="session-panel panel">
          <div className="panel-heading">
            <div><span className="eyebrow">Sessão</span><h2>Avaliação</h2></div>
            <span className="step-chip">01</span>
          </div>

          <label className="field"><span>Paciente / sessão</span><input value={patientId} onChange={(e) => setPatientId(e.target.value)} /></label>
          <label className="field"><span>Avaliador autenticado</span><input value={authUser.username} readOnly /></label>
          <label className="field"><span>Movimento habilitado nesta fase</span><select value={movement} onChange={(e) => setMovement(e.target.value)}>{movementConfigs.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}</select></label>
          <label className="field"><span>Lado</span><select value={side} onChange={(e) => setSide(e.target.value)}><option value="direito">Direito</option><option value="esquerdo">Esquerdo</option></select></label>

          {movementConfig && (
            <div className="capture-protocol" role="note">
              <div className="capture-protocol-title">Protocolo de captura</div>
              <div className="capture-protocol-meta">Referência: {movementConfig.referenceRomDeg}° · {movementConfig.recommendedView} · {movementConfig.measurementPlane}</div>
              <ul>{movementConfig.captureInstructions.map((item) => <li key={item}>{item}</li>)}</ul>
            </div>
          )}

          <label className="file-field">
            <span className="file-title">Vídeo da avaliação</span>
            <span className="file-hint">{movementConfig ? `${movementConfig.recommendedView} · plano ${movementConfig.measurementPlane}` : "Câmera estável e articulação visível."}</span>
            <input type="file" accept="video/*" onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)} />
            <span className="file-button">{videoFile ? videoFile.name : "Selecionar vídeo"}</span>
          </label>

          <div className={`preflight-card ${preflight ? `status-${preflight.status}` : ""}`}>
            <div className="preflight-heading">
              <div><span className="eyebrow">Pré-checagem</span><strong>Qualidade da captura</strong></div>
              {preflight && <span className="preflight-score">{preflight.score.toFixed(0)}/100</span>}
            </div>
            {!preflight ? (
              <>
                <p>Valida enquadramento, landmarks, plano anatômico, FPS, iluminação e orientação antes do processamento completo.</p>
                <button className="secondary-button" disabled={!videoFile || preflightBusy || running || !backendOnline} onClick={runPreflight}>
                  {preflightBusy ? "Verificando…" : "Verificar captura"}
                </button>
              </>
            ) : (
              <>
                <div className={`preflight-summary ${preflight.status}`}>
                  <strong>{preflight.status === "pass" ? "Captura aprovada" : preflight.status === "warn" ? "Captura com ressalvas" : "Captura reprovada"}</strong>
                  <span>{preflight.detected_frames}/{preflight.sampled_frames} frames amostrados com pose detectada</span>
                </div>
                <div className="preflight-checks">
                  {preflight.checks.map((check) => (
                    <div className={`preflight-check ${check.status}`} key={check.code}>
                      <span className="preflight-check-dot" />
                      <div><strong>{check.label}</strong><span>{check.message}</span></div>
                    </div>
                  ))}
                </div>
                {!preflight.can_analyze && (
                  <label className="field override-field">
                    <span>Justificativa de override experimental</span>
                    <textarea value={preflightOverrideReason} onChange={(e) => setPreflightOverrideReason(e.target.value)} placeholder="Descreva por que esta captura deve ser analisada apesar da reprovação…" rows={3} />
                  </label>
                )}
                <button className="ghost-button" onClick={() => { api.discardPreflight(preflight.preflight_id).catch(() => undefined); setPreflight(null); setPreflightOverrideReason(""); }}>Refazer verificação</button>
              </>
            )}
            {preflightError && <div className="preflight-error">{preflightError}</div>}
          </div>

          <div className="session-actions">
            <button className="primary-button" disabled={!videoFile || !preflight || running || !backendOnline || (!preflight.can_analyze && preflightOverrideReason.trim().length < 12)} onClick={startAnalysis}>{running ? "Analisando…" : "Iniciar análise"}</button>
            {running && <button className="secondary-button" onClick={cancelAnalysis}>Cancelar análise</button>}
          </div>

          <div className="progress-block">
            <div className="progress-meta"><span>Processamento</span><strong>{progress}%</strong></div>
            <div className="progress-track"><div className="progress-fill" style={{ width: `${progress}%` }} /></div>
          </div>
        </aside>

        <section className="clinical-stage panel">
          <div className="stage-toolbar">
            <div>
              <div className="stage-title-line"><span className="eyebrow">Análise markerless</span><span className="markerless-mini-badge">SEM MARCADORES</span></div>
              <h1>{movement} · {side === "direito" ? "Direito" : "Esquerdo"}</h1>
            </div>
            <div className="stage-actions">
              <div className="view-segmented" aria-label="Modo de visualização">
                <button className={visualMode === "video" ? "active" : ""} onClick={() => setVisualMode("video")}>Vídeo</button>
                <button className={visualMode === "split" ? "active" : ""} onClick={() => setVisualMode("split")}>2D + 3D</button>
                <button className={visualMode === "3d" ? "active" : ""} onClick={() => setVisualMode("3d")}>3D</button>
              </div>
              {selectedMeasurement && (
                <button
                  className={`review-button overlay-toggle ${anatomicalOverlayEnabled ? "active" : ""}`}
                  onClick={() => setAnatomicalOverlayEnabled((value) => !value)}
                  title="Mostrar ou ocultar a cadeia anatômica do membro selecionado"
                >
                  Membro {anatomicalOverlayEnabled ? "ON" : "OFF"}
                </button>
              )}
              {selectedMeasurement && !running && !presentationMode && (
                reviewMode
                  ? <button className="review-button active" onClick={resetReview}>Sair da revisão</button>
                  : <button className="review-button" onClick={beginReview}>Revisar frame</button>
              )}
              <div className={`quality-pill ${qualityClass(confidence)}`}><span />{selectedMeasurement?.quality === "predicted" ? "Predito" : confidenceLabel(confidence)}</div>
            </div>
          </div>

          <div className={`visual-stage-grid mode-${visualMode}`}>
            {visualMode !== "3d" && <div className="video-stage">
              {videoUrl ? (
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls={!reviewMode}
                  playsInline
                  className="clinical-video"
                  onLoadedMetadata={(event) => {
                    const video = event.currentTarget;
                    setVideoSize({ width: video.videoWidth || 1920, height: video.videoHeight || 1080 });
                  }}
                  onTimeUpdate={(event) => syncFrameFromVideo(event.currentTarget.currentTime)}
                  onSeeked={(event) => handleVideoSeeked(event.currentTarget)}
                />
              ) : (
                <div className="empty-video"><div className="empty-icon">+</div><strong>Selecione um vídeo</strong><span>A avaliação aparecerá aqui</span></div>
              )}
              <AnatomicalOverlay
                measurement={selectedMeasurement}
                movement={movement}
                side={side}
                width={videoSize.width}
                height={videoSize.height}
                enabled={anatomicalOverlayEnabled}
              />
              <ManualGoniometerOverlay
                measurement={selectedMeasurement}
                manual={manualPoints}
                onChange={setManualPoints}
                width={videoSize.width}
                height={videoSize.height}
                enabled={reviewMode}
                convention={movementConfig?.convention}
              />
              <div className="video-hud">
                <span>FRAME {String((selectedEvent?.frame ?? frameIndex) + 1).padStart(4, "0")}</span>
                <span>{result?.fps ? `${fmt(result.fps)} FPS` : selectedSystem?.processingFps ? `${fmt(selectedSystem.processingFps)} FPS PROC` : "FPS —"}</span>
                <span>{reviewMode ? "REVISÃO MANUAL" : "MARKERLESS · " + (selectedMeasurement?.source?.toUpperCase() ?? "AGUARDANDO")}</span>
              </div>
            </div>}
            {visualMode !== "video" && <div className="three-d-stage">
              <VirtualBodyViewer
                skeleton={result?.skeleton_3d?.[Math.min(frameIndex, Math.max((result?.skeleton_3d?.length ?? 1) - 1, 0))] ?? null}
                movement={movement}
                side={side}
                angle={currentAngle}
              />
            </div>}
          </div>

          <div className="transport-row">
            <button onClick={() => seekFrame(frameIndex - 1)} disabled={!totalSelectable}>−1</button>
            <input type="range" min="0" max={Math.max(totalSelectable - 1, 0)} value={Math.min(frameIndex, Math.max(totalSelectable - 1, 0))} onChange={(e) => seekFrame(Number(e.target.value))} disabled={!totalSelectable || reviewMode} />
            <button onClick={() => seekFrame(frameIndex + 1)} disabled={!totalSelectable}>+1</button>
            <span>{totalSelectable ? `${frameIndex + 1} / ${totalSelectable}` : "0 / 0"}</span>
          </div>
        </section>

        <aside className="measurement-panel panel">
          <div className="panel-heading"><div><span className="eyebrow">Goniometria digital</span><h2>{reviewMode ? "Revisão" : "Medição"}</h2></div><span className="step-chip">02</span></div>
          <div className="angle-primary">
            <span>{reviewMode ? "Automático / Manual" : "Ângulo clínico"}</span>
            {reviewMode ? (
              <div className="compare-angle">
                <strong>{fmt(currentAngle)}<small>° auto</small></strong>
                <strong>{fmt(manualAngle)}<small>° manual</small></strong>
              </div>
            ) : (
              <strong>{fmt(currentAngle)}<small>°</small></strong>
            )}
            <em>{selectedMeasurement ? "Medição auditável por frame" : "Aguardando análise"}</em>
          </div>

          {!reviewMode ? (
            presentationMode ? (
              <>
                <div className="simple-result-grid">
                  <div className="simple-result-card primary"><span>Amplitude observada</span><strong>{fmt(adm)}°</strong><small>maior excursão válida registrada</small></div>
                  <div className="simple-result-card"><span>Velocidade neste instante</span><strong>{fmt(velocity)}<small> °/s</small></strong></div>
                  <div className="simple-result-card"><span>Qualidade do rastreamento</span><strong>{confidenceLabel(confidence)}</strong><small>{confidence == null ? "sem estimativa" : `${fmt(confidence * 100, 0)}% de visibilidade mínima`}</small></div>
                  <div className="simple-result-card"><span>Repetições detectadas</span><strong>{result?.total_reps ?? "—"}</strong><small>{result?.duracao_s ? `${fmt(result.duracao_s, 1)} s de vídeo` : "aguardando resultado"}</small></div>
                </div>
                <div className="measurement-note presentation-note">
                  <strong>Leitura rápida</strong>
                  <span>O marcador acompanha o membro selecionado sem adesivos ou sensores corporais. Alterne para “Pesquisa” no topo para abrir métricas técnicas e revisão científica.</span>
                </div>
              </>
            ) : (
              <>
                <div className="metric-stack">
                  <div className="metric-row"><span>ADM observada</span><strong>{fmt(adm)}°</strong></div>
                  <div className="metric-row"><span>Velocidade angular</span><strong>{fmt(velocity)} °/s</strong></div>
                  <div className="metric-row"><span>Visibilidade mínima</span><strong>{confidence == null ? "—" : `${fmt(confidence * 100, 0)}%`}</strong></div>
                  <div className="metric-row"><span>Medição válida</span><strong>{selectedMeasurement ? (selectedMeasurement.measurementValid ? "Sim" : "Não") : "—"}</strong></div>
                  <div className="metric-row"><span>Desvio da câmera</span><strong>{fmt(selectedMeasurement?.viewDeviationDeg)}°</strong></div>
                  <div className="metric-row"><span>Componente fora do plano</span><strong>{fmt(selectedMeasurement?.outOfPlaneDeg)}°</strong></div>
                  {movementConfig?.directionRule && <div className="metric-row"><span>Sentido do movimento</span><strong>{selectedMeasurement?.directionAlignment == null ? "—" : (selectedMeasurement.directionAlignment >= (movementConfig.minDirectionAlignment ?? -0.1) ? "Compatível" : "Incompatível")}</strong></div>}
                  <div className="metric-row"><span>Repetições</span><strong>{result?.total_reps ?? "—"}</strong></div>
                  <div className="metric-row"><span>Fonte</span><strong>{selectedMeasurement?.source ?? "—"}</strong></div>
                </div>
                <div className="measurement-note">
                  <strong>Convenção clínica</strong>
                  <span>Plano {movementConfig?.measurementPlane ?? "—"} · projeção anatômica {selectedMeasurement?.planeProjectionApplied ? "ativa" : "indisponível"}. ADM usa apenas frames válidos pelo protocolo.</span>
                </div>
              </>
            )
          ) : (
            <div className="review-form">
              <div className="error-cards">
                <div><span>Auto − manual imagem</span><strong className={Math.abs(autoVsManual ?? 0) <= 5 ? "within" : "outside"}>{autoVsManual == null ? "—" : `${autoVsManual >= 0 ? "+" : ""}${fmt(autoVsManual)}°`}</strong></div>
              </div>

              <label className="field compact"><span>Observação</span><input value={reviewNotes} onChange={(e) => setReviewNotes(e.target.value)} placeholder="posição, compensação, dificuldade…" /></label>
              <div className="review-actions">
                <button className="secondary-button" onClick={restoreAutomaticPoints}>Restaurar pontos</button>
                <button className="primary-button" disabled={savingReview || manualAngle == null} onClick={saveReview}>{savingReview ? "Salvando…" : "Registrar revisão"}</button>
              </div>
              <p className="review-help">Arraste PROX, EIXO e DIST. A referência física é registrada separadamente no estudo experimental para reduzir viés durante a revisão da imagem.</p>
              {reviewStatus && <div className="review-status">{reviewStatus}</div>}
            </div>
          )}
        </aside>

        <section className="timeline-panel panel">
          <div className="section-title-row">
            <div><span className="eyebrow">Movimento ao longo do tempo</span><h2>Curva angular sincronizada ao vídeo</h2></div>
            <span className="data-count">{angles.length} frames analisados</span>
          </div>
          <MotionInsightChart
            values={angles}
            velocities={result?.velocidades_player ?? []}
            current={frameIndex}
            fps={result?.fps}
            onSelect={totalSelectable ? seekFrame : undefined}
            advanced={!presentationMode}
          />
        </section>

        {!presentationMode && <ValidationStudyPanel
          movement={movement}
          side={side}
          jobId={analysisJobId}
          frame={frameIndex}
          measurement={selectedMeasurement}
          isAdmin={authUser.role === "admin"}
          protocolVersion={protocolVersion}
          movementConfig={movementConfig}
        />}

        {!presentationMode && <section className="validation-panel panel">
          <div className="section-title-row">
            <div><span className="eyebrow">Revisão interna</span><h2>Automático × goniometria manual da imagem</h2></div>
            <div className="validation-actions">
              <span className="data-count">{validationReport?.total_reviews ?? 0} revisões · {validationReport?.unique_evaluators ?? 0} avaliadores</span>
              <button className="export-link" onClick={() => api.downloadValidationCsv(movement, side)}>Exportar revisões</button>
              <button className="export-link" disabled={!analysisJobId} onClick={() => analysisJobId && api.downloadSessionBundle(analysisJobId)}>Pacote da sessão</button>
            </div>
          </div>
          <div className="validation-grid">
            <div><span>MAE × revisão manual</span><strong>{fmt(validationReport?.automatic_vs_manual.mae_deg)}°</strong></div>
            <div><span>RMSE</span><strong>{fmt(validationReport?.automatic_vs_manual.rmse_deg)}°</strong></div>
            <div><span>Bias</span><strong>{fmt(validationReport?.automatic_vs_manual.bias_deg)}°</strong></div>
            <div><span>ICC(A,1)</span><strong>{fmt(validationReport?.automatic_vs_manual.icc_a1, 3)}</strong></div>
            <div><span>CCC de Lin</span><strong>{fmt(validationReport?.automatic_vs_manual.ccc, 3)}</strong></div>
            <div><span>N revisões</span><strong>{validationReport?.automatic_vs_manual.n ?? 0}</strong></div>
          </div>
          <p className="methodology-warning">A revisão manual da imagem é uma checagem de posicionamento dos landmarks, não o padrão-ouro físico. A referência goniométrica é registrada separadamente no estudo experimental acima.</p>
        </section>}

        {presentationMode && <section className="explainability-panel panel">
          <div className="section-title-row"><div><span className="eyebrow">Como isso foi calculado?</span><h2>Da imagem à medida</h2></div><span className="markerless-core-badge"><i /> MARKERLESS</span></div>
          <div className="explainability-flow">
            <div><span>01</span><strong>Vídeo</strong><p>O frame original permanece como referência visual.</p></div>
            <div><span>02</span><strong>Landmarks</strong><p>Pontos corporais são detectados sem marcadores físicos.</p></div>
            <div><span>03</span><strong>Plano anatômico</strong><p>O movimento é projetado no plano clínico configurado.</p></div>
            <div><span>04</span><strong>Ângulo</strong><p>A série temporal gera amplitude e velocidade observadas.</p></div>
          </div>
          <p className="presentation-scope-note">O resultado é apoio quantitativo à avaliação e pesquisa. Não constitui diagnóstico autônomo.</p>
        </section>}


        {!presentationMode && <section className="engineering-panel panel">
          <button className="engineering-toggle" onClick={() => setEngineeringOpen((value) => !value)}>
            <div><span className="eyebrow">Modo engenharia</span><strong>Telemetria e diagnóstico do pipeline</strong></div><span>{engineeringOpen ? "−" : "+"}</span>
          </button>
          {engineeringOpen && (
            <div className="engineering-grid">
              <div><span>Geométrico raw</span><strong>{fmt(selectedMeasurement?.angleRawDeg)}°</strong></div>
              <div><span>Pontos 3D / Kalman</span><strong>{fmt(selectedMeasurement?.angleLandmarkDeg)}°</strong></div>
              <div><span>Projetado no plano</span><strong>{fmt(selectedMeasurement?.anglePlaneDeg)}°</strong></div>
              <div><span>Plano anatômico</span><strong>{selectedMeasurement?.measurementPlane ?? "—"}</strong></div>
              <div><span>Desvio da câmera</span><strong>{fmt(selectedMeasurement?.viewDeviationDeg)}°</strong></div>
              <div><span>Fora do plano</span><strong>{fmt(selectedMeasurement?.outOfPlaneDeg)}°</strong></div>
              <div><span>Regra direcional</span><strong>{selectedMeasurement?.directionRule ?? "—"}</strong></div>
              <div><span>Alinhamento direcional</span><strong>{fmt(selectedMeasurement?.directionAlignment, 3)}</strong></div>
              <div><span>Referencial corporal</span><strong>{selectedMeasurement?.bodyFrameConfidence == null ? "—" : `${fmt(selectedMeasurement.bodyFrameConfidence * 100, 0)}%`}</strong></div>
              <div><span>Geométrico + EMA</span><strong>{fmt(selectedMeasurement?.angleFilteredDeg)}°</strong></div>
              <div><span>Clínico</span><strong>{fmt(selectedMeasurement?.angleClinicalDeg)}°</strong></div>
              <div><span>Inferência</span><strong>{fmt(selectedSystem?.inferenceMs)} ms</strong></div>
              <div><span>Pipeline</span><strong>{fmt(selectedSystem?.latencyMs)} ms</strong></div>
              <div><span>Processing FPS</span><strong>{fmt(selectedSystem?.processingFps)}</strong></div>
              <div><span>Espaço de medição</span><strong>{selectedMeasurement?.measurementSpace ?? "—"}</strong></div>
              <div><span>Visib. média</span><strong>{selectedMeasurement?.landmarkConfidenceMean == null ? "—" : `${fmt(selectedMeasurement.landmarkConfidenceMean * 100, 0)}%`}</strong></div>
              <div><span>Flags</span><strong>{selectedMeasurement?.qualityFlags?.join(", ") || "—"}</strong></div>
              <div><span>Protocolo</span><strong>{result?.protocol_version ?? "—"}</strong></div>
              <div><span>Hash resultado</span><strong className="hash-value">{result?.result_sha256?.slice(0, 16) ?? "—"}</strong></div>
              <div><span>CPU</span><strong>{fmt(selectedSystem?.cpuPct)}%</strong></div>
              <div><span>RAM</span><strong>{fmt(selectedSystem?.ramPct)}%</strong></div>
              <div><span>VRAM</span><strong>{fmt(selectedSystem?.vramMb)} MB</strong></div>
            </div>
          )}
        </section>}
      </main>
      ) : (
        <WorkspaceHub
          active={activeWorkspace}
          user={authUser}
          jobId={analysisJobId}
          result={result}
          imuAnalysisId={imuAnalysisId}
          onImuAnalysis={setImuAnalysisId}
        />
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
