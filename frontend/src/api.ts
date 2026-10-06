import type {
  AnalysisResult,
  AnalysisSocketEvent,
  ValidationReview,
  ValidationReviewRequest,
  ValidationStats,
  ValidationReport,
  AuthSession,
  AuthUser,
  ValidationStudy,
  ValidationStudyReport,
  ValidationStudyTrial,
  MovementConfig,
  CapturePreflightReport,
  GaitTemporalAnalysis,
  IMUAnalysisRecord,
  MotorFeatureRecord,
  LongitudinalAnalysisRecord,
  FormalReadiness,
  FormalRun,
  PublicationReadiness,
  ClinicalSessionSummary,
  ClinicalSessionDetail,
  DatasetStats,
  AuditEvent,
  MetricsSnapshot,
  SystemHealth,
} from "./types";

const backendHttp = window.location.origin;
const backendWs = backendHttp.replace(/^http/, "ws");
const TOKEN_KEY = "physioai.access_token";

function getToken() {
  return sessionStorage.getItem(TOKEN_KEY);
}

function setToken(token: string | null) {
  if (token) sessionStorage.setItem(TOKEN_KEY, token);
  else sessionStorage.removeItem(TOKEN_KEY);
}

async function jsonOrError(response: Response) {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    if (response.status === 401) setToken(null);
    throw new Error(payload.detail || response.statusText);
  }
  return response.json();
}

function authHeaders(extra?: HeadersInit): HeadersInit {
  const headers = new Headers(extra);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return headers;
}

async function authorizedFetch(input: RequestInfo | URL, init: RequestInit = {}) {
  return fetch(input, {
    ...init,
    headers: authHeaders(init.headers),
  });
}


async function downloadBlob(response: Response, filename: string) {
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export const api = {
  backendHttp,
  backendWs,
  getToken,

  async health() {
    return jsonOrError(await fetch(`${backendHttp}/health`));
  },

  async login(username: string, password: string) {
    const session = await jsonOrError(
      await fetch(`${backendHttp}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      }),
    ) as AuthSession;
    setToken(session.access_token);
    return session;
  },

  async me() {
    return jsonOrError(await authorizedFetch(`${backendHttp}/api/auth/me`)) as Promise<AuthUser>;
  },

  logout() {
    setToken(null);
  },

  async systemHealth() {
    return jsonOrError(await authorizedFetch(`${backendHttp}/api/system/health`)) as Promise<SystemHealth>;
  },

  async movements() {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/config/movimentos`),
    ) as Promise<{ items: MovementConfig[]; protocol_version?: string; clinical_spec_sha256?: string }>;
  },


  async capturePreflight(formData: FormData) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/capture/preflight`, {
        method: "POST",
        body: formData,
      }),
    ) as Promise<CapturePreflightReport>;
  },

  async discardPreflight(preflightId: string) {
    const response = await authorizedFetch(`${backendHttp}/api/capture/preflight/${preflightId}`, {
      method: "DELETE",
    });
    if (!response.ok && response.status !== 404) await jsonOrError(response);
  },

  async startAnalysis(formData: FormData) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/analise/processar`, {
        method: "POST",
        body: formData,
      }),
    ) as Promise<{ job_id: string; status: string; patient_pseudonym: string }>;
  },

  async cancelAnalysis(jobId: string) {
    const response = await authorizedFetch(`${backendHttp}/api/analise/cancelar/${jobId}`, {
      method: "POST",
    });
    if (!response.ok && response.status !== 404) await jsonOrError(response);
  },

  async getResult(jobId: string) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/analise/resultado/${jobId}`),
    ) as Promise<AnalysisResult>;
  },

  async tryGetResult(jobId: string) {
    const response = await authorizedFetch(`${backendHttp}/api/analise/resultado/${jobId}`);
    if (response.status === 404) return null;
    return jsonOrError(response) as Promise<AnalysisResult>;
  },

  async saveValidation(payload: ValidationReviewRequest) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/validacao/revisoes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    ) as Promise<ValidationReview>;
  },

  async validationStats(movement?: string, side?: string) {
    const qs = new URLSearchParams();
    if (movement) qs.set("movement", movement);
    if (side) qs.set("side", side);
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/validacao/stats${suffix}`),
    ) as Promise<ValidationStats>;
  },

  async validationReport(movement?: string, side?: string) {
    const qs = new URLSearchParams();
    if (movement) qs.set("movement", movement);
    if (side) qs.set("side", side);
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/validacao/relatorio${suffix}`),
    ) as Promise<ValidationReport>;
  },

  async downloadValidationCsv(movement?: string, side?: string) {
    const qs = new URLSearchParams();
    if (movement) qs.set("movement", movement);
    if (side) qs.set("side", side);
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    const response = await authorizedFetch(`${backendHttp}/api/validacao/export.csv${suffix}`);
    if (!response.ok) await jsonOrError(response);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "physioai_validacao.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  },

  async studies(limit = 100) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/estudos?limit=${limit}`),
    ) as Promise<{ items: ValidationStudy[] }>;
  },

  async createStudy(payload: Record<string, unknown>) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/estudos`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    ) as Promise<ValidationStudy>;
  },

  async freezeStudy(studyId: string) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/estudos/${studyId}/freeze`, { method: "POST" }),
    ) as Promise<ValidationStudy>;
  },

  async closeStudy(studyId: string) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/estudos/${studyId}/close`, { method: "POST" }),
    ) as Promise<ValidationStudy>;
  },

  async addStudyTrial(studyId: string, payload: {
    job_id: string;
    frame: number;
    reference_angle_deg: number;
    reference_device_serial?: string | null;
    notes?: string;
    manual_exclusion_reason?: string | null;
  }) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/estudos/${studyId}/ensaios`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    ) as Promise<ValidationStudyTrial>;
  },

  async studyReport(studyId: string) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/estudos/${studyId}/relatorio`),
    ) as Promise<ValidationStudyReport>;
  },

  async downloadStudyCsv(studyId: string, code = "study") {
    const response = await authorizedFetch(`${backendHttp}/api/estudos/${studyId}/export.csv`);
    if (!response.ok) await jsonOrError(response);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `physioai_${code}_trials.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  },

  async downloadStudyBundle(studyId: string, code = "study") {
    const response = await authorizedFetch(`${backendHttp}/api/estudos/${studyId}/export.zip`);
    if (!response.ok) await jsonOrError(response);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `physioai_${code}_study.zip`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  },

  async recentValidations(limit = 20) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/validacao/revisoes?limit=${limit}`),
    ) as Promise<{ items: ValidationReview[] }>;
  },

  async sessions(limit = 50) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/sessoes?limit=${limit}`),
    ) as Promise<{ items: ClinicalSessionSummary[] }>;
  },

  async downloadSessionBundle(jobId: string) {
    const response = await authorizedFetch(`${backendHttp}/api/sessoes/${jobId}/export.zip`);
    if (!response.ok) await jsonOrError(response);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `physioai_session_${jobId}.zip`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  },

  async audit(limit = 100) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/auditoria?limit=${limit}`),
    ) as Promise<{ items: AuditEvent[] }>;
  },

  async deletePatient(patientId: string) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/privacy/patient/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patient_id: patientId, confirmation: "EXCLUIR" }),
      }),
    );
  },


  async study(studyId: string) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/estudos/${studyId}`),
    ) as Promise<ValidationStudy>;
  },

  async studyTrials(studyId: string) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/estudos/${studyId}/ensaios`),
    ) as Promise<{ items: ValidationStudyTrial[] }>;
  },

  async formalReadiness(studyId: string, payload: {
    ethics_status: string;
    ethics_protocol_id?: string | null;
    reference_calibration_date?: string | null;
    reference_calibration_due_date?: string | null;
    notes?: string;
  }) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/estudos/${studyId}/formal-runs/readiness`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    ) as Promise<FormalReadiness>;
  },

  async createFormalRun(studyId: string, payload: {
    ethics_status: string;
    ethics_protocol_id?: string | null;
    reference_calibration_date?: string | null;
    reference_calibration_due_date?: string | null;
    notes?: string;
  }) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/estudos/${studyId}/formal-runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    ) as Promise<FormalRun>;
  },

  async formalRuns(studyId: string) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/estudos/${studyId}/formal-runs`),
    ) as Promise<{ items: FormalRun[] }>;
  },

  async formalRun(runId: string) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/formal-runs/${runId}`),
    ) as Promise<FormalRun>;
  },

  async closeFormalRun(runId: string) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/formal-runs/${runId}/close`, { method: "POST" }),
    ) as Promise<FormalRun>;
  },

  async downloadFormalRun(runId: string) {
    const response = await authorizedFetch(`${backendHttp}/api/formal-runs/${runId}/export.zip`);
    if (!response.ok) await jsonOrError(response);
    await downloadBlob(response, `physioai_formal_run_${runId}.zip`);
  },

  async session(jobId: string) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/sessoes/${jobId}`),
    ) as Promise<ClinicalSessionDetail>;
  },

  async datasetStats() {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/dataset/stats`),
    ) as Promise<DatasetStats>;
  },

  async systemMetrics() {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/system/metrics`),
    ) as Promise<MetricsSnapshot>;
  },

  async version() {
    return jsonOrError(await fetch(`${backendHttp}/version`)) as Promise<Record<string, unknown>>;
  },
  openSocket(jobId: string, onEvent: (event: AnalysisSocketEvent) => void, onDisconnect?: () => void) {
    const token = getToken();
    // O bearer viaja como subprotocolo, não na URL, evitando vazamento em
    // logs de acesso de proxies/reverse proxies.
    const protocols = token ? ["physioai", token] : ["physioai"];
    const socket = new WebSocket(`${backendWs}/ws/analise/${jobId}`, protocols);
    socket.onmessage = (message) => {
      try {
        onEvent(JSON.parse(message.data) as AnalysisSocketEvent);
      } catch {
        // Evento inválido não derruba uma sessão clínica em andamento.
      }
    };
    socket.onclose = () => onDisconnect?.();
    return socket;
  },

  async analyzeExperimentalGait(payload: { fps: number; coordinate_space?: string; skeleton_frames: number[][][] }) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/experimental/gait/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ coordinate_space: "relative_body_3d", ...payload }),
      }),
    ) as Promise<GaitTemporalAnalysis>;
  },

  async analyzeExperimentalImu(payload: {
    timestamps_ms: number[];
    gyro_rad_s: number[][];
    accel_m_s2?: number[][];
    band_low_hz?: number;
    band_high_hz?: number;
    job_id?: string | null;
    video_offset_ms?: number;
    device_label?: string | null;
  }) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/experimental/imu/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    ) as Promise<IMUAnalysisRecord>;
  },

  async analyzeExperimentalMotorFeatures(payload: {
    job_id: string;
    imu_analysis_id?: string | null;
    min_cycle_prominence_deg?: number;
    tremor_band_low_hz?: number;
    tremor_band_high_hz?: number;
  }) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/experimental/motor-features/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    ) as Promise<MotorFeatureRecord>;
  },

  async analyzeExperimentalLongitudinal(payload: {
    job_id: string;
    include_motor_features?: boolean;
    max_sessions?: number;
  }) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/experimental/longitudinal/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    ) as Promise<LongitudinalAnalysisRecord>;
  },

  async publicationReadiness(runId: string) {
    return jsonOrError(
      await authorizedFetch(`${backendHttp}/api/formal-runs/${runId}/publication/readiness`),
    ) as Promise<PublicationReadiness>;
  },

  async downloadPublicationPackage(runId: string) {
    const response = await authorizedFetch(`${backendHttp}/api/formal-runs/${runId}/publication.zip`);
    if (!response.ok) await jsonOrError(response);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `physioai_external_review_${runId}.zip`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  },

};
