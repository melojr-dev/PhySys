
export interface CapturePreflightCheck {
  code: string;
  label: string;
  status: "pass" | "warn" | "fail" | "info" | string;
  message: string;
  value?: number | string | null;
  threshold?: string | null;
  critical?: boolean;
}

export interface CapturePreflightReport {
  preflight_id: string;
  patient_pseudonym: string;
  expires_in_seconds: number;
  status: "pass" | "warn" | "fail" | string;
  can_analyze: boolean;
  score: number;
  movement: string;
  side: string;
  sampled_frames: number;
  detected_frames: number;
  best_frame_index?: number | null;
  best_frame_timestamp_ms?: number | null;
  video: { width?: number; height?: number; fps?: number; total_frames?: number; duration_s?: number | null };
  summary: Record<string, number | string | null>;
  checks: CapturePreflightCheck[];
}

export type MeasurementSource = "observed" | "predicted" | "manual";

export interface MovementConfig {
  name: string;
  jointCode: string;
  referenceRomDeg: number;
  validated: boolean;
  validationStatus: string;
  convention: string;
  measurementPlane: string;
  recommendedView: string;
  landmarks: { proximal: string; vertex: string; distal: string };
  captureInstructions: string[];
  minViewAlignment?: number;
  maxOutOfPlaneDeg?: number;
  requirePlaneProjection?: boolean;
  directionRule?: string | null;
  minDirectionAlignment?: number | null;
}

export interface Point2D {
  x: number;
  y: number;
  confidence?: number | null;
}

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export interface JointMeasurement {
  joint: string;
  movement: string;
  angleRawDeg: number | null;
  angleLandmarkDeg?: number | null;
  anglePlaneDeg?: number | null;
  angleFilteredDeg: number | null;
  angleClinicalDeg: number | null;
  angularVelocityDegS: number | null;
  admDeg: number | null;
  source: MeasurementSource;
  measurementSpace?: string | null;
  measurementPlane?: string | null;
  planeProjectionApplied?: boolean;
  bodyFrameConfidence?: number | null;
  viewAlignment?: number | null;
  viewDeviationDeg?: number | null;
  outOfPlaneDeg?: number | null;
  directionRule?: string | null;
  directionAlignment?: number | null;
  confidence: number | null;
  landmarkConfidenceMin?: number | null;
  landmarkConfidenceMean?: number | null;
  quality?: "excellent" | "good" | "acceptable" | "low" | "predicted" | "unavailable" | string;
  measurementValid?: boolean;
  qualityFlags?: string[];
  proximal?: Point2D | null;
  vertex?: Point2D | null;
  distal?: Point2D | null;
  proximal3d?: Point3D | null;
  vertex3d?: Point3D | null;
  distal3d?: Point3D | null;
  /** MediaPipe Pose landmarks normalizados [0,1] para o overlay visual. */
  poseLandmarks2d?: Array<Point2D | null> | null;
}

export interface SystemTelemetry {
  captureFps?: number | null;
  processingFps?: number | null;
  latencyMs?: number | null;
  inferenceMs?: number | null;
  cpuPct?: number | null;
  ramPct?: number | null;
  gpuPct?: number | null;
  vramMb?: number | null;
}

export interface AnalysisFrameEvent {
  type: "analysis.frame";
  frame: number;
  timestampMs: number;
  measurement: JointMeasurement;
  system?: SystemTelemetry;
}

export interface ProgressEvent {
  type: "progress";
  pct: number;
}

export interface DoneEvent {
  type: "done";
  job_id: string;
}

export interface CancelledEvent {
  type: "cancelled";
  job_id: string;
}

export interface ErrorEvent {
  type: "error";
  detail: string;
}

export type AnalysisSocketEvent =
  | AnalysisFrameEvent
  | ProgressEvent
  | DoneEvent
  | CancelledEvent
  | ErrorEvent;

export interface AnalysisResult {
  job_id?: string;
  status?: string;
  protocol_version?: string;
  clinical_spec_sha256?: string;
  result_sha256?: string;
  adm_max?: number;
  adm_observada?: number;
  adm_min?: number;
  adm_media?: number;
  adm_std?: number;
  angulo_clinico_min?: number;
  angulo_clinico_max?: number;
  angulo_clinico_medio?: number;
  adm_ref?: number;
  pct_adm_ref?: number;
  vel_pico?: number;
  vel_media?: number;
  total_reps?: number;
  fps?: number;
  duracao_s?: number;
  total_frames?: number;
  frames_b64?: string[];
  angulos?: number[];
  angulos_brutos?: number[];
  angulos_geometricos_filtrados?: number[];
  angulos_player?: number[];
  velocidades?: number[];
  velocidades_player?: number[];
  frame_measurements?: AnalysisFrameEvent[];
  skeleton_3d?: number[][][];
  fluidez_ia?: number;
}

export interface ValidationReviewRequest {
  job_id: string;
  patient_id?: string | null;
  evaluator_id?: string | null;
  movement?: string | null;
  side?: string | null;
  frame: number;
  timestamp_ms?: number | null;
  automatic_angle_deg?: number | null;
  manual_angle_deg?: number | null;
  reference_angle_deg?: number | null;
  manual_method?: string | null;
  reference_method?: string | null;
  confidence?: number | null;
  source?: MeasurementSource | null;
  proximal?: Point2D | null;
  vertex?: Point2D | null;
  distal?: Point2D | null;
  notes?: string;
}

export interface ValidationReview extends ValidationReviewRequest {
  id: string;
  schema_version: number;
  created_at: string;
  error_vs_manual_deg?: number | null;
  error_vs_reference_deg?: number | null;
}

export interface ErrorMetrics {
  n: number;
  mae_deg: number | null;
  median_ae_deg: number | null;
  rmse_deg: number | null;
  bias_deg: number | null;
  sd_error_deg: number | null;
  loa_lower_deg: number | null;
  loa_upper_deg: number | null;
  within_2_deg_pct: number | null;
  within_5_deg_pct: number | null;
  pearson_r: number | null;
  ccc: number | null;
  icc_a1: number | null;
  mean_automatic_deg: number | null;
  mean_comparator_deg: number | null;
}

export interface ValidationStats {
  total_reviews: number;
  unique_patients: number;
  unique_sessions: number;
  unique_evaluators: number;
  movement?: string | null;
  side?: string | null;
  automatic_vs_manual: ErrorMetrics;
  automatic_vs_reference: ErrorMetrics;
}

export interface BlandAltmanPoint {
  mean_deg: number;
  difference_deg: number;
}

export interface ValidationReport extends ValidationStats {
  bland_altman_manual: BlandAltmanPoint[];
  bland_altman_reference: BlandAltmanPoint[];
  methodology: {
    signed_error: string;
    icc: string;
    ccc: string;
    loa: string;
    warning: string;
  };
}

export interface AuthUser {
  username: string;
  role: string;
}

export interface AuthSession {
  access_token: string;
  token_type: string;
  expires_at: number;
  user: AuthUser;
}

export interface ValidationStudy {
  id: string;
  schema_version: number;
  code: string;
  title: string;
  protocol_version: string;
  created_at: string;
  created_by: string;
  status: "draft" | "frozen" | "closed";
  frozen_at?: string | null;
  closed_at?: string | null;
  movement: string;
  side: "direito" | "esquerdo" | "ambos" | string;
  target_participants?: number | null;
  repetitions_per_participant?: number | null;
  min_landmark_confidence: number;
  require_observed: boolean;
  require_valid_measurement: boolean;
  reference_method: string;
  reference_device_model?: string | null;
  reference_device_resolution_deg?: number | null;
  camera_setup?: Record<string, unknown>;
  protocol_notes?: string;
  protocol_sha256?: string | null;
}

export interface ValidationStudyTrial {
  id: string;
  study_id: string;
  job_id: string;
  patient_id: string;
  evaluator_id: string;
  reference_assessor_id?: string | null;
  movement: string;
  side: string;
  trial_number: number;
  frame: number;
  timestamp_ms: number;
  automatic_angle_deg: number;
  reference_angle_deg: number;
  error_deg: number;
  absolute_error_deg: number;
  landmark_confidence_min?: number | null;
  body_frame_confidence?: number | null;
  measurement_plane?: string | null;
  view_alignment?: number | null;
  view_deviation_deg?: number | null;
  out_of_plane_deg?: number | null;
  source?: string | null;
  measurement_valid: boolean;
  quality?: string | null;
  included: boolean;
  exclusion_reason?: string | null;
}

export interface ValidationStudyReport {
  study: ValidationStudy;
  counts: {
    participants_total: number;
    participants_included: number;
    sessions_total: number;
    trials_total: number;
    trials_included: number;
    trials_excluded: number;
    completion_pct?: number | null;
  };
  agreement: ErrorMetrics;
  participant_balanced: {
    participants: number;
    mean_patient_mae_deg?: number | null;
    median_patient_mae_deg?: number | null;
    mean_patient_bias_deg?: number | null;
    mean_patient_rmse_deg?: number | null;
  };
  cluster_bootstrap: {
    participants: number;
    iterations: number;
    mae_ci95_deg: [number | null, number | null];
    bias_ci95_deg: [number | null, number | null];
    method: string;
  };
  by_reference_angle: Record<string, ErrorMetrics>;
  by_side: Record<string, ErrorMetrics>;
  exclusion_reasons: Record<string, number>;
  bland_altman: Array<{ mean_deg: number; difference_deg: number }>;
  methodology: Record<string, string>;
}

export interface GaitEvent {
  side: "left" | "right" | string;
  kind: string;
  frame: number;
  timestamp_s: number;
  prominence: number;
}

export interface GaitTemporalAnalysis {
  schema_version: number;
  experimental: boolean;
  coordinate_space: string;
  duration_s: number;
  fps: number;
  total_frames: number;
  valid_frame_ratio: number;
  event_count: number;
  left_event_count: number;
  right_event_count: number;
  cadence_steps_min: number | null;
  mean_step_time_s: number | null;
  step_time_cv_pct: number | null;
  left_step_time_s: number | null;
  right_step_time_s: number | null;
  step_time_symmetry_ratio: number | null;
  left_stride_time_s: number | null;
  right_stride_time_s: number | null;
  stride_time_symmetry_ratio: number | null;
  stride_length_m: number | null;
  quality_score: number;
  status: string;
  quality_flags: string[];
  events: GaitEvent[];
}


export interface IMUSynchronization {
  linked_job_id: string | null;
  offset_ms: number;
  video_frame_count: number;
  aligned_frame_count: number;
  overlap_ratio: number;
  median_alignment_error_ms: number | null;
  p95_alignment_error_ms: number | null;
}

export interface IMUAnalysisSummary {
  schema_version: number;
  experimental: boolean;
  diagnostic: boolean;
  sample_count: number;
  duration_s: number;
  sampling_rate_hz: number;
  sampling_jitter_cv_pct: number;
  invalid_sample_ratio: number;
  gyro_rms_rad_s: number;
  gyro_p95_rad_s: number;
  accel_dynamic_rms_m_s2: number | null;
  dominant_frequency_hz: number | null;
  dominant_frequency_power_ratio: number | null;
  configured_band_hz: [number, number];
  configured_band_power_ratio: number | null;
  spectral_entropy: number | null;
  periodicity_index: number | null;
  quality_score: number;
  status: string;
  quality_flags: string[];
  synchronization: IMUSynchronization | null;
}

export interface IMUAnalysisRecord {
  id: string;
  schema_version: number;
  created_at: string;
  owner: string;
  job_id: string | null;
  device_label: string | null;
  sample_count: number;
  duration_s: number;
  input_sha256: string;
  summary: IMUAnalysisSummary;
}

export interface VisionMotorFeatures {
  valid_frame_count: number;
  total_frame_count: number;
  valid_frame_ratio: number;
  duration_s: number;
  sampling_rate_hz: number | null;
  angular_range_deg: number | null;
  median_abs_velocity_deg_s: number | null;
  p95_abs_velocity_deg_s: number | null;
  cycle_count: number;
  median_cycle_duration_s: number | null;
  cycle_duration_cv_pct: number | null;
  median_cycle_amplitude_deg: number | null;
  cycle_amplitude_cv_pct: number | null;
  median_cycle_peak_velocity_deg_s: number | null;
  cycle_peak_velocity_cv_pct: number | null;
  amplitude_decrement_pct: number | null;
  peak_velocity_decrement_pct: number | null;
  tremor_band_hz: [number, number];
  tremor_dominant_frequency_hz: number | null;
  tremor_band_power_ratio: number | null;
  tremor_periodicity_index: number | null;
  quality_score: number;
  status: string;
  quality_flags: string[];
}

export interface MotorFeatureSummary {
  schema_version: number;
  experimental: boolean;
  diagnostic: boolean;
  intended_use: string;
  min_cycle_prominence_deg: number;
  unsupported_inferences: string[];
  vision: VisionMotorFeatures;
  fusion_mode: "vision_only" | "video_imu" | string;
  imu_analysis_id: string | null;
  imu_status: string | null;
  imu_quality_score: number | null;
  imu_dominant_frequency_hz: number | null;
  imu_band_power_ratio: number | null;
  imu_periodicity_index: number | null;
  video_imu_overlap_ratio: number | null;
  cross_modal_frequency_delta_hz: number | null;
  fusion_quality_score: number;
  status: string;
  quality_flags: string[];
}

export interface MotorFeatureRecord {
  id: string;
  schema_version: number;
  created_at: string;
  owner: string;
  job_id: string;
  imu_analysis_id: string | null;
  source_sha256: string;
  source: Record<string, unknown>;
  summary: MotorFeatureSummary;
}


export interface LongitudinalMetricSummary {
  metric: string;
  unit: string;
  n: number;
  first: number | null;
  latest: number | null;
  delta: number | null;
  delta_pct: number | null;
  mean: number | null;
  sd: number | null;
  cv_pct: number | null;
  median_abs_consecutive_change: number | null;
  slope_per_30d: number | null;
}

export interface LongitudinalSummary {
  schema_version: number;
  experimental: boolean;
  diagnostic: boolean;
  intended_use: string;
  unsupported_inferences: string[];
  anchor_job_id: string;
  movement: string;
  side: string;
  protocol_version: string;
  clinical_spec_sha256: string;
  status: string;
  session_count: number;
  excluded_session_count: number;
  software_versions: string[];
  mixed_software_versions: boolean;
  span_days: number | null;
  quality_flags: string[];
  sessions: Array<Record<string, unknown>>;
  excluded_sessions: Array<{ job_id: string; reasons: string[] }>;
  metrics: Record<string, LongitudinalMetricSummary>;
  methodology: Record<string, string>;
}

export interface LongitudinalAnalysisRecord {
  id: string;
  schema_version: number;
  created_at: string;
  owner: string;
  patient_pseudonym: string;
  anchor_job_id: string;
  movement: string;
  side: string;
  source_sha256: string;
  source: Record<string, unknown>;
  summary: LongitudinalSummary;
}

export interface FormalReadinessCheck {
  code: string;
  ok: boolean;
  detail: string;
}

export interface FormalReadiness {
  ready: boolean;
  checks: FormalReadinessCheck[];
  software_version: string;
  clinical_protocol_version: string;
  study_protocol_sha256?: string | null;
  statistical_plan_sha256?: string | null;
  protocol_lock_sha256?: string | null;
}

export interface FormalRun {
  id: string;
  schema_version: number;
  study_id: string;
  created_at: string;
  closed_at?: string | null;
  created_by: string;
  status: "open" | "closed" | string;
  software_version: string;
  clinical_protocol_version: string;
  study_protocol_sha256: string;
  statistical_plan_sha256: string;
  protocol_lock_sha256: string;
  ethics_status: string;
  ethics_protocol_id?: string | null;
  reference_calibration_date: string;
  reference_calibration_due_date: string;
  start_trial_count: number;
  end_trial_count?: number | null;
  snapshot_sha256: string;
  close_report_sha256?: string | null;
  notes?: string;
  close_report?: ValidationStudyReport | null;
}

export interface PublicationReadiness {
  package_ready: boolean;
  checks: Array<{ code: string; ok: boolean; detail: string }>;
  warnings: Array<{ code: string; detail: string }>;
  boundary: string;
}

export interface ClinicalSessionSummary {
  id: string;
  job_id: string;
  patient_pseudonym: string;
  evaluator_id: string;
  movement: string;
  side: string;
  started_at: string;
  completed_at?: string | null;
  status: string;
  software_version: string;
  protocol_version: string;
  clinical_spec_sha256: string;
  total_frames?: number | null;
  valid_frames?: number | null;
  fps?: number | null;
  adm_deg?: number | null;
  total_reps?: number | null;
  result_sha256?: string | null;
  provenance?: Record<string, unknown>;
}

export interface ClinicalSessionDetail {
  session: ClinicalSessionSummary;
  measurement_count: number;
  valid_measurement_count: number;
  validation_review_count: number;
  validation_trial_count: number;
  imu_analysis_count: number;
  motor_feature_analysis_count: number;
  longitudinal_analysis_count: number;
  measurement_sha256: string;
  validation_sha256: string;
  validation_trials_sha256: string;
  imu_analyses_sha256: string;
  motor_features_sha256: string;
  longitudinal_analyses_sha256: string;
}

export interface DatasetStats {
  total_samples?: number;
  total_amostras?: number;
  patients?: number;
  pacientes_unicos?: number;
  labels_distintos?: number;

  movements?: Record<string, number>;
  labels?: Record<string, number>;
  distribuicao_labels?: Record<string, number>;
  sides?: Record<string, number>;

  storage?: string;
  schema_version?: string | number;
  ultimas_10?: Array<Record<string, unknown>>;

  [key: string]: unknown;
}

export interface AuditEvent {
  id: string;
  created_at: string;
  actor: string;
  role: string;
  action: string;
  resource_type?: string | null;
  resource_id?: string | null;
  status: string;
  remote_ip?: string | null;
  metadata: Record<string, unknown>;
}

export interface MetricsSnapshot {
  uptime_seconds: number;
  counters: Array<{ name: string; labels: Record<string, string>; value: number }>;
  gauges: Array<{ name: string; labels: Record<string, string>; value: number }>;
  timers: Array<{ name: string; labels: Record<string, string>; count: number; sum_seconds: number; max_seconds: number; avg_seconds: number }>;
}

export interface SystemHealth {
  status: string;
  versao: string;
  build?: Record<string, unknown>;
  readiness?: Record<string, { ok?: boolean; [key: string]: unknown }>;
  analisador_pronto?: boolean;
  modelo_ia_pronto?: boolean;
  gpu_disponivel?: boolean;
  gpu_info?: Record<string, unknown>;
  result_ttl_seconds?: number;
  preflight_ttl_seconds?: number;
  preflight_required?: boolean;
  preflights_pending?: number;
  max_inflight_jobs?: number;
  max_upload_mb?: number;
  max_total_upload_mb?: number;
  protocol_version?: string;
  clinical_spec_sha256?: string;
  experimental_inputs_enabled?: boolean;
  metrics?: MetricsSnapshot;
}
