"""PhysioAI v2.20 — backend pré-clínico de análise biomecânica.

FastAPI + WebSocket com processamento assíncrono, telemetria frame a frame,
persistência auditável, autenticação/pseudonimização e proveniência científica.
Nesta versão a via clínica fica restrita aos movimentos explicitamente definidos
em ``src.core.clinical``; recursos experimentais não são tratados como validados.
"""
from __future__ import annotations

import asyncio
import csv
import io
import base64
import gc
import hashlib
import json
import logging
import os
import tempfile
import threading
import traceback
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import torch
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
    Query,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

try:
    import psutil
except ImportError:  # telemetria de sistema é opcional
    psutil = None


from src.core.engine import AnaliseCancelada, AnalisadorADMWeb, FrameMeasurement, ResultadoAnalise
from src.core.gait import analyze_gait_temporal
from src.core.imu import analyze_imu_signal
from src.core.motor_features import build_multimodal_motor_profile
from src.core.longitudinal import build_longitudinal_profile
from src.core.capture import CapturePreflightReport, capture_preflight_spec_snapshot, evaluate_capture_preflight
from src.core.clinical import (
    clinical_angle_from_points_2d,
    get_movement_spec,
    public_movement_specs,
)
from src.validation import ValidationStore, ValidationStudyStore, FormalValidationRunStore, formal_readiness, report_for_formal_run, load_json
from src.data import DatasetStore, ClinicalSessionStore, MeasurementStore, IMUAnalysisStore, MotorFeatureStore, LongitudinalAnalysisStore
from src.security import AuthError, AuthManager, AuthUser, AuditStore
from src.observability import configure_structured_logging, metrics
from src.build_info import get_build_info
from src.runtime import data_dir, database_path, model_path, sqlite_quick_check
from src.research import (
    PROTOCOL_VERSION,
    build_session_bundle,
    build_study_bundle,
    build_formal_validation_bundle,
    build_publication_review_bundle,
    publication_readiness,
    canonical_json_sha256,
    clinical_spec_sha256,
    clinical_spec_snapshot,
    model_provenance,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURAÇÃO DE LOGGING / OBSERVABILIDADE
# ─────────────────────────────────────────────────────────────────────────────
configure_structured_logging()
logger = logging.getLogger("physioai.server")
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = data_dir(APP_DIR)
RUNTIME_DB = database_path(APP_DIR)
MODEL_PATH = model_path(APP_DIR)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTES DE RUNTIME
# ─────────────────────────────────────────────────────────────────────────────
MAX_PLAYER_FRAMES   = 600
MAX_WORKERS_ENCODE  = min(8, (os.cpu_count() or 4))

MEDIAPIPE_CONNECTIONS: list[tuple[int, int]] = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (11,12),(11,13),(13,15),(15,17),(15,19),(17,19),
    (12,14),(14,16),(16,18),(16,20),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]

JOINT_COLORS: dict[int, str] = {
    **{i: "#00C9A7" for i in range(11)},
    **{i: "#5C6BC0" for i in range(11, 23)},
    **{i: "#FF7043" for i in range(23, 33)},
}

# ─────────────────────────────────────────────────────────────────────────────
#  ESTADO GLOBAL — jobs assíncronos, telemetria, preflight e resultados efêmeros
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class _PreflightRecord:
    preflight_id: str
    owner: str
    patient_pseudonym: str
    movement: str
    side: str
    path: str
    artifact: dict
    report: dict
    created_at: float
    claimed: bool = False


class _AppState:
    analisador:      AnalisadorADMWeb | None = None
    semaphore:       asyncio.Semaphore       = None   # type: ignore[assignment]
    preflight_semaphore: asyncio.Semaphore    = None   # type: ignore[assignment]
    # { job_id: asyncio.Queue }  — fila de progresso para o WebSocket
    progress_queues: dict[str, asyncio.Queue] = {}
    # { job_id: dict | Exception }  — resultado aguardando GET
    resultados:      dict[str, dict | Exception] = {}
    # cancelamento cooperativo thread-safe por job
    cancel_flags:    dict[str, threading.Event] = {}
    # dono e TTL de cada job/result para isolamento entre usuários e limpeza de RAM
    job_owner:       dict[str, str] = {}
    result_created:  dict[str, float] = {}
    # vídeos já submetidos ao preflight; reutilizados pela análise sem segundo upload
    preflights:      dict[str, _PreflightRecord] = {}


app_state = _AppState()
validation_store = ValidationStore(RUNTIME_DB)
study_store = ValidationStudyStore(RUNTIME_DB)
formal_run_store = FormalValidationRunStore(RUNTIME_DB)
dataset_store = DatasetStore(RUNTIME_DB)
session_store = ClinicalSessionStore(RUNTIME_DB)
measurement_store = MeasurementStore(RUNTIME_DB)
imu_store = IMUAnalysisStore(RUNTIME_DB)
motor_feature_store = MotorFeatureStore(RUNTIME_DB)
longitudinal_store = LongitudinalAnalysisStore(RUNTIME_DB)
audit_store = AuditStore(RUNTIME_DB)
auth_manager = AuthManager.from_env()
RESULT_TTL_SECONDS = max(300, int(os.environ.get("PHYSIOAI_RESULT_TTL_SECONDS", "900")))
PREFLIGHT_TTL_SECONDS = max(180, int(os.environ.get("PHYSIOAI_PREFLIGHT_TTL_SECONDS", "900")))
PREFLIGHT_MAX_SAMPLES = max(5, min(24, int(os.environ.get("PHYSIOAI_PREFLIGHT_MAX_SAMPLES", "12"))))
MAX_INFLIGHT_JOBS = max(1, int(os.environ.get("PHYSIOAI_MAX_INFLIGHT_JOBS", "3")))
MIN_FREE_DISK_MB = max(64, int(os.environ.get("PHYSIOAI_MIN_FREE_DISK_MB", "512")))
DB_QUICKCHECK_INTERVAL_SECONDS = max(30, int(os.environ.get("PHYSIOAI_DB_QUICKCHECK_INTERVAL_SECONDS", "300")))
_db_check_lock = threading.Lock()
_db_check_cache: dict[str, object] = {"at": 0.0, "ok": False, "message": "not_checked"}
BUILD_INFO = get_build_info()
APP_VERSION = BUILD_INFO.version
MAX_UPLOAD_MB = max(16, int(os.environ.get("PHYSIOAI_MAX_UPLOAD_MB", "256")))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_TOTAL_UPLOAD_MB = max(MAX_UPLOAD_MB, int(os.environ.get("PHYSIOAI_MAX_TOTAL_UPLOAD_MB", "512")))
MAX_TOTAL_UPLOAD_BYTES = MAX_TOTAL_UPLOAD_MB * 1024 * 1024
WS_QUEUE_MAX = max(32, int(os.environ.get("PHYSIOAI_WS_QUEUE_MAX", "512")))
CLINICAL_SPEC_SHA256 = clinical_spec_sha256()
MODEL_PROVENANCE = model_provenance(MODEL_PATH)
ENABLE_EXPERIMENTAL_INPUTS = os.environ.get("PHYSIOAI_ENABLE_EXPERIMENTAL_INPUTS", "0") == "1"
REQUIRE_PREFLIGHT = os.environ.get("PHYSIOAI_REQUIRE_PREFLIGHT", "1") == "1"
FORMAL_VALIDATION_RESOURCES = APP_DIR / "src" / "validation" / "resources"
STATISTICAL_PLAN_PATH = FORMAL_VALIDATION_RESOURCES / "statistical_analysis_plan.json"
FORMAL_CHECKLIST_PATH = FORMAL_VALIDATION_RESOURCES / "formal_validation_checklist.json"
PROTOCOL_LOCK_PATH = FORMAL_VALIDATION_RESOURCES / "protocol_lock.json"
PUBLICATION_REQUIREMENTS_PATH = FORMAL_VALIDATION_RESOURCES / "publication_requirements.json"


# ─────────────────────────────────────────────────────────────────────────────
#  SEGURANÇA / PRIVACIDADE / TTL
# ─────────────────────────────────────────────────────────────────────────────
def _remote_ip(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else None


def _request_user(request: Request) -> AuthUser:
    user = getattr(request.state, "user", None)
    if not isinstance(user, AuthUser):
        raise HTTPException(status_code=401, detail="Não autenticado")
    return user


def _require_admin(user: AuthUser) -> None:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Acesso restrito ao administrador")


def _assert_job_access(job_id: str, user: AuthUser) -> None:
    owner = app_state.job_owner.get(job_id)
    if owner is None:
        session = session_store.get_by_job(job_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Job não encontrado")
        owner = str(session.get("evaluator_id") or "")
    if owner != user.username and user.role != "admin":
        raise HTTPException(status_code=403, detail="Job pertence a outro usuário")


async def _stage_upload(
    upload: UploadFile,
    *,
    suffix: str,
    role: str,
) -> tuple[str, dict]:
    """Grava upload em streaming, limita tamanho e calcula SHA-256 sem reter o arquivo em RAM."""
    digest = hashlib.sha256()
    size = 0
    path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
            path = fh.name
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Arquivo '{role}' excede o limite de {MAX_UPLOAD_MB} MB.",
                    )
                digest.update(chunk)
                fh.write(chunk)
        if size == 0:
            raise HTTPException(status_code=422, detail=f"Arquivo '{role}' está vazio.")
        return path, {
            "role": role,
            "content_type": upload.content_type or "application/octet-stream",
            "size_bytes": size,
            "sha256": digest.hexdigest(),
        }
    except Exception:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
        raise


async def _cleanup_expired_results() -> None:
    while True:
        await asyncio.sleep(60)
        now = time.monotonic()
        expired = [
            job_id for job_id, created in list(app_state.result_created.items())
            if now - created > RESULT_TTL_SECONDS
        ]
        for job_id in expired:
            app_state.resultados.pop(job_id, None)
            app_state.progress_queues.pop(job_id, None)
            app_state.cancel_flags.pop(job_id, None)
            app_state.job_owner.pop(job_id, None)
            app_state.result_created.pop(job_id, None)
            metrics.inc("physioai_results_expired_total")
            logger.info("Resultado expirado removido: job_id=%s", job_id)

        expired_preflights = [
            preflight_id for preflight_id, record in list(app_state.preflights.items())
            if now - record.created_at > PREFLIGHT_TTL_SECONDS
        ]
        for preflight_id in expired_preflights:
            record = app_state.preflights.pop(preflight_id, None)
            if record and record.path and os.path.exists(record.path):
                try:
                    os.unlink(record.path)
                except OSError as exc:
                    logger.warning("Falha ao remover vídeo expirado de preflight '%s': %s", record.path, exc)
            metrics.inc("physioai_preflights_expired_total")
            logger.info("Preflight expirado removido: preflight_id=%s", preflight_id)


async def _sample_system_metrics() -> None:
    """Amostra recursos do processo/host sem bloquear o event loop."""
    process = psutil.Process(os.getpid()) if psutil is not None else None
    while True:
        try:
            if psutil is not None:
                metrics.set_gauge("physioai_system_cpu_percent", float(psutil.cpu_percent(interval=None)))
                vm = psutil.virtual_memory()
                metrics.set_gauge("physioai_system_ram_percent", float(vm.percent))
                metrics.set_gauge("physioai_system_ram_available_bytes", float(vm.available))
                if process is not None:
                    mem = process.memory_info()
                    metrics.set_gauge("physioai_process_rss_bytes", float(mem.rss))
                    metrics.set_gauge("physioai_process_threads", float(process.num_threads()))
            try:
                stat = os.statvfs(APP_DIR / "src" / "utils")
                free_bytes = stat.f_bavail * stat.f_frsize
                metrics.set_gauge("physioai_disk_free_bytes", float(free_bytes))
            except Exception:
                pass
            if torch.cuda.is_available():
                metrics.set_gauge("physioai_gpu_memory_allocated_bytes", float(torch.cuda.memory_allocated(0)))
                metrics.set_gauge("physioai_gpu_memory_reserved_bytes", float(torch.cuda.memory_reserved(0)))
        except Exception as exc:
            logger.debug("Falha ao amostrar métricas do sistema: %s", exc)
        await asyncio.sleep(5)


# ─────────────────────────────────────────────────────────────────────────────
#  LIFESPAN — recursos compartilhados, limpeza e telemetria de sistema
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== PhysioAI Server — inicializando recursos ===")

    # Semáforo criado aqui porque precisa do event loop já rodando
    app_state.semaphore = asyncio.Semaphore(1)
    app_state.preflight_semaphore = asyncio.Semaphore(1)
    cleanup_task = asyncio.create_task(_cleanup_expired_results())
    metrics_task = asyncio.create_task(_sample_system_metrics())
    metrics.set_gauge("physioai_jobs_pending", 0)
    metrics.set_gauge("physioai_jobs_active", 0)
    metrics.set_gauge("physioai_websocket_connections", 0)

    usar_gpu = torch.cuda.is_available()
    app_state.analisador = AnalisadorADMWeb(
        id_paciente="__base__",
        tipo_movimento="Flexão de Cotovelo",
        lado_do_corpo="direito",
        usar_gpu=usar_gpu,
    )

    # O classificador legado de "fluidez" não participa da via clínica v2.
    # Pode ser habilitado explicitamente apenas para experimentação.
    if os.environ.get("PHYSIOAI_ENABLE_LEGACY_FLUIDITY", "0") == "1":
        modelo_carregado = app_state.analisador.carregar_modelo_ia(str(MODEL_PATH))
        logger.info("Modelo legado de fluidez: %s", "carregado" if modelo_carregado else "indisponível")
    else:
        logger.info("Modelo legado de fluidez desativado na via clínica v2.")

    backend = "GPU 🔥" if usar_gpu else "CPU 🧊"
    logger.info("Backend de inferência: %s", backend)
    logger.info("=== Servidor pronto para receber requisições ===")

    yield

    for task in (cleanup_task, metrics_task):
        task.cancel()
    for task in (cleanup_task, metrics_task):
        try:
            await task
        except asyncio.CancelledError:
            pass

    logger.info("=== Encerrando servidor — liberando recursos ===")
    for record in list(app_state.preflights.values()):
        if record.path and os.path.exists(record.path):
            try:
                os.unlink(record.path)
            except OSError:
                pass
    app_state.preflights.clear()
    if app_state.analisador is not None:
        app_state.analisador.close()
        app_state.analisador = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("=== Recursos liberados com sucesso ===")


# ─────────────────────────────────────────────────────────────────────────────
#  APLICAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PhysioAI API",
    version=APP_VERSION,
    description=(
        "Backend de análise biomecânica com telemetria frame a frame. "
        "v2.17: análise longitudinal descritiva e controle de comparabilidade entre sessões/versionamentos."
    ),
    lifespan=lifespan,
)

_allowed_origins = [
    value.strip() for value in os.environ.get("PHYSIOAI_ALLOWED_ORIGINS", "").split(",")
    if value.strip()
]
if _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


class ExperimentalGaitRequest(BaseModel):
    fps: float = Field(gt=5.0, le=240.0)
    coordinate_space: str = Field(default="relative_body_3d", max_length=64)
    skeleton_frames: list[list[list[float]]]


class ExperimentalMotorFeatureRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=128)
    imu_analysis_id: str | None = Field(default=None, max_length=128)
    min_cycle_prominence_deg: float = Field(default=8.0, ge=2.0, le=60.0)
    tremor_band_low_hz: float = Field(default=3.0, ge=0.5, le=15.0)
    tremor_band_high_hz: float = Field(default=8.0, ge=1.0, le=20.0)


class ExperimentalLongitudinalRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=128)
    include_motor_features: bool = True
    max_sessions: int = Field(default=100, ge=2, le=500)


class ExperimentalIMURequest(BaseModel):
    timestamps_ms: list[float]
    gyro_rad_s: list[list[float]]
    accel_m_s2: list[list[float]] | None = None
    band_low_hz: float = Field(default=3.0, ge=0.2, le=19.0)
    band_high_hz: float = Field(default=8.0, ge=0.3, le=20.0)
    job_id: str | None = Field(default=None, max_length=128)
    video_offset_ms: float = Field(default=0.0, ge=-10000.0, le=10000.0)
    device_label: str | None = Field(default=None, max_length=128)



def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str):
        return path
    # Antes do roteamento, mantenha apenas categorias conhecidas para evitar
    # cardinalidade causada por UUIDs/identificadores na URL.
    path = request.url.path
    if path.startswith("/api/analise/resultado/"):
        return "/api/analise/resultado/{job_id}"
    if path.startswith("/api/analise/cancelar/"):
        return "/api/analise/cancelar/{job_id}"
    return path if len(path) <= 96 else "<long-path>"


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - started
        route = _route_template(request)
        metrics.inc(
            "physioai_http_requests_total",
            method=request.method, route=route, status=str(status_code),
        )
        metrics.observe(
            "physioai_http_request_duration", elapsed,
            method=request.method, route=route,
        )
        response_obj = locals().get("response")
        if response_obj is not None:
            response_obj.headers["X-Request-ID"] = request_id
            response_obj.headers["Server-Timing"] = f'app;dur={elapsed * 1000:.2f}'
        logger.info(
            "http_request",
            extra={
                "event": "http.request",
                "request_id": request_id,
                "method": request.method,
                "route": route,
                "status": status_code,
                "duration_ms": round(elapsed * 1000, 3),
            },
        )


@app.middleware("http")
async def api_auth_middleware(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/api/") or path == "/api/auth/login":
        return await call_next(request)

    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return JSONResponse(status_code=401, content={"detail": "Autenticação obrigatória"})
    token = header.split(" ", 1)[1].strip()
    try:
        request.state.user = auth_manager.verify_token(token)
    except AuthError:
        return JSONResponse(status_code=401, content={"detail": "Sessão inválida ou expirada"})
    return await call_next(request)


@app.post("/api/auth/login")
async def auth_login(payload: LoginRequest, request: Request):
    try:
        user = auth_manager.authenticate(payload.username, payload.password)
    except AuthError as exc:
        audit_store.log(
            actor=payload.username.strip() or "unknown", role="unknown", action="auth.login",
            status="denied", remote_ip=_remote_ip(request),
        )
        raise HTTPException(status_code=401, detail="Credenciais inválidas") from exc
    token, expires_at = auth_manager.issue_token(user)
    audit_store.log(
        actor=user.username, role=user.role, action="auth.login",
        status="ok", remote_ip=_remote_ip(request),
    )
    return {"access_token": token, "token_type": "bearer", "expires_at": expires_at,
            "user": {"username": user.username, "role": user.role}}


@app.get("/api/auth/me")
async def auth_me(request: Request):
    user = _request_user(request)
    return {"username": user.username, "role": user.role}


@app.post("/api/experimental/imu/analyze", status_code=201)
async def experimental_imu_analyze(payload: ExperimentalIMURequest, request: Request):
    """Analisa sinal inercial e, opcionalmente, sincroniza com uma sessão de vídeo.

    Saídas são métricas experimentais descritivas. Nenhum campo deste endpoint
    representa diagnóstico de tremor, bradicinesia, rigidez ou doença de Parkinson.
    """
    user = _request_user(request)
    n = len(payload.timestamps_ms)
    if n < 32:
        raise HTTPException(status_code=422, detail="São necessárias ao menos 32 amostras IMU.")
    if n > 60000:
        raise HTTPException(status_code=413, detail="Máximo de 60000 amostras IMU por análise.")
    if len(payload.gyro_rad_s) != n:
        raise HTTPException(status_code=422, detail="gyro_rad_s deve ter o mesmo N de timestamps_ms.")
    if payload.accel_m_s2 is not None and len(payload.accel_m_s2) != n:
        raise HTTPException(status_code=422, detail="accel_m_s2 deve ter o mesmo N de timestamps_ms.")
    if payload.band_low_hz >= payload.band_high_hz:
        raise HTTPException(status_code=422, detail="band_low_hz deve ser menor que band_high_hz.")

    video_timestamps: list[float] | None = None
    linked_job = (payload.job_id or "").strip() or None
    if linked_job is not None:
        _assert_job_access(linked_job, user)
        measurements = measurement_store.list_for_job(linked_job)
        video_timestamps = [
            float(row["timestamp_ms"]) for row in measurements
            if row.get("timestamp_ms") is not None
        ]
        if not video_timestamps:
            raise HTTPException(status_code=409, detail="Sessão vinculada ainda não possui telemetria de frames.")

    try:
        analysis = analyze_imu_signal(
            payload.timestamps_ms,
            payload.gyro_rad_s,
            accel_m_s2=payload.accel_m_s2,
            band_hz=(payload.band_low_hz, payload.band_high_hz),
            video_timestamps_ms=video_timestamps,
            video_offset_ms=payload.video_offset_ms,
            linked_job_id=linked_job,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    raw_payload = {
        "timestamps_ms": payload.timestamps_ms,
        "gyro_rad_s": payload.gyro_rad_s,
        "accel_m_s2": payload.accel_m_s2,
        "band_hz": [payload.band_low_hz, payload.band_high_hz],
        "job_id": linked_job,
        "video_offset_ms": payload.video_offset_ms,
        "device_label": payload.device_label,
    }
    summary = analysis.to_dict()
    record = imu_store.save(
        owner=user.username,
        job_id=linked_job,
        device_label=payload.device_label,
        raw_payload=raw_payload,
        summary=summary,
    )
    audit_store.log(
        actor=user.username, role=user.role, action="experimental.imu.analyze",
        resource_type="imu_analysis", resource_id=record["id"],
        remote_ip=_remote_ip(request),
        metadata={
            "job_id": linked_job, "samples": n, "status": analysis.status,
            "input_sha256": record["input_sha256"],
        },
    )
    metrics.inc("physioai_experimental_imu_total", status=analysis.status)
    return record


def _assert_imu_access(record: dict, user: AuthUser) -> None:
    owner = str(record.get("owner") or "")
    if owner != user.username and user.role != "admin":
        raise HTTPException(status_code=403, detail="Análise IMU pertence a outro usuário")


@app.get("/api/experimental/imu/{analysis_id}")
async def experimental_imu_get(analysis_id: str, request: Request):
    user = _request_user(request)
    record = imu_store.get(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Análise IMU não encontrada")
    _assert_imu_access(record, user)
    return record


@app.get("/api/experimental/imu/{analysis_id}/export.json")
async def experimental_imu_export(analysis_id: str, request: Request):
    user = _request_user(request)
    record = imu_store.get(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Análise IMU não encontrada")
    _assert_imu_access(record, user)
    raw = imu_store.raw_payload(analysis_id)
    payload_out = {"record": record, "raw": raw}
    audit_store.log(
        actor=user.username, role=user.role, action="experimental.imu.export",
        resource_type="imu_analysis", resource_id=analysis_id, remote_ip=_remote_ip(request),
    )
    return Response(
        content=json.dumps(payload_out, ensure_ascii=False, indent=2, allow_nan=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="physioai_imu_{analysis_id}.json"'},
    )


def _assert_motor_feature_access(record: dict, user: AuthUser) -> None:
    owner = str(record.get("owner") or "")
    if owner != user.username and user.role != "admin":
        raise HTTPException(status_code=403, detail="Perfil motor pertence a outro usuário")


@app.post("/api/experimental/motor-features/analyze", status_code=201)
async def experimental_motor_features_analyze(payload: ExperimentalMotorFeatureRequest, request: Request):
    """Extrai descritores experimentais de vídeo e, opcionalmente, IMU.

    O endpoint NÃO emite diagnóstico de Parkinson, bradicinesia clínica, rigidez
    ou roda dentada. Decremento de amplitude/velocidade, ritmo e conteúdo
    espectral são descritores de pesquisa do ensaio.
    """
    user = _request_user(request)
    _assert_job_access(payload.job_id, user)
    if payload.tremor_band_low_hz >= payload.tremor_band_high_hz:
        raise HTTPException(status_code=422, detail="tremor_band_low_hz deve ser menor que tremor_band_high_hz.")

    measurements = measurement_store.list_for_job(payload.job_id)
    if len(measurements) < 12:
        raise HTTPException(status_code=409, detail="Sessão não possui medições válidas suficientes para extração motora.")

    imu_record = None
    if payload.imu_analysis_id:
        imu_record = imu_store.get(payload.imu_analysis_id)
        if imu_record is None:
            raise HTTPException(status_code=404, detail="Análise IMU não encontrada")
        _assert_imu_access(imu_record, user)
        if str(imu_record.get("job_id") or "") != payload.job_id:
            raise HTTPException(status_code=409, detail="A análise IMU deve estar vinculada à mesma sessão de vídeo.")

    profile = build_multimodal_motor_profile(
        measurements,
        imu_record=imu_record,
        min_cycle_prominence_deg=payload.min_cycle_prominence_deg,
        tremor_band_hz=(payload.tremor_band_low_hz, payload.tremor_band_high_hz),
    )
    measurement_sha = canonical_json_sha256(measurements)
    source_payload = {
        "job_id": payload.job_id,
        "measurement_sha256": measurement_sha,
        "imu_analysis_id": payload.imu_analysis_id,
        "imu_input_sha256": imu_record.get("input_sha256") if imu_record else None,
        "min_cycle_prominence_deg": payload.min_cycle_prominence_deg,
        "tremor_band_hz": [payload.tremor_band_low_hz, payload.tremor_band_high_hz],
        "software_version": APP_VERSION,
    }
    record = motor_feature_store.save(
        owner=user.username,
        job_id=payload.job_id,
        imu_analysis_id=payload.imu_analysis_id,
        source_payload=source_payload,
        summary=profile.to_dict(),
    )
    audit_store.log(
        actor=user.username, role=user.role, action="experimental.motor_features.analyze",
        resource_type="motor_feature_analysis", resource_id=record["id"], remote_ip=_remote_ip(request),
        metadata={
            "job_id": payload.job_id, "imu_analysis_id": payload.imu_analysis_id,
            "status": profile.status, "source_sha256": record["source_sha256"],
        },
    )
    metrics.inc("physioai_experimental_motor_features_total", status=profile.status)
    return record


@app.get("/api/experimental/motor-features/{analysis_id}")
async def experimental_motor_features_get(analysis_id: str, request: Request):
    user = _request_user(request)
    record = motor_feature_store.get(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Perfil motor não encontrado")
    _assert_motor_feature_access(record, user)
    return record


@app.get("/api/experimental/motor-features/{analysis_id}/export.json")
async def experimental_motor_features_export(analysis_id: str, request: Request):
    user = _request_user(request)
    record = motor_feature_store.get(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Perfil motor não encontrado")
    _assert_motor_feature_access(record, user)
    audit_store.log(
        actor=user.username, role=user.role, action="experimental.motor_features.export",
        resource_type="motor_feature_analysis", resource_id=analysis_id, remote_ip=_remote_ip(request),
    )
    return Response(
        content=json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="physioai_motor_features_{analysis_id}.json"'},
    )


def _assert_longitudinal_access(record: dict, user: AuthUser) -> None:
    owner = str(record.get("owner") or "")
    if owner != user.username and user.role != "admin":
        raise HTTPException(status_code=403, detail="Análise longitudinal pertence a outro usuário")


@app.post("/api/experimental/longitudinal/analyze", status_code=201)
async def experimental_longitudinal_analyze(payload: ExperimentalLongitudinalRequest, request: Request):
    """Resumo longitudinal descritivo de sessões comparáveis do mesmo paciente.

    Não interpreta melhora clínica, progressão de doença ou resposta terapêutica.
    O job âncora fixa movimento, lado, protocolo e hash da especificação clínica.
    """
    user = _request_user(request)
    _assert_job_access(payload.job_id, user)
    anchor = session_store.get_by_job(payload.job_id)
    if anchor is None:
        raise HTTPException(status_code=404, detail="Sessão âncora não encontrada")
    patient = str(anchor.get("patient_pseudonym") or "")
    sessions = session_store.for_patient(patient, limit=payload.max_sessions)
    if user.role != "admin":
        sessions = [row for row in sessions if str(row.get("evaluator_id") or "") == user.username]

    motor_by_job: dict[str, list[dict]] = {}
    if payload.include_motor_features:
        for row in sessions:
            job = str(row.get("job_id") or "")
            if job:
                motor_by_job[job] = motor_feature_store.list_for_job(job)

    try:
        summary = build_longitudinal_profile(
            sessions, anchor_job_id=payload.job_id, motor_features_by_job=motor_by_job
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    source_payload = {
        "anchor_job_id": payload.job_id,
        "patient_pseudonym": patient,
        "session_sources": [
            {
                "job_id": row.get("job_id"),
                "result_sha256": row.get("result_sha256"),
                "software_version": row.get("software_version"),
                "protocol_version": row.get("protocol_version"),
                "clinical_spec_sha256": row.get("clinical_spec_sha256"),
            }
            for row in sessions
        ],
        "motor_feature_sources": {
            job: [item.get("source_sha256") for item in records]
            for job, records in motor_by_job.items() if records
        },
        "software_version": APP_VERSION,
    }
    record = longitudinal_store.save(
        owner=user.username, patient_pseudonym=patient, anchor_job_id=payload.job_id,
        movement=str(anchor.get("movement") or ""), side=str(anchor.get("side") or ""),
        source_payload=source_payload, summary=summary,
    )
    audit_store.log(
        actor=user.username, role=user.role, action="experimental.longitudinal.analyze",
        resource_type="longitudinal_analysis", resource_id=record["id"], remote_ip=_remote_ip(request),
        metadata={
            "job_id": payload.job_id, "session_count": summary.get("session_count"),
            "status": summary.get("status"), "source_sha256": record.get("source_sha256"),
        },
    )
    metrics.inc("physioai_experimental_longitudinal_total", status=str(summary.get("status") or "unknown"))
    return record


@app.get("/api/experimental/longitudinal/{analysis_id}")
async def experimental_longitudinal_get(analysis_id: str, request: Request):
    user = _request_user(request)
    record = longitudinal_store.get(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Análise longitudinal não encontrada")
    _assert_longitudinal_access(record, user)
    return record


@app.get("/api/experimental/longitudinal/{analysis_id}/export.json")
async def experimental_longitudinal_export(analysis_id: str, request: Request):
    user = _request_user(request)
    record = longitudinal_store.get(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Análise longitudinal não encontrada")
    _assert_longitudinal_access(record, user)
    audit_store.log(
        actor=user.username, role=user.role, action="experimental.longitudinal.export",
        resource_type="longitudinal_analysis", resource_id=analysis_id, remote_ip=_remote_ip(request),
    )
    return Response(
        content=json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="physioai_longitudinal_{analysis_id}.json"'},
    )


@app.post("/api/experimental/gait/analyze")
async def experimental_gait_analyze(payload: ExperimentalGaitRequest, request: Request):
    """Analisa temporalidade de marcha a partir de skeleton 3D já extraído.

    Endpoint experimental: não integra a via clínica nem gera diagnóstico.
    Limita o payload para evitar uso acidental como armazenamento de vídeos.
    """
    user = _request_user(request)
    frame_count = len(payload.skeleton_frames)
    if frame_count < 2:
        raise HTTPException(status_code=422, detail="São necessários pelo menos 2 frames.")
    if frame_count > 10000:
        raise HTTPException(status_code=413, detail="Máximo de 10000 frames por análise experimental.")
    if payload.coordinate_space not in {"relative_body_3d", "metric_global"}:
        raise HTTPException(status_code=422, detail="coordinate_space inválido.")
    try:
        result = analyze_gait_temporal(
            payload.skeleton_frames,
            payload.fps,
            coordinate_space=payload.coordinate_space,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    audit_store.log(
        actor=user.username, role=user.role, action="experimental.gait.analyze",
        status="ok", metadata={
            "frames": frame_count,
            "coordinate_space": payload.coordinate_space,
            "status": result.status,
        },
    )
    metrics.inc("physioai_experimental_gait_total", status=result.status)
    return result.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITÁRIOS DE CODIFICAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
def _encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 75],
    )
    return base64.b64encode(buf).decode()


def frames_para_b64(frames: list[np.ndarray], step: int = 1) -> list[str]:
    frames_amostrados = frames[::step]
    n = len(frames_amostrados)
    if n == 0:
        return []

    out: list[str | None] = [None] * n
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_ENCODE) as ex:
        futuros = {
            ex.submit(_encode_frame, f): i
            for i, f in enumerate(frames_amostrados)
        }
        for fut in as_completed(futuros):
            out[futuros[fut]] = fut.result()

    return out  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURAS PLOTLY — compatibilidade com resultados consolidados
# ─────────────────────────────────────────────────────────────────────────────
def build_skeleton_fig(skel: np.ndarray, com: np.ndarray | None) -> str:
    fig = go.Figure()

    bone_x, bone_y, bone_z = [], [], []
    for a, b in MEDIAPIPE_CONNECTIONS:
        if a >= len(skel) or b >= len(skel):
            continue
        bone_x.extend([skel[a, 0], skel[b, 0], None])
        bone_y.extend([skel[a, 1], skel[b, 1], None])
        bone_z.extend([skel[a, 2], skel[b, 2], None])

    fig.add_trace(go.Scatter3d(
        x=bone_x, y=bone_y, z=bone_z,
        mode="lines", line=dict(color="#2A3050", width=4),
        showlegend=False, hoverinfo="skip",
    ))

    n_joints  = min(33, len(skel))
    jcolors   = [JOINT_COLORS.get(i, "#ffffff") for i in range(n_joints)]
    hover_texts = [
        f"Junta {i}<br>x={skel[i,0]:.3f} m<br>y={skel[i,1]:.3f} m<br>z={skel[i,2]:.3f} m"
        for i in range(n_joints)
    ]

    fig.add_trace(go.Scatter3d(
        x=skel[:n_joints, 0], y=skel[:n_joints, 1], z=skel[:n_joints, 2],
        mode="markers",
        marker=dict(size=5, color=jcolors, symbol="circle"),
        showlegend=False,
        hovertemplate="%{text}<extra></extra>",
        text=hover_texts,
    ))

    if com is not None:
        fig.add_trace(go.Scatter3d(
            x=[com[0]], y=[com[1]], z=[com[2]],
            mode="markers+text",
            marker=dict(size=10, color="#FF7043", symbol="diamond",
                        line=dict(width=2, color="#FFFFFF")),
            text=["CoM"], textposition="top center",
            textfont=dict(color="#FF7043", size=11, family="DM Mono"),
            name="Centro de Massa",
            hovertemplate=(
                f"CoM<br>x={com[0]:.3f} m<br>"
                f"y={com[1]:.3f} m<br>z={com[2]:.3f} m<extra></extra>"
            ),
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            bgcolor="rgba(24,28,39,0.0)",
            xaxis=dict(title="X (m)", gridcolor="#2A3050", color="#6B7A99", backgroundcolor="rgba(0,0,0,0)", showbackground=True),
            yaxis=dict(title="Y (m)", gridcolor="#2A3050", color="#6B7A99", backgroundcolor="rgba(0,0,0,0)", showbackground=True),
            zaxis=dict(title="Z (m)", gridcolor="#2A3050", color="#6B7A99", backgroundcolor="rgba(0,0,0,0)", showbackground=True),
            aspectmode="data",
        ),
        height=540,
        font=dict(family="DM Sans", color="#6B7A99"),
        legend=dict(bgcolor="rgba(30,34,53,0.8)", bordercolor="#2A3050", font=dict(color="#E8ECF4")),
    )
    return fig.to_json()


def build_com_2d_fig(com_trajetoria: list[np.ndarray]) -> str:
    com_arr = np.array(com_trajetoria)
    fig = go.Figure(go.Scatter(
        x=com_arr[:, 0], y=com_arr[:, 2],
        mode="lines+markers",
        marker=dict(size=3, color=np.arange(len(com_arr)), colorscale="Turbo", showscale=False),
        line=dict(color="#FF7043", width=1.5),
        name="CoM (X × Z)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(24,28,39,1)",
        xaxis=dict(title="X (m)", gridcolor="#2A3050", color="#6B7A99", zerolinecolor="#2A3050"),
        yaxis=dict(title="Z (m)", gridcolor="#2A3050", color="#6B7A99", zerolinecolor="#2A3050"),
        margin=dict(l=40, r=10, t=10, b=40), height=280,
        font=dict(color="#6B7A99", family="DM Sans"),
    )
    return fig.to_json()


def build_calcanhar_fig(skeleton_3d: list[np.ndarray], fps: float) -> str:
    if not skeleton_3d:
        return go.Figure().to_json()

    heel_y     = [float(skel[30, 1]) if len(skel) > 30 else 0.0 for skel in skeleton_3d]
    timestamps = [i / fps for i in range(len(heel_y))]

    fig = go.Figure(go.Scatter(
        x=timestamps, y=heel_y,
        mode="lines", line=dict(color="#00C9A7", width=2),
        name="Calcanhar D (Y)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(24,28,39,1)",
        xaxis=dict(title="Tempo (s)", gridcolor="#2A3050", color="#6B7A99", zerolinecolor="#2A3050"),
        yaxis=dict(title="Y (m)",     gridcolor="#2A3050", color="#6B7A99", zerolinecolor="#2A3050"),
        margin=dict(l=40, r=10, t=10, b=40), height=280,
        font=dict(color="#6B7A99", family="DM Sans"),
    )
    return fig.to_json()


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS DO PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def _reconfigurar_analisador(
    id_paciente: str,
    movimento:   str,
    lado:        str,
    usar_gpu:    bool,
) -> AnalisadorADMWeb:
    an = app_state.analisador
    if an is None:
        raise RuntimeError("AnalisadorADMWeb não inicializado.")

    an.id_paciente    = id_paciente
    an.tipo_movimento = movimento
    an.lado_do_corpo  = lado.lower()
    an.usar_gpu       = usar_gpu
    an.device         = torch.device("cuda" if usar_gpu and torch.cuda.is_available() else "cpu")
    an.angulo_suavizado = None
    an.angulo_anterior  = None
    an.contador_reps    = 0
    an.fase_movimento   = "repouso"

    for filtro in an.filtros_kalman:
        filtro.reset()

    an._dispatch_angulo = an._construir_dispatch_angulo()
    an._configurar_nos_anatomicos()
    return an


def _calcular_metricas_descritivas(res: ResultadoAnalise, movimento: str, step: int) -> dict:
    angulos = [
        m.angle_clinical_deg
        for m in res.frame_measurements
        if m.angle_clinical_deg is not None and m.measurement_valid
    ]
    velocidades = [
        abs(float(m.angular_velocity_deg_s))
        for m in res.frame_measurements
        if m.measurement_valid
        and m.angular_velocity_deg_s is not None
        and np.isfinite(m.angular_velocity_deg_s)
    ]
    fps = res.fps
    total_frames = res.total_frames
    duracao = (total_frames - 1) / fps if fps > 0 and total_frames > 1 else 0.0

    ang_min = float(min(angulos)) if angulos else 0.0
    ang_max = float(max(angulos)) if angulos else 0.0
    ang_media = float(np.mean(angulos)) if angulos else 0.0
    ang_std = float(np.std(angulos)) if angulos else 0.0
    adm_observada = max(0.0, ang_max - ang_min)

    vel_pico = float(max(velocidades)) if velocidades else 0.0
    vel_media = float(np.mean(velocidades)) if velocidades else 0.0

    spec = get_movement_spec(movimento)
    adm_ref = float(spec.reference_rom_deg if spec is not None else 180.0)
    pct_ref = round(adm_observada / adm_ref * 100, 1) if adm_ref else None

    pm = res.parametros_marcha or {}
    frames_no_player = (total_frames + step - 1) // step

    return {
        # aliases de compatibilidade: adm_max agora significa excursão observada
        "adm_max": round(adm_observada, 1),
        "adm_observada": round(adm_observada, 1),
        "adm_min": round(ang_min, 1),
        "adm_media": round(ang_media, 1),
        "adm_std": round(ang_std, 1),
        "angulo_clinico_min": round(ang_min, 1),
        "angulo_clinico_max": round(ang_max, 1),
        "angulo_clinico_medio": round(ang_media, 1),
        "adm_ref": adm_ref,
        "pct_adm_ref": pct_ref,
        "vel_pico": round(vel_pico, 1),
        "vel_media": round(vel_media, 1),
        "total_reps": res.total_reps,
        "fluidez_ia": round(res.fluidez_ia, 1),
        "fps": round(fps, 2),
        "fps_player": round(fps / step, 2),
        "duracao_s": round(duracao, 2),
        "total_frames": total_frames,
        "frames_no_player": frames_no_player,
        "step_subsampling": step,
        "subsampling_aplicado": step > 1,
        "stride_length_m": pm.get("stride_length_m"),
        "step_width_m": pm.get("step_width_m"),
        "cadencia_passos_min": pm.get("cadencia_passos_min"),
        "hardware": "GPU" if torch.cuda.is_available() else "CPU",
    }


def _measurement_event(measurement: FrameMeasurement) -> dict:
    def point(p):
        if p is None:
            return None
        return {"x": p.x, "y": p.y, "confidence": p.confidence}

    def point3d(p):
        if p is None:
            return None
        return {"x": p.x, "y": p.y, "z": p.z}

    cpu_pct = ram_pct = None
    if psutil is not None:
        try:
            cpu_pct = float(psutil.cpu_percent(interval=None))
            ram_pct = float(psutil.virtual_memory().percent)
        except Exception:
            pass

    vram_mb = None
    if torch.cuda.is_available():
        try:
            vram_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
        except Exception:
            pass

    return {
        "type": "analysis.frame",
        "frame": measurement.frame,
        "timestampMs": measurement.timestamp_ms,
        "measurement": {
            "joint": measurement.joint,
            "movement": measurement.movement,
            "angleRawDeg": measurement.angle_raw_deg,
            "angleLandmarkDeg": measurement.angle_landmark_deg,
            "anglePlaneDeg": measurement.angle_plane_deg,
            "angleFilteredDeg": measurement.angle_filtered_deg,
            "angleClinicalDeg": measurement.angle_clinical_deg,
            "angularVelocityDegS": measurement.angular_velocity_deg_s,
            "admDeg": measurement.adm_deg,
            "source": measurement.source,
            "measurementSpace": measurement.measurement_space,
            "measurementPlane": measurement.measurement_plane,
            "planeProjectionApplied": measurement.plane_projection_applied,
            "bodyFrameConfidence": measurement.body_frame_confidence,
            "viewAlignment": measurement.view_alignment,
            "viewDeviationDeg": measurement.view_deviation_deg,
            "outOfPlaneDeg": measurement.out_of_plane_deg,
            "directionRule": measurement.direction_rule,
            "directionAlignment": measurement.direction_alignment,
            "confidence": measurement.confidence,
            "landmarkConfidenceMin": measurement.landmark_confidence_min,
            "landmarkConfidenceMean": measurement.landmark_confidence_mean,
            "quality": measurement.quality,
            "measurementValid": measurement.measurement_valid,
            "qualityFlags": measurement.quality_flags,
            "proximal": point(measurement.proximal),
            "vertex": point(measurement.vertex),
            "distal": point(measurement.distal),
            "proximal3d": point3d(measurement.proximal_3d),
            "vertex3d": point3d(measurement.vertex_3d),
            "distal3d": point3d(measurement.distal_3d),
            "poseLandmarks2d": (
                [point(p) for p in measurement.overlay_landmarks_2d]
                if measurement.overlay_landmarks_2d
                else None
            ),
        },
        "system": {
            "processingFps": measurement.processing_fps,
            "latencyMs": measurement.pipeline_ms,
            "inferenceMs": measurement.inference_ms,
            "cpuPct": cpu_pct,
            "ramPct": ram_pct,
            "gpuPct": None,
            "vramMb": vram_mb,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  WEBSOCKET — progresso e telemetria biomecânica frame a frame
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/analise/{job_id}")
async def ws_progresso(websocket: WebSocket, job_id: str):
    # Preferimos Sec-WebSocket-Protocol para não colocar credenciais na URL,
    # que pode aparecer em access logs de proxies. O query param permanece
    # apenas como fallback de compatibilidade com clientes antigos.
    offered_protocols = [
        item.strip()
        for item in websocket.headers.get("sec-websocket-protocol", "").split(",")
        if item.strip()
    ]
    token = offered_protocols[1] if len(offered_protocols) >= 2 and offered_protocols[0] == "physioai" else ""
    if not token:
        token = websocket.query_params.get("token", "")
    try:
        user = auth_manager.verify_token(token)
    except AuthError:
        await websocket.close(code=4401, reason="Não autenticado")
        return
    owner = app_state.job_owner.get(job_id)
    if owner is None:
        await websocket.close(code=4404, reason="Job não encontrado")
        return
    if owner != user.username and user.role != "admin":
        await websocket.close(code=4403, reason="Acesso negado")
        return

    await websocket.accept(subprotocol="physioai" if offered_protocols and offered_protocols[0] == "physioai" else None)
    metrics.inc_gauge("physioai_websocket_connections", 1)
    metrics.inc("physioai_websocket_connections_total")
    # Reutiliza a fila criada pelo POST, eliminando a race condition POST × WS.
    queue = app_state.progress_queues.setdefault(job_id, asyncio.Queue(maxsize=WS_QUEUE_MAX))

    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=300.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "error", "detail": "Timeout de 5 minutos."})
                break

            await websocket.send_json(msg)
            if msg.get("type") in ("done", "error", "cancelled"):
                break
    except WebSocketDisconnect:
        logger.info("WebSocket desconectado: job_id=%s", job_id)
    finally:
        metrics.inc_gauge("physioai_websocket_connections", -1)


# ─────────────────────────────────────────────────────────────────────────────
#  TASK DE BACKGROUND — processamento e persistência server-side
# ─────────────────────────────────────────────────────────────────────────────
async def _pipeline_background(
    job_id: str,
    tmp_a: str,
    tmp_b: str | None,
    tmp_calib: str | None,
    tmp_depth: str | None,
    id_paciente: str,
    movimento: str,
    lado: str,
    offset_ms: int,
    tipo_sensor: str,
    modo_gpu: bool,
    max_player_frames: int,
    retornar_frames: bool,
    gerar_graficos: bool,
    retornar_skeleton: bool,
) -> None:
    loop = asyncio.get_running_loop()
    cancel_flag = app_state.cancel_flags.setdefault(job_id, threading.Event())

    def _enfileirar(msg: dict) -> None:
        # A fila é limitada para impedir crescimento de RAM caso o cliente
        # desconecte durante vídeos longos. Eventos frame/progress antigos podem
        # ser descartados; eventos terminais sempre ganham espaço.
        queue = app_state.progress_queues.get(job_id)
        if queue is None:
            return

        def _put_bounded() -> None:
            try:
                queue.put_nowait(msg)
                return
            except asyncio.QueueFull:
                pass

            event_type = str(msg.get("type") or "")
            if event_type in {"analysis.frame", "progress"}:
                metrics.inc("physioai_websocket_events_dropped_total", type=event_type)
                return

            # done/error/cancelled não podem ser perdidos. Remove o item mais
            # antigo até haver espaço para o evento terminal.
            while queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            try:
                queue.put_nowait(msg)
            except asyncio.QueueFull:
                metrics.inc("physioai_websocket_events_dropped_total", type="terminal")

        loop.call_soon_threadsafe(_put_bounded)

    def _progress_callback(pct: float) -> None:
        _enfileirar({"type": "progress", "pct": round(min(max(pct, 0.0), 1.0), 4)})

    def _telemetry_callback(measurement: FrameMeasurement) -> None:
        metrics.inc("physioai_analysis_frames_total", movement=movimento, side=lado)
        _enfileirar(_measurement_event(measurement))

    job_started = time.perf_counter()
    acquired = False
    active_counted = False
    try:
        wait_started = time.perf_counter()
        await app_state.semaphore.acquire()
        acquired = True
        queue_wait = time.perf_counter() - wait_started
        metrics.inc_gauge("physioai_jobs_pending", -1)
        metrics.inc_gauge("physioai_jobs_active", 1)
        active_counted = True
        metrics.observe("physioai_job_queue_wait", queue_wait, movement=movimento)
        if cancel_flag.is_set():
            raise AnaliseCancelada()
        try:
            analisador = _reconfigurar_analisador(id_paciente, movimento, lado, modo_gpu)
            resultado: ResultadoAnalise = await loop.run_in_executor(
                None,
                lambda: analisador.processar_multicamera(
                    path_a=tmp_a,
                    path_b=tmp_b,
                    path_calib=tmp_calib,
                    path_depth=tmp_depth,
                    tipo_sensor=tipo_sensor,
                    offset_ms=offset_ms,
                    progress_callback=_progress_callback,
                    telemetry_callback=_telemetry_callback,
                    cancel_callback=cancel_flag.is_set,
                    armazenar_frames=retornar_frames,
                ),
            )
        finally:
            app_state.semaphore.release()
            acquired = False
            if active_counted:
                metrics.inc_gauge("physioai_jobs_active", -1)
                active_counted = False

        total_frames = resultado.total_frames
        fps = resultado.fps
        step = 1
        if total_frames > max_player_frames:
            step = max(2, round(total_frames / max_player_frames))

        frames_b64: list[str] = []
        if retornar_frames:
            frames_b64 = await loop.run_in_executor(
                None, lambda: frames_para_b64(resultado.frames_a, step=step)
            )

        metricas = _calcular_metricas_descritivas(resultado, movimento, step)

        # Persistência científica: cada frame fica disponível para revisão e
        # exportação mesmo após o payload HTTP expirar da memória.
        measurement_rows = [m.to_dict() for m in resultado.frame_measurements]
        measurement_store.save_many(job_id, measurement_rows)
        valid_frames = sum(1 for m in resultado.frame_measurements if m.measurement_valid)
        result_sha256 = canonical_json_sha256({
            "job_id": job_id,
            "metrics": metricas,
            "measurements": measurement_rows,
        })

        # Converte o dataclass para o contrato camelCase do frontend.
        measurements_player = [
            _measurement_event(resultado.frame_measurements[i]) | {"type": "analysis.frame"}
            for i in range(0, len(resultado.frame_measurements), step)
        ]

        skeleton_3d_serialized = (
            [skel.tolist() for skel in resultado.skeleton_3d[::step]]
            if retornar_skeleton
            else []
        )

        fig_skeleton_json = fig_com2d_json = fig_calcanhar_json = None
        if gerar_graficos:
            if resultado.skeleton_3d:
                mid_idx = len(resultado.skeleton_3d) // 2
                com_mid = resultado.com_trajetoria[mid_idx] if resultado.com_trajetoria else None
                fig_skeleton_json = build_skeleton_fig(resultado.skeleton_3d[mid_idx], com_mid)
            if resultado.com_trajetoria:
                fig_com2d_json = build_com_2d_fig(resultado.com_trajetoria)
            if resultado.skeleton_3d and fps > 0:
                fig_calcanhar_json = build_calcanhar_fig(resultado.skeleton_3d, fps)

        payload = {
            "job_id": job_id,
            "status": "ok",
            "protocol_version": PROTOCOL_VERSION,
            "clinical_spec_sha256": CLINICAL_SPEC_SHA256,
            "result_sha256": result_sha256,
            **metricas,
            "angulos": [round(a, 3) for a in resultado.angulos_filtrados],
            "angulos_brutos": [round(a, 3) for a in resultado.angulos_brutos],
            "angulos_geometricos_filtrados": [round(a, 3) for a in resultado.angulos_geometricos_filtrados],
            "velocidades": [round(v, 3) for v in resultado.velocidades],
            "angulos_player": [round(a, 3) for a in resultado.angulos_filtrados[::step]],
            "velocidades_player": [round(v, 3) for v in resultado.velocidades[::step]],
            "frame_measurements": measurements_player,
            "parametros_marcha": resultado.parametros_marcha or {},
            "frames_b64": frames_b64,
            "skeleton_3d": skeleton_3d_serialized,
            "fig_skeleton_json": fig_skeleton_json,
            "fig_com2d_json": fig_com2d_json,
            "fig_calcanhar_json": fig_calcanhar_json,
        }

        app_state.resultados[job_id] = payload
        app_state.result_created[job_id] = time.monotonic()
        session_store.finish(
            job_id,
            status="completed",
            total_frames=resultado.total_frames,
            valid_frames=valid_frames,
            fps=resultado.fps,
            adm_deg=metricas.get("adm_observada") or metricas.get("adm_max"),
            total_reps=resultado.total_reps,
            result_sha256=result_sha256,
        )
        _enfileirar({"type": "done", "job_id": job_id})
        metrics.inc("physioai_jobs_total", status="completed", movement=movimento)
        logger.info("Job %s concluído com sucesso.", job_id, extra={"event": "analysis.completed", "job_id": job_id, "movement": movimento})

    except AnaliseCancelada:
        app_state.resultados.pop(job_id, None)
        app_state.result_created.pop(job_id, None)
        session_store.finish(job_id, status="cancelled")
        _enfileirar({"type": "cancelled", "job_id": job_id})
        metrics.inc("physioai_jobs_total", status="cancelled", movement=movimento)
        logger.info("Job %s cancelado.", job_id, extra={"event": "analysis.cancelled", "job_id": job_id, "movement": movimento})
    except Exception as exc:
        logger.error("=== ERRO CRÍTICO NO PROCESSAMENTO DO JOB %s ===", job_id)
        traceback.print_exc()
        app_state.resultados[job_id] = exc
        app_state.result_created[job_id] = time.monotonic()
        session_store.finish(job_id, status="error")
        metrics.inc("physioai_jobs_total", status="error", movement=movimento)
        _enfileirar({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})
    finally:
        if acquired:
            app_state.semaphore.release()
        if active_counted:
            metrics.inc_gauge("physioai_jobs_active", -1)
        metrics.observe("physioai_job_total_duration", time.perf_counter() - job_started, movement=movimento)
        app_state.cancel_flags.pop(job_id, None)
        for path in [tmp_a, tmp_b, tmp_calib, tmp_depth]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError as exc:
                    logger.warning("Falha ao deletar temporário '%s': %s", path, exc)
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
#  PREFLIGHT DE CAPTURA — valida o vídeo antes do processamento completo
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/capture/preflight")
async def capture_preflight(
    request: Request,
    video_a:     Annotated[UploadFile, File()],
    id_paciente: Annotated[str, Form()] = "pac-001",
    movimento:   Annotated[str, Form()] = "Flexão de Cotovelo",
    lado:        Annotated[str, Form()] = "direito",
):
    if not id_paciente.strip():
        raise HTTPException(status_code=422, detail="id_paciente não pode ser vazio.")
    if lado not in {"direito", "esquerdo"}:
        raise HTTPException(status_code=422, detail="lado deve ser 'direito' ou 'esquerdo'.")
    if get_movement_spec(movimento) is None:
        raise HTTPException(status_code=422, detail="Movimento não habilitado no protocolo clínico.")

    user = _request_user(request)
    patient_pseudonym = auth_manager.pseudonymize_patient_id(id_paciente)

    tmp_path: str | None = None
    preflight_id = str(uuid.uuid4())
    try:
        tmp_path, artifact = await _stage_upload(video_a, suffix=".mp4", role="video_a")
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        async with app_state.preflight_semaphore:
            report: CapturePreflightReport = await loop.run_in_executor(
                None,
                lambda: evaluate_capture_preflight(
                    tmp_path,
                    movimento,
                    lado,
                    max_samples=PREFLIGHT_MAX_SAMPLES,
                ),
            )
        elapsed = time.perf_counter() - started
        report_dict = report.to_dict()
        record = _PreflightRecord(
            preflight_id=preflight_id,
            owner=user.username,
            patient_pseudonym=patient_pseudonym,
            movement=movimento,
            side=lado,
            path=tmp_path,
            artifact=artifact,
            report=report_dict,
            created_at=time.monotonic(),
        )
        app_state.preflights[preflight_id] = record
        metrics.inc("physioai_preflights_total", status=report.status, movement=movimento)
        metrics.observe("physioai_preflight_duration", elapsed, movement=movimento)
        audit_store.log(
            actor=user.username,
            role=user.role,
            action="capture.preflight",
            resource_type="preflight",
            resource_id=preflight_id,
            remote_ip=_remote_ip(request),
            metadata={
                "patient": patient_pseudonym,
                "movement": movimento,
                "side": lado,
                "status": report.status,
                "score": report.score,
                "video_sha256": artifact["sha256"],
            },
        )
        return {
            "preflight_id": preflight_id,
            "patient_pseudonym": patient_pseudonym,
            "expires_in_seconds": PREFLIGHT_TTL_SECONDS,
            **report_dict,
        }
    except Exception:
        if tmp_path and os.path.exists(tmp_path) and preflight_id not in app_state.preflights:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


@app.delete("/api/capture/preflight/{preflight_id}")
async def discard_preflight(preflight_id: str, request: Request):
    user = _request_user(request)
    record = app_state.preflights.get(preflight_id)
    if record is None:
        return {"status": "gone"}
    if record.owner != user.username and user.role != "admin":
        raise HTTPException(status_code=403, detail="Preflight pertence a outro usuário")
    app_state.preflights.pop(preflight_id, None)
    if record.path and os.path.exists(record.path):
        try:
            os.unlink(record.path)
        except OSError:
            pass
    audit_store.log(
        actor=user.username, role=user.role, action="capture.preflight.discard",
        resource_type="preflight", resource_id=preflight_id, remote_ip=_remote_ip(request),
    )
    return {"status": "discarded"}


# ─────────────────────────────────────────────────────────────────────────────
#  ENDPOINT DE ANÁLISE — POST 202 + execução assíncrona
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/analise/processar", status_code=202)
async def processar_analise(
    request: Request,
    video_a:           Annotated[UploadFile | None, File()] = None,
    video_b:           Annotated[UploadFile | None, File()] = None,
    calib_file:        Annotated[UploadFile | None, File()] = None,
    depth_file:        Annotated[UploadFile | None, File()] = None,
    id_paciente:       Annotated[str,  Form()] = "pac-001",
    movimento:         Annotated[str,  Form()] = "Flexão de Cotovelo",
    lado:              Annotated[str,  Form()] = "direito",
    offset_ms:         Annotated[int,  Form()] = 0,
    tipo_sensor:       Annotated[str,  Form()] = "Nenhum",
    modo_gpu:          Annotated[bool, Form()] = True,
    max_player_frames: Annotated[int,  Form()] = MAX_PLAYER_FRAMES,
    retornar_frames:   Annotated[bool, Form()] = False,
    gerar_graficos:    Annotated[bool, Form()] = False,
    retornar_skeleton: Annotated[bool, Form()] = False,
    job_id:            Annotated[str | None, Form()] = None,
    preflight_id:       Annotated[str | None, Form()] = None,
    preflight_override_reason: Annotated[str | None, Form()] = None,
):
    if not id_paciente.strip():
        raise HTTPException(status_code=422, detail="id_paciente não pode ser vazio.")
    if lado not in {"direito", "esquerdo"}:
        raise HTTPException(status_code=422, detail="lado deve ser 'direito' ou 'esquerdo'.")
    if get_movement_spec(movimento) is None:
        raise HTTPException(
            status_code=422,
            detail="Movimento ainda não habilitado na via clínica pré-clínica.",
        )
    if app_state.analisador is None:
        raise HTTPException(status_code=503, detail="Servidor não inicializado.")
    if depth_file is not None or tipo_sensor != "Nenhum":
        raise HTTPException(
            status_code=422,
            detail="Fusão de profundidade/LiDAR permanece experimental e não está habilitada na via clínica.",
        )
    if (video_b is not None or calib_file is not None) and not ENABLE_EXPERIMENTAL_INPUTS:
        raise HTTPException(
            status_code=422,
            detail="Entrada estéreo é experimental. Defina PHYSIOAI_ENABLE_EXPERIMENTAL_INPUTS=1 apenas para pesquisa controlada.",
        )
    if ENABLE_EXPERIMENTAL_INPUTS and ((video_b is None) != (calib_file is None)):
        raise HTTPException(
            status_code=422,
            detail="Modo estéreo experimental requer vídeo B e arquivo de calibração juntos.",
        )

    inflight = metrics.get_gauge("physioai_jobs_pending") + metrics.get_gauge("physioai_jobs_active")
    if inflight >= MAX_INFLIGHT_JOBS:
        metrics.inc("physioai_jobs_rejected_total", reason="capacity")
        raise HTTPException(
            status_code=429,
            detail=f"Capacidade de análise atingida ({MAX_INFLIGHT_JOBS} jobs em processamento/fila). Tente novamente após a conclusão de uma sessão.",
        )

    user = _request_user(request)
    patient_pseudonym = auth_manager.pseudonymize_patient_id(id_paciente)

    if video_a is not None and preflight_id is not None:
        raise HTTPException(status_code=422, detail="Informe video_a ou preflight_id, nunca ambos.")
    if video_a is None and preflight_id is None:
        raise HTTPException(status_code=422, detail="É necessário executar o preflight ou enviar video_a.")
    if REQUIRE_PREFLIGHT and preflight_id is None:
        raise HTTPException(
            status_code=428,
            detail="Preflight de captura obrigatório. Execute /api/capture/preflight antes da análise.",
        )

    preflight_record: _PreflightRecord | None = None
    if preflight_id is not None:
        try:
            normalized_preflight_id = str(uuid.UUID(preflight_id))
        except (ValueError, AttributeError) as exc:
            raise HTTPException(status_code=422, detail="preflight_id deve ser um UUID válido.") from exc
        preflight_record = app_state.preflights.get(normalized_preflight_id)
        if preflight_record is None:
            raise HTTPException(status_code=404, detail="Preflight não encontrado ou expirado.")
        if time.monotonic() - preflight_record.created_at > PREFLIGHT_TTL_SECONDS:
            app_state.preflights.pop(normalized_preflight_id, None)
            if os.path.exists(preflight_record.path):
                try:
                    os.unlink(preflight_record.path)
                except OSError:
                    pass
            raise HTTPException(status_code=410, detail="Preflight expirado. Execute uma nova verificação de captura.")
        if preflight_record.owner != user.username and user.role != "admin":
            raise HTTPException(status_code=403, detail="Preflight pertence a outro usuário.")
        if preflight_record.claimed:
            raise HTTPException(status_code=409, detail="Preflight já consumido por outra análise.")
        if preflight_record.patient_pseudonym != patient_pseudonym:
            raise HTTPException(status_code=409, detail="Preflight pertence a outro paciente/sessão.")
        if preflight_record.movement != movimento or preflight_record.side != lado:
            raise HTTPException(status_code=409, detail="Movimento/lado diferem do preflight. Execute nova verificação.")
        if not bool(preflight_record.report.get("can_analyze")):
            reason = (preflight_override_reason or "").strip()
            if len(reason) < 12:
                raise HTTPException(
                    status_code=422,
                    detail="Captura reprovada. Para uso experimental, administrador deve justificar o override (mínimo 12 caracteres).",
                )
            _require_admin(user)
    if job_id:
        try:
            effective_job_id = str(uuid.UUID(job_id))
        except (ValueError, AttributeError) as exc:
            raise HTTPException(status_code=422, detail="job_id deve ser um UUID válido.") from exc
    else:
        effective_job_id = str(uuid.uuid4())
    if app_state.job_owner.get(effective_job_id) or session_store.get_by_job(effective_job_id):
        raise HTTPException(status_code=409, detail="job_id já existe")

    # Reserva capacidade antes do primeiro await do upload.
    metrics.inc_gauge("physioai_jobs_pending", 1)
    tmp_a = tmp_b = tmp_calib = tmp_depth = None
    staged_paths: list[str] = []
    accepted = False
    session_started = False
    try:
        if preflight_record is not None:
            # O preflight transfere a posse do temporário para o job. O arquivo
            # não é enviado novamente pelo navegador e será removido no finally
            # do pipeline, exatamente como um upload convencional.
            claimed_id = preflight_record.preflight_id
            app_state.preflights.pop(claimed_id, None)
            preflight_record.claimed = True
            tmp_a = preflight_record.path
            art_a = dict(preflight_record.artifact)
            staged_paths.append(tmp_a)
            artifacts = [art_a]
        else:
            assert video_a is not None
            tmp_a, art_a = await _stage_upload(video_a, suffix=".mp4", role="video_a")
            staged_paths.append(tmp_a)
            artifacts = [art_a]

        if video_b is not None:
            tmp_b, artifact = await _stage_upload(video_b, suffix=".mp4", role="video_b")
            staged_paths.append(tmp_b)
            artifacts.append(artifact)

        if calib_file is not None:
            tmp_calib, artifact = await _stage_upload(calib_file, suffix=".npz", role="calibration")
            staged_paths.append(tmp_calib)
            artifacts.append(artifact)

        if depth_file is not None:
            suffix = os.path.splitext(depth_file.filename or "")[-1] or ".bin"
            tmp_depth, artifact = await _stage_upload(depth_file, suffix=suffix, role="depth")
            staged_paths.append(tmp_depth)
            artifacts.append(artifact)

        total_upload_bytes = sum(int(item["size_bytes"]) for item in artifacts)
        if total_upload_bytes > MAX_TOTAL_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Uploads da sessão excedem o limite total de {MAX_TOTAL_UPLOAD_MB} MB.",
            )

        provenance = {
            "build": BUILD_INFO.public(),
            "protocol_version": PROTOCOL_VERSION,
            "clinical_spec_sha256": CLINICAL_SPEC_SHA256,
            "clinical_spec": clinical_spec_snapshot(),
            "model": MODEL_PROVENANCE,
            "inputs": artifacts,
            "capture_preflight": ({
                "preflight_id": preflight_record.preflight_id,
                "report": preflight_record.report,
                "override_reason": (preflight_override_reason or "").strip() or None,
            } if preflight_record is not None else {
                "preflight_id": None,
                "report": None,
                "override_reason": None,
            }),
            "analysis_config": {
                "movement": movimento,
                "side": lado,
                "offset_ms": offset_ms,
                "sensor_type": tipo_sensor,
                "gpu_requested": modo_gpu,
                "return_frames": retornar_frames,
                "generate_plots": gerar_graficos,
                "return_skeleton": retornar_skeleton,
            },
        }

        app_state.job_owner[effective_job_id] = user.username
        session_store.start(
            job_id=effective_job_id,
            patient_pseudonym=patient_pseudonym,
            evaluator_id=user.username,
            movement=movimento,
            side=lado,
            software_version=APP_VERSION,
            protocol_version=PROTOCOL_VERSION,
            clinical_spec_sha256=CLINICAL_SPEC_SHA256,
            provenance=provenance,
        )
        session_started = True
        audit_store.log(
            actor=user.username, role=user.role, action="analysis.start",
            resource_type="job", resource_id=effective_job_id, remote_ip=_remote_ip(request),
            metadata={
                "patient": patient_pseudonym,
                "movement": movimento,
                "side": lado,
                "video_a_sha256": art_a["sha256"],
                "protocol_version": PROTOCOL_VERSION,
                "preflight_id": preflight_record.preflight_id if preflight_record else None,
                "preflight_status": preflight_record.report.get("status") if preflight_record else "skipped",
                "preflight_override": bool((preflight_override_reason or "").strip()),
            },
        )

        app_state.progress_queues.setdefault(effective_job_id, asyncio.Queue(maxsize=WS_QUEUE_MAX))
        app_state.cancel_flags[effective_job_id] = threading.Event()
        metrics.inc("physioai_jobs_total", status="accepted", movement=movimento)

        asyncio.create_task(
            _pipeline_background(
                job_id=effective_job_id,
                tmp_a=tmp_a,
                tmp_b=tmp_b,
                tmp_calib=tmp_calib,
                tmp_depth=tmp_depth,
                id_paciente=patient_pseudonym,
                movimento=movimento,
                lado=lado,
                offset_ms=offset_ms,
                tipo_sensor=tipo_sensor,
                modo_gpu=modo_gpu,
                max_player_frames=max_player_frames,
                retornar_frames=retornar_frames,
                gerar_graficos=gerar_graficos,
                retornar_skeleton=retornar_skeleton,
            )
        )
        accepted = True
        return JSONResponse(
            content={
                "job_id": effective_job_id,
                "status": "accepted",
                "patient_pseudonym": patient_pseudonym,
                "protocol_version": PROTOCOL_VERSION,
                "preflight_id": preflight_record.preflight_id if preflight_record else None,
            },
            status_code=202,
        )
    except Exception:
        if not accepted:
            metrics.inc_gauge("physioai_jobs_pending", -1)
            app_state.progress_queues.pop(effective_job_id, None)
            app_state.cancel_flags.pop(effective_job_id, None)
            app_state.job_owner.pop(effective_job_id, None)
            if session_started:
                session_store.finish(effective_job_id, status="error")
            for path in staged_paths:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
        raise


@app.post("/api/analise/cancelar/{job_id}")
async def cancelar_analise(job_id: str, request: Request):
    user = _request_user(request)
    _assert_job_access(job_id, user)
    flag = app_state.cancel_flags.get(job_id)
    if flag is None:
        raise HTTPException(status_code=404, detail="Job não encontrado ou já finalizado.")
    flag.set()
    audit_store.log(actor=user.username, role=user.role, action="analysis.cancel",
                    resource_type="job", resource_id=job_id, remote_ip=_remote_ip(request))
    return {"status": "cancelling", "job_id": job_id}


# ─────────────────────────────────────────────────────────────────────────────
#  ENDPOINT 2 — GET resultado
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/analise/resultado/{job_id}")
async def get_resultado(job_id: str, request: Request):
    user = _request_user(request)
    _assert_job_access(job_id, user)
    if job_id not in app_state.resultados:
        raise HTTPException(status_code=404, detail="Job não encontrado ou ainda em processamento.")

    resultado = app_state.resultados.pop(job_id)
    app_state.progress_queues.pop(job_id, None)
    app_state.cancel_flags.pop(job_id, None)
    app_state.result_created.pop(job_id, None)
    app_state.job_owner.pop(job_id, None)
    audit_store.log(actor=user.username, role=user.role, action="analysis.result.read",
                    resource_type="job", resource_id=job_id, remote_ip=_remote_ip(request))

    if isinstance(resultado, Exception):
        raise HTTPException(
            status_code=500,
            detail=f"Processamento falhou: {type(resultado).__name__} — {resultado}",
        )

    return JSONResponse(content=resultado)


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDAÇÃO CLÍNICA — revisão manual e referência goniométrica
# ─────────────────────────────────────────────────────────────────────────────
class ValidationPoint(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ValidationReviewIn(BaseModel):
    job_id: str = Field(min_length=1, max_length=128)
    # Campos abaixo continuam aceitos por compatibilidade do cliente, porém
    # paciente/movimento/lado/ângulo automático são reobtidos server-side.
    patient_id: str | None = Field(default=None, max_length=128)
    evaluator_id: str | None = Field(default=None, max_length=128)
    movement: str | None = Field(default=None, max_length=128)
    side: str | None = Field(default=None, max_length=32)
    frame: int = Field(ge=0)
    timestamp_ms: float | None = Field(default=None, ge=0.0)
    automatic_angle_deg: float | None = Field(default=None, ge=0.0, le=180.0)
    manual_angle_deg: float | None = Field(default=None, ge=0.0, le=180.0)
    reference_angle_deg: float | None = Field(default=None, ge=0.0, le=180.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = Field(default=None, max_length=32)
    proximal: ValidationPoint | None = None
    vertex: ValidationPoint | None = None
    distal: ValidationPoint | None = None
    notes: str = Field(default="", max_length=2000)


@app.post("/api/validacao/revisoes", status_code=201)
async def salvar_revisao_validacao(review: ValidationReviewIn, request: Request):
    user = _request_user(request)
    _assert_job_access(review.job_id, user)

    session = session_store.get_by_job(review.job_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    measurement = measurement_store.get(review.job_id, review.frame)
    if measurement is None:
        raise HTTPException(status_code=422, detail="Frame não encontrado na telemetria persistida")
    automatic = measurement.get("angle_clinical_deg")
    if automatic is None:
        raise HTTPException(status_code=422, detail="Frame sem ângulo clínico automático")

    points = [review.proximal, review.vertex, review.distal]
    provided_count = sum(point is not None for point in points)
    if provided_count not in (0, 3):
        raise HTTPException(
            status_code=422,
            detail="Revisão manual exige os três pontos: proximal, vértice e distal.",
        )

    manual_angle: float | None = None
    proximal = vertex = distal = None
    if provided_count == 3:
        proximal = review.proximal.model_dump() if hasattr(review.proximal, "model_dump") else review.proximal.dict()
        vertex = review.vertex.model_dump() if hasattr(review.vertex, "model_dump") else review.vertex.dict()
        distal = review.distal.model_dump() if hasattr(review.distal, "model_dump") else review.distal.dict()
        try:
            manual_angle = clinical_angle_from_points_2d(
                str(session["movement"]), proximal, vertex, distal
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    payload = {
        "job_id": review.job_id,
        "patient_id": session["patient_pseudonym"],
        "evaluator_id": user.username,
        "movement": session["movement"],
        "side": session["side"],
        "frame": review.frame,
        "timestamp_ms": measurement["timestamp_ms"],
        "automatic_angle_deg": automatic,
        "manual_angle_deg": manual_angle,
        # Fase 11: referência física é registrada somente em ensaios de um
        # protocolo congelado (/api/estudos/{id}/ensaios). O campo legado
        # permanece aceito no schema para clientes antigos, mas é ignorado.
        "reference_angle_deg": None,
        "manual_method": "image_2d_three_point" if manual_angle is not None else None,
        "reference_method": None,
        "confidence": measurement.get("landmark_confidence_min"),
        "source": measurement.get("source"),
        "proximal": proximal,
        "vertex": vertex,
        "distal": distal,
        "notes": review.notes,
    }
    try:
        saved = validation_store.save_review(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    audit_store.log(
        actor=user.username,
        role=user.role,
        action="validation.review.create",
        resource_type="validation_review",
        resource_id=saved["id"],
        remote_ip=_remote_ip(request),
        metadata={
            "job_id": review.job_id,
            "frame": review.frame,
            "patient": session["patient_pseudonym"],
            "movement": session["movement"],
            "side": session["side"],
        },
    )
    return saved


@app.get("/api/validacao/revisoes")
async def listar_revisoes_validacao(
    limit: int = Query(default=20, ge=1, le=200),
):
    return {"items": validation_store.recent(limit)}


@app.get("/api/validacao/stats")
async def estatisticas_validacao(
    movement: str | None = None,
    side: str | None = None,
):
    return validation_store.stats(movement=movement, side=side)


@app.get("/api/validacao/relatorio")
async def relatorio_validacao(
    movement: str | None = None,
    side: str | None = None,
    max_points: int = Query(default=2000, ge=1, le=10000),
):
    return validation_store.report(movement=movement, side=side, max_points=max_points)


@app.get("/api/validacao/export.csv")
async def exportar_validacao_csv(
    movement: str | None = None,
    side: str | None = None,
):
    rows = validation_store.export_rows(movement=movement, side=side)
    output = io.StringIO(newline="")
    columns = [
        "id", "schema_version", "created_at", "job_id", "patient_id", "evaluator_id",
        "movement", "side", "frame", "timestamp_ms", "automatic_angle_deg",
        "manual_angle_deg", "reference_angle_deg", "manual_method",
        "reference_method", "error_vs_manual_deg", "error_vs_reference_deg",
        "confidence", "source", "notes",
    ]
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    filename = "physioai_validacao.csv"
    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )




# ─────────────────────────────────────────────────────────────────────────────
#  ESTUDOS DE VALIDAÇÃO — protocolo congelado + ensaios independentes
# ─────────────────────────────────────────────────────────────────────────────
class ValidationStudyCreateIn(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    protocol_version: str = Field(default=PROTOCOL_VERSION, min_length=1, max_length=32)
    movement: str = Field(min_length=1, max_length=128)
    side: str = Field(default="ambos", max_length=16)
    target_participants: int | None = Field(default=None, gt=0, le=10000)
    repetitions_per_participant: int | None = Field(default=None, gt=0, le=1000)
    min_landmark_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    require_observed: bool = True
    require_valid_measurement: bool = True
    reference_method: str = Field(default="physical_goniometer", max_length=64)
    reference_device_model: str | None = Field(default=None, max_length=128)
    reference_device_resolution_deg: float | None = Field(default=None, gt=0.0, le=10.0)
    camera_setup: dict = Field(default_factory=dict)
    protocol_notes: str = Field(default="", max_length=5000)


class ValidationTrialCreateIn(BaseModel):
    job_id: str = Field(min_length=1, max_length=128)
    frame: int = Field(ge=0)
    reference_angle_deg: float = Field(ge=0.0, le=180.0)
    reference_device_serial: str | None = Field(default=None, max_length=128)
    notes: str = Field(default="", max_length=2000)
    manual_exclusion_reason: str | None = Field(default=None, max_length=256)

class FormalValidationRunCreateIn(BaseModel):
    ethics_status: str = Field(default="pending", max_length=32)
    ethics_protocol_id: str | None = Field(default=None, max_length=128)
    reference_calibration_date: str | None = Field(default=None, max_length=16)
    reference_calibration_due_date: str | None = Field(default=None, max_length=16)
    notes: str = Field(default="", max_length=3000)


@app.post("/api/estudos", status_code=201)
async def criar_estudo_validacao(payload: ValidationStudyCreateIn, request: Request):
    user = _request_user(request)
    _require_admin(user)
    if get_movement_spec(payload.movement) is None:
        raise HTTPException(status_code=422, detail="Movimento clínico não suportado")
    try:
        data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        data["created_by"] = user.username
        created = study_store.create_study(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    audit_store.log(
        actor=user.username, role=user.role, action="validation.study.create",
        resource_type="validation_study", resource_id=created["id"],
        remote_ip=_remote_ip(request), metadata={"code": created["code"]},
    )
    return created


@app.get("/api/estudos")
async def listar_estudos_validacao(limit: int = Query(default=100, ge=1, le=500)):
    return {"items": study_store.list_studies(limit)}


@app.get("/api/estudos/{study_id}")
async def obter_estudo_validacao(study_id: str):
    try:
        return study_store.get_study(study_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Estudo não encontrado") from exc


@app.post("/api/estudos/{study_id}/freeze")
async def congelar_estudo_validacao(study_id: str, request: Request):
    user = _request_user(request)
    _require_admin(user)
    try:
        study = study_store.freeze(study_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Estudo não encontrado") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    audit_store.log(
        actor=user.username, role=user.role, action="validation.study.freeze",
        resource_type="validation_study", resource_id=study_id,
        remote_ip=_remote_ip(request), metadata={"protocol_sha256": study.get("protocol_sha256")},
    )
    return study


@app.post("/api/estudos/{study_id}/close")
async def fechar_estudo_validacao(study_id: str, request: Request):
    user = _request_user(request)
    _require_admin(user)
    try:
        study = study_store.close(study_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Estudo não encontrado") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    audit_store.log(
        actor=user.username, role=user.role, action="validation.study.close",
        resource_type="validation_study", resource_id=study_id,
        remote_ip=_remote_ip(request),
    )
    return study


@app.post("/api/estudos/{study_id}/ensaios", status_code=201)
async def registrar_ensaio_validacao(
    study_id: str, payload: ValidationTrialCreateIn, request: Request
):
    user = _request_user(request)
    _assert_job_access(payload.job_id, user)
    try:
        study = study_store.get_study(study_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Estudo não encontrado") from exc
    if study["status"] != "frozen":
        raise HTTPException(status_code=409, detail="O protocolo precisa estar congelado")

    session = session_store.get_by_job(payload.job_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    measurement = measurement_store.get(payload.job_id, payload.frame)
    if measurement is None:
        raise HTTPException(status_code=422, detail="Frame não encontrado na telemetria persistida")
    automatic = measurement.get("angle_clinical_deg")
    if automatic is None:
        raise HTTPException(status_code=422, detail="Frame sem ângulo clínico automático")

    try:
        trial = study_store.add_trial({
            "study_id": study_id,
            "job_id": payload.job_id,
            "patient_id": session["patient_pseudonym"],
            "evaluator_id": user.username,
            "reference_assessor_id": user.username,
            "movement": session["movement"],
            "side": session["side"],
            "frame": payload.frame,
            "timestamp_ms": measurement["timestamp_ms"],
            "automatic_angle_deg": automatic,
            "reference_angle_deg": payload.reference_angle_deg,
            "reference_method": study["reference_method"],
            "reference_device_model": study.get("reference_device_model"),
            "reference_device_serial": payload.reference_device_serial,
            "landmark_confidence_min": measurement.get("landmark_confidence_min"),
            "body_frame_confidence": measurement.get("body_frame_confidence"),
            "measurement_plane": measurement.get("measurement_plane"),
            "view_alignment": measurement.get("view_alignment"),
            "view_deviation_deg": measurement.get("view_deviation_deg"),
            "out_of_plane_deg": measurement.get("out_of_plane_deg"),
            "direction_rule": measurement.get("direction_rule"),
            "direction_alignment": measurement.get("direction_alignment"),
            "source": measurement.get("source"),
            "measurement_valid": measurement.get("measurement_valid"),
            "quality": measurement.get("quality"),
            "manual_exclusion_reason": payload.manual_exclusion_reason,
            "notes": payload.notes,
        })
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    audit_store.log(
        actor=user.username, role=user.role, action="validation.trial.create",
        resource_type="validation_trial", resource_id=trial["id"],
        remote_ip=_remote_ip(request),
        metadata={
            "study_id": study_id, "job_id": payload.job_id, "frame": payload.frame,
            "included": trial["included"], "exclusion_reason": trial.get("exclusion_reason"),
        },
    )
    return trial


@app.get("/api/estudos/{study_id}/ensaios")
async def listar_ensaios_validacao(study_id: str, included_only: bool = False):
    try:
        study_store.get_study(study_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Estudo não encontrado") from exc
    return {"items": study_store.trials(study_id, included_only=included_only)}


@app.get("/api/estudos/{study_id}/relatorio")
async def relatorio_estudo_validacao(
    study_id: str,
    bootstrap_iterations: int = Query(default=1000, ge=200, le=10000),
):
    try:
        return study_store.report(study_id, bootstrap_iterations=bootstrap_iterations)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Estudo não encontrado") from exc


@app.get("/api/estudos/{study_id}/export.csv")
async def exportar_estudo_validacao_csv(study_id: str):
    try:
        study = study_store.get_study(study_id)
        rows = study_store.export_trials(study_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Estudo não encontrado") from exc
    output = io.StringIO(newline="")
    columns = [
        "id", "study_id", "created_at", "job_id", "patient_id", "evaluator_id",
        "reference_assessor_id", "movement", "side", "trial_number", "frame",
        "timestamp_ms", "automatic_angle_deg", "reference_angle_deg", "error_deg",
        "absolute_error_deg", "reference_method", "reference_device_model",
        "reference_device_serial", "landmark_confidence_min", "source",
        "measurement_valid", "quality", "included", "exclusion_reason", "notes",
    ]
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    safe_code = "".join(ch for ch in study["code"] if ch.isalnum() or ch in "-_") or "study"
    return Response(
        content=output.getvalue(), media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="physioai_{safe_code}_trials.csv"'},
    )




@app.get("/api/estudos/{study_id}/export.zip")
async def exportar_estudo_validacao_zip(study_id: str, request: Request):
    user = _request_user(request)
    try:
        study = study_store.get_study(study_id)
        trials = study_store.export_trials(study_id)
        report = study_store.report(study_id, bootstrap_iterations=2000)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Estudo não encontrado") from exc
    bundle = build_study_bundle(study=study, trials=trials, report=report)
    safe_code = "".join(ch for ch in study["code"] if ch.isalnum() or ch in "-_") or "study"
    audit_store.log(
        actor=user.username, role=user.role, action="validation.study.export",
        resource_type="validation_study", resource_id=study_id,
        remote_ip=_remote_ip(request), metadata={"trials": len(trials)},
    )
    return Response(
        content=bundle, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="physioai_{safe_code}_study.zip"'},
    )

# ─────────────────────────────────────────────────────────────────────────────
#  EXECUÇÕES FORMAIS DE VALIDAÇÃO — snapshot imutável + gate operacional
# ─────────────────────────────────────────────────────────────────────────────
def _trials_for_formal_run(run: dict) -> list[dict]:
    rows = study_store.trials(run["study_id"])
    start = str(run["created_at"])
    end = str(run.get("closed_at") or "9999-12-31T23:59:59+00:00")
    return [row for row in rows if start <= str(row.get("created_at") or "") <= end]


def _formal_readiness_for(study: dict, payload: FormalValidationRunCreateIn) -> dict:
    return formal_readiness(
        study=study,
        ethics_status=payload.ethics_status,
        ethics_protocol_id=payload.ethics_protocol_id,
        calibration_date=payload.reference_calibration_date,
        calibration_due_date=payload.reference_calibration_due_date,
        statistical_plan_path=STATISTICAL_PLAN_PATH,
        protocol_lock_path=PROTOCOL_LOCK_PATH,
        software_version=APP_VERSION,
        clinical_protocol_version=PROTOCOL_VERSION,
    )


@app.post("/api/estudos/{study_id}/formal-runs/readiness")
async def readiness_execucao_formal(study_id: str, payload: FormalValidationRunCreateIn):
    try:
        study = study_store.get_study(study_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Estudo não encontrado") from exc
    return _formal_readiness_for(study, payload)


@app.post("/api/estudos/{study_id}/formal-runs", status_code=201)
async def iniciar_execucao_formal(study_id: str, payload: FormalValidationRunCreateIn, request: Request):
    user = _request_user(request)
    _require_admin(user)
    try:
        study = study_store.get_study(study_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Estudo não encontrado") from exc
    readiness = _formal_readiness_for(study, payload)
    if not readiness["ready"]:
        raise HTTPException(status_code=422, detail={"message": "Execução formal não está pronta", **readiness})

    statistical_plan = load_json(STATISTICAL_PLAN_PATH)
    checklist = load_json(FORMAL_CHECKLIST_PATH)
    protocol_lock = load_json(PROTOCOL_LOCK_PATH)
    trials_before = study_store.trials(study_id)
    snapshot = {
        "kind": "formal_validation_run_start",
        "created_at": time.time(),
        "build": BUILD_INFO.public(),
        "clinical_spec_sha256": CLINICAL_SPEC_SHA256,
        "model_provenance": MODEL_PROVENANCE,
        "study": study,
        "readiness": readiness,
        "statistical_analysis_plan": statistical_plan,
        "formal_validation_checklist": checklist,
        "protocol_lock": protocol_lock,
    }
    try:
        run = formal_run_store.create({
            "study_id": study_id,
            "created_by": user.username,
            "software_version": APP_VERSION,
            "clinical_protocol_version": PROTOCOL_VERSION,
            "study_protocol_sha256": study["protocol_sha256"],
            "statistical_plan_sha256": readiness["statistical_plan_sha256"],
            "protocol_lock_sha256": readiness["protocol_lock_sha256"],
            "ethics_status": payload.ethics_status,
            "ethics_protocol_id": payload.ethics_protocol_id,
            "reference_calibration_date": payload.reference_calibration_date,
            "reference_calibration_due_date": payload.reference_calibration_due_date,
            "start_trial_count": len(trials_before),
            "snapshot": snapshot,
            "notes": payload.notes,
        })
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    audit_store.log(
        actor=user.username, role=user.role, action="validation.formal_run.start",
        resource_type="formal_validation_run", resource_id=run["id"],
        remote_ip=_remote_ip(request), metadata={"study_id": study_id, "snapshot_sha256": run["snapshot_sha256"]},
    )
    return run


@app.get("/api/estudos/{study_id}/formal-runs")
async def listar_execucoes_formais(study_id: str):
    try:
        study_store.get_study(study_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Estudo não encontrado") from exc
    return {"items": formal_run_store.list_for_study(study_id)}


@app.get("/api/formal-runs/{run_id}")
async def obter_execucao_formal(run_id: str):
    try:
        return formal_run_store.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Execução formal não encontrada") from exc


@app.post("/api/formal-runs/{run_id}/close")
async def fechar_execucao_formal(run_id: str, request: Request):
    user = _request_user(request)
    _require_admin(user)
    try:
        run = formal_run_store.get(run_id)
        study = study_store.get_study(run["study_id"])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Execução formal não encontrada") from exc
    rows = _trials_for_formal_run(run)
    report = report_for_formal_run(study, rows, bootstrap_iterations=2000)
    try:
        closed = formal_run_store.close(run_id, end_trial_count=run["start_trial_count"] + len(rows), report=report)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    audit_store.log(
        actor=user.username, role=user.role, action="validation.formal_run.close",
        resource_type="formal_validation_run", resource_id=run_id,
        remote_ip=_remote_ip(request), metadata={"study_id": run["study_id"], "trials": len(rows), "report_sha256": closed["close_report_sha256"]},
    )
    return closed


@app.get("/api/formal-runs/{run_id}/export.zip")
async def exportar_execucao_formal(run_id: str, request: Request):
    user = _request_user(request)
    try:
        run = formal_run_store.get(run_id)
        study = study_store.get_study(run["study_id"])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Execução formal não encontrada") from exc
    rows = _trials_for_formal_run(run)
    report = run.get("close_report") or report_for_formal_run(study, rows, bootstrap_iterations=2000)
    bundle = build_formal_validation_bundle(
        run=run,
        study=study,
        trials=rows,
        report=report,
        statistical_plan=load_json(STATISTICAL_PLAN_PATH),
        protocol_lock=load_json(PROTOCOL_LOCK_PATH),
        checklist=load_json(FORMAL_CHECKLIST_PATH),
    )
    audit_store.log(
        actor=user.username, role=user.role, action="validation.formal_run.export",
        resource_type="formal_validation_run", resource_id=run_id,
        remote_ip=_remote_ip(request), metadata={"study_id": run["study_id"], "trials": len(rows)},
    )
    return Response(
        content=bundle,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="physioai_formal_run_{run_id}.zip"'},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PACOTE EXTERNO DE REVISÃO — desidentificado e derivado da execução formal
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/formal-runs/{run_id}/publication/readiness")
async def prontidao_publicacao(run_id: str, request: Request):
    user = _request_user(request)
    try:
        run = formal_run_store.get(run_id)
        study = study_store.get_study(run["study_id"])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Execução formal não encontrada") from exc
    rows = _trials_for_formal_run(run)
    report = run.get("close_report") or report_for_formal_run(study, rows, bootstrap_iterations=2000)
    result = publication_readiness(run=run, study=study, report=report)
    audit_store.log(
        actor=user.username, role=user.role, action="validation.publication.readiness",
        resource_type="formal_validation_run", resource_id=run_id,
        remote_ip=_remote_ip(request), metadata={"study_id": run["study_id"], "package_ready": result["package_ready"]},
    )
    return result


@app.get("/api/formal-runs/{run_id}/publication.zip")
async def exportar_pacote_revisao_externa(run_id: str, request: Request):
    user = _request_user(request)
    _require_admin(user)
    try:
        run = formal_run_store.get(run_id)
        study = study_store.get_study(run["study_id"])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Execução formal não encontrada") from exc
    rows = _trials_for_formal_run(run)
    report = run.get("close_report") or report_for_formal_run(study, rows, bootstrap_iterations=2000)
    readiness = publication_readiness(run=run, study=study, report=report)
    if not readiness["package_ready"]:
        raise HTTPException(status_code=409, detail={"message": "Pacote de revisão externa indisponível", **readiness})
    try:
        bundle = build_publication_review_bundle(
            run=run, study=study, trials=rows, report=report,
            statistical_plan=load_json(STATISTICAL_PLAN_PATH),
            protocol_lock=load_json(PROTOCOL_LOCK_PATH),
            checklist=load_json(FORMAL_CHECKLIST_PATH),
            requirements=load_json(PUBLICATION_REQUIREMENTS_PATH),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    digest = hashlib.sha256(bundle).hexdigest()
    audit_store.log(
        actor=user.username, role=user.role, action="validation.publication.export",
        resource_type="formal_validation_run", resource_id=run_id,
        remote_ip=_remote_ip(request), metadata={"study_id": run["study_id"], "trials": len(rows), "bundle_sha256": digest},
    )
    return Response(
        content=bundle, media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="physioai_external_review_{run_id}.zip"',
            "X-PhysioAI-SHA256": digest,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET CIENTÍFICO — anotação humana + métricas autoritativas do servidor
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/dataset/salvar", status_code=201)
async def salvar_dataset(
    request: Request,
    job_id: Annotated[str, Form()],
    label_qual: Annotated[str, Form()],
    notas: Annotated[str, Form()] = "",
):
    """Cataloga uma sessão concluída sem aceitar métricas automáticas do cliente.

    O navegador fornece apenas anotação humana (label/notas). Identidade da sessão,
    movimento e todas as métricas quantitativas são reconstruídas a partir do banco
    server-side para preservar a integridade científica do dataset.
    """
    user = _request_user(request)
    _assert_job_access(job_id, user)

    label = label_qual.strip()
    if not label or label in ("Selecione…", "Selecione..."):
        raise HTTPException(status_code=422, detail="label_qual inválido.")

    session = session_store.get_by_job(job_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    if session.get("status") != "completed":
        raise HTTPException(status_code=409, detail="A sessão ainda não foi concluída com sucesso.")

    measurements = measurement_store.list_for_job(job_id)
    valid = [
        row for row in measurements
        if row.get("measurement_valid") and row.get("angle_clinical_deg") is not None
    ]
    if len(valid) < 2:
        raise HTTPException(
            status_code=422,
            detail="A sessão não possui medições clínicas válidas suficientes para catalogação.",
        )

    angles = np.asarray([float(row["angle_clinical_deg"]) for row in valid], dtype=np.float64)
    finite_angles = angles[np.isfinite(angles)]
    if finite_angles.size < 2:
        raise HTTPException(status_code=422, detail="Sequência angular válida insuficiente.")

    velocities = np.asarray([
        abs(float(row["angular_velocity_deg_s"]))
        for row in valid
        if row.get("angular_velocity_deg_s") is not None
        and np.isfinite(float(row["angular_velocity_deg_s"]))
    ], dtype=np.float64)

    sequence = np.interp(
        np.linspace(0.0, 1.0, 100),
        np.linspace(0.0, 1.0, finite_angles.size),
        finite_angles,
    ).astype(float).tolist()

    spec = get_movement_spec(str(session["movement"]))
    if spec is None:
        raise HTTPException(status_code=422, detail="Movimento sem especificação clínica habilitada.")

    adm_deg = float(finite_angles.max() - finite_angles.min())
    adm_ref = float(spec.reference_rom_deg)
    provenance = session.get("provenance") or {}
    analysis_config = provenance.get("analysis_config") or {}
    inputs = provenance.get("inputs") or []
    input_roles = {str(item.get("role")) for item in inputs if isinstance(item, dict)}

    sample = {
        "job_id": job_id,
        "patient_id": session["patient_pseudonym"],
        "movement": session["movement"],
        "side": session["side"],
        "label": label,
        "label_marcha": "Não aplicável",
        # Campos legados mantidos para compatibilidade; adm_max = excursão clínica observada.
        "adm_max": round(adm_deg, 4),
        "adm_media": round(float(finite_angles.mean()), 4),
        "adm_ref": adm_ref,
        "pct_adm_ref": round(adm_deg / adm_ref * 100.0, 2) if adm_ref else None,
        "vel_pico": round(float(velocities.max()), 4) if velocities.size else 0.0,
        "total_reps": int(session.get("total_reps") or 0),
        "fluidez_ia": 0.0,
        "stride_length_m": None,
        "step_width_m": None,
        "cadencia_passos_min": None,
        "offset_cameras_ms": int(analysis_config.get("offset_ms") or 0),
        "tipo_sensor": str(analysis_config.get("sensor_type") or "Nenhum"),
        "has_lidar": "depth" in input_roles,
        "has_stereo": "video_b" in input_roles,
        "hardware": "GPU" if torch.cuda.is_available() else "CPU",
        "sequence": [round(float(v), 6) for v in sequence],
        "notes": notas,
        "valid_frames": len(valid),
        "result_sha256": session.get("result_sha256"),
        "software_version": session.get("software_version"),
        "protocol_version": session.get("protocol_version"),
        "clinical_spec_sha256": session.get("clinical_spec_sha256"),
    }

    try:
        saved = dataset_store.save(sample)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    audit_store.log(
        actor=user.username, role=user.role, action="dataset.sample.create",
        resource_type="dataset_sample", resource_id=saved["id"],
        remote_ip=_remote_ip(request),
        metadata={
            "job_id": job_id,
            "patient": session["patient_pseudonym"],
            "movement": session["movement"],
            "side": session["side"],
            "result_sha256": session.get("result_sha256"),
        },
    )
    return {"status": "ok", "mensagem": "Sessão catalogada com sucesso.", **saved}


@app.get("/api/dataset/stats")
async def dataset_stats():
    return dataset_store.stats()


@app.get("/api/config/movimentos")
async def movimentos_clinicos():
    return {
        "items": public_movement_specs(),
        "protocol_version": PROTOCOL_VERSION,
        "clinical_spec_sha256": clinical_spec_sha256(),
        "capture_preflight": capture_preflight_spec_snapshot(),
    }


def _cached_database_quick_check() -> tuple[bool, str]:
    now = time.monotonic()
    with _db_check_lock:
        age = now - float(_db_check_cache.get("at") or 0.0)
        if age >= DB_QUICKCHECK_INTERVAL_SECONDS or _db_check_cache.get("message") == "not_checked":
            ok, message = sqlite_quick_check(RUNTIME_DB)
            _db_check_cache.update({"at": now, "ok": ok, "message": message})
        return bool(_db_check_cache.get("ok")), str(_db_check_cache.get("message"))


def _readiness_state() -> tuple[bool, dict]:
    checks: dict[str, dict] = {}
    analyzer_ok = app_state.analisador is not None and app_state.semaphore is not None
    checks["analyzer"] = {"ok": analyzer_ok}

    runtime_dir = str(DATA_DIR)
    os.makedirs(runtime_dir, exist_ok=True)
    writable = os.access(runtime_dir, os.W_OK)
    checks["storage_writable"] = {"ok": writable, "path": runtime_dir}

    db_ok, db_message = _cached_database_quick_check()
    checks["database"] = {"ok": db_ok, "quick_check": db_message}

    try:
        stat = os.statvfs(runtime_dir)
        free_mb = stat.f_bavail * stat.f_frsize / (1024 * 1024)
    except Exception:
        free_mb = 0.0
    disk_ok = free_mb >= MIN_FREE_DISK_MB
    checks["disk_free"] = {"ok": disk_ok, "free_mb": round(free_mb, 1), "minimum_mb": MIN_FREE_DISK_MB}

    return all(item["ok"] for item in checks.values()), checks


@app.get("/version")
async def version_info():
    return BUILD_INFO.public()


@app.get("/health")
async def health():
    # Liveness: se este handler responde, o processo HTTP está vivo.
    return {"status": "ok", "versao": APP_VERSION, "git_sha": BUILD_INFO.git_sha}


@app.get("/ready")
async def ready():
    ok, checks = _readiness_state()
    content = {"status": "ready" if ok else "not_ready", "versao": APP_VERSION, "checks": checks}
    return JSONResponse(content=content, status_code=200 if ok else 503)


@app.get("/api/system/health")
async def system_health(request: Request):
    _request_user(request)
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "nome": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "vram_usada_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        }
    ready_ok, readiness = _readiness_state()
    return {
        "status": "ok" if ready_ok else "degraded",
        "versao": APP_VERSION,
        "build": BUILD_INFO.public(),
        "readiness": readiness,
        "analisador_pronto": app_state.analisador is not None,
        "modelo_ia_pronto": bool(app_state.analisador and app_state.analisador.modelo_ia is not None),
        "gpu_disponivel": torch.cuda.is_available(),
        "gpu_info": gpu_info,
        "result_ttl_seconds": RESULT_TTL_SECONDS,
        "preflight_ttl_seconds": PREFLIGHT_TTL_SECONDS,
        "preflight_required": REQUIRE_PREFLIGHT,
        "preflight_max_samples": PREFLIGHT_MAX_SAMPLES,
        "preflights_pending": len(app_state.preflights),
        "max_inflight_jobs": MAX_INFLIGHT_JOBS,
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_total_upload_mb": MAX_TOTAL_UPLOAD_MB,
        "protocol_version": PROTOCOL_VERSION,
        "clinical_spec_sha256": CLINICAL_SPEC_SHA256,
        "model_provenance": MODEL_PROVENANCE,
        "experimental_inputs_enabled": ENABLE_EXPERIMENTAL_INPUTS,
        "metrics": metrics.snapshot(),
    }


@app.get("/api/system/metrics")
async def system_metrics(request: Request):
    user = _request_user(request)
    _require_admin(user)
    return metrics.snapshot()


@app.get("/api/system/metrics.prom")
async def system_metrics_prom(request: Request):
    user = _request_user(request)
    _require_admin(user)
    return Response(content=metrics.to_prometheus(), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/api/sessoes")
async def listar_sessoes(request: Request, limit: int = Query(default=50, ge=1, le=500)):
    user = _request_user(request)
    items = session_store.recent(limit) if user.role == "admin" else session_store.recent_for_evaluator(user.username, limit)
    return {"items": items}


@app.get("/api/sessoes/{job_id}")
async def obter_sessao(job_id: str, request: Request):
    user = _request_user(request)
    _assert_job_access(job_id, user)
    session = session_store.get_by_job(job_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    measurements = measurement_store.list_for_job(job_id)
    reviews = validation_store.for_job(job_id)
    study_trials = study_store.for_job(job_id)
    imu_analyses = imu_store.list_for_job(job_id)
    motor_features = motor_feature_store.list_for_job(job_id)
    longitudinal_analyses = longitudinal_store.list_for_job(job_id)
    return {
        "session": session,
        "measurement_count": len(measurements),
        "valid_measurement_count": sum(1 for row in measurements if row.get("measurement_valid")),
        "validation_review_count": len(reviews),
        "validation_trial_count": len(study_trials),
        "imu_analysis_count": len(imu_analyses),
        "motor_feature_analysis_count": len(motor_features),
        "longitudinal_analysis_count": len(longitudinal_analyses),
        "measurement_sha256": canonical_json_sha256(measurements),
        "validation_sha256": canonical_json_sha256(reviews),
        "validation_trials_sha256": canonical_json_sha256(study_trials),
        "imu_analyses_sha256": canonical_json_sha256(imu_analyses),
        "motor_features_sha256": canonical_json_sha256(motor_features),
        "longitudinal_analyses_sha256": canonical_json_sha256(longitudinal_analyses),
    }


@app.get("/api/sessoes/{job_id}/export.zip")
async def exportar_sessao(job_id: str, request: Request):
    user = _request_user(request)
    _assert_job_access(job_id, user)
    session = session_store.get_by_job(job_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    measurements = measurement_store.list_for_job(job_id)
    reviews = validation_store.for_job(job_id)
    study_trials = study_store.for_job(job_id)
    imu_analyses = imu_store.list_for_job(job_id)
    imu_payloads = {item["id"]: imu_store.raw_payload(item["id"]) for item in imu_analyses}
    motor_features = motor_feature_store.list_for_job(job_id)
    longitudinal_analyses = longitudinal_store.list_for_job(job_id)
    manifest = {
        "bundle_schema_version": 2,
        "session": session,
        "exported_by": user.username,
        "export_build": BUILD_INFO.public(),
        "protocol": {
            "version": PROTOCOL_VERSION,
            "clinical_spec_sha256": CLINICAL_SPEC_SHA256,
            "clinical_spec": clinical_spec_snapshot(),
        },
        "counts": {
            "measurements": len(measurements),
            "valid_measurements": sum(1 for row in measurements if row.get("measurement_valid")),
            "validation_reviews": len(reviews),
            "validation_trials": len(study_trials),
            "imu_analyses": len(imu_analyses),
            "motor_feature_analyses": len(motor_features),
            "longitudinal_analyses": len(longitudinal_analyses),
        },
        "content_sha256": {
            "measurements": canonical_json_sha256(measurements),
            "validation_reviews": canonical_json_sha256(reviews),
            "validation_trials": canonical_json_sha256(study_trials),
            "imu_analyses": canonical_json_sha256(imu_analyses),
            "motor_features": canonical_json_sha256(motor_features),
            "longitudinal_analyses": canonical_json_sha256(longitudinal_analyses),
        },
        "interpretation": {
            "landmark_visibility": "tracking quality; not clinical accuracy",
            "manual_method": "2D three-point image goniometry",
            "reference_method": "physical goniometer when provided",
        },
    }
    bundle = build_session_bundle(
        manifest=manifest, measurements=measurements, validation_reviews=reviews,
        validation_trials=study_trials, imu_analyses=imu_analyses, imu_payloads=imu_payloads,
        motor_feature_analyses=motor_features, longitudinal_analyses=longitudinal_analyses
    )
    audit_store.log(
        actor=user.username, role=user.role, action="session.export",
        resource_type="job", resource_id=job_id, remote_ip=_remote_ip(request),
        metadata={"measurements": len(measurements), "reviews": len(reviews)},
    )
    return Response(
        content=bundle,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="physioai_session_{job_id}.zip"'},
    )


@app.get("/api/auditoria")
async def listar_auditoria(request: Request, limit: int = Query(default=100, ge=1, le=1000)):
    user = _request_user(request)
    _require_admin(user)
    return {"items": audit_store.recent(limit)}


class PrivacyDeleteRequest(BaseModel):
    patient_id: str = Field(min_length=1, max_length=128)
    confirmation: str = Field(min_length=1, max_length=64)


@app.post("/api/privacy/patient/delete")
async def apagar_dados_paciente(payload: PrivacyDeleteRequest, request: Request):
    user = _request_user(request)
    _require_admin(user)
    if payload.confirmation != "EXCLUIR":
        raise HTTPException(status_code=422, detail="Confirmação inválida")
    pseudonym = auth_manager.pseudonymize_patient_id(payload.patient_id)
    patient_jobs = session_store.jobs_for_patient(pseudonym)
    active_jobs = [job_id for job_id in patient_jobs if job_id in app_state.cancel_flags]
    if active_jobs:
        for job_id in active_jobs:
            app_state.cancel_flags[job_id].set()
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Existem análises ativas deste paciente. O cancelamento foi solicitado; repita a exclusão após o encerramento dos jobs.",
                "active_jobs": active_jobs,
            },
        )
    deleted_measurements = sum(measurement_store.delete_job(job_id) for job_id in patient_jobs)
    deleted_imu = sum(imu_store.delete_job(job_id) for job_id in patient_jobs)
    deleted_motor_features = sum(motor_feature_store.delete_job(job_id) for job_id in patient_jobs)
    deleted_longitudinal = longitudinal_store.delete_patient(pseudonym)
    deleted = {
        "frame_measurements": deleted_measurements,
        "imu_analyses": deleted_imu,
        "motor_feature_analyses": deleted_motor_features,
        "longitudinal_analyses": deleted_longitudinal,
        "validation_reviews": validation_store.delete_patient(pseudonym),
        "validation_trials": study_store.delete_patient(pseudonym),
        "dataset_samples": dataset_store.delete_patient(pseudonym),
        "clinical_sessions": session_store.delete_patient(pseudonym),
        "audit_events_redacted": audit_store.redact_patient(pseudonym),
    }
    audit_store.log(
        actor=user.username, role=user.role, action="privacy.patient.delete",
        resource_type="patient", resource_id="[REDACTED]", remote_ip=_remote_ip(request),
        metadata={"deleted": deleted},
    )
    return {"status": "ok", "patient_pseudonym": pseudonym, "deleted": deleted}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, workers=1, log_level="info")