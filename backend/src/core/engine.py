"""PhysioAI — motor biomecânico pré-clínico.

A via clínica é definida exclusivamente por ``src.core.clinical.MOVEMENT_SPECS``.
Cada movimento habilitado declara landmarks, convenção angular, referência
pré-clínica e protocolo de captura. Recursos estéreo/LiDAR e métricas de marcha
permanecem experimentais e não são tratados como clinicamente validados.

Fase 15:
- Flexão de Cotovelo;
- Flexão de Joelho;
- Flexão de Quadril;
- Dorsiflexão de Tornozelo;
- Flexão de Ombro;
- Abdução de Ombro.

Movimentos não especificados são rejeitados em vez de usar heurísticas legadas.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import time
import warnings
from typing import Callable, ClassVar

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.python.solutions import pose as mp_pose
from scipy.signal import find_peaks
from src.core.anatomy import (
    alignment_deviation_deg,
    build_body_frame,
    camera_view_alignment,
    max_out_of_plane_deg,
    movement_direction_alignment,
    projected_internal_angle_deg,
)
from src.core.clinical import (
    MOVEMENT_SPECS,
    apply_plane_quality,
    get_movement_spec,
    tracking_quality,
)

logger = logging.getLogger(__name__)

# ── Compatibilidade de autocast PyTorch ≥ 2.0 vs < 2.0 ──────────────────────
try:
    from torch.amp import autocast as _amp_autocast
    def _make_autocast(device_type: str, enabled: bool):
        return _amp_autocast(device_type=device_type, enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast as _amp_autocast  # type: ignore
    def _make_autocast(device_type: str, enabled: bool):
        return _amp_autocast(enabled=enabled)


# ═════════════════════════════════════════════════════════════════════════════
#  WRAPPER DE PONTO 3D — nível de módulo (v4.0: removido do loop quente)
# ═════════════════════════════════════════════════════════════════════════════

class _PtWrapper:
    """
    Wrapper leve para acesso via atributo a um array (x, y, z).

    FIX v4.0: na v3.1 esta classe era redefinida DENTRO do loop por frame em
    processar_estereo(), causando overhead de definição de classe × N frames.
    Movida para nível de módulo — definida uma única vez por importação.
    """
    __slots__ = ("x", "y", "z")

    def __init__(self, arr: np.ndarray) -> None:
        self.x = float(arr[0])
        self.y = float(arr[1])
        self.z = float(arr[2])

    def __repr__(self) -> str:  # pragma: no cover
        return f"_PtWrapper(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"


# ═════════════════════════════════════════════════════════════════════════════
#  CONTRATOS DE DADOS
# ═════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class Calibracao:
    """Parâmetros completos de calibração estéreo."""
    K_A: np.ndarray            # Intrínseca câmera A (3×3)
    d_A: np.ndarray            # Distorção câmera A (5,)
    K_B: np.ndarray            # Intrínseca câmera B (3×3)
    d_B: np.ndarray            # Distorção câmera B (5,)
    R:   np.ndarray            # Rotação A→B (3×3)
    T:   np.ndarray            # Translação A→B (3×1), em metros
    P_A: np.ndarray            # Matriz de projeção câmera A (3×4)
    P_B: np.ndarray            # Matriz de projeção câmera B (3×4)
    baseline_m: float          # Distância entre centros ópticos (metros)
    rms_reprojection: float    # Erro RMS de calibração (pixels)

    def __repr__(self) -> str:
        return (
            f"Calibracao(baseline={self.baseline_m:.4f}m, "
            f"rms={self.rms_reprojection:.4f}px)"
        )


@dataclasses.dataclass
class ConfigLiDAR:
    """Configuração do sensor de profundidade."""
    tipo: str                       # "realsense_d435" | "zed2" | "custom"
    extrinsica_rgb: np.ndarray      # Transformação LiDAR→câmera A (4×4)
    escala_profundidade: float      # Fator de conversão raw→metros (ex: 0.001 mm→m)

    def __repr__(self) -> str:
        return f"ConfigLiDAR(tipo='{self.tipo}', escala={self.escala_profundidade})"


@dataclasses.dataclass
class FrameSincronizado:
    """Par de frames com metadados de sincronização."""
    timestamp: float
    frame_a: np.ndarray
    frame_b: np.ndarray | None
    delta_ms: float
    qualidade_sinc: float          # 1.0 = perfeito, 0.0 = delta ≥ 1 período
    depth_map: np.ndarray | None   # Mapa de profundidade associado (opcional)

    def __repr__(self) -> str:
        return (
            f"FrameSincronizado(ts={self.timestamp:.1f}ms, "
            f"delta={self.delta_ms:.2f}ms, q={self.qualidade_sinc:.2f})"
        )




@dataclasses.dataclass
class Point2DTelemetry:
    """Ponto normalizado [0, 1] usado pelo overlay goniométrico."""
    x: float
    y: float
    confidence: float | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Point3DTelemetry:
    """Ponto 3D no espaço indicado por ``measurement_space``."""
    x: float
    y: float
    z: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class FrameMeasurement:
    """Medição biomecânica auditável produzida para um único frame."""
    frame: int
    timestamp_ms: float
    joint: str
    movement: str
    side: str
    angle_raw_deg: float | None
    angle_landmark_deg: float | None
    angle_plane_deg: float | None
    angle_filtered_deg: float | None
    angle_clinical_deg: float | None
    angular_velocity_deg_s: float | None
    adm_deg: float | None
    confidence: float | None
    source: str
    measurement_space: str
    measurement_plane: str | None
    plane_projection_applied: bool
    body_frame_confidence: float | None
    view_alignment: float | None
    view_deviation_deg: float | None
    out_of_plane_deg: float | None
    direction_rule: str | None
    direction_alignment: float | None
    landmark_confidence_min: float | None
    landmark_confidence_mean: float | None
    quality: str
    measurement_valid: bool
    quality_flags: list[str]
    proximal: Point2DTelemetry | None = None
    vertex: Point2DTelemetry | None = None
    distal: Point2DTelemetry | None = None
    proximal_3d: Point3DTelemetry | None = None
    vertex_3d: Point3DTelemetry | None = None
    distal_3d: Point3DTelemetry | None = None
    inference_ms: float | None = None
    pipeline_ms: float | None = None
    processing_fps: float | None = None
    # Landmarks 2D normalizados usados exclusivamente pela camada visual do
    # frontend. Eles não entram em ``to_dict`` para não alterar o contrato
    # científico/persistido das medições já validado nas fases anteriores.
    overlay_landmarks_2d: list[Point2DTelemetry | None] = dataclasses.field(
        default_factory=list, repr=False
    )

    def to_dict(self) -> dict:
        payload = dataclasses.asdict(self)
        payload.pop("overlay_landmarks_2d", None)
        return payload


class AnaliseCancelada(RuntimeError):
    """Sinaliza cancelamento solicitado pelo cliente durante o processamento."""

@dataclasses.dataclass
class ResultadoAnalise:
    """Saída completa do pipeline para o app.py."""
    frames_a: list[np.ndarray]
    skeleton_3d: list[np.ndarray]
    com_trajetoria: list[np.ndarray]
    angulos_filtrados: list[float]
    angulos_brutos: list[float]
    angulos_geometricos_filtrados: list[float]
    frame_measurements: list[FrameMeasurement]
    velocidades: list[float]
    total_frames: int
    total_reps: int
    fps: float
    point_cloud: np.ndarray | None
    parametros_marcha: dict
    fluidez_ia: float

    def __repr__(self) -> str:
        return (
            f"ResultadoAnalise(frames={len(self.frames_a)}, "
            f"reps={self.total_reps}, fps={self.fps:.1f}, "
            f"fluidez={self.fluidez_ia:.1f}%)"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  FILTRO DE KALMAN 6D — Posição + Velocidade por landmark
# ═════════════════════════════════════════════════════════════════════════════

class KalmanLandmark3D:
    """
    Filtro de Kalman 6D para suavização de trajetória 3D de um único landmark.

    Estado: x = [X, Y, Z, Vx, Vy, Vz]ᵀ

    Modelo de Transição (velocidade constante):
        F = | I₃  dt·I₃ |
            | 0₃    I₃  |

    Matrizes de Observação intercambiáveis por sensor:
        H_full  (3×6)  — estéreo:   observa X, Y, Z
        H_xy    (2×6)  — monocular: observa X, Y (profundidade ambígua)
        H_z     (1×6)  — LiDAR:    observa apenas Z (mais preciso)

    Ruído de Medição (R) adaptativo:
        R_stereo  — erro padrão de triangulação
        R_mono    — 10× maior (profundidade estimada, não medida)
        R_lidar   — 10× menor (sensor de tempo de voo)

    Em oclusão (conf < limiar), executa apenas predict() e extrapola o estado.
    A covariância P cresce monotonicamente sinalizando incerteza crescente.

    Forma de Joseph para update de P — numericamente estável:
        P = (I - KH) P (I - KH)ᵀ + K R Kᵀ

    v3.1: update() usa np.linalg.solve em vez de np.linalg.inv para ganho K.
    """

    def __init__(
        self,
        dt: float = 0.033,
        sigma_processo: float = 1e-2,
        sigma_medicao: float = 1e-3,
    ) -> None:
        # ── Estado e covariância ──────────────────────────────────────────
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 1.0

        # ── Matriz de transição F (velocidade constante) ──────────────────
        self.F = np.eye(6, dtype=np.float32)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt

        # ── Ruído de processo Q ───────────────────────────────────────────
        q = sigma_processo ** 2
        self.Q = np.diag([q, q, q, q * 10, q * 10, q * 10]).astype(np.float32)

        # ── Matrizes de observação H por tipo de sensor ───────────────────
        self.H_full = np.zeros((3, 6), dtype=np.float32)
        self.H_full[0, 0] = self.H_full[1, 1] = self.H_full[2, 2] = 1.0

        self.H_xy = np.zeros((2, 6), dtype=np.float32)
        self.H_xy[0, 0] = self.H_xy[1, 1] = 1.0

        self.H_z = np.zeros((1, 6), dtype=np.float32)
        self.H_z[0, 2] = 1.0

        # ── Matrizes de ruído de medição R por sensor ─────────────────────
        r = sigma_medicao ** 2
        self.R_stereo = np.eye(3, dtype=np.float32) * r

        # R_mono (2×2): par de H_xy — quando só X, Y são observados
        self.R_mono   = np.eye(2, dtype=np.float32) * (r * 10.0)

        # R_mono_full (3×3): par de H_full no modo monocular —
        # pose_world_landmarks fornece X, Y, Z, mas Z é muito incerto.
        # BUG v3.1/v4.0: usar R_mono (2×2) com H_full (3×6) causava
        # ValueError ao calcular S = H·PH_t + R → (3,3)+(2,2) incompatível.
        self.R_mono_full = np.diag([r * 10.0, r * 10.0, r * 100.0]).astype(np.float32)

        self.R_lidar  = np.array([[r * 0.1]], dtype=np.float32)

        # ── Cache: identidade 6×6 ─────────────────────────────────────────
        self._I6 = np.eye(6, dtype=np.float32)

    def predict(self) -> np.ndarray:
        """Etapa de predição. Retorna posição predita [X, Y, Z]."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3, 0]

    def update(
        self,
        z: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> np.ndarray:
        """
        Etapa de atualização com medição z, observação H e ruído R.

        K = P·Hᵀ·S⁻¹ calculado via np.linalg.solve(Sᵀ, (P·Hᵀ)ᵀ)ᵀ em vez de
        np.linalg.inv(S) — evita materializar a inversa e é numericamente mais
        estável para matrizes S mal condicionadas.
        Para S escalar (1×1), usa divisão direta sem overhead de solve.
        """
        z    = z.reshape(-1, 1).astype(np.float32)
        PH_t = self.P @ H.T                     # (6, m)
        S    = H @ PH_t + R                      # (m, m)
        y    = z - (H @ self.x)                  # Inovação

        if S.shape[0] == 1:
            K = PH_t / float(S[0, 0])
        else:
            K = np.linalg.solve(S.T, PH_t.T).T

        self.x = self.x + K @ y

        # Forma de Joseph — numericamente estável
        I_KH  = self._I6 - (K @ H)
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        return self.x[:3, 0]

    def set_dt(self, dt: float) -> None:
        """Atualiza o passo temporal do modelo de velocidade constante."""
        dt = max(float(dt), 1e-6)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt

    def reset(self) -> None:
        """Reinicia o filtro para uma nova sessão de captura."""
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 1.0


# ═════════════════════════════════════════════════════════════════════════════
#  MOTOR CORE: AnalisadorADMWeb v4.0
# ═════════════════════════════════════════════════════════════════════════════

class AnalisadorADMWeb:
    """
    Motor de captura e análise biomecânica.

    Suporta três modos de operação, com diferentes fontes de coordenadas:
      1. Estéreo calibrado          — triangulação DLT experimental
      2. Monocular                  — pose_world_landmarks do MediaPipe, relativo

    A via clínica principal desta versão usa apenas movimentos explicitamente
    definidos em ``MOVEMENT_SPECS``. LiDAR/profundidade permanece experimental
    e não integra a via clínica habilitada.

    Uso como context manager recomendado para garantir liberação de recursos:

        with AnalisadorADMWeb("pac-001", "Flexão de Cotovelo", "direito") as analisador:
            resultado = analisador.processar_multicamera(path_a=..., ...)
    """

    # Compatibilidade legada. A fonte clínica autoritativa é MOVEMENT_SPECS.
    _LIMITES_ADM: ClassVar[dict] = {
        name: {
            "max_adm_ref": spec.reference_rom_deg,
            "th_flex": spec.enter_flexion_deg,
            "th_ext": spec.complete_extension_deg,
            "angulo_diminui": spec.converter_id == "flexion_from_internal_angle",
        }
        for name, spec in MOVEMENT_SPECS.items()
    }

    # Tabela de pesos segmentares Winter (1990)
    _WINTER_PROXIMAL: ClassVar[np.ndarray] = np.array(
        [0, 11, 12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27], dtype=np.int32
    )
    _WINTER_DISTAL: ClassVar[np.ndarray] = np.array(
        [0, 24, 14, 16, 18, 13, 15, 17, 26, 28, 32, 25, 27, 31], dtype=np.int32
    )
    _WINTER_PESOS: ClassVar[np.ndarray] = np.array([
        0.081,  # Cabeça
        0.497,  # Tronco
        0.028,  # Braço D
        0.016,  # Antebraço D
        0.006,  # Mão D
        0.028,  # Braço E
        0.016,  # Antebraço E
        0.006,  # Mão E
        0.100,  # Coxa D
        0.0465, # Perna D
        0.0145, # Pé D
        0.100,  # Coxa E
        0.0465, # Perna E
        0.0145, # Pé E
    ], dtype=np.float32)

    _WINTER_PESOS_SOMA: ClassVar[float] = float(_WINTER_PESOS.sum())

    _VISIBILIDADE_MIN:   ClassVar[float] = 0.50
    _QUALIDADE_SINC_MIN: ClassVar[float] = 0.60
    _RESOLUCAO_PLAYER:   ClassVar[tuple[int, int]] = (640, 480)

    # Escalas LiDAR por tipo de sensor (raw → metros)
    _ESCALAS_LIDAR: ClassVar[dict[str, float]] = {
        "Intel RealSense": 0.001,   # mm → m
        "Azure Kinect":    0.001,
        "Velodyne VLP-16": 1.0,     # já em metros
        "Ouster OS1":      1.0,
    }

    def __init__(
        self,
        id_paciente: str,
        tipo_movimento: str,
        lado_do_corpo: str,
        usar_gpu: bool = True,
        alpha_ema: float = 0.30,
    ) -> None:
        self.id_paciente    = id_paciente
        self.tipo_movimento = tipo_movimento
        self.lado_do_corpo  = lado_do_corpo.lower()
        self.alpha          = alpha_ema
        self.usar_gpu       = usar_gpu

        # ── Estado clínico ────────────────────────────────────────────────
        self.angulo_suavizado: float | None = None
        self.angulo_anterior:  float | None = None
        self.contador_reps:    int = 0
        self.fase_movimento:   str = "repouso"

        # ── Hardware e modelo IA ──────────────────────────────────────────
        self.device    = torch.device("cuda" if usar_gpu and torch.cuda.is_available() else "cpu")
        self.modelo_ia = None
        self._configurar_nos_anatomicos()

        # ── String de backend cacheada (v4.0: evita condicional por frame) ─
        self._backend_str = self._resolver_backend_str()

        # ── Dispatch dict de ângulos (v4.0: O(1) lookup em vez de if/elif) ─
        self._dispatch_angulo = self._construir_dispatch_angulo()

        # ── Estimadores de pose (um por feed) ─────────────────────────────
        self.pose_estimators: list[mp_pose.Pose] = [
            self._criar_estimador_pose() for _ in range(2)
        ]

        # ── Filtros de Kalman 6D (um por landmark) ────────────────────────
        self.filtros_kalman: list[KalmanLandmark3D] = [
            KalmanLandmark3D(dt=0.033) for _ in range(33)
        ]

        # ── Calibração estéreo ────────────────────────────────────────────
        self.calibracao:           Calibracao | None = None
        self.config_lidar:         ConfigLiDAR | None = None
        self.is_stereo_calibrated: bool = False

        # Atributos legados para compatibilidade com código V2.0
        self.K_A = self.K_B = np.eye(3, dtype=np.float32)
        self.d_A = self.d_B = np.zeros(5, dtype=np.float32)
        self.R   = np.eye(3, dtype=np.float32)
        self.T   = np.zeros((3, 1), dtype=np.float32)
        self.P_A = self.P_B = np.zeros((3, 4), dtype=np.float32)

        # ── Buffer HUD pré-alocado (v4.0: evita np.full_like por frame) ───
        target_w, target_h = self._RESOLUCAO_PLAYER
        hud_h = 54
        self._hud_dark_buf = np.full(
            (hud_h, target_w, 3), (12, 14, 22), dtype=np.uint8
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  CONTEXT MANAGER (v4.0 — adicionado para gestão segura de recursos)
    # ─────────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Libera todos os recursos alocados: pose estimators e memória GPU.

        Chamado automaticamente ao sair do bloco `with`.
        """
        for est in self.pose_estimators:
            try:
                est.close()
            except Exception as exc:
                logger.debug("Erro ao fechar pose estimator: %s", exc)
        self.pose_estimators.clear()

        if self.modelo_ia is not None:
            try:
                del self.modelo_ia
                self.modelo_ia = None
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                logger.debug("Erro ao liberar modelo IA: %s", exc)

        logger.info("AnalisadorADMWeb[%s] encerrado.", self.id_paciente)

    def __enter__(self) -> "AnalisadorADMWeb":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"AnalisadorADMWeb(paciente='{self.id_paciente}', "
            f"movimento='{self.tipo_movimento}', "
            f"lado='{self.lado_do_corpo}', "
            f"backend='{self._backend_str}')"
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  INICIALIZAÇÃO
    # ─────────────────────────────────────────────────────────────────────────

    def _resolver_backend_str(self) -> str:
        """
        Resolve string de backend uma única vez no __init__.

        v4.0: na v3.1 era computada a cada frame em _desenhar_overlay().
        Diferencia corretamente NVIDIA CUDA, AMD ROCm e CPU.
        """
        if self.device.type != "cuda":
            return "CPU"
        # torch.version.hip existe apenas em builds ROCm
        return "GPU-ROCm" if getattr(torch.version, "hip", None) else "GPU-CUDA"

    def _construir_dispatch_angulo(self) -> dict:
        """
        Constrói dict de dispatch para _calcular_angulo_para_movimento.

        v4.0: substitui cadeia if/elif por lookup O(1). Os valores são tuplas
        (fn_ou_None, key_a, key_b, key_c); fn_ou_None é None para o caso padrão
        (angulo direto) ou uma função de pós-processamento do ângulo bruto.
        """
        return {
            name: (None, spec.proximal_node, spec.vertex_node, spec.distal_node)
            for name, spec in MOVEMENT_SPECS.items()
        }

    def _criar_estimador_pose(self) -> mp_pose.Pose:
        """
        Tenta criar estimador com complexidade decrescente (2→1→0).

        v4.0: captura Exception (não apenas PermissionError) para cobrir
        RuntimeError e falhas de inicialização TFLite em ambientes restritos.
        """
        last_exc: Exception | None = None
        for complexity in (2, 1, 0):
            try:
                return mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=complexity,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=0.65,
                    min_tracking_confidence=0.65,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Falha ao criar estimador de pose com complexity=%d: %s",
                    complexity, exc,
                )
        raise RuntimeError(
            f"Não foi possível inicializar o estimador de pose. "
            f"Último erro: {last_exc}"
        )

    def _configurar_nos_anatomicos(self) -> None:
        """Mapeia nomes anatômicos para índices MediaPipe PoseLandmark."""
        is_left = self.lado_do_corpo == "esquerdo"
        p = mp_pose.PoseLandmark
        self.nodes: dict = {
            "ombro":     p.LEFT_SHOULDER  if is_left else p.RIGHT_SHOULDER,
            "cotovelo":  p.LEFT_ELBOW     if is_left else p.RIGHT_ELBOW,
            "pulso":     p.LEFT_WRIST     if is_left else p.RIGHT_WRIST,
            "quadril":   p.LEFT_HIP       if is_left else p.RIGHT_HIP,
            "joelho":    p.LEFT_KNEE      if is_left else p.RIGHT_KNEE,
            "tornozelo": p.LEFT_ANKLE     if is_left else p.RIGHT_ANKLE,
            "pe":        p.LEFT_FOOT_INDEX if is_left else p.RIGHT_FOOT_INDEX,
            "indicador": p.LEFT_INDEX     if is_left else p.RIGHT_INDEX,
            "mindinho":  p.LEFT_PINKY     if is_left else p.RIGHT_PINKY,
        }

    def _resetar_estado_sessao(self) -> None:
        """
        Reinicia estado clínico e filtros para uma nova sessão de análise.

        v3.1: pose_estimators recriados para evitar state bleed do tracking
        interno do MediaPipe entre análises de pacientes distintos.
        """
        self.fase_movimento   = "repouso"
        self.contador_reps    = 0
        self.angulo_anterior  = None
        self.angulo_suavizado = None
        for kf in self.filtros_kalman:
            kf.reset()

        for i in range(len(self.pose_estimators)):
            try:
                self.pose_estimators[i].close()
            except Exception:
                pass
            self.pose_estimators[i] = self._criar_estimador_pose()

    # ─────────────────────────────────────────────────────────────────────────
    #  BLOCO 1: CALIBRAÇÃO
    # ─────────────────────────────────────────────────────────────────────────

    def carregar_calibracao(
        self,
        K_A: np.ndarray,
        d_A: np.ndarray,
        K_B: np.ndarray,
        d_B: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        baseline_m: float = 0.0,
        rms_reprojection: float = 0.0,
    ) -> None:
        """
        Carrega parâmetros de calibração estéreo e pré-computa P_A / P_B.

        P_A = K_A · [I | 0]
        P_B = K_B · [R | T]
        """
        K_A = np.float32(K_A)
        d_A = np.float32(d_A)
        K_B = np.float32(K_B)
        d_B = np.float32(d_B)
        R   = np.float32(R)
        T   = np.float32(T).reshape(3, 1)

        P_A = K_A @ np.hstack((np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)))
        P_B = K_B @ np.hstack((R, T))

        self.calibracao = Calibracao(
            K_A=K_A, d_A=d_A, K_B=K_B, d_B=d_B,
            R=R, T=T, P_A=P_A, P_B=P_B,
            baseline_m=baseline_m,
            rms_reprojection=rms_reprojection,
        )
        self.K_A, self.d_A = K_A, d_A
        self.K_B, self.d_B = K_B, d_B
        self.R, self.T     = R, T
        self.P_A, self.P_B = P_A, P_B
        self.is_stereo_calibrated = True
        logger.info("Calibração estéreo carregada: %r", self.calibracao)

    def carregar_config_lidar(self, config: ConfigLiDAR) -> None:
        """Registra a configuração do sensor de profundidade para fusão."""
        self.config_lidar = config
        logger.info("Configuração LiDAR carregada: %r", config)

    # ─────────────────────────────────────────────────────────────────────────
    #  BLOCO 2: SINCRONIZAÇÃO
    # ─────────────────────────────────────────────────────────────────────────

    def iterar_frames_sincronizados(
        self,
        cap_a: cv2.VideoCapture,
        cap_b: cv2.VideoCapture | None,
        offset_ms: float = 0.0,
    ):
        """Gera frames sincronizados sob demanda, sem carregar o vídeo inteiro na RAM."""
        fps = float(cap_a.get(cv2.CAP_PROP_FPS)) or 30.0
        periodo_ms = 1000.0 / max(fps, 1e-6)

        if cap_b is not None and abs(offset_ms) > 1.0:
            frames_offset = int(abs(offset_ms) / 1000.0 * fps)
            cap_adiantar = cap_b if offset_ms > 0 else cap_a
            for _ in range(frames_offset):
                if not cap_adiantar.read()[0]:
                    break

        while cap_a.isOpened():
            ret_a, frame_a = cap_a.read()
            if not ret_a:
                break

            ts_a = float(cap_a.get(cv2.CAP_PROP_POS_MSEC))
            frame_b: np.ndarray | None = None
            delta = 0.0

            if cap_b is not None and cap_b.isOpened():
                ret_b, f_b = cap_b.read()
                if ret_b:
                    frame_b = f_b
                    delta = ts_a - float(cap_b.get(cv2.CAP_PROP_POS_MSEC))

            q_sinc = max(0.0, 1.0 - (abs(delta) / periodo_ms)) if cap_b is not None else 1.0
            yield FrameSincronizado(
                timestamp=ts_a,
                frame_a=frame_a,
                frame_b=frame_b,
                delta_ms=delta,
                qualidade_sinc=q_sinc,
                depth_map=None,
            )

    def sincronizar_frames(
        self,
        cap_a: cv2.VideoCapture,
        cap_b: cv2.VideoCapture | None,
        offset_ms: float = 0.0,
    ) -> list[FrameSincronizado]:
        """Compatibilidade: materializa o iterador de sincronização em lista."""
        return list(self.iterar_frames_sincronizados(cap_a, cap_b, offset_ms))

    # ─────────────────────────────────────────────────────────────────────────
    #  BLOCO 3: TRIANGULAÇÃO DLT
    # ─────────────────────────────────────────────────────────────────────────

    def triangular_landmarks(
        self,
        lm_a: np.ndarray,    # (33, 2) — coordenadas pixel câmera A
        lm_b: np.ndarray,    # (33, 2) — coordenadas pixel câmera B
        conf_a: np.ndarray,  # (33,)   — visibilidade câmera A
        conf_b: np.ndarray,  # (33,)   — visibilidade câmera B
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pipeline DLT completo:

        1. Undistort — cv2.undistortPoints com (K, d) de cada câmera.
        2. Triangulação — cv2.triangulatePoints → homogêneo → euclidiano.
        3. Kalman update dinâmico — H_full com R escalado pela confiança.
        """
        if not self.is_stereo_calibrated or lm_a.size == 0 or lm_b.size == 0:
            return np.zeros((33, 3), dtype=np.float32), np.zeros(33, dtype=np.float32)

        cal = self.calibracao

        pts_ud_a = cv2.undistortPoints(
            lm_a.reshape(-1, 1, 2).astype(np.float32), cal.K_A, cal.d_A
        )
        pts_ud_b = cv2.undistortPoints(
            lm_b.reshape(-1, 1, 2).astype(np.float32), cal.K_B, cal.d_B
        )

        P1     = np.hstack((np.eye(3), np.zeros((3, 1)))).astype(np.float32)
        P2     = np.hstack((cal.R, cal.T)).astype(np.float32)
        pts_4d = cv2.triangulatePoints(P1, P2, pts_ud_a, pts_ud_b)
        w4     = pts_4d[3]
        pts_3d = (pts_4d[:3] / np.where(np.abs(w4) > 1e-8, w4, 1e-8)).T

        conf_3d = conf_a * conf_b

        for i in range(33):
            kf = self.filtros_kalman[i]
            kf.predict()
            if conf_3d[i] >= 0.20:
                R_dyn     = kf.R_stereo / max(float(conf_3d[i]), 0.1)
                pts_3d[i] = kf.update(pts_3d[i], kf.H_full, R_dyn)
            else:
                pts_3d[i] = kf.x[:3, 0]

        return np.float32(pts_3d), np.float32(conf_3d)

    # ─────────────────────────────────────────────────────────────────────────
    #  BLOCO 4: FUSÃO LiDAR
    # ─────────────────────────────────────────────────────────────────────────

    def fundir_lidar(
        self,
        pts_3d: np.ndarray,     # (33, 3)
        depth_map: np.ndarray,  # (H, W)
        lm_pixel: np.ndarray,   # (33, 2)
    ) -> np.ndarray:
        """
        Fusão de profundidade LiDAR via update exclusivo do eixo Z.

        px/py convertidos para int com numpy antes do loop.
        Máscara de validade computada vetorialmente. O loop Kalman por
        landmark não é eliminável sem reestruturar o filtro, mas iteramos
        apenas sobre os índices válidos (~2–10 dos 33).
        """
        if self.config_lidar is None or depth_map is None:
            return pts_3d

        cfg          = self.config_lidar
        h_d, w_d     = depth_map.shape[:2]
        pts_out      = pts_3d.copy()

        px = lm_pixel[:, 0].astype(np.int32)
        py = lm_pixel[:, 1].astype(np.int32)
        validos = np.where(
            (px >= 0) & (px < w_d) & (py >= 0) & (py < h_d)
        )[0]

        for i in validos:
            raw_depth = float(depth_map[py[i], px[i]])
            if raw_depth <= 0 or not np.isfinite(raw_depth):
                continue
            z_metros = raw_depth * cfg.escala_profundidade
            kf = self.filtros_kalman[i]
            pts_out[i] = kf.update(
                np.array([z_metros], dtype=np.float32),
                kf.H_z,
                kf.R_lidar,
            )

        return np.float32(pts_out)

    # ─────────────────────────────────────────────────────────────────────────
    #  BLOCO 5: CÁLCULO DE ÂNGULO
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _calcular_angulo_clinico(
        p_a: np.ndarray,
        p_b: np.ndarray,
        p_c: np.ndarray,
    ) -> float:
        """Ângulo no vértice p_b formado pelos segmentos p_b→p_a e p_b→p_c."""
        v1, v2 = p_a - p_b, p_c - p_b
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_theta = float(np.dot(v1, v2)) / (n1 * n2)
        return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

    def _calcular_angulo_para_movimento(
        self,
        pts_3d: dict,
    ) -> tuple[float, str, str, str]:
        """
        Retorna (ângulo_bruto, nome_a, nome_b, nome_c) para o tipo de movimento.

        O dispatch é derivado exclusivamente de MOVEMENT_SPECS. Movimentos sem
        especificação clínica explícita são rejeitados em vez de cair em
        heurísticas geométricas legadas.
        """
        mv = self.tipo_movimento

        # Lookup O(1) para movimentos simples
        if mv in self._dispatch_angulo:
            _, key_a, key_b, key_c = self._dispatch_angulo[mv]
            ang = self._calcular_angulo_clinico(
                pts_3d.get(key_a, np.zeros(3)),
                pts_3d.get(key_b, np.zeros(3)),
                pts_3d.get(key_c, np.zeros(3)),
            )
            return ang, key_a, key_b, key_c

        raise ValueError(
            f"Movimento '{mv}' não possui especificação clínica autoritativa. "
            "Defina-o em MOVEMENT_SPECS antes de habilitá-lo no pipeline."
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  BLOCO 5b: PONTOS CLÍNICOS
    # ─────────────────────────────────────────────────────────────────────────

    def _obter_pontos_clinicos(
        self,
        landmarks_2d,
        landmarks_3d,
        w: int,
        h: int,
    ) -> tuple[dict, dict]:
        """
        Extrai dicionários de pontos clínicos 2D (pixel) e 3D (metros).

        Retorna:
          dict_3d — {nome: np.ndarray(3,)} para _calcular_angulo_para_movimento
          dict_2d — {nome: (int, int)} para _desenhar_overlay
        """
        dict_2d: dict = {}
        dict_3d: dict = {}

        for nome, enum_val in self.nodes.items():
            idx        = enum_val.value
            lm2d       = landmarks_2d[idx]
            lm3d       = landmarks_3d[idx]
            dict_2d[nome] = (int(lm2d.x * w), int(lm2d.y * h))
            dict_3d[nome] = np.array([lm3d.x, lm3d.y, lm3d.z], dtype=np.float32)

        # Centro da mão — midpoint entre indicador e mindinho
        if "indicador" in dict_3d and "mindinho" in dict_3d:
            dict_3d["centro_mao"] = (dict_3d["indicador"] + dict_3d["mindinho"]) * 0.5

        return dict_3d, dict_2d

    def _visibilidade_media(
        self,
        landmarks,
        n_a: str,
        n_b: str,
        n_c: str,
    ) -> float:
        """
        Retorna a visibilidade média dos três landmarks clínicos.

        v4.0: elimina lista intermediária — acumula soma e contagem diretamente.
        """
        total = 0.0
        count = 0
        for nome in (n_a, n_b, n_c):
            if nome in self.nodes:
                total += float(landmarks[self.nodes[nome].value].visibility)
                count += 1
        return total / count if count > 0 else 0.0

    def _angulo_clinico(self, angulo_geometrico: float) -> float:
        """Converte ângulo segmentar para a convenção clínica definida.

        Movimentos sem especificação explícita permanecem no modo legado e não
        devem ser apresentados como clinicamente habilitados pela interface.
        """
        spec = get_movement_spec(self.tipo_movimento)
        if spec is not None:
            return spec.clinical_angle(angulo_geometrico)
        return float(angulo_geometrico)

    @staticmethod
    def _point2d_telemetry(
        pt: tuple[int, int] | None,
        w: int,
        h: int,
        confidence: float | None = None,
    ) -> Point2DTelemetry | None:
        if pt is None or w <= 0 or h <= 0:
            return None
        return Point2DTelemetry(
            x=float(np.clip(pt[0] / w, 0.0, 1.0)),
            y=float(np.clip(pt[1] / h, 0.0, 1.0)),
            confidence=confidence,
        )

    @staticmethod
    def _point3d_telemetry(pt: np.ndarray | None) -> Point3DTelemetry | None:
        if pt is None:
            return None
        arr = np.asarray(pt, dtype=np.float64).reshape(-1)
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            return None
        return Point3DTelemetry(
            x=float(arr[0]),
            y=float(arr[1]),
            z=float(arr[2]),
        )

    def _atualizar_repeticoes_clinicas(self, angulo_clinico: float) -> None:
        """Contagem com histerese na escala clínica do movimento.

        Os limiares são operacionais para segmentação de ciclo, não limites
        diagnósticos ou valores normativos do paciente.
        """
        spec = get_movement_spec(self.tipo_movimento)
        if spec is None:
            return
        if angulo_clinico >= spec.enter_flexion_deg and self.fase_movimento != "flexao":
            self.fase_movimento = "flexao"
        elif angulo_clinico <= spec.complete_extension_deg and self.fase_movimento == "flexao":
            self.fase_movimento = "extensao"
            self.contador_reps += 1

    # ─────────────────────────────────────────────────────────────────────────
    #  BLOCO 6: CENTRO DE MASSA — VETORIZADO
    # ─────────────────────────────────────────────────────────────────────────

    def calcular_centro_de_massa_3d(self, skeleton: np.ndarray) -> np.ndarray:
        """
        CoM global — modelo segmentar de Winter (1990). Versão single-frame.

        Uses _WINTER_PESOS_SOMA (class constant) em vez de recomputar sum().
        """
        com_segs  = (skeleton[self._WINTER_PROXIMAL] + skeleton[self._WINTER_DISTAL]) * 0.5
        com_total = np.einsum("sd,s->d", com_segs, self._WINTER_PESOS)
        return (com_total / self._WINTER_PESOS_SOMA).astype(np.float32)

    def calcular_com_batch(self, skeleton_frames: np.ndarray) -> np.ndarray:
        """
        Versão vetorizada do CoM para pós-processamento em lote.

        skeleton_frames: (N, 33, 3) → retorna (N, 3).
        ~40× mais rápido que N chamadas de calcular_centro_de_massa_3d().
        """
        proximal = skeleton_frames[:, self._WINTER_PROXIMAL, :]
        distal   = skeleton_frames[:, self._WINTER_DISTAL, :]
        com_segs = (proximal + distal) * 0.5
        com_3d   = np.einsum("nsd,s->nd", com_segs, self._WINTER_PESOS)
        return (com_3d / self._WINTER_PESOS_SOMA).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    #  BLOCO 7: ANÁLISE DE MARCHA
    # ─────────────────────────────────────────────────────────────────────────

    def calcular_parametros_marcha(
        self,
        skeleton_frames: list[np.ndarray],
        fps: float,
    ) -> dict:
        """
        Calcula parâmetros espaço-temporais de marcha a partir do skeleton 3D.

        Retorna: stride_length_m, step_width_m, cadencia_passos_min,
                 simetria_vetorial.

        v4.0: cálculo de `widths` vetorizado com numpy indexing em vez de
        list comprehension Python — elimina loop explícito sobre strikes_d.
        """
        vazio = {
            "stride_length_m":      0.0,
            "step_width_m":         0.0,
            "cadencia_passos_min":  0.0,
            "simetria_vetorial":    0.0,
        }
        if len(skeleton_frames) < int(fps):
            return vazio

        skel = np.array(skeleton_frames, dtype=np.float32)  # (N, 33, 3)

        # MediaPipe Pose: LEFT_HEEL=29, RIGHT_HEEL=30.
        heel_d_y = skel[:, 30, 1]
        heel_e_y = skel[:, 29, 1]

        min_dist_frames = int(fps * 0.4)
        strikes_d, _ = find_peaks(-heel_d_y, distance=min_dist_frames, prominence=0.02)
        strikes_e, _ = find_peaks(-heel_e_y, distance=min_dist_frames, prominence=0.02)

        def _stride_lengths(strikes: np.ndarray, heel_idx: int) -> list[float]:
            if len(strikes) < 2:
                return []
            return [
                float(np.linalg.norm(
                    skel[strikes[i + 1], heel_idx, [0, 2]] - skel[strikes[i], heel_idx, [0, 2]]
                ))
                for i in range(len(strikes) - 1)
            ]

        strides_d = _stride_lengths(strikes_d, 30)
        strides_e = _stride_lengths(strikes_e, 29)

        # v4.0: vetorizado — elimina list comprehension Python sobre strikes_d
        if len(strikes_d) > 0:
            widths = np.abs(skel[strikes_d, 30, 0] - skel[strikes_d, 29, 0])
            mean_width = float(widths.mean())
        else:
            mean_width = 0.0

        simetria = 0.0
        n_min    = min(len(strides_d), len(strides_e))
        if n_min >= 2:
            arr_d = np.array(strides_d[:n_min])
            arr_e = np.array(strides_e[:n_min])
            if arr_d.std() > 1e-6 and arr_e.std() > 1e-6:
                simetria = float(np.corrcoef(arr_d, arr_e)[0, 1])

        duracao       = len(skeleton_frames) / fps
        todos_strides = strides_d + strides_e

        return {
            "stride_length_m":      float(np.mean(todos_strides)) if todos_strides else 0.0,
            "step_width_m":         mean_width,
            "cadencia_passos_min":  float(((len(strikes_d) + len(strikes_e)) / duracao) * 60) if duracao > 0 else 0.0,
            "simetria_vetorial":    simetria,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  BLOCO 8: PIPELINE ESTÉREO PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────────

    def processar_estereo(
        self,
        video_path_a: str,
        video_path_b: str | None,
        depth_path: str | None,
        tipo_lidar: str | None,
        offset_ms: float,
        progress_callback: Callable[[float], None] | None,
        telemetry_callback: Callable[[FrameMeasurement], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
        armazenar_frames: bool = False,
    ) -> ResultadoAnalise:
        """Processa o vídeo em streaming e emite uma medição auditável por frame."""
        cap_a = cv2.VideoCapture(video_path_a)
        if not cap_a.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo A: {video_path_a}")

        cap_b = cv2.VideoCapture(video_path_b) if video_path_b else None
        if cap_b is not None and not cap_b.isOpened():
            logger.warning("Vídeo B não pôde ser aberto; continuando em modo monocular.")
            cap_b.release()
            cap_b = None

        fps = float(cap_a.get(cv2.CAP_PROP_FPS)) or 30.0
        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        total_f = max(int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

        self._resetar_estado_sessao()
        for filtro in self.filtros_kalman:
            filtro.set_dt(1.0 / fps)

        res_frames: list[np.ndarray] = []
        skel_3d: list[np.ndarray] = []
        angulos_clinicos: list[float] = []
        angulos_brutos: list[float] = []
        angulos_geo_filtrados: list[float] = []
        vels: list[float] = []
        measurements: list[FrameMeasurement] = []
        clinical_min: float | None = None
        clinical_max: float | None = None
        target_w, target_h = self._RESOLUCAO_PLAYER
        movement_spec = get_movement_spec(self.tipo_movimento)
        if movement_spec is None:
            raise ValueError(
                f"Movimento '{self.tipo_movimento}' não possui especificação clínica autoritativa."
            )

        frame_stream = self.iterar_frames_sincronizados(cap_a, cap_b, offset_ms)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                for i, fs in enumerate(frame_stream):
                    if cancel_callback and cancel_callback():
                        raise AnaliseCancelada("Análise cancelada pelo usuário.")

                    frame_t0 = time.perf_counter()
                    img_a = cv2.cvtColor(fs.frame_a, cv2.COLOR_BGR2RGB)
                    h, w = img_a.shape[:2]

                    raw_geo: float | None = None
                    measured_geo: float | None = None
                    plane_geo: float | None = None
                    filtered_geo: float | None = None
                    clinical: float | None = None
                    vel_ang: float | None = None
                    conf: float | None = None
                    source = "predicted"
                    measurement_space = "kalman_predicted_3d"
                    body_frame_confidence: float | None = None
                    view_alignment: float | None = None
                    view_deviation: float | None = None
                    out_of_plane: float | None = None
                    direction_alignment: float | None = None
                    projection_applied = False
                    dict_2d: dict = {}
                    dict_3d: dict = {}
                    overlay_landmarks_2d: list[Point2DTelemetry | None] = []
                    n_a, n_b, n_c = "ombro", "cotovelo", "pulso"

                    inference_t0 = time.perf_counter()
                    res_b: object = None
                    fut_a = executor.submit(self.pose_estimators[0].process, img_a)

                    if fs.frame_b is not None and fs.qualidade_sinc >= self._QUALIDADE_SINC_MIN:
                        img_b = cv2.cvtColor(fs.frame_b, cv2.COLOR_BGR2RGB)
                        fut_b = executor.submit(self.pose_estimators[1].process, img_b)
                        res_a = fut_a.result()
                        res_b = fut_b.result()
                    else:
                        res_a = fut_a.result()
                    inference_ms = (time.perf_counter() - inference_t0) * 1000.0

                    pts_3d = np.zeros((33, 3), dtype=np.float32)

                    if res_a.pose_landmarks:
                        landmarks_2d = res_a.pose_landmarks.landmark
                        lm_a = np.array(
                            [[lm.x * w, lm.y * h] for lm in landmarks_2d],
                            dtype=np.float32,
                        )
                        cf_a = np.array([lm.visibility for lm in landmarks_2d], dtype=np.float32)
                        # Coordenadas normalizadas da mesma detecção que alimenta
                        # a medição clínica. O frontend usa esta lista apenas para
                        # desenhar a cadeia anatômica selecionada sobre o vídeo.
                        overlay_landmarks_2d = [
                            Point2DTelemetry(
                                x=float(np.clip(lm.x, 0.0, 1.0)),
                                y=float(np.clip(lm.y, 0.0, 1.0)),
                                confidence=float(lm.visibility),
                            )
                            for lm in landmarks_2d
                        ]

                        usa_estereo = (
                            res_b is not None
                            and getattr(res_b, "pose_landmarks", None) is not None
                            and self.is_stereo_calibrated
                            and fs.qualidade_sinc >= self._QUALIDADE_SINC_MIN
                        )

                        raw_world_pts: np.ndarray | None = None
                        if usa_estereo:
                            img_b_h, img_b_w = fs.frame_b.shape[:2]
                            lm_b = np.array(
                                [[lm.x * img_b_w, lm.y * img_b_h] for lm in res_b.pose_landmarks.landmark],
                                dtype=np.float32,
                            )
                            cf_b = np.array(
                                [lm.visibility for lm in res_b.pose_landmarks.landmark],
                                dtype=np.float32,
                            )
                            pts_3d, _ = self.triangular_landmarks(lm_a, lm_b, cf_a, cf_b)
                            measurement_space = "stereo_3d"
                            if fs.depth_map is not None and self.config_lidar is not None:
                                pts_3d = self.fundir_lidar(pts_3d, fs.depth_map, lm_a)
                        elif res_a.pose_world_landmarks:
                            measurement_space = "world_3d"
                            raw_world_pts = np.array(
                                [[lm.x, lm.y, lm.z] for lm in res_a.pose_world_landmarks.landmark],
                                dtype=np.float32,
                            )
                            for j in range(33):
                                kf = self.filtros_kalman[j]
                                predicted = kf.predict()
                                visibility = float(cf_a[j])
                                if visibility >= 0.20:
                                    r_dyn = kf.R_mono_full / max(visibility, 0.10)
                                    pts_3d[j] = kf.update(raw_world_pts[j], kf.H_full, r_dyn)
                                else:
                                    pts_3d[j] = predicted

                        world_wrappers = [_PtWrapper(pts_3d[j]) for j in range(33)]
                        dict_3d, dict_2d = self._obter_pontos_clinicos(
                            landmarks_2d, world_wrappers, w, h
                        )
                        measured_geo, n_a, n_b, n_c = self._calcular_angulo_para_movimento(dict_3d)
                        conf_values = [
                            float(landmarks_2d[self.nodes[name].value].visibility)
                            for name in (n_a, n_b, n_c)
                            if name in self.nodes
                        ]
                        conf = min(conf_values) if conf_values else 0.0
                        # Se algum landmark clínico não é observável, o vetor 3D
                        # correspondente depende da predição temporal do Kalman.
                        source = "observed" if len(conf_values) == 3 and conf >= 0.20 else "predicted"
                        if source == "predicted":
                            measurement_space = "kalman_predicted_3d"

                        if raw_world_pts is not None:
                            raw_wrappers = [_PtWrapper(raw_world_pts[j]) for j in range(33)]
                            raw_dict, _ = self._obter_pontos_clinicos(
                                landmarks_2d, raw_wrappers, w, h
                            )
                            raw_geo, _, _, _ = self._calcular_angulo_para_movimento(raw_dict)
                        else:
                            raw_geo = measured_geo

                        body_frame = build_body_frame(pts_3d)
                        bilateral_conf = [
                            float(landmarks_2d[idx].visibility)
                            for idx in (11, 12, 23, 24)
                            if idx < len(landmarks_2d)
                        ]
                        body_frame_confidence = min(bilateral_conf) if len(bilateral_conf) == 4 else None
                        if body_frame is not None:
                            plane_geo = projected_internal_angle_deg(
                                dict_3d.get(n_a), dict_3d.get(n_b), dict_3d.get(n_c),
                                frame=body_frame, plane=movement_spec.measurement_plane,
                            )
                            if plane_geo is not None:
                                projection_applied = True
                            view_alignment = camera_view_alignment(
                                body_frame, movement_spec.measurement_plane
                            )
                            view_deviation = alignment_deviation_deg(view_alignment)
                            out_of_plane = max_out_of_plane_deg(
                                dict_3d.get(n_a), dict_3d.get(n_b), dict_3d.get(n_c),
                                frame=body_frame, plane=movement_spec.measurement_plane,
                            )
                            if movement_spec.direction_rule and dict_3d.get(n_b) is not None and dict_3d.get(n_c) is not None:
                                direction_alignment = movement_direction_alignment(
                                    np.asarray(dict_3d.get(n_c), dtype=np.float64) - np.asarray(dict_3d.get(n_b), dtype=np.float64),
                                    frame=body_frame,
                                    plane=movement_spec.measurement_plane,
                                    direction_rule=movement_spec.direction_rule,
                                    side=self.lado_do_corpo,
                                )

                        angle_for_filter = plane_geo if plane_geo is not None else measured_geo
                        filtered_geo = (
                            self.alpha * angle_for_filter + (1.0 - self.alpha) * self.angulo_suavizado
                            if self.angulo_suavizado is not None
                            else angle_for_filter
                        )
                        self.angulo_suavizado = filtered_geo

                    elif self.angulo_suavizado is not None:
                        for j, kf in enumerate(self.filtros_kalman):
                            pts_3d[j] = kf.predict()
                        pred = {
                            nome: pts_3d[enum_val.value]
                            for nome, enum_val in self.nodes.items()
                        }
                        if "indicador" in pred and "mindinho" in pred:
                            pred["centro_mao"] = (pred["indicador"] + pred["mindinho"]) * 0.5
                        dict_3d = pred
                        measured_geo, n_a, n_b, n_c = self._calcular_angulo_para_movimento(pred)
                        raw_geo = measured_geo
                        body_frame = build_body_frame(pts_3d)
                        if body_frame is not None:
                            plane_geo = projected_internal_angle_deg(
                                dict_3d.get(n_a), dict_3d.get(n_b), dict_3d.get(n_c),
                                frame=body_frame, plane=movement_spec.measurement_plane,
                            )
                            projection_applied = plane_geo is not None
                            view_alignment = camera_view_alignment(
                                body_frame, movement_spec.measurement_plane
                            )
                            view_deviation = alignment_deviation_deg(view_alignment)
                            out_of_plane = max_out_of_plane_deg(
                                dict_3d.get(n_a), dict_3d.get(n_b), dict_3d.get(n_c),
                                frame=body_frame, plane=movement_spec.measurement_plane,
                            )
                            if movement_spec.direction_rule and dict_3d.get(n_b) is not None and dict_3d.get(n_c) is not None:
                                direction_alignment = movement_direction_alignment(
                                    np.asarray(dict_3d.get(n_c), dtype=np.float64) - np.asarray(dict_3d.get(n_b), dtype=np.float64),
                                    frame=body_frame,
                                    plane=movement_spec.measurement_plane,
                                    direction_rule=movement_spec.direction_rule,
                                    side=self.lado_do_corpo,
                                )
                        angle_for_filter = plane_geo if plane_geo is not None else measured_geo
                        filtered_geo = (
                            self.alpha * angle_for_filter
                            + (1.0 - self.alpha) * self.angulo_suavizado
                        )
                        self.angulo_suavizado = filtered_geo
                        conf = 0.0
                        source = "predicted"

                    # Qualidade é calculada a partir da menor visibilidade dos
                    # três landmarks. Isso NÃO representa acurácia clínica.
                    def lm_conf(nome: str) -> float | None:
                        if not res_a.pose_landmarks or nome not in self.nodes:
                            return None
                        return float(
                            res_a.pose_landmarks.landmark[self.nodes[nome].value].visibility
                        )

                    quality_info = tracking_quality(
                        [lm_conf(n_a), lm_conf(n_b), lm_conf(n_c)],
                        source=source,
                    )
                    quality_info = apply_plane_quality(
                        quality_info,
                        body_frame_available=view_alignment is not None,
                        body_frame_confidence=body_frame_confidence,
                        view_alignment=view_alignment,
                        min_view_alignment=movement_spec.min_view_alignment,
                        out_of_plane_deg=out_of_plane,
                        max_out_of_plane_deg=movement_spec.max_out_of_plane_deg,
                        projection_applied=projection_applied,
                        direction_rule=movement_spec.direction_rule,
                        direction_alignment=direction_alignment,
                        min_direction_alignment=movement_spec.min_direction_alignment,
                    )

                    if filtered_geo is not None:
                        clinical = self._angulo_clinico(filtered_geo)
                        previous_clinical = self.angulo_anterior
                        vel_ang = (
                            (clinical - previous_clinical) * fps
                            if previous_clinical is not None
                            else 0.0
                        )
                        self.angulo_anterior = clinical

                        if bool(quality_info["measurement_valid"]):
                            clinical_min = (
                                clinical if clinical_min is None else min(clinical_min, clinical)
                            )
                            clinical_max = (
                                clinical if clinical_max is None else max(clinical_max, clinical)
                            )
                            self._atualizar_repeticoes_clinicas(clinical)

                    adm = (
                        clinical_max - clinical_min
                        if clinical_min is not None and clinical_max is not None
                        else 0.0
                    )

                    proximal = self._point2d_telemetry(
                        dict_2d.get(n_a), w, h, lm_conf(n_a)
                    )
                    vertex = self._point2d_telemetry(
                        dict_2d.get(n_b), w, h, lm_conf(n_b)
                    )
                    distal = self._point2d_telemetry(
                        dict_2d.get(n_c), w, h, lm_conf(n_c)
                    )
                    proximal_3d = self._point3d_telemetry(dict_3d.get(n_a))
                    vertex_3d = self._point3d_telemetry(dict_3d.get(n_b))
                    distal_3d = self._point3d_telemetry(dict_3d.get(n_c))

                    if armazenar_frames:
                        if filtered_geo is not None and dict_2d:
                            self._desenhar_overlay(
                                img_a, dict_2d, n_a, n_b, n_c,
                                clinical if clinical is not None else filtered_geo,
                                vel_ang or 0.0,
                                conf or 0.0,
                            )
                        if img_a.shape[1] != target_w or img_a.shape[0] != target_h:
                            img_a = cv2.resize(
                                img_a, (target_w, target_h), interpolation=cv2.INTER_AREA
                            )

                    pipeline_ms = (time.perf_counter() - frame_t0) * 1000.0
                    processing_fps = 1000.0 / pipeline_ms if pipeline_ms > 0 else None
                    measurement = FrameMeasurement(
                        frame=i,
                        timestamp_ms=float(
                            fs.timestamp
                            if np.isfinite(fs.timestamp)
                            else (i / fps * 1000.0)
                        ),
                        joint=(
                            f"{self.lado_do_corpo}_{get_movement_spec(self.tipo_movimento).joint_code}"
                            if get_movement_spec(self.tipo_movimento) is not None
                            else n_b
                        ),
                        movement=self.tipo_movimento,
                        side=self.lado_do_corpo,
                        angle_raw_deg=(
                            round(float(raw_geo), 4) if raw_geo is not None else None
                        ),
                        angle_landmark_deg=(
                            round(float(measured_geo), 4)
                            if measured_geo is not None
                            else None
                        ),
                        angle_plane_deg=(
                            round(float(plane_geo), 4) if plane_geo is not None else None
                        ),
                        angle_filtered_deg=(
                            round(float(filtered_geo), 4)
                            if filtered_geo is not None
                            else None
                        ),
                        angle_clinical_deg=(
                            round(float(clinical), 4) if clinical is not None else None
                        ),
                        angular_velocity_deg_s=(
                            round(float(vel_ang), 4) if vel_ang is not None else None
                        ),
                        adm_deg=round(float(adm), 4),
                        confidence=(
                            quality_info["landmark_confidence_min"]
                            if quality_info["landmark_confidence_min"] is not None
                            else None
                        ),
                        source=source,
                        measurement_space=measurement_space,
                        measurement_plane=movement_spec.measurement_plane,
                        plane_projection_applied=projection_applied,
                        body_frame_confidence=(
                            round(float(body_frame_confidence), 4)
                            if body_frame_confidence is not None else None
                        ),
                        view_alignment=(
                            round(float(view_alignment), 4) if view_alignment is not None else None
                        ),
                        view_deviation_deg=(
                            round(float(view_deviation), 4) if view_deviation is not None else None
                        ),
                        out_of_plane_deg=(
                            round(float(out_of_plane), 4) if out_of_plane is not None else None
                        ),
                        direction_rule=movement_spec.direction_rule,
                        direction_alignment=(
                            round(float(direction_alignment), 4)
                            if direction_alignment is not None else None
                        ),
                        landmark_confidence_min=quality_info["landmark_confidence_min"],
                        landmark_confidence_mean=quality_info["landmark_confidence_mean"],
                        quality=str(quality_info["quality"]),
                        measurement_valid=bool(quality_info["measurement_valid"]),
                        quality_flags=list(quality_info["quality_flags"]),
                        proximal=proximal,
                        vertex=vertex,
                        distal=distal,
                        proximal_3d=proximal_3d,
                        vertex_3d=vertex_3d,
                        distal_3d=distal_3d,
                        inference_ms=round(inference_ms, 3),
                        pipeline_ms=round(pipeline_ms, 3),
                        processing_fps=(
                            round(processing_fps, 2)
                            if processing_fps is not None
                            else None
                        ),
                        overlay_landmarks_2d=overlay_landmarks_2d,
                    )

                    if armazenar_frames:
                        res_frames.append(img_a)
                    skel_3d.append(pts_3d)
                    angulos_brutos.append(float(raw_geo or 0.0))
                    angulos_geo_filtrados.append(float(filtered_geo or 0.0))
                    angulos_clinicos.append(float(clinical or 0.0))
                    vels.append(float(vel_ang or 0.0))
                    measurements.append(measurement)

                    if telemetry_callback:
                        telemetry_callback(measurement)
                    if progress_callback:
                        progress_callback(min((i + 1) / total_f, 1.0))
        finally:
            cap_a.release()
            if cap_b:
                cap_b.release()

        if not measurements:
            raise RuntimeError("O vídeo não contém frames decodificáveis.")

        skel_arr = np.asarray(skel_3d, dtype=np.float32)
        com_traj = list(self.calcular_com_batch(skel_arr)) if len(skel_arr) else []

        # Classificador de fluidez legado desativado até treinamento/validação.
        fluidez = 0.0
        parametros_marcha = (
            self.calcular_parametros_marcha(skel_3d, fps)
            if self.tipo_movimento == "Marcha"
            else {}
        )

        return ResultadoAnalise(
            frames_a=res_frames,
            skeleton_3d=skel_3d,
            com_trajetoria=com_traj,
            angulos_filtrados=angulos_clinicos,
            angulos_brutos=angulos_brutos,
            angulos_geometricos_filtrados=angulos_geo_filtrados,
            frame_measurements=measurements,
            velocidades=vels,
            total_frames=len(measurements),
            total_reps=self.contador_reps,
            fps=fps,
            point_cloud=None,
            parametros_marcha=parametros_marcha,
            fluidez_ia=fluidez,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  BLOCO 8b: ENTRY POINT PÚBLICO — processar_multicamera
    # ─────────────────────────────────────────────────────────────────────────

    def processar_multicamera(
        self,
        path_a: str,
        path_b: str | None = None,
        path_calib: str | None = None,
        path_depth: str | None = None,
        tipo_sensor: str = "Nenhum",
        offset_ms: float = 0.0,
        progress_callback: Callable[[float], None] | None = None,
        telemetry_callback: Callable[[FrameMeasurement], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
        armazenar_frames: bool = False,
    ) -> ResultadoAnalise:
        """
        Entry point público chamado pelo app.py.

        v3.1: método inexistente na v3.0 — o app.py recebia AttributeError.

        v4.0 — BUG CORRIGIDO: np.lib.npyio.NpzFile não implementa .get().
        dados.get("baseline_m", 0.0) levantava AttributeError silencioso ao
        carregar calibração com chaves opcionais. Corrigido com guarda
        `"chave" in dados` antes do acesso.

        Parâmetros:
            path_a          — vídeo câmera A (obrigatório)
            path_b          — câmera B (opcional, modo estéreo)
            path_calib      — arquivo .npz com parâmetros de calibração
            path_depth      — arquivo de profundidade LiDAR (npy/npz)
            tipo_sensor     — "Nenhum" | "Intel RealSense" | "Azure Kinect" | ...
            offset_ms       — offset temporal A→B em milissegundos
            progress_callback — callback(float 0→1) para barra de progresso
        """
        # ── 1. Carrega calibração estéreo (se fornecida) ──────────────────
        if path_calib is not None:
            try:
                dados = np.load(path_calib)
                # FIX v4.0: NpzFile não tem .get() — usar guard "key in dados"
                self.carregar_calibracao(
                    K_A=dados["K_A"],
                    d_A=dados["d_A"],
                    K_B=dados["K_B"],
                    d_B=dados["d_B"],
                    R=dados["R"],
                    T=dados["T"],
                    baseline_m=float(dados["baseline_m"]) if "baseline_m" in dados else 0.0,
                    rms_reprojection=float(dados["rms_reprojection"]) if "rms_reprojection" in dados else 0.0,
                )
            except Exception as exc:
                warnings.warn(
                    f"[PhysioAI] Falha ao carregar calibração '{path_calib}': {exc}. "
                    "Continuando em modo monocular.",
                    stacklevel=2,
                )
                logger.warning("Calibração inválida: %s", exc)

        # ── 2. Configura LiDAR (se fornecido e sensor ≠ "Nenhum") ─────────
        if path_depth is not None and tipo_sensor != "Nenhum":
            escala = self._ESCALAS_LIDAR.get(tipo_sensor, 0.001)
            self.carregar_config_lidar(ConfigLiDAR(
                tipo=tipo_sensor,
                extrinsica_rgb=np.eye(4, dtype=np.float32),
                escala_profundidade=escala,
            ))

        # ── 3. Delega para o pipeline principal ───────────────────────────
        return self.processar_estereo(
            video_path_a=path_a,
            video_path_b=path_b,
            depth_path=path_depth,
            tipo_lidar=tipo_sensor if tipo_sensor != "Nenhum" else None,
            offset_ms=offset_ms,
            progress_callback=progress_callback,
            telemetry_callback=telemetry_callback,
            cancel_callback=cancel_callback,
            armazenar_frames=armazenar_frames,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  MODELO IA
    # ─────────────────────────────────────────────────────────────────────────

    def carregar_modelo_ia(self, caminho_modelo: str = "src/models/modelo_pibic.pt") -> bool:
        """
        Carrega modelo TorchScript para predição de fluidez.

        Retorna True em sucesso, False se o arquivo não for encontrado ou
        o modelo for incompatível (logged como warning).
        """
        try:
            self.modelo_ia = torch.jit.load(caminho_modelo, map_location=self.device)
            self.modelo_ia.eval()
            if self.device.type == "cuda":
                self.modelo_ia = self.modelo_ia.half()
                try:
                    self.modelo_ia = torch.compile(self.modelo_ia)
                except Exception as exc:
                    logger.debug("torch.compile não disponível: %s", exc)
            else:
                self.modelo_ia = self.modelo_ia.float()
            logger.info("Modelo IA carregado: %s [%s]", caminho_modelo, self._backend_str)
            return True
        except Exception as exc:
            logger.warning("Falha ao carregar modelo IA '%s': %s", caminho_modelo, exc)
            return False

    def prever_fluidez(self, angulos: list) -> float:
        """
        Prevê índice de fluidez do movimento (0–100) via modelo TorchScript.

        Retorna 0.0 se o modelo não estiver carregado ou se a sequência for
        muito curta (<2 ângulos). Erros de inferência são logados e não
        propagados para não interromper o pipeline.
        """
        if self.modelo_ia is None or len(angulos) < 2:
            return 0.0
        seq    = self.normalizar_sequencia(angulos)
        tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
        if self.device.type == "cuda":
            tensor = tensor.half()
        tensor = tensor.view(1, 100, 1)
        try:
            with torch.no_grad(), _make_autocast(self.device.type, enabled=(self.device.type == "cuda")):
                output = self.modelo_ia(tensor)
            return round(float(torch.sigmoid(output).item()) * 100, 2)
        except Exception as exc:
            logger.warning("Erro na inferência de fluidez: %s", exc)
            return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    #  UTILITÁRIOS
    # ─────────────────────────────────────────────────────────────────────────

    def normalizar_sequencia(self, angulos: list, tamanho_alvo: int = 100) -> list:
        """Interpola a sequência de ângulos para tamanho_alvo pontos equidistantes."""
        arr = np.asarray(angulos, dtype=np.float32)
        if len(arr) < 2:
            return [0.0] * tamanho_alvo
        return np.interp(
            np.linspace(0, 1, tamanho_alvo),
            np.linspace(0, 1, len(arr)),
            arr,
        ).tolist()

    def limites_do_movimento(self) -> dict:
        """
        Retorna os limites clínicos de ADM para o tipo de movimento configurado.

        Usa valores padrão conservadores se o tipo não estiver mapeado, com
        aviso de log para facilitar diagnóstico.
        """
        limites = self._LIMITES_ADM.get(self.tipo_movimento)
        if limites is None:
            logger.warning(
                "Tipo de movimento '%s' não encontrado em _LIMITES_ADM. "
                "Usando padrão genérico.",
                self.tipo_movimento,
            )
            return {"max_adm_ref": 180, "th_flex": 90, "th_ext": 140, "angulo_diminui": False}
        return limites

    def _desenhar_overlay(
        self,
        img: np.ndarray,
        pts_2d: dict,
        nome_a: str,
        nome_b: str,
        nome_c: str,
        angulo: float,
        velocidade: float,
        confianca: float,
    ) -> None:
        """
        Renderiza esqueleto clínico + HUD no frame RGB.

        v3.1: eliminado img.copy() para o fundo semitransparente.
        v4.0: _hud_dark_buf pré-alocado no __init__ — elimina np.full_like()
        por frame. _backend_str cacheado no __init__ — elimina condicional
        device.type por frame.
        """
        h, w    = img.shape[:2]
        pa      = pts_2d.get(nome_a, (0, 0))
        pb      = pts_2d.get(nome_b, (0, 0))
        pc      = pts_2d.get(nome_c, (0, 0))
        cor_seg = (0, 220, 140) if confianca >= 0.70 else (255, 195, 0)

        cv2.line(img, pa, pb, cor_seg, 4, cv2.LINE_AA)
        cv2.line(img, pb, pc, cor_seg, 4, cv2.LINE_AA)
        for pt, r, c in [
            (pb, 11, (255, 255, 255)),
            (pb,  7, (30, 80, 220)),
            (pa,  6, cor_seg),
            (pc,  6, cor_seg),
        ]:
            cv2.circle(img, pt, r, c, -1, cv2.LINE_AA)

        if angulo > 0:
            tx  = int(np.clip(pb[0] - 45, 5, w - 100))
            ty  = int(np.clip(pb[1] + 58, 20, h - 10))
            txt = f"{angulo:.1f}°"
            cv2.putText(img, txt, (tx + 1, ty + 1), cv2.FONT_HERSHEY_DUPLEX, 0.95, (0, 0, 0),     2, cv2.LINE_AA)
            cv2.putText(img, txt, (tx,     ty    ), cv2.FONT_HERSHEY_DUPLEX, 0.95, (0, 230, 175),  2, cv2.LINE_AA)

        # v4.0: usa buffer pré-alocado em vez de np.full_like() por frame.
        # FIX: _hud_dark_buf tem largura fixa (target_w=640). Se o frame ainda
        # não foi redimensionado (resize acontece APÓS _desenhar_overlay no loop),
        # w pode ser maior que 640 → dark_roi e roi teriam shapes diferentes →
        # cv2.addWeighted levantava cv2.error (-209).
        # Solução: hud_w = min(w, buf_w) garante que ambos os arrays têm a
        # mesma largura, independente da resolução original do vídeo.
        hud_h = 54
        buf_w = self._hud_dark_buf.shape[1]          # largura do buffer (640)
        hud_w = min(w, buf_w)                         # largura segura para ambos
        roi      = img[:hud_h, :hud_w]
        dark_roi = self._hud_dark_buf[:hud_h, :hud_w]
        cv2.addWeighted(dark_roi, 0.74, roi, 0.26, 0, roi)
        img[:hud_h, :hud_w] = roi

        cv2.putText(
            img,
            f"PhysioAI v4.0  [{self._backend_str}]  |  Reps: {self.contador_reps}  |"
            f"  Fase: {self.fase_movimento.upper()}  |  Vel: {velocidade:.0f}°/s  |  {self.tipo_movimento}",
            (10, 34), cv2.FONT_HERSHEY_DUPLEX, 0.58, (200, 235, 255), 1, cv2.LINE_AA,
        )

        bx, by    = w - 115, 8
        bar_color = (0, 220, 100) if confianca >= 0.70 else (255, 170, 0)
        cv2.rectangle(img, (bx, by), (bx + 100,                  by + 12), (40, 44, 58), -1)
        cv2.rectangle(img, (bx, by), (bx + int(confianca * 100), by + 12), bar_color,    -1)
        cv2.putText(
            img,
            f"Conf {confianca:.0%}",
            (bx - 2, by + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 200, 255), 1, cv2.LINE_AA,
        )