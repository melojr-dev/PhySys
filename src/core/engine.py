import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from torch.cuda.amp import autocast
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# ESTRUTURA DE RESULTADO POR FRAME
# ---------------------------------------------------------------------------
@dataclass
class ResultadoFrame:
    angulo: float = 0.0
    velocidade: float = 0.0
    confianca: float = 0.0          # 0-1: média de visibilidade dos landmarks
    fase: str = "repouso"
    plano_movimento: str = ""       # sagital / frontal / transversal
    compensacao: bool = False       # detectou compensação postural?
    landmarks_2d: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CONFIGURAÇÕES POR MOVIMENTO
# ---------------------------------------------------------------------------
CONFIGS_MOVIMENTO = {
    "Flexão de Cotovelo": {
        "nos": ["ombro", "cotovelo", "pulso"],
        "plano": "sagital",
        "th_flex": 50,          # ângulo mínimo = flexão completa
        "th_ext": 155,          # ângulo máximo = extensão completa
        "adm_referencia": 145,  # ADM normal (graus)
        "inverter": False,      # ângulo já representa flexão diretamente
        "descricao": "Arco de movimento cotovelo: 0° (estendido) → 145° (fletido)",
    },
    "Abdução de Ombro": {
        "nos": ["cotovelo", "ombro", "quadril"],
        "plano": "frontal",
        "th_flex": 30,
        "th_ext": 160,
        "adm_referencia": 180,
        "inverter": False,
        "descricao": "Braço lateral: 0° (neutro) → 180° (vertical)",
    },
    "Flexão de Ombro": {
        "nos": ["cotovelo", "ombro", "quadril"],
        "plano": "sagital",
        "th_flex": 30,
        "th_ext": 160,
        "adm_referencia": 180,
        "inverter": False,
        "descricao": "Braço frontal: 0° (neutro) → 180° (vertical)",
    },
    "Desvio de Punho": {
        "nos": ["cotovelo", "pulso", "centro_mao"],
        "plano": "frontal",
        "th_flex": 15,
        "th_ext": 25,
        "adm_referencia": 55,
        "inverter": True,       # 180° = neutro → subtrai para obter desvio
        "descricao": "Desvio radial/ulnar: 0° (neutro) → ±25° (desvio)",
    },
    "Flexão de Punho": {
        "nos": ["cotovelo", "pulso", "centro_mao"],
        "plano": "sagital",
        "th_flex": 15,
        "th_ext": 25,
        "adm_referencia": 80,
        "inverter": True,
        "descricao": "Flexão/extensão punho: 0° (neutro) → 80° (flexão)",
    },
}


# ---------------------------------------------------------------------------
# FILTRO EMA ADAPTATIVO
# ---------------------------------------------------------------------------
class FiltroEMAAdaptativo:
    """
    EMA com alpha dinâmico: movimentos rápidos → segue mais fielmente,
    movimentos lentos / ruído → suaviza mais agressivamente.
    """
    def __init__(self, alpha_base=0.35, limiar_mudanca=8.0):
        self.alpha_base = alpha_base
        self.limiar = limiar_mudanca
        self.valor = None

    def atualizar(self, medicao: float, confianca: float = 1.0) -> float:
        if self.valor is None:
            self.valor = medicao
            return medicao

        delta = abs(medicao - self.valor)

        # Alpha aumenta proporcionalmente com a velocidade de mudança
        alpha_dinamico = self.alpha_base + (1 - self.alpha_base) * min(delta / (self.limiar * 3), 1.0)

        # Pondera alpha pela confiança: landmark pouco visível → mais suavização
        alpha_final = alpha_dinamico * confianca + self.alpha_base * (1 - confianca)
        alpha_final = float(np.clip(alpha_final, 0.05, 0.95))

        self.valor = alpha_final * medicao + (1 - alpha_final) * self.valor
        return self.valor

    def resetar(self):
        self.valor = None


# ---------------------------------------------------------------------------
# DETECTOR DE COMPENSAÇÃO POSTURAL
# ---------------------------------------------------------------------------
class DetectorCompensacao:
    """
    Detecta compensações comuns: inclinação de tronco, elevação de ombro
    e rotação de quadril durante o movimento.
    """
    def __init__(self, limiar_graus=12.0):
        self.limiar = limiar_graus
        self.postura_base: Optional[Dict] = None
        self.n_frames_base = 0
        self._acumulador: Dict[str, List[float]] = {"inclinacao": [], "elevacao": []}

    def calibrar(self, landmarks_mundo):
        """Chama nos primeiros N frames parado para estabelecer postura de referência."""
        lm = landmarks_mundo
        p = mp_pose.PoseLandmark

        ombro_e = np.array([lm[p.LEFT_SHOULDER.value].x, lm[p.LEFT_SHOULDER.value].y])
        ombro_d = np.array([lm[p.RIGHT_SHOULDER.value].x, lm[p.RIGHT_SHOULDER.value].y])
        quadril_e = np.array([lm[p.LEFT_HIP.value].x, lm[p.LEFT_HIP.value].y])
        quadril_d = np.array([lm[p.RIGHT_HIP.value].x, lm[p.RIGHT_HIP.value].y])

        inclinacao = abs(ombro_e[1] - ombro_d[1])
        altura_ombros = ((ombro_e[1] + ombro_d[1]) / 2)

        self._acumulador["inclinacao"].append(inclinacao)
        self._acumulador["elevacao"].append(altura_ombros)
        self.n_frames_base += 1

        if self.n_frames_base >= 10:
            self.postura_base = {
                "inclinacao": float(np.median(self._acumulador["inclinacao"])),
                "elevacao":   float(np.median(self._acumulador["elevacao"])),
            }

    def verificar(self, landmarks_mundo) -> bool:
        if self.postura_base is None:
            return False

        lm = landmarks_mundo
        p = mp_pose.PoseLandmark

        ombro_e = lm[p.LEFT_SHOULDER.value]
        ombro_d = lm[p.RIGHT_SHOULDER.value]

        inclinacao_atual = abs(ombro_e.y - ombro_d.y)
        elevacao_atual   = (ombro_e.y + ombro_d.y) / 2

        delta_inclinacao = abs(inclinacao_atual - self.postura_base["inclinacao"])
        delta_elevacao   = abs(elevacao_atual   - self.postura_base["elevacao"])

        # Elevação de ombro > limiar em coordenadas normalizadas ≈ compensação
        return (delta_inclinacao > 0.06) or (delta_elevacao > 0.05)


# ---------------------------------------------------------------------------
# ANALISADOR PRINCIPAL
# ---------------------------------------------------------------------------
class AnalisadorADMWeb:

    def __init__(self, id_paciente: str, tipo_movimento: str,
                 lado_do_corpo: str, usar_gpu: bool = True):
        self.id_paciente    = id_paciente
        self.tipo_movimento = tipo_movimento
        self.lado           = lado_do_corpo.lower()
        self.config         = CONFIGS_MOVIMENTO.get(tipo_movimento, {})

        # Hardware
        if usar_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.modelo_ia = None

        # Pose engine — tenta model_complexity=2, cai para 1 se falhar
        for complexity in (2, 1, 0):
            try:
                self.pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=complexity,
                    smooth_landmarks=True,          # suavização nativa do MediaPipe
                    enable_segmentation=False,
                    min_detection_confidence=0.55,
                    min_tracking_confidence=0.55,
                )
                break
            except Exception:
                continue

        # Estado por sessão de vídeo
        self._filtro       = FiltroEMAAdaptativo(alpha_base=0.35, limiar_mudanca=8.0)
        self._compensacao  = DetectorCompensacao()
        self._angulo_ant   = None
        self._contador_reps = 0
        self._fase          = "repouso"
        self._frames_calib  = 0

        self._configurar_nos()

    # ------------------------------------------------------------------
    # NODOS ANATÔMICOS
    # ------------------------------------------------------------------
    def _configurar_nos(self):
        esq = self.lado == "esquerdo"
        p = mp_pose.PoseLandmark
        self.nos = {
            "ombro":    p.LEFT_SHOULDER  if esq else p.RIGHT_SHOULDER,
            "cotovelo": p.LEFT_ELBOW     if esq else p.RIGHT_ELBOW,
            "pulso":    p.LEFT_WRIST     if esq else p.RIGHT_WRIST,
            "quadril":  p.LEFT_HIP       if esq else p.RIGHT_HIP,
            "joelho":   p.LEFT_KNEE      if esq else p.RIGHT_KNEE,
            "tornozelo":p.LEFT_ANKLE     if esq else p.RIGHT_ANKLE,
            "indicador":p.LEFT_INDEX     if esq else p.RIGHT_INDEX,
            "mindinho": p.LEFT_PINKY     if esq else p.RIGHT_PINKY,
            "polegar":  p.LEFT_THUMB     if esq else p.RIGHT_THUMB,
        }

    # ------------------------------------------------------------------
    # EXTRAÇÃO DE PONTOS
    # ------------------------------------------------------------------
    def _extrair_pontos(self, lm_norm, lm_mundo, w: int, h: int
                        ) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple], float]:
        """
        Retorna:
          pts_3d  — coordenadas 3D em metros (espaço mundo do MediaPipe)
          pts_2d  — pixels na imagem
          confianca — média de visibilidade dos landmarks usados
        """
        pts_3d: Dict[str, np.ndarray] = {}
        pts_2d: Dict[str, Tuple]      = {}
        visibilidades: List[float]     = []

        for nome, no in self.nos.items():
            idx = no.value
            lm_n = lm_norm[idx]
            lm_w = lm_mundo[idx]

            pts_3d[nome] = np.array([lm_w.x, lm_w.y, lm_w.z], dtype=np.float64)
            pts_2d[nome] = (int(lm_n.x * w), int(lm_n.y * h))
            visibilidades.append(getattr(lm_n, "visibility", 1.0))

        # Centro da mão (média de indicador + mindinho + polegar)
        pts_3d["centro_mao"] = (pts_3d["indicador"] + pts_3d["mindinho"] + pts_3d["polegar"]) / 3.0
        pts_2d["centro_mao"] = (
            int((pts_2d["indicador"][0] + pts_2d["mindinho"][0] + pts_2d["polegar"][0]) / 3),
            int((pts_2d["indicador"][1] + pts_2d["mindinho"][1] + pts_2d["polegar"][1]) / 3),
        )

        confianca = float(np.mean(visibilidades))
        return pts_3d, pts_2d, confianca

    # ------------------------------------------------------------------
    # CÁLCULO DE ÂNGULO
    # ------------------------------------------------------------------
    @staticmethod
    def _angulo_entre(p_a: np.ndarray, p_b: np.ndarray, p_c: np.ndarray) -> float:
        """Ângulo no vértice B formado pelos vetores BA e BC (0–180°)."""
        v1 = p_a - p_b
        v2 = p_c - p_b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_t = np.dot(v1, v2) / (n1 * n2)
        return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))

    def _calcular_angulo_movimento(self, pts_3d: Dict[str, np.ndarray]) -> float:
        """
        Seleciona os três pontos corretos e aplica transformação específica
        para cada tipo de movimento.
        """
        cfg = self.config
        if not cfg:
            return 0.0

        nos_mov = cfg["nos"]  # lista de 3 nomes

        try:
            p_a = pts_3d[nos_mov[0]]
            p_b = pts_3d[nos_mov[1]]
            p_c = pts_3d[nos_mov[2]]
        except KeyError:
            return 0.0

        angulo_bruto = self._angulo_entre(p_a, p_b, p_c)

        if cfg.get("inverter"):
            # Para punho: 180° = neutro, desvios são diferenças em relação a 180°
            angulo_bruto = abs(180.0 - angulo_bruto)

        # Garante intervalo 0–180
        return float(np.clip(angulo_bruto, 0.0, 180.0))

    # ------------------------------------------------------------------
    # DETECÇÃO DE FASE E REPETIÇÕES
    # ------------------------------------------------------------------
    def _atualizar_fase(self, angulo: float) -> Tuple[str, int]:
        cfg  = self.config
        th_f = cfg.get("th_flex", 60)
        th_e = cfg.get("th_ext", 140)

        mov = self.tipo_movimento

        # Para ombro e cotovelo: ângulo PEQUENO = flexão (cotovelo) ou neutro (ombro)
        # Para abdução: ângulo GRANDE = braço levantado
        if "Abdução" in mov or "Flexão de Ombro" in mov:
            # rep: começa em baixo (ângulo pequeno) → sobe (ângulo grande) → volta
            if angulo >= th_e and self._fase != "pico":
                self._fase = "pico"
            elif angulo <= th_f and self._fase == "pico":
                self._fase = "repouso"
                self._contador_reps += 1
        else:
            # Flexão de cotovelo / punho: ângulo diminui na flexão
            if angulo <= th_f and self._fase != "flexao":
                self._fase = "flexao"
            elif angulo >= th_e and self._fase == "flexao":
                self._fase = "extensao"
                self._contador_reps += 1
            elif angulo < th_e and self._fase == "extensao":
                self._fase = "repouso"

        return self._fase, self._contador_reps

    # ------------------------------------------------------------------
    # RENDERIZAÇÃO SOBRE O FRAME
    # ------------------------------------------------------------------
    def _renderizar(self, img: np.ndarray, pts_2d: Dict,
                    angulo: float, vel: float,
                    fase: str, reps: int, compensacao: bool) -> np.ndarray:

        cfg = self.config
        h, w = img.shape[:2]

        # Pontos do arco de movimento
        nos_mov = cfg.get("nos", [])
        pontos_arco = [pts_2d[n] for n in nos_mov if n in pts_2d]

        if len(pontos_arco) >= 2:
            for i in range(len(pontos_arco) - 1):
                cv2.line(img, pontos_arco[i], pontos_arco[i+1], (0, 230, 120), 3)
            # Vértice (articulação principal)
            vertice = pontos_arco[1] if len(pontos_arco) > 1 else pontos_arco[0]
            cv2.circle(img, vertice, 9, (0, 60, 255), -1)
            cv2.circle(img, vertice, 12, (255, 255, 255), 2)

            # Label de ângulo próximo ao vértice
            off_x, off_y = 15, -20
            cv2.putText(img, f"{angulo:.1f}°",
                        (vertice[0] + off_x, vertice[1] + off_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 240, 255), 2, cv2.LINE_AA)

        # Barra de progresso de ADM (lateral direita)
        adm_ref = cfg.get("adm_referencia", 180)
        pct     = min(angulo / max(adm_ref, 1), 1.0)
        bar_x, bar_y, bar_h = w - 28, int(h * 0.1), int(h * 0.8)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + 18, bar_y + bar_h),
                      (50, 50, 50), -1)
        cor_barra = (
            (0, 200, 80)   if pct < 0.6 else
            (0, 200, 220)  if pct < 0.85 else
            (0, 80, 255)
        )
        preench = int(bar_h * pct)
        cv2.rectangle(img,
                      (bar_x, bar_y + bar_h - preench),
                      (bar_x + 18, bar_y + bar_h),
                      cor_barra, -1)
        cv2.putText(img, "ADM", (bar_x - 4, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        # HUD superior
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 58), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        hw_str  = "GPU" if self.device.type == "cuda" else "CPU"
        alerta  = "  ⚠ COMPENSACAO" if compensacao else ""
        hud     = (f"{hw_str}  |  {self.tipo_movimento} ({self.lado.upper()})  |  "
                   f"REPS: {reps}  |  FASE: {fase.upper()}  |  "
                   f"VEL: {vel:.0f} deg/s{alerta}")
        cv2.putText(img, hud, (12, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.58,
                    (255, 80, 80) if compensacao else (255, 255, 255),
                    1, cv2.LINE_AA)

        return img

    # ------------------------------------------------------------------
    # CARREGAMENTO DO MODELO IA
    # ------------------------------------------------------------------
    def carregar_modelo_ia(self, caminho_modelo: str = "src/models/modelo_pibic.pt") -> bool:
        try:
            self.modelo_ia = torch.jit.load(caminho_modelo, map_location=self.device)
            if self.device.type == "cuda":
                self.modelo_ia = self.modelo_ia.half()
                try:
                    self.modelo_ia = torch.compile(self.modelo_ia)
                except Exception:
                    pass
            else:
                self.modelo_ia = self.modelo_ia.float()
            self.modelo_ia.eval()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # INFERÊNCIA DE FLUIDEZ
    # ------------------------------------------------------------------
    def prever_fluidez(self, angulos: List[float]) -> float:
        if self.modelo_ia is None or not angulos:
            return 0.0
        seq   = self.normalizar_sequencia(angulos)
        is_cu = self.device.type == "cuda"
        with torch.no_grad(), autocast(enabled=is_cu):
            t = torch.tensor(seq, dtype=torch.float32).to(self.device)
            if is_cu:
                t = t.half()
            t      = t.view(1, 100, 1)
            output = self.modelo_ia(t)
            return round(torch.sigmoid(output).item() * 100, 2)

    # ------------------------------------------------------------------
    # PROCESSAMENTO DE VÍDEO
    # ------------------------------------------------------------------
    def processar_video_para_memoria(
        self,
        video_path: str,
        progress_callback=None,
        largura_saida: int = 640,
        altura_saida: int = 480,
    ) -> Tuple[List, List[float], List[float], int, float]:
        """
        Processa o vídeo frame a frame.

        Retorna:
          frames      — lista de np.ndarray RGB redimensionados
          angulos     — ângulo suavizado por frame (graus)
          velocidades — velocidade angular por frame (graus/s)
          total_reps  — repetições detectadas
          fps         — taxa de quadros do vídeo
        """
        # Reseta estado
        self._filtro.resetar()
        self._compensacao = DetectorCompensacao()
        self._angulo_ant  = None
        self._contador_reps = 0
        self._fase        = "repouso"
        self._frames_calib = 0

        cap      = cv2.VideoCapture(video_path)
        total_f  = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        fps_orig = cap.get(cv2.CAP_PROP_FPS)
        fps      = fps_orig if fps_orig and fps_orig > 0 else 30.0

        frames:      List[np.ndarray] = []
        angulos:     List[float]      = []
        velocidades: List[float]      = []

        n_frame = 0

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res     = self.pose.process(img_rgb)

            angulo_frame  = 0.0
            vel_frame     = 0.0
            compensacao   = False
            fase          = self._fase

            if res.pose_landmarks and res.pose_world_landmarks:
                h, w, _ = img_rgb.shape
                lm_norm  = res.pose_landmarks.landmark
                lm_mundo = res.pose_world_landmarks.landmark

                # Calibração postural nos primeiros 15 frames
                if self._frames_calib < 15:
                    self._compensacao.calibrar(lm_mundo)
                    self._frames_calib += 1

                pts_3d, pts_2d, confianca = self._extrair_pontos(lm_norm, lm_mundo, w, h)

                # Ângulo bruto do movimento
                angulo_bruto = self._calcular_angulo_movimento(pts_3d)

                # Filtragem adaptativa (confiança pondera suavização)
                angulo_filtrado = self._filtro.atualizar(angulo_bruto, confianca)

                # Velocidade angular (derivada central quando possível)
                if self._angulo_ant is not None:
                    vel_frame = abs(angulo_filtrado - self._angulo_ant) * fps
                self._angulo_ant = angulo_filtrado

                angulo_frame = angulo_filtrado

                # Fase e repetições
                fase, reps = self._atualizar_fase(angulo_frame)

                # Compensação postural
                compensacao = self._compensacao.verificar(lm_mundo)

                # Renderização
                img_rgb = self._renderizar(
                    img_rgb, pts_2d,
                    angulo_frame, vel_frame,
                    fase, self._contador_reps, compensacao
                )

            # Reduz resolução e salva
            frame_out = cv2.resize(img_rgb, (largura_saida, altura_saida))
            frames.append(frame_out)
            angulos.append(round(angulo_frame, 3))
            velocidades.append(round(vel_frame, 3))

            n_frame += 1
            if progress_callback:
                progress_callback(min(n_frame / total_f, 1.0))

        cap.release()
        return frames, angulos, velocidades, self._contador_reps, fps

    # ------------------------------------------------------------------
    # UTILITÁRIOS
    # ------------------------------------------------------------------
    def normalizar_sequencia(self, angulos: List[float], tamanho_alvo: int = 100) -> List[float]:
        """Interpola a sequência de ângulos para tamanho fixo (entrada do modelo IA)."""
        arr = np.array(angulos, dtype=np.float64)
        if len(arr) < 2:
            return [0.0] * tamanho_alvo
        return np.interp(
            np.linspace(0, 1, tamanho_alvo),
            np.linspace(0, 1, len(arr)),
            arr,
        ).tolist()

    def relatorio_movimento(self, angulos: List[float], velocidades: List[float]) -> Dict:
        """Gera um dicionário com estatísticas clínicas da sessão."""
        validos = [a for a in angulos if a > 1.0]
        vels_v  = [v for v in velocidades if v > 0.0]
        cfg     = self.config

        return {
            "tipo_movimento":    self.tipo_movimento,
            "lado":              self.lado,
            "adm_maxima":        round(max(validos), 2)       if validos else 0.0,
            "adm_minima":        round(min(validos), 2)       if validos else 0.0,
            "adm_media":         round(float(np.mean(validos)), 2) if validos else 0.0,
            "adm_dp":            round(float(np.std(validos)), 2)  if validos else 0.0,
            "adm_referencia":    cfg.get("adm_referencia", 180),
            "pct_adm_normal":    round(max(validos) / cfg.get("adm_referencia", 180) * 100, 1) if validos else 0.0,
            "vel_pico":          round(max(vels_v), 2)        if vels_v else 0.0,
            "vel_media":         round(float(np.mean(vels_v)), 2) if vels_v else 0.0,
            "total_reps":        self._contador_reps,
            "plano_movimento":   cfg.get("plano", ""),
            "descricao":         cfg.get("descricao", ""),
        }
