import cv2
import numpy as np
import torch
from collections import deque
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from torch.cuda.amp import autocast

class AnalisadorADMWeb:
    def __init__(self, id_paciente, tipo_movimento, lado_do_corpo, usar_gpu=True):
        self.id_paciente = id_paciente
        self.tipo_movimento = tipo_movimento
        self.lado_do_corpo = lado_do_corpo.lower()
        
        # --- FILTRO DE ESTABILIDADE EXTREMA (EMA) ---
        # Alpha 0.35: equilíbrio entre resposta imediata e remoção de ruído (jitter)
        self.angulo_suavizado = None
        self.alpha = 0.35 

        self.angulo_anterior = None
        self.contador_reps = 0
        self.fase_movimento = "repouso"
        
        # --- CONFIGURAÇÃO DE HARDWARE (ROCm/CPU) ---
        if usar_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.modelo_ia = None
        
        # --- MOTOR DE POSE: AUTO-ADAPTÁVEL (Nuvem vs Local) ---
        try:
            # Tenta usar o modelo Heavy (Nível 2 - Ideal para o seu PC Local em Belém)
            self.pose = mp_pose.Pose(
                static_image_mode=False, 
                model_complexity=2, 
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
        except PermissionError:
            # Fallback automático para a Nuvem (Streamlit Cloud bloqueia downloads na pasta base)
            # O modelo Full (Nível 1) já vem embutido e resolve o problema
            self.pose = mp_pose.Pose(
                static_image_mode=False, 
                model_complexity=1, 
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            
        self._configurar_nos_anatomicos()

    def _configurar_nos_anatomicos(self):
        """Mapeia os pontos de interesse baseando-se na lateralidade do paciente."""
        is_left = self.lado_do_corpo == 'esquerdo'
        p = mp_pose.PoseLandmark
        self.nodes = {
            'ombro': p.LEFT_SHOULDER if is_left else p.RIGHT_SHOULDER,
            'cotovelo': p.LEFT_ELBOW if is_left else p.RIGHT_ELBOW,
            'pulso': p.LEFT_WRIST if is_left else p.RIGHT_WRIST,
            'quadril': p.LEFT_HIP if is_left else p.RIGHT_HIP,
            'indicador': p.LEFT_INDEX if is_left else p.RIGHT_INDEX,
            'mindinho': p.LEFT_PINKY if is_left else p.RIGHT_PINKY
        }

    def _obter_pontos_clinicos(self, lm, w_lm, w, h):
        """Extrai coordenadas 3D (World) para cálculo e 2D (Pixel) para desenho."""
        pts_3d, pts_2d = {}, {}
        for nome, node in self.nodes.items():
            # Pontos 3D em metros reais (compensação de profundidade)
            pts_3d[nome] = np.array([w_lm[node.value].x, w_lm[node.value].y, w_lm[node.value].z])
            # Pontos 2D para renderização na tela
            pts_2d[nome] = (int(lm[node.value].x * w), int(lm[node.value].y * h))
            
        # PONTO VIRTUAL DA PALMA: Resolve a instabilidade do punho ignorando os dedos
        pts_3d['centro_mao'] = (pts_3d['indicador'] + pts_3d['mindinho']) / 2.0
        pts_2d['centro_mao'] = (
            int((pts_2d['indicador'][0] + pts_2d['mindinho'][0]) / 2),
            int((pts_2d['indicador'][1] + pts_2d['mindinho'][1]) / 2)
        )
        return pts_3d, pts_2d

    def _calcular_angulo_clinico(self, p_a, p_b, p_c):
        """Calcula o ângulo vetorial no espaço 3D (Goniometria Digital)."""
        v1 = p_a - p_b
        v2 = p_c - p_b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0: return 0.0
        cos_theta = np.dot(v1, v2) / (n1 * n2)
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    def carregar_modelo_ia(self, caminho_modelo="src/models/modelo_pibic.pt"):
        try:
            self.modelo_ia = torch.jit.load(caminho_modelo, map_location=self.device)
            if self.device.type == 'cuda':
                self.modelo_ia = self.modelo_ia.half() 
                try: self.modelo_ia = torch.compile(self.modelo_ia)
                except: pass
