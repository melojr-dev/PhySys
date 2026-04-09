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
            self.pose = mp_pose.Pose(
                static_image_mode=False, 
                model_complexity=2, 
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
        except PermissionError:
            self.pose = mp_pose.Pose(
                static_image_mode=False, 
                model_complexity=1, 
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            
        self._configurar_nos_anatomicos()

    def _configurar_nos_anatomicos(self):
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
        pts_3d, pts_2d = {}, {}
        for nome, node in self.nodes.items():
            pts_3d[nome] = np.array([w_lm[node.value].x, w_lm[node.value].y, w_lm[node.value].z])
            pts_2d[nome] = (int(lm[node.value].x * w), int(lm[node.value].y * h))
            
        pts_3d['centro_mao'] = (pts_3d['indicador'] + pts_3d['mindinho']) / 2.0
        pts_2d['centro_mao'] = (
            int((pts_2d['indicador'][0] + pts_2d['mindinho'][0]) / 2),
            int((pts_2d['indicador'][1] + pts_2d['mindinho'][1]) / 2)
        )
        return pts_3d, pts_2d

    def _calcular_angulo_clinico(self, p_a, p_b, p_c):
        v1 = p_a - p_b
        v2 = p_c - p_b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0: 
            return 0.0
        cos_theta = np.dot(v1, v2) / (n1 * n2)
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    def carregar_modelo_ia(self, caminho_modelo="src/models/modelo_pibic.pt"):
        try:
            self.modelo_ia = torch.jit.load(caminho_modelo, map_location=self.device)
            if self.device.type == 'cuda':
                self.modelo_ia = self.modelo_ia.half() 
                try: 
                    self.modelo_ia = torch.compile(self.modelo_ia)
                except: 
                    pass 
            else:
                self.modelo_ia = self.modelo_ia.float()
            self.modelo_ia.eval()
            return True
        except: 
            return False

    def prever_fluidez(self, angulos):
        if self.modelo_ia is None or not angulos: 
            return 0.0
        input_data = self.normalizar_sequencia(angulos)
        with torch.no_grad():
            is_cuda = self.device.type == 'cuda'
            with autocast(enabled=is_cuda):
                tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
                if is_cuda: 
                    tensor = tensor.half()
                tensor = tensor.view(1, 100, 1)
                output = self.modelo_ia(tensor)
                return round(torch.sigmoid(output).item() * 100, 2)

    def processar_video_para_memoria(self, video_path, progress_callback=None):
        cap = cv2.VideoCapture(video_path)
        frames, angulos, velocidades = [], [], []
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        self.angulo_suavizado = None
        self.angulo_anterior = None
        self.contador_reps = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(img_rgb)
            val_ang, vel_ang = 0.0, 0.0

            if res.pose_landmarks and res.pose_world_landmarks:
                h, w, _ = img_rgb.shape
                pts_3d, pts_2d = self._obter_pontos_clinicos(res.pose_landmarks.landmark, 
                                                             res.pose_world_landmarks.landmark, w, h)
                
                if self.tipo_movimento == 'Flexão de Cotovelo':
                    p_a, p_b, p_c = pts_3d['ombro'], pts_3d['cotovelo'], pts_3d['pulso']
                    d_a, d_b, d_c = pts_2d['ombro'], pts_2d['cotovelo'], pts_2d['pulso']
                    val_ang = self._calcular_angulo_clinico(p_a, p_b, p_c)
                    
                elif 'Ombro' in self.tipo_movimento:
                    p_a, p_b, p_c = pts_3d['quadril'], pts_3d['ombro'], pts_3d['cotovelo']
                    d_a, d_b, d_c = pts_2d['quadril'], pts_2d['ombro'], pts_2d['cotovelo']
                    val_ang = self._calcular_angulo_clinico(p_a, p_b, p_c)

                elif 'Punho' in self.tipo_movimento:
                    p_a, p_b, p_c = pts_3d['cotovelo'], pts_3d['pulso'], pts_3d['centro_mao']
                    d_a, d_b, d_c = pts_2d['cotovelo'], pts_2d['pulso'], pts_2d['centro_mao']
                    val_ang = abs(180 - self._calcular_angulo_clinico(p_a, p_b, p_c))

                if self.angulo_suavizado is None:
                    self.angulo_suavizado = val_ang
                else:
                    self.angulo_suavizado = (self.alpha * val_ang) + ((1 - self.alpha) * self.angulo_suavizado)
                val_ang = self.angulo_suavizado

                if self.angulo_anterior is not None:
                    vel_ang = abs(val_ang - self.angulo_anterior) * fps
                self.angulo_anterior = val_ang

                th_flex = 30 if 'Punho' in self.tipo_movimento else 90
                th_ext = 10 if 'Punho' in self.tipo_movimento else 140
                
                if val_ang < th_flex and self.fase_movimento != "flexao": 
                    self.fase_movimento = "flexao"
                elif val_ang > th_ext and self.fase_movimento == "flexao":
                    self.fase_movimento = "extensao"
                    self.contador_reps += 1

                overlay = img_rgb.copy()
                cv2.rectangle(overlay, (20, 20), (660, 70), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, img_rgb, 0.4, 0, img_rgb)
                
                cv2.line(img_rgb, d_a, d_b, (0, 255, 0), 4)
                cv2.line(img_rgb, d_b, d_c, (0, 255, 0), 4)
                cv2.circle(img_rgb, d_b, 8, (0, 0, 255), -1)
                
                hw = "GPU-ROCm" if self.device.type == "cuda" else "CPU"
                status = f"MODO: {hw} | REPS: {self.contador_reps} | FASE: {self.fase_movimento.upper()}"
                cv2.putText(img_rgb, status, (35, 53), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img_rgb, f"{int(val_ang)} deg", (d_b[0]-40, d_b[1]+45), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
            
            frames.append(img_rgb)
            angulos.append(val_ang)
            velocidades.append(vel_ang)
            if progress_callback: 
                progress_callback(min(len(frames) / total_f, 1.0))

        cap.release()
        return frames, angulos, velocidades, self.contador_reps, fps

    def normalizar_sequencia(self, angulos, tamanho_alvo=100):
        if len(angulos) < 2: 
            return [0.0] * tamanho_alvo
        return np.interp(np.linspace(0, 1, tamanho_alvo), np.linspace(0, 1, len(angulos)), angulos).tolist()
