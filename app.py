import streamlit as st
import tempfile
import time
import pandas as pd
import numpy as np
import torch
import os
import gc
from src.core.engine import AnalisadorADMWeb

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PhysioAI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ---- Reset e base ---- */
section[data-testid="stSidebar"] { background: #f8f9fb; border-right: 1px solid #e8eaf0; }
div[data-testid="metric-container"] {
    background: #ffffff; border: 1px solid #eef0f5;
    padding: 12px 16px; border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
/* Botões com mais estilo */
.stButton > button {
    width: 100%; font-weight: 600; border-radius: 8px;
    transition: all 0.15s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }

/* Progress bar verde */
.stProgress > div > div > div { background: linear-gradient(90deg, #1D9E75, #5DCAA5); }

/* Slider mais limpo */
.stSlider [data-baseweb="slider"] { padding: 0 4px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #f0f2f6; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 6px 16px; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }

/* Métricas com delta colorido */
[data-testid="stMetricDelta"] svg { display: none; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# FILTRO DE KALMAN (melhora captação de movimento)
# ---------------------------------------------------------------------------
class FiltroKalman1D:
    """Suaviza ângulos frame a frame eliminando jitter sem delay perceptível."""
    def __init__(self, q=0.01, r=0.1):
        self.q = q      # ruído do processo (menor = mais suave)
        self.r = r      # ruído da medição  (maior = mais confiança no modelo)
        self.p = 1.0
        self.x = None

    def atualizar(self, medicao):
        if self.x is None:
            self.x = medicao
            return medicao
        # Predição
        p_pred = self.p + self.q
        # Ganho de Kalman
        k = p_pred / (p_pred + self.r)
        # Atualização
        self.x = self.x + k * (medicao - self.x)
        self.p = (1 - k) * p_pred
        return self.x

    def resetar(self):
        self.p = 1.0
        self.x = None


def filtrar_angulos(angulos_brutos, q=0.008, r=0.12):
    """Aplica Kalman e remove outliers antes (janela de mediana)."""
    arr = np.array(angulos_brutos, dtype=float)

    # 1. Remove outliers com filtro de mediana adaptativo
    janela = 5
    arr_limpo = arr.copy()
    for i in range(len(arr)):
        inicio = max(0, i - janela // 2)
        fim = min(len(arr), i + janela // 2 + 1)
        vizinhos = arr[inicio:fim]
        mediana = np.median(vizinhos[vizinhos > 1])  # ignora frames sem detecção
        if mediana > 0 and abs(arr[i] - mediana) > 25:  # spike > 25°
            arr_limpo[i] = mediana

    # 2. Kalman
    filtro = FiltroKalman1D(q=q, r=r)
    return [filtro.atualizar(v) for v in arr_limpo]


def calcular_velocidade_suave(angulos_filtrados, fps):
    """Velocidade angular com derivada central (mais precisa que diferença simples)."""
    if len(angulos_filtrados) < 3:
        return [0.0] * len(angulos_filtrados)
    vels = [0.0]
    for i in range(1, len(angulos_filtrados) - 1):
        vel = (angulos_filtrados[i + 1] - angulos_filtrados[i - 1]) * fps / 2.0
        vels.append(abs(vel))
    vels.append(vels[-1])
    # Suaviza velocidade também com média móvel pequena
    vels_np = np.array(vels)
    kernel = np.ones(5) / 5
    vels_suave = np.convolve(vels_np, kernel, mode='same')
    return vels_suave.tolist()


def detectar_repeticoes(angulos, fps, limiar_min=15.0, limiar_pico=30.0):
    """Conta repetições por detecção de picos com histeria."""
    reps = 0
    em_movimento = False
    ultimo_pico = 0
    min_frames_entre_reps = int(fps * 0.8)

    for i, ang in enumerate(angulos):
        if ang >= limiar_pico and not em_movimento and (i - ultimo_pico) > min_frames_entre_reps:
            em_movimento = True
            ultimo_pico = i
            reps += 1
        elif ang < limiar_min:
            em_movimento = False
    return reps


# ---------------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------------
@st.cache_resource
def get_analisador(id_p, mov, lado, gpu_on):
    analisador = AnalisadorADMWeb(id_p, mov, lado, usar_gpu=gpu_on)
    analisador.carregar_modelo_ia()
    return analisador


# ---------------------------------------------------------------------------
# ESTADO DA SESSÃO
# ---------------------------------------------------------------------------
def inicializar_estado():
    defaults = {
        'frames_salvos': [],
        'angulos_salvos': [],        # brutos
        'angulos_filtrados': [],     # pós-Kalman
        'velocidades_salvas': [],
        'total_reps': 0,
        'fps': 30.0,
        'analise_feita': False,
        'frame_atual': 0,
        'ultimo_arquivo': None,
        'playing': False,
        'velocidade_play': 1.0,
        'suavizacao_q': 0.008,
        'suavizacao_r': 0.12,
        'outliers_removidos': 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def resetar_estado(nome_arquivo):
    if st.session_state.ultimo_arquivo != nome_arquivo:
        for k in ['frames_salvos', 'angulos_salvos', 'angulos_filtrados',
                  'velocidades_salvas']:
            st.session_state[k] = []
        st.session_state.update({
            'total_reps': 0, 'fps': 30.0, 'analise_feita': False,
            'frame_atual': 0, 'playing': False, 'ultimo_arquivo': nome_arquivo,
            'outliers_removidos': 0,
        })


inicializar_estado()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ PhysioAI")
    st.caption("Análise biomecânica assistida por IA")
    st.divider()

    # Hardware
    with st.expander("🚀 Hardware", expanded=True):
        if torch.cuda.is_available():
            nome_gpu = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_usada = torch.cuda.memory_allocated(0) / 1e9
            pct_vram = vram_usada / vram_total if vram_total > 0 else 0
            st.success(f"GPU ativa: {nome_gpu}")
            st.progress(pct_vram, text=f"VRAM: {vram_usada:.1f} / {vram_total:.0f} GB")
        else:
            st.warning("CPU Mode — ROCm não detectado")

    modo_gpu = st.toggle("Aceleração GPU (ROCm / CUDA)", value=True)
    if modo_gpu and torch.cuda.is_available():
        st.caption("🔥 FP16 + otimizações RDNA 2 ativas")
    else:
        st.caption("🧊 FP32 na CPU")

    st.divider()

    # Paciente
    st.subheader("👤 Paciente")
    id_paciente = st.text_input("ID", "PAC-001", label_visibility="collapsed",
                                 placeholder="ID do paciente")
    movimento = st.selectbox("Movimento", [
        "Flexão de Cotovelo", "Abdução de Ombro", "Flexão de Ombro",
        "Desvio de Punho", "Flexão de Punho",
    ])
    lado = st.radio("Lado", ["Direito", "Esquerdo"], horizontal=True)

    st.divider()

    # Qualidade de captação
    st.subheader("🔬 Captação de Movimento")
    suav_q = st.slider(
        "Suavização (Filtro Kalman)", 0.001, 0.05, 0.008, 0.001,
        help="Menor = mais suave. Maior = segue o movimento mais fielmente."
    )
    suav_r = st.slider(
        "Confiança na medição", 0.05, 0.5, 0.12, 0.01,
        help="Maior = confia mais no modelo. Menor = segue mais o sensor."
    )
    st.session_state['suavizacao_q'] = suav_q
    st.session_state['suavizacao_r'] = suav_r

    limiar_rep = st.slider("Limiar de repetição (°)", 20, 80, 30,
                            help="Ângulo mínimo para contar uma repetição completa.")

    st.divider()
    uploaded_file = st.file_uploader("📂 Vídeo", type=["mp4", "avi", "mov"])

    # Dataset stats
    st.divider()
    st.subheader("📊 Dataset PIBIC")
    try:
        df_ds = pd.read_csv("src/utils/dataset_pibic.csv")
        col_a, col_b = st.columns(2)
        col_a.metric("Amostras", len(df_ds))
        col_b.metric("Labels", df_ds['label'].nunique())
        with st.expander("Distribuição"):
            st.bar_chart(df_ds['label'].value_counts())
    except Exception:
        st.caption("Nenhuma amostra ainda.")


# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------
st.title("🩺 PhysioAI — Análise Biomecânica")

if not uploaded_file:
    st.info("⬅️ Faça upload de um vídeo na barra lateral para começar.")
    st.stop()

resetar_estado(uploaded_file.name)

# --- BOTÃO PROCESSAR ---
if st.sidebar.button("▶️ Processar Vídeo", type="primary"):
    st.session_state.frames_salvos = []
    gc.collect()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())
        tmp_path = tfile.name

    barra = st.progress(0)
    status = st.empty()

    def cb(pct):
        barra.progress(pct)
        hw = "GPU 🔥" if modo_gpu else "CPU 🧊"
        status.markdown(f"**Processando ({hw}):** `{int(pct * 100)}%`")

    analisador = get_analisador(id_paciente, movimento, lado, modo_gpu)
    frames, angulos_brutos, _, total_reps_bruto, fps = \
        analisador.processar_video_para_memoria(tmp_path, progress_callback=cb)

    # Pós-processamento de sinal melhorado
    angulos_brutos_arr = np.array(angulos_brutos)
    angulos_antes = np.sum(np.abs(np.diff(angulos_brutos_arr[angulos_brutos_arr > 1])) > 25)

    angulos_filtrados = filtrar_angulos(angulos_brutos, q=suav_q, r=suav_r)
    velocidades = calcular_velocidade_suave(angulos_filtrados, fps)
    total_reps = detectar_repeticoes(angulos_filtrados, fps, limiar_pico=limiar_rep)

    angulos_depois = np.sum(np.abs(np.diff(np.array(angulos_filtrados)[np.array(angulos_filtrados) > 1])) > 25)
    outliers_removidos = int(max(0, angulos_antes - angulos_depois))

    st.session_state.update({
        'frames_salvos': frames,
        'angulos_salvos': angulos_brutos,
        'angulos_filtrados': angulos_filtrados,
        'velocidades_salvas': velocidades,
        'total_reps': total_reps,
        'fps': fps,
        'analise_feita': True,
        'outliers_removidos': outliers_removidos,
    })
    barra.empty()
    status.empty()
    st.toast(f"✅ Processamento concluído — {total_reps} repetições detectadas", icon="✅")


# ---------------------------------------------------------------------------
# PLAYER + RELATÓRIO
# ---------------------------------------------------------------------------
if not (st.session_state.analise_feita and st.session_state.frames_salvos):
    st.stop()

frames = st.session_state.frames_salvos
angulos = st.session_state.angulos_filtrados or st.session_state.angulos_salvos
velocidades = st.session_state.velocidades_salvas
total_frames = len(frames)
fps = st.session_state.fps
duracao = (total_frames - 1) / fps

# Validados (com movimento real)
validos = [a for a in angulos if a > 1]
adm_max = max(validos) if validos else 0
adm_media = np.mean(validos) if validos else 0
vel_pico = max(velocidades) if velocidades else 0

st.divider()
tab_player, tab_relatorio, tab_pibic = st.tabs(
    ["▶️  Player", "📄  Relatório Clínico", "🧪  Dataset PIBIC"]
)

# ===== TAB PLAYER =====
with tab_player:
    col_vid, col_info = st.columns([2, 1], gap="large")

    with col_vid:
        st.subheader(f"{movimento} — {lado}")

        # Controles de reprodução
        c_play, c_vel, c_skip_l, c_skip_r = st.columns([2, 1, 1, 1])

        if c_play.button(
            "⏸ Pausar" if st.session_state.playing else "▶️ Reproduzir",
            use_container_width=True
        ):
            st.session_state.playing = not st.session_state.playing
            st.rerun()

        vel_play = c_vel.selectbox(
            "Velocidade", [0.25, 0.5, 1.0, 1.5, 2.0],
            index=2, format_func=lambda x: f"{x}×",
            label_visibility="collapsed"
        )
        st.session_state.velocidade_play = vel_play

        frames_pulo = max(1, int(fps * 0.5))
        if c_skip_l.button("⏪ −0.5s", use_container_width=True):
            st.session_state.update({
                'frame_atual': max(0, st.session_state.frame_atual - frames_pulo),
                'playing': False,
            })
            st.rerun()
        if c_skip_r.button("+0.5s ⏩", use_container_width=True):
            st.session_state.update({
                'frame_atual': min(total_frames - 1, st.session_state.frame_atual + frames_pulo),
                'playing': False,
            })
            st.rerun()

        # Slider de linha do tempo
        idx = st.slider(
            "Linha do tempo",
            0, total_frames - 1,
            st.session_state.frame_atual,
            label_visibility="collapsed",
        )
        if idx != st.session_state.frame_atual:
            st.session_state.update({'frame_atual': idx, 'playing': False})
            st.rerun()

        # Frame atual
        img_placeholder = st.empty()
        img_placeholder.image(
            frames[st.session_state.frame_atual],
            channels="RGB",
            use_column_width=True,
        )

        # Info do frame
        t_atual = st.session_state.frame_atual / fps
        st.caption(
            f"⏱ {t_atual:.2f}s / {duracao:.2f}s  |  "
            f"🎞 Frame {st.session_state.frame_atual}/{total_frames - 1}  |  "
            f"📐 {angulos[st.session_state.frame_atual]:.1f}°  |  "
            f"💨 {int(velocidades[st.session_state.frame_atual])}°/s  |  "
            f"🚀 {'GPU' if modo_gpu else 'CPU'}"
        )

        # Loop do player
        if st.session_state.playing:
            if st.session_state.frame_atual < total_frames - 1:
                pulo = max(1, int(vel_play * 2))
                st.session_state.frame_atual = min(
                    st.session_state.frame_atual + pulo, total_frames - 1
                )
                delay = max(0.01, 0.033 / vel_play)
                time.sleep(delay)
                st.rerun()
            else:
                st.session_state.update({'playing': False, 'frame_atual': 0})
                st.rerun()

    with col_info:
        st.subheader("📊 Métricas ao Vivo")

        ang_frame = angulos[st.session_state.frame_atual]
        vel_frame = velocidades[st.session_state.frame_atual]

        m1, m2 = st.columns(2)
        m1.metric("Ângulo", f"{ang_frame:.1f}°")
        m2.metric("Velocidade", f"{int(vel_frame)}°/s")
        m3, m4 = st.columns(2)
        m3.metric("ADM máx.", f"{adm_max:.1f}°")
        m4.metric("Reps", st.session_state.total_reps)

        st.markdown("**Curva de ângulo**")
        chart_df = pd.DataFrame({
            "Ângulo (°)": angulos,
            "Frame": range(len(angulos)),
        }).set_index("Frame")
        st.line_chart(chart_df, height=180, color="#1D9E75")

        # Indicadores de qualidade de sinal
        st.markdown("**Qualidade do sinal**")
        col_sig_a, col_sig_b = st.columns(2)
        col_sig_a.metric("Outliers corrigidos", st.session_state.outliers_removidos)
        consistencia = 100 - min(100, int(np.std(validos) / max(1, adm_max) * 100)) if validos else 0
        col_sig_b.metric("Consistência", f"{consistencia}%")


# ===== TAB RELATÓRIO =====
with tab_relatorio:
    st.subheader(f"Relatório Clínico — {id_paciente}")
    st.caption(f"Gerado em: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')} | Movimento: {movimento} ({lado})")

    # KPIs
    ka, kb, kc, kd = st.columns(4)
    ka.metric("ADM máxima", f"{adm_max:.1f}°")
    kb.metric("ADM média", f"{adm_media:.1f}°")
    kc.metric("Velocidade pico", f"{int(vel_pico)}°/s")
    kd.metric("Repetições", st.session_state.total_reps)

    st.divider()
    col_rel_a, col_rel_b = st.columns(2)

    with col_rel_a:
        st.markdown("**Curva Angular Completa**")
        df_ang = pd.DataFrame({"Ângulo (°)": angulos})
        st.line_chart(df_ang, height=200, color="#1D9E75")

    with col_rel_b:
        st.markdown("**Perfil de Velocidade**")
        df_vel = pd.DataFrame({"Velocidade (°/s)": velocidades})
        st.line_chart(df_vel, height=200, color="#BA7517")

    st.divider()
    st.markdown("**Análise descritiva**")
    desc = {
        "Amplitude máxima (°)": f"{adm_max:.1f}",
        "Amplitude mínima (°)": f"{min(validos):.1f}" if validos else "—",
        "Amplitude média (°)": f"{adm_media:.1f}",
        "Desvio padrão (°)": f"{np.std(validos):.1f}" if validos else "—",
        "Velocidade pico (°/s)": f"{int(vel_pico)}",
        "Velocidade média (°/s)": f"{int(np.mean([v for v in velocidades if v > 0]))}" if velocidades else "—",
        "Total de repetições": str(st.session_state.total_reps),
        "Duração do vídeo (s)": f"{duracao:.1f}",
        "Outliers corrigidos": str(st.session_state.outliers_removidos),
        "FPS": f"{fps:.1f}",
    }
    df_desc = pd.DataFrame(desc.items(), columns=["Parâmetro", "Valor"])
    st.dataframe(df_desc, use_container_width=True, hide_index=True)


# ===== TAB PIBIC =====
with tab_pibic:
    st.subheader("🧪 Coleta para o Dataset PIBIC")
    st.info(
        "Rotule esta sequência de movimento para enriquecer o dataset de treinamento. "
        "Os dados são normalizados antes de salvar.",
        icon="ℹ️"
    )

    col_pibic_a, col_pibic_b = st.columns([2, 1])

    with col_pibic_a:
        label_qual = st.selectbox(
            "Diagnóstico visual do movimento:",
            ["Selecione...", "Fluido / Normal", "Tremor Leve",
             "Tremor Acentuado", "Braquicinesia"],
        )
        notas = st.text_area("Observações clínicas (opcional)", height=80, placeholder="Ex.: paciente relatou dor no arco final...")

    with col_pibic_b:
        st.metric("ADM máx. desta sessão", f"{adm_max:.1f}°")
        st.metric("Reps detectadas", st.session_state.total_reps)

    if st.button("💾 Salvar no dataset", type="primary", use_container_width=True):
        if label_qual == "Selecione...":
            st.error("Selecione um diagnóstico antes de salvar.")
        elif not st.session_state.angulos_filtrados:
            st.error("Nenhuma sequência disponível para salvar.")
        else:
            analisador_pibic = get_analisador(id_paciente, movimento, lado, modo_gpu)
            seq_norm = analisador_pibic.normalizar_sequencia(st.session_state.angulos_filtrados)

            novo = {
                "id_paciente": id_paciente,
                "movimento": movimento,
                "lado": lado,
                "label": label_qual,
                "adm_max": round(adm_max, 2),
                "adm_media": round(adm_media, 2),
                "vel_pico": round(vel_pico, 2),
                "total_reps": st.session_state.total_reps,
                "outliers_corrigidos": st.session_state.outliers_removidos,
                "filtro_q": suav_q,
                "filtro_r": suav_r,
                "notas": notas,
                "hardware": "GPU" if modo_gpu else "CPU",
                "sequencia": str(seq_norm),
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            path_ds = "src/utils/dataset_pibic.csv"
            os.makedirs(os.path.dirname(path_ds), exist_ok=True)
            header = not os.path.exists(path_ds)
            pd.DataFrame([novo]).to_csv(path_ds, mode='a', index=False, header=header)
            st.success("✅ Sequência catalogada com sucesso!")
            st.balloons()
