import os
try:
    import cv2
except ImportError as e:
    if "libGL" in str(e):
        print("Instalando dependências headless...")
        os.system("pip uninstall -y opencv-python opencv-contrib-python")
        os.system("pip install opencv-python-headless opencv-contrib-python-headless")
        import cv2
import streamlit as st
import tempfile
import time
import pandas as pd
import numpy as np
import torch
from src.core.engine import AnalisadorADMWeb


# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="PhysioAI - Player Completo", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    div[data-testid="metric-container"] {
        background-color: #ffffff; border: 1px solid #e6e9ef; padding: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stButton button { width: 100%; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNÇÃO DE CACHE ÚNICA (Gerencia CPU/GPU) ---
@st.cache_resource
def get_analisador(id_p, mov, lado, gpu_on):
    """Mantém o modelo carregado no hardware selecionado sem recarregar no rerun."""
    analisador = AnalisadorADMWeb(id_p, mov, lado, usar_gpu=gpu_on)
    analisador.carregar_modelo_ia()
    return analisador

def inicializar_estado():
    chaves = ['frames_salvos', 'angulos_salvos', 'velocidades_salvas', 'total_reps', 'fps', 'analise_feita', 'frame_atual', 'ultimo_arquivo', 'playing']
    valores_padrao = [[], [], [], 0, 30.0, False, 0, None, False]
    for chave, valor in zip(chaves, valores_padrao):
        if chave not in st.session_state:
            st.session_state[chave] = valor

def resetar_estado(nome_arquivo):
    if st.session_state.ultimo_arquivo != nome_arquivo:
        st.session_state.update({
            'frames_salvos': [], 'angulos_salvos': [], 'velocidades_salvas': [], 'total_reps': 0,
            'fps': 30.0, 'analise_feita': False, 'frame_atual': 0, 'playing': False, 'ultimo_arquivo': nome_arquivo
        })

inicializar_estado()

# --- BARRA LATERAL ---
st.sidebar.title("⚙️ Configurações")

# MONITOR DE HARDWARE (Consolidado)
with st.sidebar.expander("🚀 Monitor de Hardware", expanded=True):
    if torch.cuda.is_available():
        st.success(f"GPU AMD Ativa: {torch.cuda.get_device_name(0)}")
        st.caption("Aceleração RDNA 2 via ROCm ativa.")
    else:
        st.warning("IA rodando na CPU. Verifique o driver ROCm.")

# PERFORMANCE TOGGLE
st.sidebar.subheader("🚀 Performance")
modo_execucao = st.sidebar.toggle("Ativar Aceleração por GPU (ROCm)", value=True)
if modo_execucao and torch.cuda.is_available():
    st.sidebar.caption("🔥 Modo: GPU Turbo (FP16 + Salto de Frames)")
else:
    st.sidebar.caption("🧊 Modo: CPU Normal (FP32)")

id_paciente = st.sidebar.text_input("ID do Paciente", "PAC-001")
movimento = st.sidebar.selectbox("Tipo de Movimento", [
    "Flexão de Cotovelo", "Abdução de Ombro", "Flexão de Ombro", 
    "Desvio de Punho", "Flexão de Punho"
])
lado = st.sidebar.radio("Lado do Corpo", ["Direito", "Esquerdo"])
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📂 Carregar Vídeo", type=["mp4", "avi", "mov"])

# --- LÓGICA DE PROCESSAMENTO ---
st.title("🩺 PhysioAI: Análise & Player")

if uploaded_file:
    resetar_estado(uploaded_file.name)
    
    if st.sidebar.button("▶️ Processar Vídeo"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            
            barra_progresso = st.progress(0)
            texto_progresso = st.empty()
            
            def atualizar_progresso(percentual):
                barra_progresso.progress(percentual)
                label_hw = "GPU Turbo" if modo_execucao else "CPU"
                texto_progresso.text(f"Processando ({label_hw}): {int(percentual * 100)}%")
            
            # Obtém o analisador correto do cache
            analisador = get_analisador(id_paciente, movimento, lado, modo_execucao)
            
            # Processamento com Unpack de 5 valores
            frames, angulos, velocidades, total_reps, fps = analisador.processar_video_para_memoria(
                tfile.name, 
                progress_callback=atualizar_progresso
            )
            
            st.session_state.update({
                'frames_salvos': frames, 
                'angulos_salvos': angulos, 
                'velocidades_salvas': velocidades,
                'total_reps': total_reps,
                'fps': fps, 
                'analise_feita': True
            })
            barra_progresso.empty()
            texto_progresso.empty()

st.sidebar.divider()
st.sidebar.subheader("📊 Estatísticas do Dataset")
try:
    df_temp = pd.read_csv("src/utils/dataset_pibic.csv")
    st.sidebar.write(f"Total de amostras: {len(df_temp)}")
    st.sidebar.write(df_temp['label'].value_counts())
except:
    st.sidebar.write("Amostras coletadas: 0")

# --- ÁREA DO PLAYER E RELATÓRIO ---
if st.session_state.analise_feita and st.session_state.frames_salvos:
    st.divider()
    col_player, col_stats = st.columns([2, 1.2])
    
    total_frames = len(st.session_state.frames_salvos)
    fps = st.session_state.fps
    duracao_total_segundos = (total_frames - 1) / fps

    with col_player:
        st.subheader(f"Visualização: {movimento}")
        
        c1, c2 = st.columns([1, 4])
        if c1.button("⏸️ Pausar" if st.session_state.playing else "▶️ Reproduzir"):
            st.session_state.playing = not st.session_state.playing
            st.rerun()
            
        idx = c2.slider("Linha do Tempo", 0, total_frames - 1, st.session_state.frame_atual, label_visibility="collapsed")
        
        if idx != st.session_state.frame_atual:
            st.session_state.update({'frame_atual': idx, 'playing': False})
            st.rerun()

        image_placeholder = st.empty()
        st.caption(f"**⏱️ Tempo:** {st.session_state.frame_atual / fps:.2f}s / {duracao_total_segundos:.2f}s | **🎞️ Frame:** {st.session_state.frame_atual} | **🚀 Hardware:** {'GPU' if modo_execucao else 'CPU'}")
        
        frames_pulo = int(fps * 0.5) 
        cp, _, cn = st.columns([1, 2, 1])
        if cp.button(f"⏪ -0.5s"):
            st.session_state.update({'frame_atual': max(0, st.session_state.frame_atual - frames_pulo), 'playing': False})
            st.rerun()
        if cn.button(f"+0.5s ⏩"):
            st.session_state.update({'frame_atual': min(total_frames - 1, st.session_state.frame_atual + frames_pulo), 'playing': False})
            st.rerun()

        if st.session_state.playing:
            for i in range(st.session_state.frame_atual, total_frames):
                if not st.session_state.playing: break
                image_placeholder.image(st.session_state.frames_salvos[i], channels="RGB", use_container_width=True)
                st.session_state.frame_atual = i
                time.sleep(1/st.session_state.fps * 0.8) 
            st.session_state.playing = False
            st.rerun()
        else:
            image_placeholder.image(st.session_state.frames_salvos[st.session_state.frame_atual], channels="RGB", use_container_width=True)

    with col_stats:
        st.subheader("📊 Relatório Clínico")
        
        angulo_atual = st.session_state.angulos_salvos[st.session_state.frame_atual]
        velocidade_atual = st.session_state.velocidades_salvas[st.session_state.frame_atual]
        validos = [x for x in st.session_state.angulos_salvos if x > 1]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Ângulo", f"{angulo_atual:.1f}°")
        m2.metric("Velocidade", f"{int(velocidade_atual)}°/s")
        m3.metric("Repetições", st.session_state.total_reps)
        
        if validos:
            adm_max = max(validos)
            with st.expander("📄 Ver Relatório Detalhado", expanded=True):
                st.markdown(f"**Paciente:** `{id_paciente}` | **Repetições:** `{st.session_state.total_reps}`")
                st.markdown(f"**Avaliação:** {movimento} ({lado})")
                st.markdown("---")
                st.markdown(f"- **Amplitude Máxima:** {adm_max:.1f}°")
                st.markdown(f"- **Média de Movimento:** {np.mean(validos):.1f}°")
                st.markdown(f"- **Velocidade Pico:** {int(max(st.session_state.velocidades_salvas))}°/s")
                
            st.line_chart(pd.DataFrame(st.session_state.angulos_salvos, columns=["Graus"]))

            st.divider()
            st.subheader("🧪 Coleta para o PIBIC")
            col_label, col_save = st.columns([2, 1])

            with col_label:
                label_qualidade = st.selectbox(
                    "Diagnóstico visual:",
                    ["Selecione...", "Fluido/Normal", "Tremor Leve", "Tremor Acentuado", "Braquicinesia"]
                )

            if col_save.button("💾 Salvar Dataset"):
                if label_qualidade != "Selecione..." and st.session_state.angulos_salvos:
                    # Usa o analisador cacheado para normalizar e salvar
                    analisador_pibic = get_analisador(id_paciente, movimento, lado, modo_execucao)
                    dados_norm = analisador_pibic.normalizar_sequencia(st.session_state.angulos_salvos)
                    
                    novo_dado = {
                        "id_paciente": id_paciente,
                        "movimento": movimento,
                        "lado": lado,
                        "label": label_qualidade,
                        "sequencia": str(dados_norm),
                        "hardware": "GPU" if modo_execucao else "CPU"
                    }
                    
                    path_ds = "src/utils/dataset_pibic.csv"
                    os.makedirs(os.path.dirname(path_ds), exist_ok=True)
                    pd.DataFrame([novo_dado]).to_csv(path_ds, mode='a', index=False, header=not os.path.exists(path_ds))
                    st.success(f"✅ Dados catalogados!")
