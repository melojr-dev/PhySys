
## 2.22.1-rc.5 — Live overlay sync fix

- Corrige dessincronização entre vídeo e landmarks durante processamento ao vivo.
- O overlay agora avança somente após o navegador concluir o seek para o timestamp analisado.
- Usa fila `latest-wins` para evitar acúmulo de seeks quando o backend processa mais rápido que o decoder consegue apresentar.
- Protocolo clínico 1.4 e cálculos biomecânicos permanecem inalterados.

## 2.22.0-rc.5 — Fase 24: Markerless Experience

- Reposiciona a experiência principal como **Markerless Human Motion Analysis**.
- Adiciona Home de apresentação com narrativa vídeo → landmarks → biomecânica → resultado.
- Adiciona alternância entre modo Apresentação e modo Pesquisa sem alterar o protocolo clínico 1.4.
- Evolui o overlay anatômico do membro selecionado com cadeia completa, mão/pé e leitura angular vinculada ao frame.
- Restaura a visualização 3D no frontend com projeção interativa dos landmarks e modo de volumes segmentares estimados.
- Solicita `skeleton_3d` no resultado principal para manter vídeo 2D e modelo 3D sincronizados.
- Substitui o gráfico básico por curva temporal interativa com pico, cursor e resumo de movimento.
- Simplifica métricas para leigos no modo Apresentação e preserva telemetria/validação no modo Pesquisa.
- Fixa `websockets==13.1` e pré-carrega o MediaPipe Pose Heavy no build do backend para evitar falhas de runtime não-root.
- Protocolo clínico permanece **1.4**; não há alteração de fórmulas biomecânicas nesta fase.

# Changelog

## 2.21.0-rc.4 — Fase 23

- Consolidação completa do frontend em cinco módulos: Avaliação, Estudos, Experimental, Pesquisa e Administração.
- Navegação responsiva com barra inferior no mobile e navegação compacta no desktop.
- Tela dedicada para gestão de estudos, execução formal, readiness e pacote de revisão externa.
- Tela experimental com marcha temporal, IMU, perfil motor multimodal e seguimento longitudinal.
- Navegador de sessões com proveniência, hashes e exportação do bundle científico.
- Visão do dataset SQLite e amostras catalogadas.
- Painel administrativo de health/readiness, métricas, auditoria e exclusão LGPD.
- Paridade de API frontend com endpoints acumulados até a Fase 22.
- Nenhuma alteração na biomecânica ou no protocolo clínico 1.4.

## 2.21.0-rc.4 — Fase 22

- adiciona pacote externo desidentificado derivado exclusivamente de execução formal fechada;
- re-alias de participantes/sessões e remoção de job IDs, avaliadores, serial do instrumento e notas livres;
- adiciona tabelas CSV padronizadas, resumo técnico e figuras SVG determinísticas de comparação de métodos e Bland–Altman;
- adiciona `publication_requirements.json` e matriz de rastreabilidade requisito→implementação→teste→evidência;
- adiciona prontidão técnica do pacote com alertas explícitos para amostra pequena/incompleta sem fabricação de dados;
- adiciona endpoints autenticados de prontidão e exportação do pacote de revisão externa;
- mantém protocolo clínico 1.4, golden baseline e escopo biomecânico congelados;
- não introduz nova articulação, diagnóstico ou alegação de validade clínica.

## 2.19.0-rc.2 — Fase 21

- adiciona execução formal de validação com snapshot imutável de software, protocolo, estudo e plano estatístico;
- adiciona gate operacional de prontidão para ética declarada, calibração do instrumento, setup de câmera e completude do estudo;
- congela `statistical_analysis_plan.json` e `formal_validation_checklist.json`;
- relatórios de execução formal usam somente ensaios coletados dentro da janela daquela execução;
- adiciona exportação ZIP autocontida com checksums para auditoria/reprodutibilidade;
- mantém protocolo clínico 1.4 e golden baseline sem mudanças biomecânicas;
- não introduz diagnóstico, nova articulação ou alegação de validade clínica.

## 2.18.0-rc.1 — Fase 20

- congela o contrato biomecânico do protocolo clínico 1.4 em `release/protocol_lock.json`;
- adiciona `release/clinical_baseline.json` com casos canônicos de regressão para os seis movimentos pré-clínicos;
- adiciona verificador independente do baseline/protocol lock;
- formaliza política de governança de mudanças biomecânicas;
- promove o pacote a release candidate de software para validação formal, sem alegação de validação clínica.

## 2.17.0-alpha.1 — Fase 19
- Novo módulo `src.core.longitudinal` para seguimento descritivo entre sessões do mesmo paciente.
- Sessão âncora define o cohort comparável: mesmo movimento, lado, protocolo clínico e `clinical_spec_sha256`.
- Sessões incompatíveis são excluídas explicitamente e preservadas com a razão de exclusão.
- Versões diferentes de software sob o mesmo protocolo são permitidas apenas com flag de comparabilidade, sem correção silenciosa.
- Métricas longitudinais: primeira/última medida, delta, delta percentual, média, SD, CV, mudança absoluta consecutiva e slope por 30 dias.
- Integração opcional de descritores motores multimodais já persistidos por sessão.
- Contrato explícito `diagnostic=false`; melhora clínica, progressão, resposta terapêutica e MCID permanecem fora das inferências suportadas.
- Persistência SQLite de análises longitudinais com SHA-256 das fontes e exportação JSON autenticada.
- Bundles científicos passam a incluir `longitudinal/index.json`.
- Exclusão LGPD remove também análises longitudinais do paciente.
- Listagem de sessões corrigida para isolamento por avaliador em usuários não administradores.
- Frontend recebe painel de seguimento longitudinal para a sessão atual.
- Regressão ampliada para 136 testes de backend antes do fechamento do release.

## 2.16.0-alpha.1 — Fase 18
- Novo módulo `src.core.motor_features` para descritores motores experimentais derivados da telemetria clínica frame a frame.
- Extração de ciclos de movimento repetitivo com duração, amplitude e velocidade de pico por ciclo.
- Variabilidade temporal/amplitude e decremento início→fim do ensaio como descritores longitudinais, sem classificação clínica de bradicinesia.
- Análise espectral angular em banda configurável, separada da análise IMU e condicionada à taxa/qualidade do vídeo.
- Fusão opcional vídeo + IMU com overlap temporal, qualidade multimodal e diferença de frequência dominante.
- Contrato explícito `diagnostic=false` e lista de inferências não suportadas: Parkinson, rigidez/roda dentada e bradicinesia clínica.
- Persistência SQLite dos perfis motores com SHA-256 das fontes, sessão e análise IMU vinculadas.
- Endpoints autenticados para extração, consulta e exportação JSON dos descritores.
- Bundle científico da sessão passa a incluir `motor_features/index.json`.
- Exclusão LGPD remove perfis motores vinculados às sessões do paciente.
- Frontend ganha painel experimental de perfil motor multimodal após a importação IMU.
- Regressão ampliada para 124 testes de backend antes do fechamento do release.

## 2.15.0-alpha.1 — Fase 17
- Novo módulo `src.core.imu` para análise inercial experimental de smartphone/IMU.
- Qualidade temporal do sensor: taxa efetiva, jitter, gaps e proporção de amostras inválidas.
- Sinal giroscópico projetado por PCA antes da análise espectral para reduzir dependência da orientação do aparelho.
- PSD/Welch com frequência dominante, potência em banda configurável, entropia espectral e índice de periodicidade.
- RMS/p95 do giroscópio e RMS de aceleração dinâmica, sem promover essas métricas a diagnóstico.
- Sincronização opcional IMU↔vídeo por `job_id` e `video_offset_ms`, com overlap e erro temporal auditáveis.
- Persistência SQLite do sinal bruto comprimido, resumo, SHA-256, usuário e sessão vinculada.
- Endpoints autenticados de análise, consulta e exportação JSON do IMU.
- Bundle científico de sessão passa a carregar índice e payloads IMU reprodutíveis com checksums.
- Exclusão LGPD de paciente remove também análises IMU vinculadas às sessões.
- Frontend recebe painel experimental para importação CSV (`timestamp_ms,gx,gy,gz[,ax,ay,az]`) e vínculo com a sessão atual.
- Regressão ampliada para 115 testes de backend.

## 2.14.0-alpha.1 — Fase 16
- Novo módulo `src.core.gait` para análise temporal experimental de marcha.
- Eventos estimados de calcanhar, cadência, tempos de passo/ciclo, variabilidade e simetria temporal.
- Métricas espaciais bloqueadas em coordenadas relativas; comprimento de passada apenas em `metric_global`.
- Endpoint autenticado `POST /api/experimental/gait/analyze`, separado da via clínica.
- Contratos TypeScript para análise temporal de marcha.
- Regressão ampliada para 106 testes de backend.


## 2.13.0-alpha.1 — Fase 15
- Preflight obrigatório de captura antes da análise clínica.
- Upload do vídeo ocorre uma única vez; o `preflight_id` reutiliza o mesmo temporário/hash no job.
- Amostragem temporal avalia resolução, FPS, detecção corporal, visibilidade, enquadramento, referencial bilateral, alinhamento de câmera e movimento fora do plano.
- Iluminação, contraste e nitidez entram como alertas operacionais não diagnósticos.
- Captura reprovada é bloqueada; override experimental exige justificativa administrativa auditada.
- Relatório completo de preflight é preservado na proveniência da sessão e no bundle científico.
- Critérios de preflight passam a fazer parte do snapshot/hash da especificação clínica.
- Protocolo clínico atualizado para 1.4; protocolo de preflight versionado como 1.0.
- Nginx encaminha o upload de preflight sem buffering para evitar cópia temporária duplicada.
- Frontend ganha fluxo responsivo de “Verificar captura” com checklist e score antes de habilitar a análise.
- Smoke E2E atualizado para autenticação → preflight → job → WebSocket → resultado.
- Regressão ampliada para 101 testes de backend antes do fechamento do release.

## 2.12.0-alpha.1 — Fase 14
- Via clínica expandida para Flexão de Ombro e Abdução de Ombro.
- Flexão do ombro usa plano sagital; abdução usa plano frontal.
- Convenção de ombro passa a usar diretamente o ângulo projetado entre tronco inferior e braço.
- Gate direcional diferencia flexão de extensão posterior e abdução de adução cruzada.
- Telemetria adiciona `direction_rule` e `direction_alignment`.
- Schema de medições incrementado para v3 com migração automática.
- Ensaios e bundles científicos preservam metadados direcionais.
- Frontend mostra compatibilidade de sentido e suporta a convenção `direct_internal_angle`.
- Protocolo clínico atualizado para 1.3.
- Punho permanece bloqueado por insuficiência de referência palmar robusta nos landmarks Pose.
- Regressão ampliada para 97 testes de backend.

## 2.11.0-alpha.1 — Fase 13
- Referencial anatômico corporal derivado de ombros e quadris bilaterais.
- Projeção explícita de ângulos no plano anatômico configurado.
- Cadeia auditável passa a separar `angle_landmark_deg`, `angle_plane_deg`, EMA e ângulo clínico.
- Gate geométrico por alinhamento da câmera, disponibilidade do referencial e componente fora do plano.
- Telemetria adiciona plano, confiança do referencial, alinhamento, desvio da câmera e movimento fora do plano.
- Schema de medições incrementado para v2 com migração automática.
- Ensaios de validação preservam metadados geométricos de captura.
- Bundles científicos exportam os novos campos.
- Frontend mostra qualidade geométrica no modo clínico e engenharia.
- Protocolo clínico atualizado para 1.2.
- Ombro e punho continuam bloqueados até modelagem específica de plano/eixo.

## 2.10.0-alpha.1 — Fase 12
- Via clínica expandida para Flexão de Quadril e Dorsiflexão de Tornozelo.
- `MOVEMENT_SPECS` passa a ser a única fonte autoritativa do dispatch clínico.
- Heurísticas legadas para ombro/punho deixam de ser fallback do motor; movimentos não especificados são rejeitados.
- Protocolo clínico atualizado para 1.1 com convenção angular específica por movimento.
- Especificações publicam plano, vista recomendada, landmarks e instruções de captura.
- Frontend carrega movimentos dinamicamente do backend e exibe protocolo de captura.
- Preview do goniômetro manual passa a respeitar a convenção angular selecionada.
- Novos estudos experimentais registram a configuração de captura derivada da especificação clínica.
- Ombro, punho, marcha, LiDAR e Parkinson permanecem fora da via clínica principal.
- Regressão ampliada para 78 testes de backend.

## 2.9.0-alpha.1 — Fase 11
- Estudos de validação com protocolo imutável após congelamento e hash SHA-256.
- Ensaios de referência física vinculados a sessão/frame e reconstruídos com telemetria server-side.
- Separação entre revisão manual da imagem e referência goniométrica física.
- Critérios explícitos de inclusão/exclusão por fonte, validade e visibilidade de landmarks.
- Resumo de erro balanceado por participante para reduzir pseudo-replicação por múltiplos frames.
- Bootstrap determinístico em cluster de participante com IC95% de MAE e bias.
- Estratificação de concordância por faixa do ângulo de referência e lado corporal.
- Exportação científica de estudo (`protocol.json`, `trials.csv`, `report.json`, checksums).
- Bundle de sessão passa a incluir ensaios de estudos associados.
- Painel frontend para criação/congelamento de protocolo e registro separado da referência física.
- Registros de referência legados enviados no endpoint de revisão são ignorados na nova via experimental.
- Regressão ampliada para 72 testes de backend.

## 2.8.0-alpha.1 — Fase 10
- Integração end-to-end entre Nginx, frontend, FastAPI, WebSocket e persistência.
- Runtime clínico movido para volume Docker dedicado (`/app/data`).
- Backend executado como usuário não-root e containers com `no-new-privileges`.
- Readiness inclui `PRAGMA quick_check` do SQLite e espaço livre do volume clínico.
- Healthchecks encadeados para backend, frontend e Nginx.
- Rota `/version` corrigida no reverse proxy.
- Upload de análise com `proxy_request_buffering off` para preservar streaming.
- Fila WebSocket limitada para impedir crescimento não controlado de RAM.
- Correção de frame sem pose inicial (`measured_geo=None` em vez de variável não inicializada).
- Frontend exibe versão injetada pelo build, removendo string de versão obsoleta.
- Smoke E2E com autenticação, upload MP4, handshake WebSocket, telemetria, resultado e persistência.
- Scripts cross-platform para teste completo da stack.
- Backup SQLite consistente com `quick_check` e SHA-256.
- Doctor de consistência de release/runtime.
- Regressão ampliada para 63 testes de backend.

## 2.7.0-alpha.1 — Fase 9
- Persistência de telemetria frame a frame em SQLite.
- Proveniência por sessão com protocolo e hashes SHA-256 dos inputs.
- Upload em streaming com limite configurável.
- Revisão manual e ângulo automático autoritativos no backend.
- Métodos de comparação registrados (`image_2d_three_point` e `physical_goniometer`).
- Exportação científica ZIP por sessão com checksums.
- Hash determinístico do resultado consolidado.
- Pontos clínicos 3D persistidos por frame, com identificação do espaço de medição.
- Ângulo geométrico dos landmarks 3D/Kalman (`angle_landmark_deg`) persistido separadamente do ângulo bruto, EMA e ângulo clínico.
- Catalogação de dataset derivada exclusivamente de sessão/telemetria server-side.
- Dataset vinculado a `job_id`, hash do resultado, versão do software e protocolo.
- Token do WebSocket transportado por subprotocolo em vez de query string no frontend atual.
- Regressão geométrica e testes de integridade do pacote científico.
- Build frontend mantém dependências diretas fixadas; o workflow de CI executa sua compilação.

## 2.6.0-alpha.1 — Fase 8
- Build metadata único (`version`, `git_sha`, `build_time`, `environment`).
- Endpoint público `/version` e versão/revisão em health.
- Dependências diretas fixadas para backend e frontend.
- OCI labels nas imagens Docker.
- Manifesto SHA-256 verificável para releases.
- CI de backend, frontend e validação do Compose.
- Workflow de release por tag.
- Scripts cross-platform para gerar/verificar release.

## 2.5.0 — Fase 7
- Observabilidade, readiness, backpressure, métricas e smoke/load tests.
