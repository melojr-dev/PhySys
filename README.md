# PhysioAI v2.22 — Markerless Human Motion Analysis


## Fase 24 — experiência markerless

A versão `2.22.0-rc.5` reorganiza a experiência principal para deixar explícito que o **núcleo do PhysioAI é markerless**: a análise parte de vídeo comum e não exige marcadores corporais nem sensores vestíveis. A mudança desta fase é de produto e visualização; o protocolo clínico permanece congelado em `1.4`.

Principais mudanças: Home orientada a leigos, modo Apresentação/Pesquisa, overlay anatômico completo do membro selecionado, curva angular interativa, retorno do visualizador 3D e representação volumétrica segmentar estimada. A visualização 3D não deve ser interpretada como escaneamento anatômico de precisão. Consulte `docs/PHASE24_VALIDATION.md`.

## Fase 22 — pacote científico para revisão externa

A versão 2.20 mantém o protocolo clínico 1.4 e o escopo biomecânico congelado. A mudança é de **saída científica e auditabilidade**: uma execução formal fechada pode gerar um pacote externo desidentificado contendo ensaios anonimizados, tabelas estatísticas, figuras SVG determinísticas (comparação de métodos e Bland–Altman), resumo técnico, governança congelada e matriz requisito→implementação→teste.

O pacote omite IDs internos de paciente/sessão/avaliador, serial do instrumento e notas livres. Participantes e sessões recebem aliases locais (`P001`, `S001`). Coleta incompleta é sinalizada como alerta e **não é preenchida ou extrapolada**. Consulte `docs/PHASE22_VALIDATION.md` e `docs/EXTERNAL_REVIEW_PACKAGE.md`.

> O pacote é material de comparação de métodos/revisão externa. Ele não declara validade clínica, desempenho diagnóstico, eficácia terapêutica ou aceitação para publicação.

## Fase 21 — pacote de execução formal de validação

A versão 2.19 mantém o protocolo clínico 1.4 e o golden baseline da Fase 20, mas adiciona uma camada operacional para **execuções formais de comparação de métodos**. Cada execução congela versão do software, protocolo do estudo, plano estatístico, situação ética declarada e calibração do instrumento de referência. Consulte `docs/PHASE21_VALIDATION.md` e `docs/FORMAL_VALIDATION.md`.

> O software não verifica a validade jurídica de uma aprovação ética e não declara validade clínica. O gate existe para impedir coleta formal quando os metadados mínimos de governança estiverem incompletos.

## Fase 20 — candidato de software para validação formal

A versão 2.18 congela o escopo funcional e adiciona um **golden baseline clínico de software** e um **protocol lock** para o protocolo 1.4. Mudanças silenciosas em landmarks, convenções angulares, plano anatômico ou thresholds críticos de preflight passam a falhar na verificação da release. Isso é controle de regressão computacional, **não validação clínica**. Consulte `docs/PHASE20_VALIDATION.md` e `docs/PROTOCOL_GOVERNANCE.md`.

## Fase 19 — seguimento longitudinal descritivo

A versão 2.17 adiciona um módulo experimental de seguimento longitudinal do mesmo paciente. O job atual funciona como sessão âncora e o backend só agrega sessões que mantenham **movimento, lado, versão do protocolo clínico e hash da especificação biomecânica** compatíveis. Mudanças de versão de software são preservadas e sinalizadas; mudanças de protocolo/especificação são excluídas do pool principal.

As saídas incluem delta entre primeira/última sessão, média, desvio-padrão, CV descritivo, mudança absoluta consecutiva e inclinação por 30 dias para ADM, proporção de frames válidos, repetições e, quando disponíveis, descritores motores multimodais. Esses valores **não significam melhora clínica, progressão de doença, resposta terapêutica ou MCID**. Consulte `docs/PHASE19_VALIDATION.md`.



## Fase 18 — descritores motores multimodais experimentais

A versão 2.16 combina a telemetria angular frame a frame com a análise IMU da Fase 17 para extrair **descritores de pesquisa**, não diagnósticos: ciclos de movimento repetitivo, duração/variabilidade dos ciclos, amplitude, velocidade de pico, decremento ao longo do ensaio e conteúdo espectral em banda configurável. Quando há IMU vinculada à mesma sessão, o sistema registra qualidade de sincronização e diferença entre frequências dominantes observadas nas duas modalidades.

O módulo mantém explicitamente fora de escopo: diagnóstico de Parkinson, classificação clínica de bradicinesia, rigidez e roda dentada. Consulte `docs/PHASE18_VALIDATION.md`.

## Fase 17 — IMU experimental

A versão 2.15 adiciona análise inercial auditável para smartphone/IMU, opcionalmente sincronizada a uma sessão de vídeo. O módulo calcula qualidade de amostragem e descritores espectrais, persiste o sinal bruto com SHA-256 e o inclui nos bundles científicos. **Não produz diagnóstico neurológico.** Consulte `docs/PHASE17_VALIDATION.md`.


## Fase 16 — marcha temporal experimental

A Fase 16 adiciona um módulo separado para análise temporal da marcha. Ele não integra a via clínica de ROM e permanece explicitamente experimental. Métricas espaciais em metros são bloqueadas quando as coordenadas são relativas ao corpo. Detalhes: `docs/PHASE16_VALIDATION.md`.

## Fase 15 — preflight de captura e gate pré-clínico

A Fase 15 impede que uma sessão de baixa qualidade chegue ao processamento completo sem ser explicitamente identificada. O vídeo é enviado **uma única vez**: o backend executa uma amostragem temporal, calcula a qualidade de captura e mantém o arquivo temporariamente vinculado a um `preflight_id`. Quando a análise é iniciada, esse mesmo arquivo/hash é reutilizado.

Movimentos habilitados permanecem:

- **Flexão de Cotovelo**;
- **Flexão de Joelho**;
- **Flexão de Quadril**;
- **Dorsiflexão de Tornozelo**;
- **Flexão de Ombro**;
- **Abdução de Ombro**.

### O preflight verifica

- decodificação do vídeo;
- resolução e FPS;
- taxa de detecção corporal;
- visibilidade dos três landmarks clínicos;
- enquadramento anatômico;
- disponibilidade do referencial corporal bilateral;
- alinhamento da câmera com o plano esperado;
- permanência do movimento no plano configurado;
- iluminação, contraste e nitidez como alertas operacionais.

Os critérios fazem parte do **protocolo clínico 1.4** e entram no hash SHA-256 da especificação científica. O preflight não substitui o gate frame a frame: a validade de cada medição continua sendo decidida pela telemetria do motor.

Capturas reprovadas são bloqueadas por padrão. Um administrador pode executar um **override experimental** apenas com justificativa textual auditada; o motivo e o relatório completo ficam preservados na proveniência da sessão.

> Todos os movimentos permanecem **pré-clínicos**. O preflight reduz erros de aquisição, mas não demonstra validade ou acurácia clínica.

Detalhes: `docs/PHASE15_VALIDATION.md`.

## Execução

```bash
cp .env.example .env
# Configure PHYSIOAI_ADMIN_PASSWORD, PHYSIOAI_AUTH_SECRET e PHYSIOAI_PSEUDONYM_SECRET.
docker compose up --build -d
```

Acesse `http://localhost`.

## Fluxo de uma avaliação

1. Autenticar o avaliador.
2. Selecionar paciente/sessão, movimento e lado.
3. Selecionar o vídeo.
4. Clicar em **Verificar captura**.
5. Corrigir a captura quando houver falha crítica; ou registrar override experimental quando justificável.
6. Iniciar a análise reutilizando o `preflight_id`, sem segundo upload.
7. Revisar frame a frame e, quando aplicável, registrar referência física no estudo experimental.

## Verificação da árvore

```bash
python scripts/verify_phase24.py
```

## Teste end-to-end Docker

Linux/macOS/WSL:

```bash
./scripts/e2e_stack.sh
```

PowerShell:

```powershell
./scripts/e2e_stack.ps1
```

O smoke E2E executa autenticação → preflight → análise → WebSocket → resultado → persistência.

## API clínica

- `POST /api/capture/preflight` — envia e avalia o vídeo uma vez;
- `DELETE /api/capture/preflight/{preflight_id}` — descarta vídeo não utilizado;
- `POST /api/analise/processar` — consome `preflight_id` e inicia o job;
- `GET /api/config/movimentos` — publica especificações biomecânicas e protocolo de preflight;
- `GET /api/sessoes/{job_id}/export.zip` — bundle científico da sessão.

## Persistência e privacidade

- o vídeo de preflight fica somente em arquivo temporário e possui TTL configurável;
- ao iniciar a análise, a posse do temporário é transferida ao job e o arquivo é apagado ao término;
- preflights abandonados são removidos automaticamente;
- relatório de captura, hashes, override e configuração entram na proveniência da sessão;
- runtime clínico estruturado permanece em SQLite no volume `physioai_data`.

## Limites atuais

- nenhum movimento é declarado clinicamente validado;
- thresholds de iluminação/nitidez são controles operacionais e devem ser calibrados empiricamente no protocolo de coleta;
- punho, rotação axial do ombro, marcha clínica, LiDAR e Parkinson permanecem fora da via clínica principal;
- o modelo legado de “fluidez” continua desativado para resultados clínicos.