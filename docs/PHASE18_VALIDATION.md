# Fase 18 — descritores motores multimodais experimentais

Versão: `2.16.0-alpha.1`

O protocolo clínico de ROM permanece `1.4`. Esta fase acrescenta uma camada **experimental de extração de features**, sem promover nenhum descritor a diagnóstico ou escore clínico.

## Objetivo

Combinar a série angular auditável do vídeo com, quando disponível, a telemetria IMU persistida da mesma sessão. O objetivo é produzir variáveis quantitativas para futuros estudos de validade de construto/critério, especialmente em tarefas motoras repetitivas.

## Entrada autoritativa

`POST /api/experimental/motor-features/analyze`

- `job_id`: sessão já processada e pertencente ao usuário;
- `imu_analysis_id`: opcional, mas quando fornecido deve estar vinculado ao mesmo `job_id`;
- `min_cycle_prominence_deg`: prominência mínima para eventos cíclicos;
- `tremor_band_low_hz` / `tremor_band_high_hz`: banda espectral exploratória.

O navegador não envia ângulos calculados. O backend lê as medições persistidas em `frame_measurements` e, opcionalmente, o resumo IMU persistido.

## Features de vídeo

Somente frames marcados como `measurement_valid` entram na série experimental.

- proporção de frames válidos;
- duração e taxa de amostragem efetiva;
- faixa angular robusta (P95−P5);
- velocidade angular absoluta mediana e P95;
- número de ciclos detectados;
- duração mediana e CV dos ciclos;
- amplitude mediana e CV por ciclo;
- velocidade de pico mediana e CV por ciclo;
- decremento de amplitude início→fim;
- decremento de velocidade de pico início→fim;
- potência/frequência angular na banda configurada quando o FPS é suficiente.

Decrementos positivos significam redução do descritor ao longo do ensaio. Isso **não equivale** a classificação clínica de bradicinesia ou fatigabilidade.

## Fusão com IMU

Quando há uma análise IMU da mesma sessão:

- qualidade do IMU;
- frequência dominante e potência na banda;
- periodicidade;
- overlap temporal vídeo↔IMU;
- diferença absoluta entre frequências dominantes vídeo↔IMU, quando ambas estão disponíveis.

Não há regra que transforme concordância espectral em diagnóstico.

## Gate de qualidade

O perfil carrega `fusion_quality_score`, `status` e `quality_flags`. Séries com menos de 70% de frames válidos nunca recebem status de vídeo `usable`, mesmo que outras métricas estejam boas.

## Inferências explicitamente não suportadas

- diagnóstico de Doença de Parkinson;
- classificação clínica de bradicinesia;
- rigidez;
- roda dentada.

Essas inferências exigiriam protocolo clínico/instrumental próprio, amostra adequada, referência e validação independente.

## Rastreabilidade

Cada perfil persiste:

- `job_id`;
- `imu_analysis_id` quando aplicável;
- SHA-256 canônico das fontes/configuração;
- resumo completo da extração;
- usuário e timestamp.

O bundle científico da sessão inclui `motor_features/index.json` com checksums.

## Regressão

- testes sintéticos de ciclos e duração;
- teste de decremento de amplitude;
- gate de frames válidos;
- contrato não diagnóstico;
- modo vídeo-only explícito;
- persistência/remoção por job;
- migração aditiva em banco existente;
- API autenticada vídeo + IMU;
- exportação científica do perfil.
