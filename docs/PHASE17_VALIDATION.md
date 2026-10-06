# Fase 17 — sensor fusion inercial experimental

Versão: `2.15.0-alpha.1`

Protocolo clínico de ROM permanece `1.4`. O módulo IMU é **experimental** e não altera a via clínica validável de goniometria.

## Objetivo

Adicionar uma camada auditável de telemetria inercial para estudos futuros de movimento fino e sincronização câmera + smartphone, sem inferir diagnóstico neurológico.

## Entrada

`POST /api/experimental/imu/analyze`

- `timestamps_ms`: timestamps monotônicos em milissegundos;
- `gyro_rad_s`: giroscópio `(N,3)` em rad/s;
- `accel_m_s2`: aceleração opcional `(N,3)` em m/s²;
- `band_low_hz` / `band_high_hz`: banda espectral configurável;
- `job_id`: sessão de vídeo opcional para sincronização;
- `video_offset_ms`: convenção `video_time ~= imu_time + offset`;
- `device_label`: rótulo operacional do sensor.

Máximo: 60.000 amostras por análise.

## Análise

1. validação de timestamps e vetores;
2. cálculo da taxa de amostragem efetiva, jitter e gaps;
3. reamostragem uniforme para evitar que jitter do clock seja interpretado como conteúdo espectral;
4. projeção do giroscópio no primeiro componente principal (PCA);
5. PSD por Welch;
6. frequência dominante, razão de potência na banda configurada, entropia espectral e periodicidade;
7. RMS/p95 giroscópico e RMS de aceleração dinâmica;
8. sincronização opcional com timestamps dos frames persistidos.

## Rastreabilidade

O sinal bruto é armazenado comprimido em SQLite, acompanhado de SHA-256 e resumo analítico. Quando vinculado a um `job_id`, o bundle científico da sessão inclui:

- `imu/index.json`;
- `imu/<analysis_id>.json` com registro + payload bruto;
- checksums SHA-256.

## Limite científico

Nenhuma saída desta fase constitui diagnóstico ou classificação de Parkinson, tremor, bradicinesia ou rigidez. Os descritores são sinais experimentais a serem validados contra protocolo clínico/instrumental próprio antes de qualquer interpretação clínica.

## Regressão

- 115 testes de backend;
- testes sintéticos de frequência dominante e jitter;
- teste de sincronização vídeo↔IMU;
- round-trip da persistência comprimida;
- exportação científica com checksum;
- API autenticada e exportação do sinal bruto;
- migração aditiva em banco existente.
