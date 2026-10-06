# PhysioAI — Fase 13: referencial anatômico e análise multiplanar

Versão: `2.11.0-alpha.1`  
Protocolo clínico: `1.2`

## Objetivo

A Fase 13 introduz um referencial anatômico corporal aproximado e projeção explícita no plano de medição. O objetivo é reduzir a aceitação de ângulos produzidos em capturas geometricamente inadequadas e preparar a arquitetura para movimentos que dependem de plano/eixo bem definidos.

Esta fase **não habilita ombro ou punho**. Cotovelo, joelho, quadril e dorsiflexão de tornozelo permanecem pré-clínicos.

## Referencial corporal

O referencial é derivado dos pares bilaterais do MediaPipe Pose:

- ombros: landmarks 11 e 12;
- quadris: landmarks 23 e 24.

São construídos três eixos ortogonais:

- mediolateral;
- superior-inferior;
- anteroposterior.

A projeção clínica usa a normal do plano anatômico:

- sagital → normal mediolateral;
- frontal → normal anteroposterior;
- transverso → normal superior-inferior.

## Cadeia angular auditável

Cada frame passa a distinguir:

1. `angle_raw_deg` — ângulo interno em landmarks 3D brutos quando disponível;
2. `angle_landmark_deg` — ângulo interno nos landmarks 3D após Kalman;
3. `angle_plane_deg` — ângulo interno após projeção no plano anatômico;
4. `angle_filtered_deg` — EMA aplicada ao ângulo projetado;
5. `angle_clinical_deg` — conversão para a convenção clínica do movimento.

O sistema continua expondo os três pontos 2D/3D que originaram a medição.

## Gate geométrico de validade

Visibilidade de landmarks continua sendo um sinal de rastreamento, não de acurácia clínica. Além dela, um frame só entra nas métricas clínicas quando satisfaz o protocolo geométrico:

- referencial corporal disponível;
- visibilidade bilateral mínima do referencial >= 0,65;
- projeção anatômica disponível;
- alinhamento da câmera com a normal do plano >= `minViewAlignment`;
- componente máxima fora do plano <= `maxOutOfPlaneDeg`;
- fonte observada e landmarks clínicos dentro do limiar de visibilidade.

Valores padrão da Fase 13:

- `minViewAlignment = 0.80`;
- `maxOutOfPlaneDeg = 35°`.

Esses limiares são operacionais e pré-clínicos. Não representam limites diagnósticos nem foram validados como pontos de corte clínicos.

## Telemetria adicional

Cada `analysis.frame` passa a publicar:

- `anglePlaneDeg`;
- `measurementPlane`;
- `planeProjectionApplied`;
- `bodyFrameConfidence`;
- `viewAlignment`;
- `viewDeviationDeg`;
- `outOfPlaneDeg`.

Flags possíveis incluem:

- `body_frame_unavailable`;
- `low_body_frame_visibility`;
- `anatomical_plane_projection_unavailable`;
- `camera_alignment_unavailable`;
- `camera_view_misaligned`;
- `out_of_plane_unavailable`;
- `out_of_plane_motion`.

## Persistência e pesquisa

O schema de telemetria foi incrementado para `2` e migra automaticamente bancos anteriores. Ensaios de validação física preservam também metadados de plano/alinhamento para auditoria posterior.

Os bundles científicos exportam essas novas colunas em `measurements.csv` e `trials.csv`.

## Interface

O modo clínico mostra:

- validade do frame;
- desvio angular estimado da câmera em relação à vista recomendada;
- componente do movimento fora do plano.

O modo engenharia mostra ainda o ângulo 3D não projetado e o ângulo projetado lado a lado.

## Limitações

- O referencial é estimado a partir de landmarks corporais, não de um sistema óptico de motion capture calibrado.
- O alinhamento da câmera é uma estimativa geométrica e não uma calibração extrínseca certificada.
- Compensações corporais podem alterar o referencial local.
- Frames geometricamente inválidos podem continuar exibindo telemetria para inspeção, mas são excluídos de ADM/repetições/estudos quando `measurement_valid=false`.
- Nenhum movimento nesta fase é declarado clinicamente validado.

## Verificação

Execute:

```bash
python scripts/verify_phase13.py
```

A verificação executa compilação Python, regressão de backend, manifesto SHA-256, doctor de integração e validação TypeScript quando disponível.
