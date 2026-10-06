# Fase 12 — Expansão clínica controlada

Versão: `2.10.0-alpha.1`  
Protocolo clínico: `1.1`

## Objetivo

Ampliar a via pré-clínica apenas para movimentos que possam ser descritos por uma convenção angular explícita usando três landmarks e um plano de captura definido. A expansão não é tratada como validação clínica.

## Fonte autoritativa

`backend/src/core/clinical.py::MOVEMENT_SPECS` define, para cada movimento:

- código articular;
- landmark proximal;
- centro articular;
- landmark distal;
- referência pré-clínica de amplitude;
- limiares operacionais de ciclo;
- função de conversão geométrica → clínica;
- plano de medição;
- vista recomendada;
- instruções de captura;
- estado de validação.

O motor deriva seu dispatch dessa tabela. Um movimento não presente em `MOVEMENT_SPECS` é rejeitado.

## Convenções angulares

### Flexão de cotovelo, joelho e quadril

A convenção continua:

`ângulo clínico = 180° − ângulo interno segmentar`

Ela transforma a posição segmentar aproximadamente alinhada em 0° de flexão.

### Dorsiflexão de tornozelo

A medição usa joelho → tornozelo → antepé. A convenção operacional é:

`dorsiflexão = max(0°, 90° − ângulo interno tíbia–pé)`

Assim, aproximadamente 90° internos correspondem a 0° de dorsiflexão e 70° internos correspondem a aproximadamente 20°. Valores associados à flexão plantar não recebem sinal negativo nesta métrica; flexão plantar deve ser implementada como movimento próprio.

## Protocolo 1.1

O hash da especificação clínica passa a incluir a convenção e o protocolo de captura de cada movimento. Novos estudos criados pela interface usam `protocol_version=1.1`; estudos antigos permanecem preservados com sua versão original.

## Interface

O frontend não mantém mais uma lista clínica independente. Após autenticação ele consulta:

`GET /api/config/movimentos`

A resposta inclui:

- `protocol_version`;
- `clinical_spec_sha256`;
- movimentos habilitados;
- referência de ROM;
- convenção angular;
- plano e vista;
- landmarks;
- instruções de captura.

O mesmo identificador de convenção é usado no preview do goniômetro manual.

## Movimentos deliberadamente não habilitados

Ombro e punho continuam bloqueados porque um ângulo entre três pontos, isoladamente, não diferencia adequadamente todos os movimentos de interesse sem controlar plano/eixo e compensações. Marcha, LiDAR e sinais parkinsonianos permanecem módulos experimentais/futuros.

## Regressão

A Fase 12 adiciona testes para:

- especificação explícita de quadril e tornozelo;
- conversão de dorsiflexão;
- geometria manual 2D de quadril e tornozelo;
- publicação do protocolo de captura;
- endpoint de configuração e protocolo 1.1.

Resultado esperado: **78/78 testes**.

## Interpretação

Nenhum dos quatro movimentos deve ser descrito como clinicamente validado até a conclusão de estudos com referência física e protocolo aprovado. `reference_rom_deg` serve para interface e análise pré-clínica; não deve ser usado como limiar diagnóstico individual.
