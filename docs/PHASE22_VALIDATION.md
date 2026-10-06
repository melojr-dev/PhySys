# Fase 22 — pacote científico para revisão externa

Versão: `2.21.0-rc.4`  
Protocolo clínico: `1.4` (inalterado)

## Objetivo

Transformar uma **execução formal fechada** em um artefato técnico reproduzível e desidentificado para revisão por orientador, banca, estatístico ou colaborador externo, sem alterar a biomecânica e sem transformar resultados preliminares em alegações clínicas.

## Pré-condições

O pacote só é liberado quando:

- a execução formal está `closed`;
- existe snapshot inicial e relatório de fechamento;
- o estudo permanece `frozen` ou `closed`;
- o protocolo congelado possui SHA-256;
- o relatório usa apenas ensaios coletados dentro da execução formal;
- existe ao menos um ensaio e participante incluído.

Conclusão amostral **não é falsificada**. Se o alvo de participantes não foi atingido, o pacote é gerado com alerta explícito `target_not_reached`.

## Desidentificação

O bundle externo:

- substitui `patient_id` por aliases locais `P001`, `P002`, ...;
- substitui `job_id`/sessão por `S001`, `S002`, ...;
- substitui o identificador da execução por `RUN001` dentro do conteúdo de revisão;
- omite identificadores internos de ensaio;
- omite avaliadores e assessor de referência;
- omite serial do instrumento;
- omite notas livres;
- preserva apenas modelo do instrumento e dados metodológicos necessários à análise.

Os aliases são válidos apenas dentro daquele bundle e não são uma chave reversível fornecida ao revisor.

## Conteúdo

- `manifest.json` — proveniência técnica e limites de interpretação;
- `publication_readiness.json` — checks e alertas;
- `trials_anonymized.csv` — dados usados para os cálculos;
- `report.json` — relatório estatístico sem IDs internos;
- `manuscript_summary.md` — resumo técnico derivado exclusivamente do relatório;
- `tables/*.csv` — contagens, concordância, balanceamento por participante, bootstrap e exclusões;
- `figures/method_comparison.svg` — automático × referência;
- `figures/bland_altman.svg` — diferenças e limites de concordância;
- `traceability/requirements.json` e `traceability/matrix.csv`;
- `governance/` — plano estatístico, protocol lock e checklist da coleta formal;
- `checksums.sha256` — hash de todo conteúdo exportado.

## Estatística

A Fase 22 **não altera** as métricas da Fase 21. Ela empacota os resultados existentes:

- MAE e mediana do erro absoluto;
- RMSE;
- viés `PhysioAI − referência`;
- SD do erro e limites de concordância de Bland–Altman;
- ICC(A,1);
- CCC de Lin;
- Pearson `r` como medida complementar, não como prova de concordância;
- proporção dentro de ±2° e ±5°;
- resumo balanceado por participante;
- bootstrap por cluster de participante.

## Limites

O pacote não deve ser descrito como:

- validação clínica automática;
- diagnóstico;
- eficácia clínica;
- certificação regulatória;
- garantia de publicação;
- evidência de desempenho em populações não estudadas.

## Endpoints

- `GET /api/formal-runs/{run_id}/publication/readiness`
- `GET /api/formal-runs/{run_id}/publication.zip` — restrito a administrador

## Regressão

```bash
python scripts/verify_phase22.py
```
