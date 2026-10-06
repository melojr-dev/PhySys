# Fase 11 — Validação clínica experimental assistida

Versão: `2.9.0-alpha.1`

Esta fase **não afirma validação clínica concluída**. Ela prepara o PhysioAI para executar um estudo de comparação de métodos após as aprovações éticas e institucionais aplicáveis.

## Unidade de coleta

A hierarquia adotada é:

`estudo → participante pseudonimizado → sessão → ensaio → frame`

Um ensaio associa exatamente um frame persistido pelo backend a uma leitura física de referência. O ângulo automático, timestamp, qualidade de rastreamento e identidade da sessão são recuperados server-side; o navegador não fornece essas métricas.

## Protocolo congelado

Antes de registrar ensaios, o protocolo precisa ser congelado. O congelamento gera `protocol_sha256` sobre:

- movimento e lado;
- número-alvo de participantes;
- repetições previstas por participante;
- limiar mínimo de visibilidade dos landmarks;
- exigência de landmark observado e medição válida;
- método de referência;
- modelo e resolução do instrumento físico;
- configuração da câmera e notas do protocolo.

Após congelado, o sistema não oferece edição do protocolo. Alterações metodológicas exigem novo estudo/código de protocolo.

## Separação entre revisão e padrão-ouro

A revisão manual da imagem e a referência física são fluxos separados:

- `validation_reviews`: ajuste manual de três pontos sobre a imagem para verificar posicionamento do algoritmo;
- `validation_trials`: comparação PhysioAI × referência física dentro de um protocolo congelado.

O endpoint de revisão ainda aceita `reference_angle_deg` por compatibilidade com clientes antigos, mas o valor é ignorado na Fase 11.

## Controle de qualidade do ensaio

O ensaio é armazenado mesmo quando falha nos critérios, porém fica marcado como `included=false` com motivo explícito. Critérios atuais:

- `source == observed`, quando exigido pelo protocolo;
- `measurement_valid == true`, quando exigido;
- `landmark_confidence_min >= min_landmark_confidence`;
- exclusão manual documentada, quando aplicável.

Isso evita remover silenciosamente observações problemáticas.

## Relatório estatístico

O relatório do estudo fornece:

- MAE, mediana do erro absoluto, RMSE e bias em graus;
- limites de concordância de Bland–Altman;
- ICC(A,1), CCC de Lin e correlação de Pearson;
- percentual dos ensaios dentro de ±2° e ±5°;
- métricas estratificadas pela faixa do ângulo físico e por lado;
- contagens de exclusão e motivos;
- resumo balanceado por participante;
- IC95% por bootstrap em cluster de participante.

O bootstrap reamostra **participantes completos**, mantendo juntos todos os ensaios de cada participante. Isso não substitui o plano estatístico do protocolo aprovado; serve para evitar a interpretação ingênua de frames repetidos como indivíduos independentes.

## Endpoints principais

- `POST /api/estudos`
- `POST /api/estudos/{id}/freeze`
- `POST /api/estudos/{id}/close`
- `POST /api/estudos/{id}/ensaios`
- `GET /api/estudos/{id}/relatorio`
- `GET /api/estudos/{id}/export.csv`
- `GET /api/estudos/{id}/export.zip`

## Exportação científica

O ZIP de estudo contém:

- `protocol.json`;
- `trials.csv`;
- `report.json`;
- `README.txt`;
- `checksums.sha256`.

O bundle por sessão também inclui `validation_trials.csv` quando houver ensaios associados.

## Condição para coleta com participantes

A existência desta infraestrutura não autoriza coleta clínica. O estudo com participantes deve ser executado apenas após as aprovações e documentos exigidos para o protocolo correspondente.
