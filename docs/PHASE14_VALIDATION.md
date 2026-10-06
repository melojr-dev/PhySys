# PhysioAI — Fase 14: ombro multiplanar e gate direcional

Versão: `2.12.0-alpha.1`  
Protocolo clínico: `1.3`

## Objetivo

A Fase 14 amplia a via pré-clínica para flexão e abdução do ombro utilizando o referencial corporal e a projeção por plano introduzidos na Fase 13. O objetivo é impedir que movimentos geometricamente semelhantes, porém executados no sentido ou plano incorreto, sejam aceitos como a grandeza clínica solicitada.

## Flexão de ombro

- landmarks: quadril → ombro → cotovelo;
- plano: sagital;
- vista recomendada: lateral;
- referência operacional: 180°;
- conversão: o ângulo interno projetado é usado diretamente;
- regra direcional: `anterior_positive`.

Na posição neutra, braço e tronco inferior apontam aproximadamente para baixo e o ângulo tende a 0°. A elevação anterior aumenta o ângulo em direção a 180°. Extensão posterior produz componente anterior negativa e é rejeitada quando ultrapassa a tolerância direcional.

## Abdução de ombro

- landmarks: quadril → ombro → cotovelo;
- plano: frontal;
- vista recomendada: anterior ou posterior;
- referência operacional: 180°;
- conversão: ângulo projetado direto;
- regra direcional: `lateral_outward`, dependente do lado corporal.

Adução cruzando a linha média produz alinhamento lateral oposto e deixa o frame inválido para a métrica de abdução.

## Gate direcional

`directionAlignment` varia de -1 a +1 após projeção do braço no plano anatômico:

- positivo: segmento aponta no sentido esperado;
- aproximadamente zero: posição neutra ou próxima do eixo vertical;
- negativo: sentido oposto.

O limiar inicial é `-0.10`, deliberadamente tolerante ao ruído perto da posição neutra. Esse valor é operacional e deve ser avaliado empiricamente antes de qualquer interpretação clínica.

Flags adicionais:

- `movement_direction_unavailable`;
- `movement_direction_mismatch`.

## Cadeia auditável

Permanece:

`raw 3D → landmarks/Kalman → projeção anatômica → EMA → conversão clínica`

A Fase 14 acrescenta ao gate de validade o sentido esperado do segmento distal quando o movimento exigir essa distinção.

## Persistência

O schema frame-a-frame passa para v3 e adiciona:

- `direction_rule`;
- `direction_alignment`.

Esses campos também são preservados nos ensaios de validação e nos CSVs dos bundles científicos. Bancos anteriores são migrados automaticamente.

## Por que o punho não foi habilitado

O MediaPipe Pose fornece punho, indicador e mindinho, mas não uma orientação palmar/metacarpal suficientemente robusta para separar com segurança:

- flexão × extensão do punho;
- desvio radial × ulnar;
- efeitos de pronação/supinação do antebraço.

Habilitar esses movimentos apenas pelo ângulo cotovelo–punho–centro da mão recriaria a ambiguidade removida nas fases anteriores. Por isso, o punho permanece experimental/bloqueado até existir um referencial local da mão adequado.

## Limitações

- a referência do tronco é estimada por landmarks, não por marcadores rígidos;
- compensações escapulares e torácicas podem afetar a medida;
- a Fase 14 não quantifica rotação interna/externa do ombro;
- `directionAlignment` é controle geométrico, não probabilidade de acurácia;
- nenhum movimento é declarado clinicamente validado.

## Verificação

```bash
python scripts/verify_phase14.py
```

A verificação executa compilação Python, regressão de backend, manifesto SHA-256, doctor de integração e validação TypeScript quando disponível.
