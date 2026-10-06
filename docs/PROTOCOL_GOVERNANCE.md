# Governança do protocolo biomecânico

O PhysioAI separa **versão de software** de **versão do protocolo clínico**. A versão de software pode mudar por correções de infraestrutura, interface ou observabilidade sem alterar a definição biomecânica da medição. Já qualquer alteração semântica em landmarks, plano anatômico, conversão angular, gate direcional, referência ROM ou critérios críticos de preflight deve ser tratada como mudança de protocolo.

## Lock do protocolo

`release/protocol_lock.json` congela o contrato utilizado pela release. O hash cobre:

- especificações públicas dos movimentos;
- landmarks proximal/vértice/distal;
- plano de medição e vista recomendada;
- conversor angular;
- limites de qualidade geométrica;
- versão e thresholds do preflight.

`python scripts/verify_clinical_baseline.py` recalcula esse contrato e falha se houver divergência.

## Baseline dourado

`release/clinical_baseline.json` contém casos geométricos canônicos com saída esperada. Eles verificam estabilidade computacional da cadeia `3 pontos -> ângulo interno -> conversão clínica`.

Esse baseline é **teste de regressão de software**. Ele não mede erro contra goniômetro, não demonstra validade clínica e não substitui os estudos experimentais definidos no projeto.

## Política de mudança

1. Correção sem efeito no contrato biomecânico: manter a versão do protocolo e passar todos os baselines.
2. Mudança intencional que altera o resultado clínico: elevar a versão do protocolo, documentar a justificativa, regenerar deliberadamente lock/baseline e preservar estudos antigos com a versão anterior.
3. Nunca regenerar lock/baseline apenas para “fazer os testes passarem”. A mudança deve estar documentada no changelog e na documentação de validação.
