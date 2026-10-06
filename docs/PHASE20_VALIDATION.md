# Fase 20 — candidato de software para validação formal

Versão: `2.18.0-rc.1`  
Protocolo clínico: `1.4` (inalterado)

## Objetivo

Congelar o escopo funcional e tornar mudanças biomecânicas detectáveis antes de uma coleta formal. Esta fase não adiciona novo movimento clínico nem declara validação clínica.

## Controles adicionados

- golden baseline versionado para todos os seis movimentos pré-clínicos;
- lock SHA-256 do contrato do protocolo 1.4;
- verificação independente `scripts/verify_clinical_baseline.py`;
- testes de regressão do baseline dentro da suíte do backend;
- política explícita de governança de mudanças do protocolo;
- release classificada como `rc` de software, não como dispositivo validado.

## Interpretação correta

Passar o baseline significa que a implementação continua produzindo as mesmas saídas nos casos canônicos. Não significa que essas saídas sejam clinicamente corretas em pacientes reais. Validade, concordância e erro continuam dependendo do protocolo experimental PhysioAI × referência física.
