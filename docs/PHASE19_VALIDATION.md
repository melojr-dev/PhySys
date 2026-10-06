# Fase 19 — validação do seguimento longitudinal

Versão: `2.17.0-alpha.1`

## Escopo

A Fase 19 adiciona análise **descritiva e experimental** entre sessões do mesmo paciente. Ela não estabelece eficácia terapêutica nem validade longitudinal clínica.

### Regra de comparabilidade

A sessão selecionada é a âncora. O resumo principal inclui somente sessões concluídas que coincidam em:

- paciente pseudonimizado;
- movimento;
- lado;
- `protocol_version`;
- `clinical_spec_sha256`.

Sessões que falham em qualquer critério são listadas em `excluded_sessions` com a razão correspondente. Mudanças de `software_version` com protocolo/especificação estáveis são mantidas, mas produzem a flag `mixed_software_versions_same_protocol`.

## Métricas

Para cada grandeza disponível, o backend calcula:

- primeira e última observação;
- delta absoluto e percentual;
- média e desvio-padrão amostral;
- coeficiente de variação descritivo;
- mediana da mudança absoluta entre sessões consecutivas;
- inclinação OLS por 30 dias.

São suportadas ADM, proporção de frames válidos, repetições e descritores motores previamente extraídos (quando existentes).

## Limites de interpretação

O contrato retorna `experimental=true` e `diagnostic=false`. Não são inferências suportadas:

- melhora clínica;
- progressão de doença;
- resposta a tratamento;
- diferença mínima clinicamente importante (MCID);
- diagnóstico.

O CV e a inclinação temporal são descritivos. Não substituem ICC/teste-reteste desenhado prospectivamente, nem um modelo longitudinal com controle de confundidores.

## Segurança e proveniência

- endpoint autenticado e isolado por avaliador para usuários não administradores;
- persistência SQLite com `source_sha256`;
- source payload registra hashes de resultados, protocolo, especificação e descritores motores utilizados;
- exportação JSON autenticada;
- bundle científico da sessão inclui `longitudinal/index.json`;
- exclusão LGPD remove os registros longitudinais ligados ao paciente.

## Verificação

```bash
python scripts/verify_phase19.py
```
