# Pacote de revisão externa

## Uso recomendado

1. Finalize e feche uma execução formal de validação.
2. Consulte `/api/formal-runs/{run_id}/publication/readiness`.
3. Revise todos os warnings, principalmente amostra incompleta e exclusões.
4. Um administrador exporta `/api/formal-runs/{run_id}/publication.zip`.
5. Verifique `checksums.sha256` antes de compartilhar.
6. Compartilhe o ZIP externo, **não** o banco SQLite nem o bundle interno de sessão.

## O que o revisor recebe

O revisor consegue reproduzir as tabelas a partir de `trials_anonymized.csv`, conferir as métricas em `report.json`, inspecionar o plano estatístico/protocolo e rastrear requisitos até testes. Não recebe identificadores internos ou notas livres.

## O que este pacote não resolve

Desidentificação técnica não substitui decisão institucional sobre compartilhamento de dados, CEP, LGPD, acordo de colaboração ou política de publicação. O operador continua responsável por verificar se o compartilhamento é permitido no protocolo aplicável.
