# Operação de uma validação formal

1. Crie o estudo com movimento, lado, alvo de participantes, repetições, instrumento de referência e setup de câmera.
2. Congele o estudo; após o freeze, o protocolo possui SHA-256 imutável.
3. Verifique `POST /api/estudos/{study_id}/formal-runs/readiness` com situação ética e datas de calibração.
4. Somente quando `ready=true`, um administrador inicia `POST /api/estudos/{study_id}/formal-runs`.
5. Colete os ensaios normalmente pelo endpoint de referência física do estudo.
6. Encerre a execução em `POST /api/formal-runs/{run_id}/close`.
7. Arquive `GET /api/formal-runs/{run_id}/export.zip` conforme a política de retenção aprovada.

O sistema preserva metadados de aprovação ética declarados pelo operador, mas não valida nem substitui CEP/Plataforma Brasil.
