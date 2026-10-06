# Pacote científico de sessão

O endpoint `GET /api/sessoes/{job_id}/export.zip` gera um artefato destinado à auditoria e reprodução de uma análise.

## Conteúdo

- `manifest.json`: sessão, build, protocolo, hashes de entrada, regras clínicas e contagens.
- `measurements.csv`: uma linha por frame processado, incluindo ângulo bruto, ângulo geométrico dos landmarks 3D/Kalman, ângulo filtrado, ângulo clínico, os três pontos clínicos 2D/3D quando disponíveis e o espaço de coordenadas usado.
- `validation_reviews.csv`: revisões manuais e referências físicas vinculadas à sessão.
- `README.txt`: interpretação resumida dos campos.
- `checksums.sha256`: integridade SHA-256 dos arquivos do pacote.

## Proveniência

A sessão registra:

- versão do PhysioAI;
- Git SHA e build time quando fornecidos pelo processo de release;
- versão do protocolo clínico;
- hash canônico da especificação clínica;
- SHA-256 e tamanho dos arquivos de entrada;
- configuração solicitada para a análise;
- hash do resultado consolidado;
- status do modelo legado de fluidez, que permanece desativado para saída clínica.

Nomes originais dos arquivos não são persistidos para reduzir risco de dados identificáveis em metadados.

## Revisão manual

Quando `job_id` e `frame` são revisados, o backend recupera o ângulo automático diretamente da telemetria persistida. O ângulo manual é calculado server-side pelos três pontos normalizados marcados na imagem. A leitura de goniômetro físico é armazenada separadamente.
