# Fase 10 — registro de validação

Versão: `2.8.0-alpha.1`

## Validado neste ambiente

- **63/63 testes de backend** aprovados.
- E2E in-process aprovado com FastAPI real, autenticação, upload MP4, job assíncrono, WebSocket, telemetria frame a frame, resultado e sessão persistida.
- Caminho sem pose detectada no primeiro frame aprovado (regressão do `measured_geo`).
- `ResourceWarning` do SQLite corrigido e E2E repetido com `ResourceWarning` tratado como erro.
- Python `compileall` aprovado.
- TypeScript validado sintaticamente com `tsc --noCheck`.
- Configuração do reverse proxy Nginx aprovada por `nginx -t`.
- Configuração do Nginx do frontend aprovada por `nginx -t`.
- `docker-compose.yml` e workflows GitHub Actions parseados como YAML válido.
- manifesto SHA-256 gerado e verificado.
- doctor: zero falhas; um aviso conhecido sobre ausência de `package-lock.json`.

## Preparado, mas não executável neste ambiente

### Docker E2E real

O host de execução desta revisão não possui Docker. A Fase 10 inclui:

- `scripts/e2e_stack.sh`;
- `scripts/e2e_stack.ps1`;
- `backend/scripts/e2e_smoke.py`;
- `backend/scripts/generate_smoke_video.py`;
- job `e2e-stack` em CI.

O E2E Docker é isolado pelo projeto Compose `physioai_e2e`, usa porta `18080` por padrão e remove apenas seu próprio volume sintético ao finalizar.

### Build React/Vite completo

A instalação NPM excedeu o limite do ambiente, portanto não foi declarado como aprovado localmente. O código passou na verificação sintática TypeScript. CI e o Dockerfile executam a instalação e `npm run build`.

A ausência de `package-lock.json` permanece como limitação de reprodutibilidade das dependências transitivas do frontend; dependências diretas estão fixadas por versão.
