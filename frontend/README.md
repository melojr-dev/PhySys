# PhysioAI Frontend v2.8 — Fase 10

SPA React/TypeScript da Clinical Motion Station.

- desktop e mobile;
- login obrigatório;
- REST autenticado por Bearer token;
- WebSocket autenticado por subprotocolo;
- telemetria biomecânica frame a frame;
- goniômetro manual e comparação com referência física;
- versão exibida deriva de `VITE_PHYSIOAI_VERSION` no build;
- o identificador digitado do participante é pseudonimizado no backend antes da persistência.

Build:

```bash
npm install --no-audit --no-fund
npm run build
```

A imagem Docker injeta versão, revisão Git e horário de build por `ARG`/`VITE_*`.
