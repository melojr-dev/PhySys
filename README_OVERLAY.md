# PhysioAI — marcador anatômico do membro

Alteração adicionada sobre o pacote enviado pelo usuário.

## O que foi implementado

- Cadeia anatômica 2D acompanhando o membro selecionado frame a frame.
- Ombro/cotovelo: ombro, cotovelo, punho, polegar, indicador e mindinho.
- Quadril/joelho: quadril, joelho, tornozelo, calcanhar e antepé.
- Dorsiflexão: joelho, tornozelo, calcanhar e antepé.
- Articulação principal destacada com halo.
- Landmarks de baixa confiança aparecem com estilo de alerta.
- Botão `Marcador ON/OFF` na barra da medição.
- Sincronização do overlay com o tempo do vídeo após a análise.
- Compatibilidade com sessões sem a nova telemetria: fallback para os três pontos goniométricos.

## Arquivos novos

- `frontend/src/components/AnatomicalOverlay.tsx`
- `frontend/src/lib/anatomicalChains.ts`

## Arquivos alterados

- `backend/src/core/engine.py`
- `backend/server.py`
- `frontend/src/types.ts`
- `frontend/src/main.tsx`
- `frontend/src/styles.css`

Os 33 landmarks 2D são enviados apenas na telemetria visual. O campo foi deliberadamente excluído de `FrameMeasurement.to_dict()` para não modificar o contrato persistido/hashes científicos das medições.

## Aplicação no projeto

Extraia este ZIP sobre a raiz `physioai_v2_phase23`, preservando a estrutura e sobrescrevendo os arquivos.
Depois reconstrua backend e frontend:

```bash
docker compose up --build -d
```

Faça uma nova análise: resultados antigos não contêm `poseLandmarks2d` e usarão apenas o fallback de 3 pontos.
