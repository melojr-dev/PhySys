# Fase 24 — Markerless Experience

Versão: `2.22.0-rc.5`

## Escopo

Esta fase altera a experiência de produto e a visualização, sem modificar as fórmulas clínicas do protocolo `1.4`.

## Entregas

- Home orientada ao conceito markerless.
- Modos Apresentação e Pesquisa.
- Overlay anatômico completo do membro selecionado.
- Leitura angular associada ao frame no overlay.
- Visualizador 3D interativo sincronizado com a análise.
- Representação volumétrica segmentar estimada (visual, não anatômica).
- Curva angular temporal interativa e sincronizada ao vídeo.
- Solicitação de `skeleton_3d` no pipeline principal.
- Dependência WebSocket explícita e modelo MediaPipe Heavy pré-carregado no build.

## Restrições científicas

A representação volumétrica/3D é estimada a partir dos landmarks e não deve ser descrita como escaneamento anatômico, reconstrução volumétrica de precisão ou mensuração clínica validada em profundidade.

O modo markerless permanece a via principal. IMU, LiDAR/depth e demais sensores são extensões experimentais opcionais.

## Critérios de aceite

1. `npm run build` deve concluir sem erros TypeScript/Vite.
2. Backend deve iniciar com suporte a WebSocket sem `Unsupported upgrade request`.
3. MediaPipe Heavy deve existir antes da troca para usuário não-root no container.
4. Nova análise deve retornar `skeleton_3d` e permitir alternância Vídeo / 2D+3D / 3D.
5. Curva temporal deve mover o mesmo `frameIndex` utilizado pelo vídeo e pelo 3D.
6. Modo Apresentação não deve exibir painéis de validação/engenharia; modo Pesquisa deve preservá-los.
7. Baseline clínica e protocolo `1.4` devem permanecer inalterados.


## Patch 2.22.1-rc.5 — live video/overlay synchronization

The live markerless preview now uses a latest-wins seek queue. Anatomical landmarks advance only after the browser fires `seeked` for the corresponding analyzed timestamp. This prevents frame-N landmarks from being drawn over an older decoded video frame while backend processing is slower than source playback. Clinical computation and protocol 1.4 are unchanged.
