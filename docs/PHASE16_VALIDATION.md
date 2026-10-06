# Fase 16 — marcha temporal experimental

Versão: `2.14.0-alpha.1`

A Fase 16 introduz um módulo **experimental** de análise temporal da marcha sem promover suas saídas a métricas clínicas validadas.

## Métricas

- candidatos de contato de calcanhar esquerdo/direito;
- cadência estimada (passos/min);
- tempo médio de passo e coeficiente de variação;
- tempos de passo por lado e razão de simetria;
- tempos de ciclo/stride por lado e razão de simetria;
- razão de frames válidos e score de qualidade.

## Restrição espacial

No espaço `relative_body_3d`, comprimento de passada em metros fica **desabilitado**. O campo espacial só é calculado quando o chamador declara `coordinate_space=metric_global`, pois `pose_world_landmarks` monocular não constitui referencial global confiável de deslocamento entre frames.

## API

`POST /api/experimental/gait/analyze`

Requer autenticação e recebe `fps`, `coordinate_space` e `skeleton_frames` no formato `(N,33,3)`. O endpoint é separado da via clínica de ROM e gera auditoria operacional.

## Estado científico

Os eventos são estimados por extremos temporais da trajetória dos calcanhares. Devem ser tratados como candidatos experimentais até validação contra referência instrumentada/observacional específica de marcha.
