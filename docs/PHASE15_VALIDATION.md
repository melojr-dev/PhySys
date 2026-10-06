# PhysioAI — Fase 15: preflight de captura

Versão: `2.13.0-alpha.1`  
Protocolo clínico: `1.4`  
Protocolo de preflight: `1.0`

## Objetivo

Detectar problemas de aquisição **antes** do processamento biomecânico completo, sem tratar qualidade de imagem como acurácia clínica. O preflight é um gate operacional e científico de coleta; a validade de cada frame continua sendo determinada pelo motor durante a análise.

## Fluxo

`upload único → amostragem temporal → relatório de captura → preflight_id → análise usando o mesmo arquivo → exclusão do temporário`

O vídeo não é enviado novamente entre preflight e análise. O SHA-256 calculado no upload é o mesmo registrado nos inputs da sessão.

## Critérios críticos

Uma captura pode ser bloqueada por:

- vídeo não decodificável;
- resolução abaixo do mínimo operacional;
- FPS abaixo do mínimo;
- baixa taxa de detecção corporal;
- visibilidade insuficiente dos landmarks clínicos;
- enquadramento insuficiente;
- ausência recorrente de referencial corporal bilateral;
- câmera incompatível com o plano do movimento;
- movimento predominantemente fora do plano esperado.

Iluminação, contraste e nitidez são avaliados, porém permanecem controles operacionais não críticos porque os thresholds ainda precisam de calibração empírica específica por câmera/ambiente.

## Rastreabilidade

`clinical_spec_snapshot()` inclui `capture_preflight` com versão e thresholds. Consequentemente, o SHA-256 da especificação clínica muda se os critérios de preflight forem alterados.

Cada sessão preserva:

- `preflight_id`;
- relatório completo e score;
- hash/tamanho do vídeo;
- versão do protocolo;
- eventual justificativa de override;
- usuário responsável pelo início da análise.

## Override

Capturas com falha crítica não podem iniciar uma análise clínica normal. Um usuário administrador pode prosseguir apenas como exceção experimental, fornecendo justificativa de no mínimo 12 caracteres. O evento entra na trilha de auditoria e na proveniência da sessão.

## Testes adicionados

- captura sintética adequada;
- captura sem pose;
- vídeo inválido;
- análise direta sem preflight retorna HTTP 428;
- incompatibilidade entre preflight e movimento/lado/paciente é rejeitada;
- descarte explícito do temporário;
- E2E com captura reprovada e override controlado;
- snapshot científico inclui o protocolo de preflight.

## Limitações

- o preflight atual avalia vídeos enviados, não streaming ao vivo da câmera;
- métricas fotométricas são heurísticas operacionais e não estimativas de erro angular;
- o MediaPipe ainda é responsável pela detecção inicial de pose;
- a etapa não substitui calibração clínica contra referência física.
