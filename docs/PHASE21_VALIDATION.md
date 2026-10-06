# Fase 21 — execução formal de validação e freeze operacional

Versão: `2.19.0-rc.2`  
Protocolo clínico: `1.4` (inalterado)

## Objetivo

Transformar o candidato de software da Fase 20 em um pacote operacional para coleta formal de comparação de métodos, sem introduzir novas capacidades clínicas.

## Conceito principal: formal validation run

O **estudo** define o protocolo e seus critérios de inclusão/exclusão. Uma **execução formal** representa uma janela concreta de coleta e congela, no início:

- versão do software;
- versão e SHA-256 do protocolo clínico;
- SHA-256 do protocolo do estudo;
- plano estatístico versionado;
- situação ética declarada e identificador do protocolo quando aplicável;
- datas de calibração/validade do instrumento de referência;
- proveniência do modelo legado (que permanece desabilitado para saída clínica);
- checklist operacional da coleta.

A execução só é criada quando todos os gates de prontidão passam. O software não verifica a legitimidade jurídica de uma aprovação ética: ele apenas exige e preserva os metadados declarados pelo pesquisador.

## Gate de prontidão

A coleta formal é bloqueada quando houver, entre outros:

- estudo não congelado;
- versão de protocolo diferente do protocolo clínico corrente;
- ausência de alvo amostral ou repetições;
- ausência de modelo/resolução do instrumento de referência;
- setup de câmera não definido;
- ética pendente para coleta com participantes humanos;
- calibração vencida/ausente;
- plano estatístico ou protocol lock ausente/incompatível.

## Plano estatístico

`release/statistical_analysis_plan.json` congela o plano de análise da comparação de métodos. O endpoint primário é MAE em graus em ensaios incluídos pelo protocolo. As análises secundárias incluem bias, RMSE, ICC(A,1), CCC de Lin, Bland–Altman, resumo balanceado por participante e bootstrap por cluster de participante.

Frames/repetições do mesmo participante não são tratados como participantes independentes e não há imputação automática de dados ausentes.

## Exportação

Cada execução formal pode ser exportada como ZIP autocontido com:

- snapshot da execução e do estudo;
- ensaios coletados somente dentro daquela execução;
- relatório estatístico da execução;
- plano estatístico;
- protocol lock;
- checklist formal;
- checksums SHA-256.

## Limite de interpretação

Esta fase melhora reprodutibilidade, auditabilidade e disciplina de coleta. **Ela não demonstra validade clínica, não cria diagnóstico e não substitui aprovação ética, desenho amostral ou análise científica do estudo real.**
