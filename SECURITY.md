# Security Policy

PhysioAI ainda é software de pesquisa pré-clínica. Não publique credenciais, vídeos, identificadores ou bancos SQLite em issues/commits.

## Relato de vulnerabilidade
Use canal privado institucional do projeto/orientação. Inclua versão (`GET /version`), revisão Git e passos de reprodução, sem dados pessoais.

## Segredos
`.env` nunca deve ser versionado. `PHYSIOAI_AUTH_SECRET`, `PHYSIOAI_ADMIN_PASSWORD` e `PHYSIOAI_PSEUDONYM_SECRET` devem ser únicos por ambiente.


## Pacotes científicos
Os ZIPs de sessão contêm pseudônimos, telemetria e referências clínicas. Eles devem ser tratados como dados de pesquisa controlados e não publicados em repositórios públicos.

## WebSocket
O frontend envia o token de sessão em `Sec-WebSocket-Protocol`, evitando colocar credenciais na query string e, portanto, no request URI normalmente registrado pelo proxy. O backend mantém query string somente como compatibilidade temporária com clientes antigos.

## Vídeos de preflight
O vídeo submetido ao preflight é temporário e reutilizado pela análise para evitar um segundo upload. Preflights não consumidos expiram automaticamente (`PHYSIOAI_PREFLIGHT_TTL_SECONDS`) e o arquivo é removido. Não copie diretórios temporários para backups ou bundles científicos. Overrides de captura reprovada são auditados e ficam registrados na proveniência da sessão.

## Dados inerciais experimentais (Fase 17)

Telemetria IMU vinculada a uma sessão é tratada como dado de pesquisa da mesma sessão. O backend persiste o payload bruto comprimido em SQLite apenas para reprodutibilidade, registra SHA-256 e aplica o mesmo controle de acesso autenticado. O bundle científico pode incluir esses sinais; portanto, sua exportação deve seguir a política de armazenamento/compartilhamento aprovada para o estudo. A exclusão de paciente remove análises IMU vinculadas aos respectivos `job_id`.

## Isolamento longitudinal (Fase 19)

Análises longitudinais são derivadas de sessões pseudonimizadas. Usuários comuns só podem agregar sessões do mesmo paciente que pertençam ao próprio avaliador; administradores podem executar revisão global. A listagem geral de sessões também é filtrada por avaliador fora do papel `admin`.

## Pacote externo de revisão (Fase 22)

Use `publication.zip` para revisão externa em vez de compartilhar banco/runtime ou bundles internos. O export re-aliasa participantes e sessões e omite IDs internos, avaliadores, serial do instrumento e notas livres. Isso reduz exposição, mas não substitui a avaliação institucional/CEP/LGPD sobre compartilhamento do conjunto de dados.
