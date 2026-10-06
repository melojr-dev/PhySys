# Operação — PhysioAI v2.8

## Persistência

O banco clínico fica em `PHYSIOAI_DATA_DIR` (Docker: `/app/data`) e, por padrão, no arquivo `physioai_clinical.sqlite3`. O Compose usa o volume nomeado `physioai_data`.

Nunca remova o volume de produção sem backup validado.

## Backup

```bash
docker compose exec backend python scripts/backup_db.py --output-dir /app/data/backups
```

O script usa `sqlite3.Connection.backup()`, executa `PRAGMA quick_check` no arquivo de destino e produz um manifesto JSON com SHA-256.

## Readiness

`GET /ready` valida:

1. analisador/semaphore inicializados;
2. diretório de runtime gravável;
3. integridade rápida do SQLite;
4. espaço livre mínimo configurado.

## Smoke E2E

```bash
./scripts/e2e_stack.sh
```

ou:

```powershell
./scripts/e2e_stack.ps1
```

O script sempre usa `docker compose down -v` ao final porque opera exclusivamente sobre dados sintéticos de teste.

## Exposição de rede

Por padrão, o Nginx publica apenas em `127.0.0.1:80`. Para LAN confiável, configure `PHYSIOAI_BIND_ADDRESS=0.0.0.0`. O Cloudflare Tunnel permanece no profile `public` e não é iniciado por padrão.
