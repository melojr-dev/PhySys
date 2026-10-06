#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v docker >/dev/null || { echo "docker não encontrado" >&2; exit 2; }
docker compose version >/dev/null

export PHYSIOAI_ADMIN_USER="${PHYSIOAI_ADMIN_USER:-admin}"
export PHYSIOAI_ADMIN_PASSWORD="${PHYSIOAI_ADMIN_PASSWORD:-E2E-only-PhysioAI-2026!}"
export PHYSIOAI_AUTH_SECRET="${PHYSIOAI_AUTH_SECRET:-E2E-auth-secret-PhysioAI-0123456789abcdef}"
export PHYSIOAI_PSEUDONYM_SECRET="${PHYSIOAI_PSEUDONYM_SECRET:-E2E-pseudo-secret-PhysioAI-fedcba9876543210}"
export PHYSIOAI_BIND_ADDRESS="127.0.0.1"
export PHYSIOAI_HTTP_PORT="${PHYSIOAI_E2E_PORT:-18080}"
export COMPOSE_PROJECT_NAME="${PHYSIOAI_E2E_PROJECT:-physioai_e2e}"
export PHYSIOAI_VERSION="${PHYSIOAI_VERSION:-$(cat VERSION)}"

cleanup() {
  code=$?
  if [[ $code -ne 0 ]]; then
    echo "\n--- docker compose ps ---" >&2
    docker compose ps >&2 || true
    echo "\n--- backend logs ---" >&2
    docker compose logs --tail=160 backend >&2 || true
    echo "\n--- nginx logs ---" >&2
    docker compose logs --tail=80 nginx >&2 || true
  fi
  docker compose down -v --remove-orphans >/dev/null 2>&1 || true
  exit $code
}
trap cleanup EXIT INT TERM

docker compose up --build -d

echo "Aguardando readiness..."
for _ in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:${PHYSIOAI_HTTP_PORT}/ready >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:${PHYSIOAI_HTTP_PORT}/ready >/dev/null

docker compose exec -T backend python scripts/generate_smoke_video.py /tmp/physioai_smoke.mp4 --frames 24 --fps 12
docker compose exec -T backend python scripts/e2e_smoke.py \
  --base-url http://nginx \
  --user "$PHYSIOAI_ADMIN_USER" \
  --password "$PHYSIOAI_ADMIN_PASSWORD" \
  --video /tmp/physioai_smoke.mp4 \
  --timeout 240

echo "PhysioAI E2E stack: PASS"
