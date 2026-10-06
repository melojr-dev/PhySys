$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw "docker não encontrado" }
docker compose version | Out-Null

if (-not $env:PHYSIOAI_ADMIN_USER) { $env:PHYSIOAI_ADMIN_USER = "admin" }
if (-not $env:PHYSIOAI_ADMIN_PASSWORD) { $env:PHYSIOAI_ADMIN_PASSWORD = "E2E-only-PhysioAI-2026!" }
if (-not $env:PHYSIOAI_AUTH_SECRET) { $env:PHYSIOAI_AUTH_SECRET = "E2E-auth-secret-PhysioAI-0123456789abcdef" }
if (-not $env:PHYSIOAI_PSEUDONYM_SECRET) { $env:PHYSIOAI_PSEUDONYM_SECRET = "E2E-pseudo-secret-PhysioAI-fedcba9876543210" }
$env:PHYSIOAI_BIND_ADDRESS = "127.0.0.1"
if (-not $env:PHYSIOAI_E2E_PORT) { $env:PHYSIOAI_E2E_PORT = "18080" }
$env:PHYSIOAI_HTTP_PORT = $env:PHYSIOAI_E2E_PORT
if (-not $env:PHYSIOAI_E2E_PROJECT) { $env:PHYSIOAI_E2E_PROJECT = "physioai_e2e" }
$env:COMPOSE_PROJECT_NAME = $env:PHYSIOAI_E2E_PROJECT
if (-not $env:PHYSIOAI_VERSION) { $env:PHYSIOAI_VERSION = (Get-Content VERSION -Raw).Trim() }

try {
    docker compose up --build -d
    Write-Host "Aguardando readiness..."
    $ready = $false
    for ($i = 0; $i -lt 90; $i++) {
        try {
            $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$($env:PHYSIOAI_HTTP_PORT)/ready" -TimeoutSec 3
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch {}
        Start-Sleep -Seconds 2
    }
    if (-not $ready) { throw "Stack não ficou ready dentro da janela de teste" }

    docker compose exec -T backend python scripts/generate_smoke_video.py /tmp/physioai_smoke.mp4 --frames 24 --fps 12
    docker compose exec -T backend python scripts/e2e_smoke.py --base-url http://nginx --user $env:PHYSIOAI_ADMIN_USER --password $env:PHYSIOAI_ADMIN_PASSWORD --video /tmp/physioai_smoke.mp4 --timeout 240
    if ($LASTEXITCODE -ne 0) { throw "E2E smoke falhou" }
    Write-Host "PhysioAI E2E stack: PASS"
}
catch {
    docker compose ps
    docker compose logs --tail=160 backend
    docker compose logs --tail=80 nginx
    throw
}
finally {
    docker compose down -v --remove-orphans | Out-Null
}
