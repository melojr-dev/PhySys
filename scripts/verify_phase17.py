#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, command: list[str], cwd: Path = ROOT) -> None:
    print(f"\n== {label} ==")
    subprocess.run(command, cwd=cwd, check=True)


def clean_generated() -> None:
    for path in ROOT.rglob("*.pyc"):
        path.unlink(missing_ok=True)
    for path in sorted(ROOT.rglob("__pycache__"), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass
    (ROOT / "frontend" / "tsconfig.tsbuildinfo").unlink(missing_ok=True)


def assert_phase17_contract() -> None:
    server = (ROOT / "backend/server.py").read_text(encoding="utf-8")
    capture = (ROOT / "backend/src/core/capture.py").read_text(encoding="utf-8")
    provenance = (ROOT / "backend/src/research/provenance.py").read_text(encoding="utf-8")
    nginx = (ROOT / "nginx/default.conf").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    frontend = (ROOT / "frontend/src/main.tsx").read_text(encoding="utf-8")

    checks = {
        "capture module": (ROOT / "backend/src/core/capture.py").is_file(),
        "preflight route": '@app.post("/api/capture/preflight")' in server,
        "preflight delete route": '@app.delete("/api/capture/preflight/{preflight_id}")' in server,
        "preflight required": "PHYSIOAI_REQUIRE_PREFLIGHT" in server,
        "preflight protocol 1.0": 'PREFLIGHT_SPEC_VERSION = "1.0"' in capture,
        "clinical protocol 1.4": 'PROTOCOL_VERSION = "1.4"' in provenance,
        "snapshot carries preflight": '"capture_preflight"' in provenance,
        "nginx streams preflight": "location = /api/capture/preflight" in nginx and "proxy_request_buffering off;" in nginx,
        "compose preflight settings": all(k in compose for k in (
            "PHYSIOAI_PREFLIGHT_TTL_SECONDS",
            "PHYSIOAI_PREFLIGHT_MAX_SAMPLES",
            "PHYSIOAI_REQUIRE_PREFLIGHT",
        )),
        "env preflight settings": all(k in env for k in (
            "PHYSIOAI_PREFLIGHT_TTL_SECONDS=",
            "PHYSIOAI_PREFLIGHT_MAX_SAMPLES=",
            "PHYSIOAI_REQUIRE_PREFLIGHT=1",
        )),
        "frontend invokes preflight": "capturePreflight" in frontend,
        "phase15 continuity doc": (ROOT / "docs/PHASE15_VALIDATION.md").is_file(),
    }
    checks.update({
        "gait module": (ROOT / "backend/src/core/gait.py").is_file(),
        "gait endpoint": '@app.post("/api/experimental/gait/analyze")' in server,
        "phase16 continuity doc": (ROOT / "docs/PHASE16_VALIDATION.md").is_file(),
        "imu module": (ROOT / "backend/src/core/imu.py").is_file(),
        "imu store": (ROOT / "backend/src/data/imu_store.py").is_file(),
        "imu endpoint": '@app.post("/api/experimental/imu/analyze"' in server,
        "imu export endpoint": '/api/experimental/imu/{analysis_id}/export.json' in server,
        "sensor fusion frontend": (ROOT / "frontend/src/components/SensorFusionPanel.tsx").is_file(),
        "phase17 validation doc": (ROOT / "docs/PHASE17_VALIDATION.md").is_file(),
        "version 2.15": (ROOT / "VERSION").read_text(encoding="utf-8").strip() == "2.15.0-alpha.1",
    })
    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        raise SystemExit("Contrato da Fase 17 incompleto: " + ", ".join(failed))


def main() -> int:
    assert_phase17_contract()
    run("Python compileall", [sys.executable, "-m", "compileall", "-q", "backend", "scripts"])
    clean_generated()
    run(
        "Backend regression",
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT / "backend",
    )
    clean_generated()

    tsc = shutil.which("tsc")
    if tsc:
        run("Frontend TypeScript syntax", [tsc, "--noEmit", "--noCheck"], cwd=ROOT / "frontend")
    else:
        print("\nSKIP TypeScript syntax: tsc não está disponível neste ambiente.")

    node_modules = ROOT / "frontend" / "node_modules"
    npm = shutil.which("npm")
    if npm and node_modules.is_dir():
        run("Frontend build", [npm, "run", "build"], cwd=ROOT / "frontend")
    else:
        print("\nSKIP frontend build completo: node_modules não está disponível neste ambiente.")
        print("CI/Docker executam npm install + npm run build.")

    clean_generated()
    run("Release manifest", [sys.executable, "scripts/release_manifest.py"])
    run("Release integrity", [sys.executable, "scripts/verify_release.py"])
    run("Static integration doctor", [sys.executable, "scripts/doctor.py"])
    clean_generated()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
