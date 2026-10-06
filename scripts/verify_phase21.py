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


def assert_contract() -> None:
    server = (ROOT / "backend/server.py").read_text(encoding="utf-8")
    checks = {
        "version": (ROOT / "VERSION").read_text(encoding="utf-8").strip() == "2.19.0-rc.2",
        "phase21 doc": (ROOT / "docs/PHASE21_VALIDATION.md").is_file(),
        "formal ops doc": (ROOT / "docs/FORMAL_VALIDATION.md").is_file(),
        "formal store": (ROOT / "backend/src/validation/formal.py").is_file(),
        "statistical plan": (ROOT / "release/statistical_analysis_plan.json").is_file(),
        "formal checklist": (ROOT / "release/formal_validation_checklist.json").is_file(),
        "runtime statistical plan": (ROOT / "backend/src/validation/resources/statistical_analysis_plan.json").is_file(),
        "runtime checklist": (ROOT / "backend/src/validation/resources/formal_validation_checklist.json").is_file(),
        "runtime protocol lock": (ROOT / "backend/src/validation/resources/protocol_lock.json").is_file(),
        "readiness endpoint": '/formal-runs/readiness' in server,
        "start endpoint": '@app.post("/api/estudos/{study_id}/formal-runs", status_code=201)' in server,
        "close endpoint": '@app.post("/api/formal-runs/{run_id}/close")' in server,
        "export endpoint": '/api/formal-runs/{run_id}/export.zip' in server,
        "protocol unchanged": 'PROTOCOL_VERSION = "1.4"' in (ROOT / "backend/src/research/provenance.py").read_text(encoding="utf-8"),
        "clinical baseline retained": (ROOT / "release/clinical_baseline.json").is_file(),
        "protocol lock retained": (ROOT / "release/protocol_lock.json").is_file(),
    }
    runtime = ROOT / "backend/src/validation/resources"
    checks["statistical plan runtime parity"] = (ROOT / "release/statistical_analysis_plan.json").read_bytes() == (runtime / "statistical_analysis_plan.json").read_bytes()
    checks["checklist runtime parity"] = (ROOT / "release/formal_validation_checklist.json").read_bytes() == (runtime / "formal_validation_checklist.json").read_bytes()
    checks["protocol lock runtime parity"] = (ROOT / "release/protocol_lock.json").read_bytes() == (runtime / "protocol_lock.json").read_bytes()
    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        raise SystemExit("Contrato da Fase 21 incompleto: " + ", ".join(failed))


def main() -> int:
    assert_contract()
    run("Clinical golden baseline", [sys.executable, "scripts/verify_clinical_baseline.py"])
    run("Python compileall", [sys.executable, "-m", "compileall", "-q", "backend", "scripts"])
    clean_generated()
    run("Backend regression", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT / "backend")
    clean_generated()
    tsc = shutil.which("tsc")
    if tsc:
        run("Frontend TypeScript syntax", [tsc, "--noEmit", "--noCheck"], cwd=ROOT / "frontend")
    else:
        print("\nSKIP TypeScript syntax: tsc não está disponível neste ambiente.")
    npm = shutil.which("npm")
    if npm and (ROOT / "frontend/node_modules").is_dir():
        run("Frontend build", [npm, "run", "build"], cwd=ROOT / "frontend")
    else:
        print("\nSKIP frontend build completo: node_modules não está disponível neste ambiente.")
    clean_generated()
    run("Release manifest", [sys.executable, "scripts/release_manifest.py"])
    run("Release integrity", [sys.executable, "scripts/verify_release.py"])
    run("Static integration doctor", [sys.executable, "scripts/doctor.py"])
    clean_generated()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
