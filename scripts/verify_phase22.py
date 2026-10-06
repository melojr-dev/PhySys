#!/usr/bin/env python3
from __future__ import annotations

import json
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


def assert_traceability_contract() -> None:
    req_path = ROOT / "release/publication_requirements.json"
    runtime_path = ROOT / "backend/src/validation/resources/publication_requirements.json"
    req = json.loads(req_path.read_text(encoding="utf-8"))
    if req_path.read_bytes() != runtime_path.read_bytes():
        raise SystemExit("publication_requirements.json diverge entre release e runtime")
    ids: set[str] = set()
    failures: list[str] = []
    for item in req.get("requirements", []):
        rid = str(item.get("id") or "")
        if not rid or rid in ids:
            failures.append(f"requirement id inválido/duplicado: {rid!r}")
        ids.add(rid)
        for rel in item.get("implementation") or []:
            if not (ROOT / rel).is_file():
                failures.append(f"{rid}: implementação ausente: {rel}")
        for rel in item.get("verification") or []:
            if not (ROOT / rel).is_file():
                failures.append(f"{rid}: verificação ausente: {rel}")
        if not str(item.get("evidence") or "").strip():
            failures.append(f"{rid}: evidência vazia")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS  traceability requirements: {len(ids)}")


def assert_contract() -> None:
    server = (ROOT / "backend/server.py").read_text(encoding="utf-8")
    publication = (ROOT / "backend/src/research/publication.py").read_text(encoding="utf-8")
    checks = {
        "version": (ROOT / "VERSION").read_text(encoding="utf-8").strip() == "2.21.0-rc.4",
        "phase22 doc": (ROOT / "docs/PHASE22_VALIDATION.md").is_file(),
        "external review doc": (ROOT / "docs/EXTERNAL_REVIEW_PACKAGE.md").is_file(),
        "publication module": (ROOT / "backend/src/research/publication.py").is_file(),
        "publication tests": (ROOT / "backend/tests/test_publication_package.py").is_file(),
        "publication requirements": (ROOT / "release/publication_requirements.json").is_file(),
        "runtime publication requirements": (ROOT / "backend/src/validation/resources/publication_requirements.json").is_file(),
        "readiness endpoint": '/publication/readiness' in server,
        "publication endpoint": '/api/formal-runs/{run_id}/publication.zip' in server,
        "admin gate": '_require_admin(user)' in server[server.find('/api/formal-runs/{run_id}/publication.zip'):],
        "deidentification aliases": 'participant_alias' in publication and 'session_alias' in publication,
        "serial omitted": 'reference_device_serials": "omitted"' in publication,
        "free text omitted": 'free_text_notes": "omitted"' in publication,
        "deterministic figures": 'build_bland_altman_svg' in publication and 'build_method_comparison_svg' in publication,
        "checksums": 'checksums.sha256' in publication,
        "protocol unchanged": 'PROTOCOL_VERSION = "1.4"' in (ROOT / "backend/src/research/provenance.py").read_text(encoding="utf-8"),
        "clinical baseline retained": (ROOT / "release/clinical_baseline.json").is_file(),
        "protocol lock retained": (ROOT / "release/protocol_lock.json").is_file(),
        "formal validation retained": (ROOT / "backend/src/validation/formal.py").is_file(),
    }
    runtime = ROOT / "backend/src/validation/resources"
    checks["requirements runtime parity"] = (
        (ROOT / "release/publication_requirements.json").read_bytes()
        == (runtime / "publication_requirements.json").read_bytes()
    )
    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        raise SystemExit("Contrato da Fase 22 incompleto: " + ", ".join(failed))
    assert_traceability_contract()


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
