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


def clean_bytecode() -> None:
    for path in ROOT.rglob("*.pyc"):
        path.unlink(missing_ok=True)
    for path in sorted(ROOT.rglob("__pycache__"), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def main() -> int:
    run("Python compileall", [sys.executable, "-m", "compileall", "-q", "backend", "scripts"])
    clean_bytecode()
    run(
        "Backend regression",
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT / "backend",
    )
    clean_bytecode()
    (ROOT / "frontend" / "tsconfig.tsbuildinfo").unlink(missing_ok=True)
    # Regenerate after all source changes, then verify exact contents.
    run("Release manifest", [sys.executable, "scripts/release_manifest.py"])
    run("Release integrity", [sys.executable, "scripts/verify_release.py"])
    run("Static integration doctor", [sys.executable, "scripts/doctor.py"])

    node_modules = ROOT / "frontend" / "node_modules"
    npm = shutil.which("npm")
    if npm and node_modules.is_dir():
        run("Frontend build", [npm, "run", "build"], cwd=ROOT / "frontend")
    else:
        print("\nSKIP frontend build: node_modules não está disponível neste ambiente.")
        print("CI/Docker executam npm install + npm run build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
