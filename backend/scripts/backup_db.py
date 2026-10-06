#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from src.runtime import database_path


BACKEND_DIR = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Backup consistente do banco cl�nico PhysioAI")
    ap.add_argument("--output-dir", default="backups")
    args = ap.parse_args()

    source = database_path(BACKEND_DIR)
    if not source.exists():
        raise SystemExit(f"Banco n�o encontrado: {source}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = out_dir / f"physioai_clinical_{stamp}.sqlite3"

    with closing(sqlite3.connect(source)) as src, closing(sqlite3.connect(target)) as dst:
        src.backup(dst)
        check = dst.execute("PRAGMA quick_check").fetchone()
    if not check or str(check[0]).lower() != "ok":
        target.unlink(missing_ok=True)
        raise SystemExit("Backup criado, mas falhou no PRAGMA quick_check")

    digest = sha256(target)
    manifest = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "backup": target.name,
        "sha256": digest,
        "quick_check": "ok",
    }
    manifest_path = target.with_suffix(target.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(target)
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
