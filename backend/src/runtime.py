from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path


def data_dir(app_dir: str | os.PathLike[str]) -> Path:
    """Return the writable runtime data directory.

    Production containers should set ``PHYSIOAI_DATA_DIR`` (default in the
    Compose stack: ``/app/data``). Local source-tree execution retains the
    historical ``src/utils`` fallback for developer convenience.
    """
    root = Path(app_dir).resolve()
    configured = os.environ.get("PHYSIOAI_DATA_DIR", "").strip()
    return Path(configured).expanduser().resolve() if configured else root / "src" / "utils"


def database_path(app_dir: str | os.PathLike[str]) -> Path:
    configured = os.environ.get("PHYSIOAI_DB_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return data_dir(app_dir) / "physioai_clinical.sqlite3"


def model_path(app_dir: str | os.PathLike[str]) -> Path:
    configured = os.environ.get("PHYSIOAI_MODEL_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(app_dir).resolve() / "src" / "models" / "modelo_pibic.pt"


def sqlite_quick_check(path: str | os.PathLike[str]) -> tuple[bool, str]:
    """Run SQLite PRAGMA quick_check without mutating clinical data."""
    db = Path(path)
    if not db.exists():
        return False, "database_missing"
    try:
        with closing(sqlite3.connect(db, timeout=5.0)) as con:
            row = con.execute("PRAGMA quick_check").fetchone()
        message = str(row[0]) if row else "no_result"
        return message.lower() == "ok", message
    except sqlite3.Error as exc:
        return False, f"sqlite_error:{type(exc).__name__}"
