from __future__ import annotations

import os
import sqlite3
from contextlib import closing
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.runtime import data_dir, database_path, model_path, sqlite_quick_check


class RuntimePathTests(unittest.TestCase):
    def test_source_tree_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            app = Path(tmp)
            self.assertEqual(data_dir(app), app / "src" / "utils")
            self.assertEqual(database_path(app), app / "src" / "utils" / "physioai_clinical.sqlite3")
            self.assertEqual(model_path(app), app / "src" / "models" / "modelo_pibic.pt")

    def test_environment_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "PHYSIOAI_DATA_DIR": str(Path(tmp) / "data"),
                "PHYSIOAI_DB_PATH": str(Path(tmp) / "db" / "clinical.sqlite3"),
                "PHYSIOAI_MODEL_PATH": str(Path(tmp) / "models" / "model.pt"),
            }
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(data_dir(tmp), Path(env["PHYSIOAI_DATA_DIR"]).resolve())
                self.assertEqual(database_path(tmp), Path(env["PHYSIOAI_DB_PATH"]).resolve())
                self.assertEqual(model_path(tmp), Path(env["PHYSIOAI_MODEL_PATH"]).resolve())

    def test_sqlite_quick_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clinical.sqlite3"
            with closing(sqlite3.connect(path)) as con:
                con.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
                con.execute("INSERT INTO test(value) VALUES ('ok')")
                con.commit()
            ok, message = sqlite_quick_check(path)
            self.assertTrue(ok)
            self.assertEqual(message, "ok")

    def test_sqlite_quick_check_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, message = sqlite_quick_check(Path(tmp) / "missing.sqlite3")
            self.assertFalse(ok)
            self.assertEqual(message, "database_missing")


if __name__ == "__main__":
    unittest.main()
