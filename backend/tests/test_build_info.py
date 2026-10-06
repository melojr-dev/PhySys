import os
import unittest
from unittest.mock import patch
from src.build_info import DEFAULT_VERSION, get_build_info

class BuildInfoTests(unittest.TestCase):
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            info=get_build_info()
            self.assertEqual(info.version, DEFAULT_VERSION)
            self.assertEqual(info.git_sha, "unknown")
            self.assertEqual(info.environment, "development")

    def test_environment_metadata(self):
        env={"PHYSIOAI_VERSION":"9.9.9","PHYSIOAI_GIT_SHA":"abc123","PHYSIOAI_BUILD_TIME":"2026-09-11T00:00:00Z","PHYSIOAI_ENV":"test"}
        with patch.dict(os.environ, env, clear=True):
            info=get_build_info()
        self.assertEqual(info.public()["version"], "9.9.9")
        self.assertEqual(info.git_sha, "abc123")
        self.assertEqual(info.environment, "test")

if __name__ == "__main__": unittest.main()
