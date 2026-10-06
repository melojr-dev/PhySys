from __future__ import annotations

import os
from dataclasses import asdict, dataclass

DEFAULT_VERSION = "2.21.0-rc.4"

@dataclass(frozen=True)
class BuildInfo:
    version: str
    git_sha: str
    build_time: str
    environment: str

    def public(self) -> dict[str, str]:
        return asdict(self)

def get_build_info() -> BuildInfo:
    return BuildInfo(
        version=os.environ.get("PHYSIOAI_VERSION", DEFAULT_VERSION).strip() or DEFAULT_VERSION,
        git_sha=os.environ.get("PHYSIOAI_GIT_SHA", "unknown").strip() or "unknown",
        build_time=os.environ.get("PHYSIOAI_BUILD_TIME", "unknown").strip() or "unknown",
        environment=os.environ.get("PHYSIOAI_ENV", "development").strip() or "development",
    )
