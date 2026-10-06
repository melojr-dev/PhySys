#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Doctor:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, name: str, detail: str = "") -> None:
        print(f"PASS  {name:30s} {detail}")

    def fail(self, name: str, detail: str = "") -> None:
        self.failures += 1
        print(f"FAIL  {name:30s} {detail}")

    def warn(self, name: str, detail: str = "") -> None:
        self.warnings += 1
        print(f"WARN  {name:30s} {detail}")

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.ok(name, detail) if condition else self.fail(name, detail)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    d = Doctor()
    required = [
        "VERSION", "docker-compose.yml", ".env.example",
        "backend/Dockerfile", "backend/server.py", "backend/src/core/capture.py", "frontend/Dockerfile",
        "frontend/package.json", "nginx/default.conf", "release/manifest.json",
    ]
    for rel in required:
        d.check(f"required:{rel}", (ROOT / rel).is_file())

    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    package = json.loads((ROOT / "frontend/package.json").read_text(encoding="utf-8"))
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    build_info = (ROOT / "backend/src/build_info.py").read_text(encoding="utf-8")
    d.check("version:frontend", package.get("version") == version, f"{package.get('version')} == {version}")
    d.check("version:compose", version in compose, version)
    d.check("version:env", f"PHYSIOAI_VERSION={version}" in env_example, version)
    d.check("version:backend_default", version in build_info, version)

    # Runtime persistence and health dependencies.
    d.check("compose:named_data_volume", "physioai_data:/app/data" in compose)
    d.check("compose:frontend_healthy", "frontend:\n        condition: service_healthy" in compose)
    d.check("compose:backend_healthy", "backend:\n        condition: service_healthy" in compose)
    d.check("compose:no_new_privileges", compose.count("no-new-privileges:true") >= 3)
    d.check("backend:nonroot", "USER physioai" in (ROOT / "backend/Dockerfile").read_text())

    nginx = (ROOT / "nginx/default.conf").read_text(encoding="utf-8")
    d.check("nginx:websocket_upgrade", 'proxy_set_header Upgrade $http_upgrade;' in nginx)
    d.check("nginx:csp", "Content-Security-Policy" in nginx)
    d.check("nginx:auth_rate_limit", "physioai_login" in nginx)
    d.check("nginx:preflight_route", "location = /api/capture/preflight" in nginx)
    d.check("nginx:preflight_streaming", "location = /api/capture/preflight" in nginx and "proxy_request_buffering off;" in nginx)

    server = (ROOT / "backend/server.py").read_text(encoding="utf-8")
    provenance = (ROOT / "backend/src/research/provenance.py").read_text(encoding="utf-8")
    capture = (ROOT / "backend/src/core/capture.py").read_text(encoding="utf-8")
    d.check("preflight:route", '@app.post("/api/capture/preflight")' in server)
    d.check("preflight:required_default", "PHYSIOAI_REQUIRE_PREFLIGHT" in server and "PHYSIOAI_REQUIRE_PREFLIGHT" in compose)
    d.check("preflight:env", all(key in env_example for key in (
        "PHYSIOAI_PREFLIGHT_TTL_SECONDS=",
        "PHYSIOAI_PREFLIGHT_MAX_SAMPLES=",
        "PHYSIOAI_REQUIRE_PREFLIGHT=1",
    )))
    d.check("preflight:spec_1_0", 'PREFLIGHT_SPEC_VERSION = "1.0"' in capture)
    d.check("clinical:protocol_1_4", 'PROTOCOL_VERSION = "1.4"' in provenance)
    d.check("clinical:preflight_snapshot", '"capture_preflight"' in provenance)
    d.check("experimental:gait_module", (ROOT / "backend/src/core/gait.py").is_file())
    d.check("experimental:gait_route", '@app.post("/api/experimental/gait/analyze")' in server)
    d.check("experimental:imu_module", (ROOT / "backend/src/core/imu.py").is_file())
    d.check("experimental:imu_store", (ROOT / "backend/src/data/imu_store.py").is_file())
    d.check("experimental:imu_route", '@app.post("/api/experimental/imu/analyze"' in server)
    d.check("experimental:imu_export", '/api/experimental/imu/{analysis_id}/export.json' in server)
    d.check("experimental:imu_frontend", (ROOT / "frontend/src/components/SensorFusionPanel.tsx").is_file())
    d.check("experimental:imu_doc", (ROOT / "docs/PHASE17_VALIDATION.md").is_file())
    d.check("experimental:motor_features_module", (ROOT / "backend/src/core/motor_features.py").is_file())
    d.check("experimental:motor_features_store", (ROOT / "backend/src/data/motor_feature_store.py").is_file())
    d.check("experimental:motor_features_route", '@app.post("/api/experimental/motor-features/analyze"' in server)
    d.check("experimental:motor_features_frontend", (ROOT / "frontend/src/components/MotorFeaturesPanel.tsx").is_file())
    d.check("experimental:motor_features_doc", (ROOT / "docs/PHASE18_VALIDATION.md").is_file())
    d.check("experimental:longitudinal_module", (ROOT / "backend/src/core/longitudinal.py").is_file())
    d.check("experimental:longitudinal_store", (ROOT / "backend/src/data/longitudinal_store.py").is_file())
    d.check("experimental:longitudinal_route", '@app.post("/api/experimental/longitudinal/analyze"' in server)
    d.check("experimental:longitudinal_frontend", (ROOT / "frontend/src/components/LongitudinalPanel.tsx").is_file())
    d.check("experimental:longitudinal_doc", (ROOT / "docs/PHASE19_VALIDATION.md").is_file())
    d.check("privacy:session_listing_scoped", "recent_for_evaluator" in server)
    d.check("phase20:clinical_baseline", (ROOT / "release/clinical_baseline.json").is_file())
    d.check("phase20:protocol_lock", (ROOT / "release/protocol_lock.json").is_file())
    d.check("phase20:governance", (ROOT / "docs/PROTOCOL_GOVERNANCE.md").is_file())
    d.check("phase21:formal_store", (ROOT / "backend/src/validation/formal.py").is_file())
    d.check("phase21:statistical_plan", (ROOT / "release/statistical_analysis_plan.json").is_file())
    d.check("phase21:formal_checklist", (ROOT / "release/formal_validation_checklist.json").is_file())
    d.check("phase21:readiness_route", "/formal-runs/readiness" in server)
    d.check("phase21:formal_export", "/api/formal-runs/{run_id}/export.zip" in server)
    d.check("phase22:publication_module", (ROOT / "backend/src/research/publication.py").is_file())
    d.check("phase22:publication_requirements", (ROOT / "release/publication_requirements.json").is_file())
    d.check("phase22:publication_readiness", "/publication/readiness" in server)
    d.check("phase22:publication_export", "/api/formal-runs/{run_id}/publication.zip" in server)
    d.check("phase22:external_review_doc", (ROOT / "docs/EXTERNAL_REVIEW_PACKAGE.md").is_file())
    pub_release = ROOT / "release/publication_requirements.json"
    pub_runtime = ROOT / "backend/src/validation/resources/publication_requirements.json"
    d.check("phase22:requirements_parity", pub_release.is_file() and pub_runtime.is_file() and pub_release.read_bytes() == pub_runtime.read_bytes())

    # No clinical/runtime data or local secrets in a release tree.
    forbidden = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.name == ".env" or path.suffix in {".sqlite", ".sqlite3", ".db"}:
            forbidden.append(str(path.relative_to(ROOT)))
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            forbidden.append(str(path.relative_to(ROOT)))
    d.check("release:no_runtime_data", not forbidden, ", ".join(forbidden[:5]))

    # Dependencies are pinned directly. Lockfile absence is reported explicitly
    # because transitive npm dependencies are not fully reproducible without it.
    deps = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
    floating = {k: v for k, v in deps.items() if isinstance(v, str) and re.match(r"^[~^*]|latest$", v)}
    d.check("frontend:direct_pins", not floating, str(floating) if floating else "exact versions")
    if (ROOT / "frontend/package-lock.json").exists():
        d.ok("frontend:lockfile", "package-lock.json")
    else:
        d.warn("frontend:lockfile", "ausente; npm install ainda resolve dependências transitivas")

    # Validate the checked-in release manifest.
    try:
        manifest = json.loads((ROOT / "release/manifest.json").read_text(encoding="utf-8"))
        changed = []
        missing = []
        for rel, expected in manifest.get("files", {}).items():
            path = ROOT / rel
            if not path.exists():
                missing.append(rel)
            elif sha256(path) != expected:
                changed.append(rel)
        d.check("release:manifest_version", manifest.get("version") == version, str(manifest.get("version")))
        d.check("release:manifest_integrity", not missing and not changed, f"missing={len(missing)} changed={len(changed)}")
    except Exception as exc:
        d.fail("release:manifest", f"{type(exc).__name__}: {exc}")

    print(f"\nDoctor: failures={d.failures} warnings={d.warnings}")
    return 1 if d.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
