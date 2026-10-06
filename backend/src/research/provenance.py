from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.core.clinical import public_movement_specs
from src.core.capture import capture_preflight_spec_snapshot

PROTOCOL_VERSION = "1.4"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def clinical_spec_snapshot() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "movements": public_movement_specs(),
        "angle_definition": "movement-specific convention applied after anatomical-plane projection and temporal filtering",
        "validity_rule": "observed landmarks only; minimum landmark visibility >= 0.65",
        "capture_protocol_rule": "measurement plane, anatomical projection, camera alignment and out-of-plane limits are versioned with each movement",
        "capture_preflight": capture_preflight_spec_snapshot(),
    }


def clinical_spec_sha256() -> str:
    return canonical_json_sha256(clinical_spec_snapshot())


def model_provenance(path: str | Path) -> dict[str, Any]:
    model_path = Path(path)
    if not model_path.exists():
        return {
            "status": "absent",
            "enabled_for_clinical_output": False,
            "sha256": None,
        }
    return {
        "status": "present_unvalidated_disabled",
        "enabled_for_clinical_output": False,
        "sha256": file_sha256(model_path),
    }
