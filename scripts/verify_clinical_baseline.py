#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from src.core.clinical import get_movement_spec, internal_angle_from_points_2d, public_movement_specs  # noqa: E402


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def load_preflight_constants() -> dict[str, object]:
    path = ROOT / "backend/src/core/capture.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in {"PREFLIGHT_SPEC_VERSION", "PREFLIGHT_THRESHOLDS"}:
                found[target.id] = ast.literal_eval(value)
    if set(found) != {"PREFLIGHT_SPEC_VERSION", "PREFLIGHT_THRESHOLDS"}:
        raise RuntimeError("Não foi possível extrair o contrato do preflight")
    return {
        "version": found["PREFLIGHT_SPEC_VERSION"],
        "thresholds": found["PREFLIGHT_THRESHOLDS"],
    }


def protocol_contract() -> dict[str, object]:
    return {
        "protocol_version": "1.4",
        "movement_specs": public_movement_specs(),
        "capture_preflight": load_preflight_constants(),
    }


def constructed_angle(angle_deg: float) -> float:
    theta = math.radians(angle_deg)
    return internal_angle_from_points_2d((1.0, 0.0), (0.0, 0.0), (math.cos(theta), math.sin(theta)))


def verify_baseline() -> int:
    baseline = json.loads((ROOT / "release/clinical_baseline.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "release/protocol_lock.json").read_text(encoding="utf-8"))

    contract = protocol_contract()
    contract_hash = sha(contract)
    if lock.get("protocol_version") != "1.4":
        raise SystemExit("protocol_lock: versão inesperada")
    if lock.get("contract_sha256") != contract_hash:
        raise SystemExit(f"protocol_lock divergente: expected={lock.get('contract_sha256')} actual={contract_hash}")

    tolerance = float(baseline.get("tolerance_deg", 1e-6))
    failures: list[str] = []
    for case in baseline.get("cases", []):
        movement = str(case["movement"])
        internal = float(case["internal_angle_deg"])
        expected = float(case["expected_clinical_deg"])
        spec = get_movement_spec(movement)
        if spec is None:
            failures.append(f"{movement}: spec ausente")
            continue
        geometry_internal = constructed_angle(internal)
        actual = spec.clinical_angle(geometry_internal)
        if abs(actual - expected) > tolerance:
            failures.append(f"{movement} internal={internal}: expected={expected} actual={actual}")

    if baseline.get("protocol_contract_sha256") != contract_hash:
        failures.append("clinical_baseline foi gerado para outro protocol contract")
    if failures:
        raise SystemExit("Baseline clínico divergente:\n- " + "\n- ".join(failures))

    print(f"PASS clinical baseline: {len(baseline.get('cases', []))} casos")
    print(f"PASS protocol lock: {contract_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(verify_baseline())
