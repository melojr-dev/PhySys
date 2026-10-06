from __future__ import annotations

import ast
import hashlib
import json
import math
import unittest
from pathlib import Path

from src.core.clinical import get_movement_spec, internal_angle_from_points_2d, public_movement_specs

ROOT = Path(__file__).resolve().parents[2]


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _preflight_contract() -> dict[str, object]:
    source = ROOT / "backend/src/core/capture.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in {"PREFLIGHT_SPEC_VERSION", "PREFLIGHT_THRESHOLDS"}:
                    values[target.id] = ast.literal_eval(node.value)
    return {"version": values["PREFLIGHT_SPEC_VERSION"], "thresholds": values["PREFLIGHT_THRESHOLDS"]}


class ReleaseClinicalBaselineTests(unittest.TestCase):
    def test_protocol_lock_matches_current_contract(self) -> None:
        lock = json.loads((ROOT / "release/protocol_lock.json").read_text(encoding="utf-8"))
        contract = {
            "protocol_version": "1.4",
            "movement_specs": public_movement_specs(),
            "capture_preflight": _preflight_contract(),
        }
        self.assertEqual(lock["protocol_version"], "1.4")
        self.assertEqual(lock["contract_sha256"], sha(contract))

    def test_golden_cases_match_current_geometry_and_converters(self) -> None:
        baseline = json.loads((ROOT / "release/clinical_baseline.json").read_text(encoding="utf-8"))
        tolerance = float(baseline["tolerance_deg"])
        self.assertGreaterEqual(len(baseline["cases"]), 18)
        for case in baseline["cases"]:
            with self.subTest(case=case):
                internal = float(case["internal_angle_deg"])
                theta = math.radians(internal)
                geometry = internal_angle_from_points_2d(
                    (1.0, 0.0), (0.0, 0.0), (math.cos(theta), math.sin(theta))
                )
                spec = get_movement_spec(case["movement"])
                self.assertIsNotNone(spec)
                actual = spec.clinical_angle(geometry)
                self.assertAlmostEqual(actual, float(case["expected_clinical_deg"]), delta=tolerance)

    def test_baseline_does_not_claim_clinical_validation(self) -> None:
        baseline = json.loads((ROOT / "release/clinical_baseline.json").read_text(encoding="utf-8"))
        claim = baseline["claim"].lower()
        self.assertIn("does not establish clinical validity", claim)


if __name__ == "__main__":
    unittest.main()
