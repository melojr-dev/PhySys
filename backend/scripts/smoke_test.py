#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def request_json(url: str, method: str = "GET", token: str | None = None, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, method=method, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode()
        return resp.status, json.loads(raw) if raw else {}


def main() -> int:
    ap = argparse.ArgumentParser(description="PhysioAI smoke test")
    ap.add_argument("--base-url", default="http://localhost")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", required=True)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    checks: list[tuple[str, bool, str]] = []
    try:
        status, health = request_json(base + "/health")
        checks.append(("liveness", status == 200 and health.get("status") == "ok", str(health)))

        status, ready = request_json(base + "/ready")
        checks.append(("readiness", status == 200 and ready.get("status") == "ready", str(ready)))

        status, login = request_json(
            base + "/api/auth/login", "POST", body={"username": args.user, "password": args.password}
        )
        token = login.get("access_token")
        checks.append(("login", status == 200 and bool(token), f"user={login.get('user')}"))

        status, system = request_json(base + "/api/system/health", token=token)
        checks.append(("system_health", status == 200 and system.get("status") in {"ok", "degraded"}, str(system.get("status"))))

        status, metrics = request_json(base + "/api/system/metrics", token=token)
        checks.append(("metrics", status == 200 and "counters" in metrics, f"uptime={metrics.get('uptime_seconds')}"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        print(f"SMOKE ERROR: {exc}", file=sys.stderr)
        return 2

    failed = False
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:16s} {detail}")
        failed |= not ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
