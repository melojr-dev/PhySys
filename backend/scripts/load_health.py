#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import math
import statistics
import time
import urllib.request


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return ordered[idx]


def hit(url: str, timeout: float) -> tuple[bool, float]:
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            resp.read()
            ok = resp.status == 200
    except Exception:
        ok = False
    return ok, (time.perf_counter() - start) * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description="Carga leve para liveness/readiness do PhysioAI")
    ap.add_argument("--url", default="http://localhost/health")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(lambda _: hit(args.url, args.timeout), range(args.requests)))
    elapsed = time.perf_counter() - started
    lat = [ms for _, ms in results]
    ok = sum(1 for success, _ in results if success)
    errors = len(results) - ok

    print(f"requests={len(results)} ok={ok} errors={errors} concurrency={args.concurrency}")
    print(f"throughput={len(results)/elapsed:.2f} req/s total={elapsed:.3f}s")
    print(
        f"latency_ms min={min(lat):.2f} mean={statistics.mean(lat):.2f} "
        f"p50={percentile(lat, .50):.2f} p95={percentile(lat, .95):.2f} "
        f"p99={percentile(lat, .99):.2f} max={max(lat):.2f}"
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
