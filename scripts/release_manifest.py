#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_PARTS = {".git", "node_modules", "dist", "__pycache__", ".pytest_cache"}
EXCLUDE_FILES = {"physioai_clinical.sqlite3", ".env"}

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_sha() -> str:
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("PHYSIOAI_GIT_SHA", "unknown")

def files():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.name in EXCLUDE_FILES or any(x in EXCLUDE_PARTS for x in p.parts):
            continue
        if p.suffix == ".tsbuildinfo":
            continue
        if p.parts[-2:] == ("release", "manifest.json"):
            continue
        yield p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output", default="release/manifest.json")
    args=ap.parse_args()
    version=(ROOT/"VERSION").read_text().strip()
    build_time=os.environ.get("PHYSIOAI_BUILD_TIME") or datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    items={str(p.relative_to(ROOT)).replace("\\","/"):sha256(p) for p in files()}
    manifest={"schema":1,"product":"PhysioAI","version":version,"git_sha":git_sha(),"build_time":build_time,"files":items}
    out=ROOT/args.output; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(out)
if __name__=="__main__": main()
