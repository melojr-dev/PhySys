#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, subprocess
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def git(*args):
    try:return subprocess.check_output(["git",*args],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return "unknown"
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest-only",action="store_true"); ap.add_argument("--allow-dirty",action="store_true"); a=ap.parse_args()
    if not a.allow_dirty and git("status","--porcelain") not in ("","unknown"):
        raise SystemExit("Working tree suja; use --allow-dirty apenas conscientemente.")
    env=os.environ.copy(); env["PHYSIOAI_VERSION"]=(ROOT/"VERSION").read_text().strip(); env["PHYSIOAI_GIT_SHA"]=git("rev-parse","HEAD"); env["PHYSIOAI_BUILD_TIME"]=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    subprocess.run(["python",str(ROOT/"scripts/release_manifest.py")],cwd=ROOT,env=env,check=True)
    if not a.manifest_only: subprocess.run(["docker","compose","build","--pull"],cwd=ROOT,env=env,check=True)
    print(f"PHYSIOAI_VERSION={env['PHYSIOAI_VERSION']}")
    print(f"PHYSIOAI_GIT_SHA={env['PHYSIOAI_GIT_SHA']}")
    print(f"PHYSIOAI_BUILD_TIME={env['PHYSIOAI_BUILD_TIME']}")
if __name__=="__main__":main()
