#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha256(p):
    h=hashlib.sha256();
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("manifest",nargs="?",default="release/manifest.json"); a=ap.parse_args()
    data=json.loads((ROOT/a.manifest).read_text())
    errors=[]
    for rel, expected in data["files"].items():
        p=ROOT/rel
        if not p.exists(): errors.append(f"missing: {rel}")
        elif sha256(p)!=expected: errors.append(f"changed: {rel}")
    if errors:
        print("\n".join(errors)); raise SystemExit(1)
    print(f"OK: PhysioAI {data['version']} ({len(data['files'])} files)")
if __name__=="__main__": main()
