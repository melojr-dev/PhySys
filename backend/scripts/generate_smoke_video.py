#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Gera v�deo MP4 pequeno para smoke test do pipeline")
    ap.add_argument("output", nargs="?", default="/tmp/physioai_smoke.mp4")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--fps", type=float, default=12.0)
    args = ap.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 320, 240
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit("OpenCV n�o conseguiu abrir VideoWriter MP4")

    for i in range(max(2, args.frames)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Padr�o simples em movimento. N�o pretende gerar pose detect�vel; o
        # objetivo � exercitar upload, decode, infer�ncia, telemetria e t�rmino
        # mesmo quando nenhum corpo � encontrado.
        x = 30 + int((width - 60) * i / max(args.frames - 1, 1))
        cv2.circle(frame, (x, height // 2), 18, (255, 255, 255), -1)
        cv2.putText(frame, f"SMOKE {i:02d}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        writer.write(frame)
    writer.release()

    if not output.exists() or output.stat().st_size <= 0:
        raise SystemExit("V�deo de smoke n�o foi criado")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
