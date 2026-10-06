from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
errors = []

def require(condition: bool, message: str):
    if not condition:
        errors.append(message)

version = (ROOT / "VERSION").read_text().strip()
require(version == "2.22.1-rc.5", f"VERSION inesperada: {version}")

package = json.loads((ROOT / "frontend" / "package.json").read_text())
require(package.get("version") == version, "frontend/package.json não acompanha VERSION")

main = (ROOT / "frontend" / "src" / "main.tsx").read_text()
require('retornar_skeleton", "true"' in main, "frontend não solicita skeleton_3d")
require('MarkerlessHome' in main, "Home markerless ausente")
require('MotionInsightChart' in main, "gráfico temporal novo ausente")
require('VirtualBodyViewer' in main, "visualizador 3D ausente")
require('presentationMode' in main, "modo Apresentação/Pesquisa ausente")

for rel in [
    "frontend/src/components/MarkerlessHome.tsx",
    "frontend/src/components/MotionInsightChart.tsx",
    "frontend/src/components/VirtualBodyViewer.tsx",
    "frontend/src/components/AnatomicalOverlay.tsx",
    "frontend/src/lib/anatomicalChains.ts",
    "docs/PHASE24_VALIDATION.md",
]:
    require((ROOT / rel).exists(), f"arquivo ausente: {rel}")

req = (ROOT / "backend" / "requirements.lock").read_text()
require("websockets==13.1" in req, "websockets não está fixado em requirements.lock")

dockerfile = (ROOT / "backend" / "Dockerfile").read_text()
require("MediaPipe Pose Heavy pronto" in dockerfile, "modelo MediaPipe Heavy não é pré-carregado no build")

# A Fase 24 não deve mudar o protocolo clínico.
protocol_lock = json.loads((ROOT / "release" / "protocol_lock.json").read_text())
protocol_version = str(protocol_lock.get("protocol_version") or protocol_lock.get("version") or "")
require(protocol_version == "1.4", f"protocol_lock mudou inesperadamente: {protocol_version!r}")

# Garantia simples de que a UI não vende a visualização 3D como escaneamento preciso.
viewer = (ROOT / "frontend" / "src" / "components" / "VirtualBodyViewer.tsx").read_text()
require("não representa escaneamento anatômico" in viewer, "disclaimer 3D ausente")

if errors:
    print("PHASE24 FAIL")
    for error in errors:
        print(f" - {error}")
    raise SystemExit(1)

print("PHASE24 OK")
print(f"version={version}")
print("clinical_protocol=1.4 (unchanged)")
print("markerless_home=ok")
print("anatomical_overlay=ok")
print("motion_chart=ok")
print("virtual_body_3d=ok")
print("websocket_runtime=declared")
print("mediapipe_heavy_preload=declared")
