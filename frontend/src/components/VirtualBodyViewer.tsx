import React, { useEffect, useMemo, useRef, useState } from "react";
import { anatomicalChainFor } from "../lib/anatomicalChains";

const CONNECTIONS: Array<[number, number]> = [
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],
  [11,12],[11,13],[13,15],[15,17],[15,19],[17,19],[15,21],
  [12,14],[14,16],[16,18],[16,20],[18,20],[16,22],
  [11,23],[12,24],[23,24],[23,25],[25,27],[27,29],[29,31],[27,31],
  [24,26],[26,28],[28,30],[30,32],[28,32],
];

interface ProjectedPoint { x: number; y: number; z: number; index: number; }

function finiteSkeleton(frame: number[][] | null | undefined) {
  return Boolean(frame?.length && frame.some((p) => p?.length >= 3 && p.every(Number.isFinite)));
}

export function VirtualBodyViewer({
  skeleton,
  movement,
  side,
  angle,
}: {
  skeleton: number[][] | null;
  movement: string;
  side: string;
  angle?: number | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{ x: number; y: number; yaw: number; pitch: number } | null>(null);
  const [yaw, setYaw] = useState(-0.2);
  const [pitch, setPitch] = useState(0.08);
  const [mode, setMode] = useState<"skeleton" | "volume">("volume");
  const chain = useMemo(() => anatomicalChainFor(movement, side), [movement, side]);
  const highlighted = useMemo(() => new Set(chain.segments.map(([a,b]) => `${Math.min(a,b)}-${Math.max(a,b)}`)), [chain]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !finiteSkeleton(skeleton)) return;
    const ctx = canvas.getContext("2d");
    if (!ctx || !skeleton) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const width = rect.width;
    const height = rect.height;
    ctx.clearRect(0, 0, width, height);

    const pts = skeleton.map((p, index) => ({ x: p[0], y: p[1], z: p[2], index })).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z));
    if (!pts.length) return;
    const torso = [11,12,23,24].map((i) => pts.find((p) => p.index === i)).filter(Boolean) as typeof pts;
    const centerSource = torso.length ? torso : pts;
    const cx = centerSource.reduce((s,p) => s + p.x, 0) / centerSource.length;
    const cy = centerSource.reduce((s,p) => s + p.y, 0) / centerSource.length;
    const cz = centerSource.reduce((s,p) => s + p.z, 0) / centerSource.length;

    const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
    const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
    const rotated = pts.map((p) => {
      const x0 = p.x - cx, y0 = p.y - cy, z0 = p.z - cz;
      const x1 = x0 * cosY + z0 * sinY;
      const z1 = -x0 * sinY + z0 * cosY;
      const y1 = y0 * cosP - z1 * sinP;
      const z2 = y0 * sinP + z1 * cosP;
      return { x: x1, y: y1, z: z2, index: p.index };
    });
    const spanX = Math.max(...rotated.map((p) => p.x)) - Math.min(...rotated.map((p) => p.x));
    const spanY = Math.max(...rotated.map((p) => p.y)) - Math.min(...rotated.map((p) => p.y));
    const scale = Math.min(width / Math.max(spanX, 0.15), height / Math.max(spanY, 0.2)) * 0.72;
    const projected: ProjectedPoint[] = rotated.map((p) => ({
      index: p.index,
      x: width / 2 + p.x * scale,
      y: height / 2 + p.y * scale,
      z: p.z,
    }));
    const byIndex = new Map(projected.map((p) => [p.index, p]));

    // grid / virtual floor
    ctx.save();
    ctx.strokeStyle = "rgba(114,166,255,.09)";
    ctx.lineWidth = 1;
    for (let i = 1; i < 7; i += 1) {
      const gy = height * 0.12 + (height * 0.76 * i) / 7;
      ctx.beginPath(); ctx.moveTo(width * 0.08, gy); ctx.lineTo(width * 0.92, gy); ctx.stroke();
    }
    for (let i = 1; i < 7; i += 1) {
      const gx = width * 0.08 + (width * 0.84 * i) / 7;
      ctx.beginPath(); ctx.moveTo(gx, height * 0.08); ctx.lineTo(gx, height * 0.92); ctx.stroke();
    }
    ctx.restore();

    const segments = CONNECTIONS.map(([a,b]) => {
      const pa = byIndex.get(a), pb = byIndex.get(b);
      return pa && pb ? { a, b, pa, pb, z: (pa.z + pb.z) / 2 } : null;
    }).filter(Boolean) as Array<{a:number;b:number;pa:ProjectedPoint;pb:ProjectedPoint;z:number}>;
    segments.sort((a,b) => a.z - b.z);

    if (mode === "volume") {
      // Corpo virtual: volumes segmentares estimados, sem alegação de reconstrução anatômica.
      for (const seg of segments) {
        const key = `${Math.min(seg.a, seg.b)}-${Math.max(seg.a, seg.b)}`;
        const active = highlighted.has(key);
        const lower = seg.a >= 23 || seg.b >= 23;
        const torso = (seg.a === 11 && seg.b === 12) || (seg.a === 23 && seg.b === 24) || (seg.a === 11 && seg.b === 23) || (seg.a === 12 && seg.b === 24);
        ctx.lineCap = "round";
        ctx.strokeStyle = active ? "rgba(66,231,197,.42)" : "rgba(105,138,168,.18)";
        ctx.lineWidth = torso ? 34 : lower ? 22 : 18;
        ctx.beginPath(); ctx.moveTo(seg.pa.x, seg.pa.y); ctx.lineTo(seg.pb.x, seg.pb.y); ctx.stroke();
      }
    }

    for (const seg of segments) {
      const key = `${Math.min(seg.a, seg.b)}-${Math.max(seg.a, seg.b)}`;
      const active = highlighted.has(key);
      ctx.lineCap = "round";
      ctx.strokeStyle = active ? "#42e7c5" : "rgba(137,167,193,.46)";
      ctx.lineWidth = active ? 4 : 2;
      ctx.beginPath(); ctx.moveTo(seg.pa.x, seg.pa.y); ctx.lineTo(seg.pb.x, seg.pb.y); ctx.stroke();
    }

    const activeIndices = new Set(chain.landmarkIndices);
    [...projected].sort((a,b) => a.z - b.z).forEach((p) => {
      const active = activeIndices.has(p.index);
      const vertex = p.index === chain.vertexIndex;
      if (vertex) {
        ctx.beginPath(); ctx.arc(p.x, p.y, 15, 0, Math.PI * 2); ctx.fillStyle = "rgba(114,166,255,.16)"; ctx.fill();
      }
      ctx.beginPath(); ctx.arc(p.x, p.y, vertex ? 7 : active ? 5.5 : 3.3, 0, Math.PI * 2);
      ctx.fillStyle = vertex ? "#edf5ff" : active ? "#42e7c5" : "rgba(168,191,212,.72)";
      ctx.fill();
    });
  }, [skeleton, movement, side, angle, yaw, pitch, mode, chain, highlighted]);

  const available = finiteSkeleton(skeleton);
  return (
    <div className="virtual-body-viewer">
      <div className="virtual-body-toolbar">
        <div><span className="eyebrow">Modelo corporal virtual</span><strong>Reconstrução 3D estimada</strong></div>
        <div className="virtual-body-actions">
          <button className={mode === "volume" ? "active" : ""} onClick={() => setMode("volume")}>Volume</button>
          <button className={mode === "skeleton" ? "active" : ""} onClick={() => setMode("skeleton")}>Esqueleto</button>
          <button onClick={() => { setYaw(0); setPitch(0); }}>Frontal</button>
          <button onClick={() => { setYaw(Math.PI / 2); setPitch(0); }}>Lateral</button>
        </div>
      </div>
      <div className="virtual-body-canvas-wrap">
        {available ? <canvas
          ref={canvasRef}
          className="virtual-body-canvas"
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture(e.pointerId);
            dragRef.current = { x: e.clientX, y: e.clientY, yaw, pitch };
          }}
          onPointerMove={(e) => {
            const drag = dragRef.current;
            if (!drag) return;
            setYaw(drag.yaw + (e.clientX - drag.x) * 0.009);
            setPitch(Math.max(-1.1, Math.min(1.1, drag.pitch + (e.clientY - drag.y) * 0.006)));
          }}
          onPointerUp={() => { dragRef.current = null; }}
          onPointerCancel={() => { dragRef.current = null; }}
        /> : <div className="virtual-body-empty"><div className="virtual-body-orbit"><span/><span/><span/></div><strong>3D aguardando nova análise</strong><p>Execute a análise com dados 3D habilitados para visualizar o corpo virtual sincronizado.</p></div>}
        {available && <div className="virtual-body-hud"><span>{movement} · {side}</span><strong>{angle == null ? "—" : `${angle.toFixed(1)}°`}</strong></div>}
      </div>
      <p className="virtual-body-disclaimer">Visualização estimada a partir dos landmarks do vídeo; não representa escaneamento anatômico ou reconstrução volumétrica de precisão.</p>
    </div>
  );
}
