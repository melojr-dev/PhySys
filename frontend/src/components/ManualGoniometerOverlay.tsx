import React, { useEffect, useRef, useState } from "react";
import type { JointMeasurement, Point2D } from "../types";
import { clamp, clinicalAngleDeg } from "../lib/geometry";

export interface ManualPoints {
  proximal: Point2D;
  vertex: Point2D;
  distal: Point2D;
}

type PointKey = keyof ManualPoints;

function px(point: Point2D, width: number, height: number) {
  return { x: point.x * width, y: point.y * height };
}

function AngleGeometry({ points, width, height, classPrefix }: {
  points: ManualPoints;
  width: number;
  height: number;
  classPrefix: string;
}) {
  const a = px(points.proximal, width, height);
  const b = px(points.vertex, width, height);
  const c = px(points.distal, width, height);
  const start = Math.atan2(a.y - b.y, a.x - b.x);
  const finish = Math.atan2(c.y - b.y, c.x - b.x);
  let delta = finish - start;
  while (delta > Math.PI) delta -= Math.PI * 2;
  while (delta < -Math.PI) delta += Math.PI * 2;
  const radius = Math.max(24, Math.min(width, height) * 0.085);
  const end = start + delta;
  const sx = b.x + Math.cos(start) * radius;
  const sy = b.y + Math.sin(start) * radius;
  const ex = b.x + Math.cos(end) * radius;
  const ey = b.y + Math.sin(end) * radius;
  const sweep = delta >= 0 ? 1 : 0;

  return (
    <>
      <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className={`${classPrefix}-line`} />
      <line x1={b.x} y1={b.y} x2={c.x} y2={c.y} className={`${classPrefix}-line`} />
      <path d={`M ${sx} ${sy} A ${radius} ${radius} 0 0 ${sweep} ${ex} ${ey}`} className={`${classPrefix}-arc`} />
    </>
  );
}

export function ManualGoniometerOverlay({
  measurement,
  manual,
  onChange,
  width,
  height,
  enabled,
  convention,
}: {
  measurement: JointMeasurement | null;
  manual: ManualPoints | null;
  onChange: (points: ManualPoints) => void;
  width: number;
  height: number;
  enabled: boolean;
  convention?: string | null;
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [selected, setSelected] = useState<PointKey | null>(null);

  const automatic = measurement?.proximal && measurement.vertex && measurement.distal
    ? {
        proximal: measurement.proximal,
        vertex: measurement.vertex,
        distal: measurement.distal,
      }
    : null;

  function clientToNormalized(clientX: number, clientY: number): Point2D | null {
    const svg = svgRef.current;
    const matrix = svg?.getScreenCTM();
    if (!svg || !matrix || width <= 0 || height <= 0) return null;
    const p = svg.createSVGPoint();
    p.x = clientX;
    p.y = clientY;
    const local = p.matrixTransform(matrix.inverse());
    return {
      x: clamp(local.x / width),
      y: clamp(local.y / height),
      confidence: null,
    };
  }

  function movePoint(key: PointKey, clientX: number, clientY: number) {
    if (!manual) return;
    const next = clientToNormalized(clientX, clientY);
    if (!next) return;
    onChange({ ...manual, [key]: next });
  }

  useEffect(() => {
    if (!enabled || !manual || !selected) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
      event.preventDefault();
      const baseX = 1 / Math.max(width, 1);
      const baseY = 1 / Math.max(height, 1);
      const scale = event.shiftKey ? 0.25 : event.altKey ? 5 : 1;
      const current = manual[selected];
      const dx = event.key === "ArrowLeft" ? -baseX * scale : event.key === "ArrowRight" ? baseX * scale : 0;
      const dy = event.key === "ArrowUp" ? -baseY * scale : event.key === "ArrowDown" ? baseY * scale : 0;
      onChange({
        ...manual,
        [selected]: {
          ...current,
          x: clamp(current.x + dx),
          y: clamp(current.y + dy),
        },
      });
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [enabled, manual, selected, width, height, onChange]);

  if (!automatic || width <= 0 || height <= 0) return null;
  const manualAngle = manual
    ? clinicalAngleDeg(convention, manual.proximal, manual.vertex, manual.distal)
    : null;

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      className={`goniometer-overlay ${enabled ? "review-enabled" : ""}`}
      tabIndex={enabled ? 0 : -1}
      aria-label={enabled ? "Goniômetro manual editável" : "Goniômetro automático"}
    >
      <AngleGeometry points={automatic} width={width} height={height} classPrefix="auto-gonio" />

      {!enabled && (
        <text
          x={automatic.vertex.x * width + 18}
          y={automatic.vertex.y * height - 18}
          className="goniometer-label-px"
        >
          {measurement?.angleClinicalDeg?.toFixed(1) ?? "—"}°
        </text>
      )}

      {enabled && manual && (
        <>
          <AngleGeometry points={manual} width={width} height={height} classPrefix="manual-gonio" />
          {(["proximal", "vertex", "distal"] as PointKey[]).map((key) => {
            const point = px(manual[key], width, height);
            return (
              <g key={key}>
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={selected === key ? 14 : 11}
                  className={`manual-handle ${selected === key ? "selected" : ""}`}
                  onPointerDown={(event) => {
                    event.preventDefault();
                    setSelected(key);
                    event.currentTarget.setPointerCapture(event.pointerId);
                  }}
                  onPointerMove={(event) => {
                    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                      movePoint(key, event.clientX, event.clientY);
                    }
                  }}
                  onPointerUp={(event) => {
                    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                      event.currentTarget.releasePointerCapture(event.pointerId);
                    }
                  }}
                />
                <text x={point.x + 14} y={point.y - 14} className="manual-point-label">
                  {key === "proximal" ? "PROX" : key === "vertex" ? "EIXO" : "DIST"}
                </text>
              </g>
            );
          })}
          <text
            x={manual.vertex.x * width + 20}
            y={manual.vertex.y * height + 34}
            className="manual-angle-label"
          >
            MANUAL {manualAngle == null ? "—" : manualAngle.toFixed(1)}°
          </text>
        </>
      )}
    </svg>
  );
}
