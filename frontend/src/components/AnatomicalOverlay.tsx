import React, { useMemo } from "react";
import type { JointMeasurement, Point2D } from "../types";
import { anatomicalChainFor, safeLandmark } from "../lib/anatomicalChains";

function toPx(point: Point2D, width: number, height: number) {
  return { x: point.x * width, y: point.y * height };
}

function confidenceClass(point: Point2D | null) {
  const confidence = point?.confidence;
  if (confidence == null) return "unknown";
  if (confidence >= 0.75) return "good";
  if (confidence >= 0.5) return "moderate";
  return "low";
}

export function AnatomicalOverlay({
  measurement,
  movement,
  side,
  width,
  height,
  enabled = true,
}: {
  measurement: JointMeasurement | null;
  movement: string;
  side: string;
  width: number;
  height: number;
  enabled?: boolean;
}) {
  const chain = useMemo(() => anatomicalChainFor(movement, side), [movement, side]);
  const landmarks = measurement?.poseLandmarks2d;
  const clinicalVertex = measurement?.vertex ? toPx(measurement.vertex, width, height) : null;

  if (!enabled || !measurement || width <= 0 || height <= 0) return null;

  // Compatibilidade com sessões antigas: ainda mostramos os três pontos clínicos
  // caso o backend não tenha enviado a cadeia completa.
  if (!landmarks?.length) {
    const fallback = [measurement.proximal, measurement.vertex, measurement.distal].filter(
      (point): point is Point2D => Boolean(point),
    );
    if (fallback.length < 2) return null;
    return (
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" className="anatomical-overlay" aria-label="Marcador do membro analisado">
        {fallback.slice(0, -1).map((point, index) => {
          const a = toPx(point, width, height);
          const b = toPx(fallback[index + 1], width, height);
          return <line key={index} x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="anatomical-segment fallback" />;
        })}
        {fallback.map((point, index) => {
          const p = toPx(point, width, height);
          return <circle key={index} cx={p.x} cy={p.y} r={index === 1 ? 10 : 7} className={`anatomical-point ${confidenceClass(point)} ${index === 1 ? "vertex" : ""}`} />;
        })}
      </svg>
    );
  }

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      className="anatomical-overlay"
      aria-label={`Cadeia anatômica: ${movement} ${side}`}
    >
      <g className={`anatomical-chain ${measurement.source === "predicted" ? "predicted" : "observed"}`}>
        {chain.segments.map(([from, to]) => {
          const a0 = safeLandmark(landmarks, from);
          const b0 = safeLandmark(landmarks, to);
          if (!a0 || !b0) return null;
          const a = toPx(a0, width, height);
          const b = toPx(b0, width, height);
          const low = Math.min(a0.confidence ?? 1, b0.confidence ?? 1) < 0.5;
          return (
            <line
              key={`${from}-${to}`}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              className={`anatomical-segment ${low ? "low-confidence" : ""}`}
            />
          );
        })}

        {chain.landmarkIndices.map((index) => {
          const point = safeLandmark(landmarks, index);
          if (!point) return null;
          const p = toPx(point, width, height);
          const isVertex = index === chain.vertexIndex;
          return (
            <g key={index}>
              {isVertex && <circle cx={p.x} cy={p.y} r={18} className="anatomical-vertex-halo" />}
              <circle
                cx={p.x}
                cy={p.y}
                r={isVertex ? 10 : 7}
                className={`anatomical-point ${confidenceClass(point)} ${isVertex ? "vertex" : ""}`}
              />
              {chain.labels[index] && (
                <text x={p.x + 13} y={p.y - 12} className="anatomical-label">
                  {chain.labels[index]}
                </text>
              )}
            </g>
          );
        })}
        {clinicalVertex && measurement.angleClinicalDeg != null && (
          <g className="anatomical-angle-readout" transform={`translate(${clinicalVertex.x + 28} ${clinicalVertex.y + 34})`}>
            <rect x="0" y="0" width="112" height="42" rx="12" />
            <text x="14" y="27">{measurement.angleClinicalDeg.toFixed(1)}°</text>
          </g>
        )}
      </g>
    </svg>
  );
}
