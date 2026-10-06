import React, { useMemo } from "react";

function fmt(value: number | null | undefined, digits = 1) {
  return value == null || Number.isNaN(value) ? "—" : value.toFixed(digits);
}

export function MotionInsightChart({
  values,
  velocities = [],
  current,
  fps,
  onSelect,
  advanced = false,
}: {
  values: number[];
  velocities?: number[];
  current: number;
  fps?: number | null;
  onSelect?: (index: number) => void;
  advanced?: boolean;
}) {
  const width = 1000;
  const height = advanced ? 300 : 250;
  const pad = { left: 54, right: 24, top: 28, bottom: 40 };
  const safe = values.length > 1 ? values : [0, 0];

  const stats = useMemo(() => {
    if (!values.length) return { min: 0, max: 0, minIndex: 0, maxIndex: 0, excursion: 0 };
    let min = values[0]; let max = values[0]; let minIndex = 0; let maxIndex = 0;
    values.forEach((value, index) => {
      if (value < min) { min = value; minIndex = index; }
      if (value > max) { max = value; maxIndex = index; }
    });
    return { min, max, minIndex, maxIndex, excursion: max - min };
  }, [values]);

  const range = Math.max(stats.max - stats.min, 1);
  const x = (index: number) => pad.left + (index / Math.max(safe.length - 1, 1)) * (width - pad.left - pad.right);
  const y = (value: number) => height - pad.bottom - ((value - stats.min) / range) * (height - pad.top - pad.bottom);
  const points = safe.map((value, index) => `${x(index)},${y(value)}`).join(" ");
  const area = `${x(0)},${height - pad.bottom} ${points} ${x(safe.length - 1)},${height - pad.bottom}`;
  const boundedCurrent = Math.min(Math.max(current, 0), Math.max(safe.length - 1, 0));
  const currentX = x(boundedCurrent);
  const currentY = y(safe[boundedCurrent] ?? 0);
  const duration = fps && fps > 0 && values.length > 1 ? (values.length - 1) / fps : null;
  const peakVelocity = velocities.length ? Math.max(...velocities.map((v) => Math.abs(v))) : null;

  return (
    <div className="motion-chart-wrap">
      <div className="motion-chart-summary">
        <div><span>Excursão visual</span><strong>{fmt(stats.excursion)}°</strong></div>
        <div><span>Pico</span><strong>{fmt(stats.max)}°</strong></div>
        <div><span>Duração</span><strong>{duration == null ? "—" : `${fmt(duration, 2)} s`}</strong></div>
        {advanced && <div><span>Vel. pico</span><strong>{peakVelocity == null ? "—" : `${fmt(peakVelocity)}°/s`}</strong></div>}
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className={`motion-chart ${onSelect ? "interactive" : ""}`}
        role="img"
        aria-label="Curva temporal do ângulo clínico"
        onPointerDown={(event) => {
          if (!onSelect || values.length === 0) return;
          const rect = event.currentTarget.getBoundingClientRect();
          const local = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
          const svgX = local * width;
          const fraction = Math.max(0, Math.min(1, (svgX - pad.left) / (width - pad.left - pad.right)));
          onSelect(Math.round(fraction * Math.max(values.length - 1, 0)));
        }}
      >
        <defs>
          <linearGradient id="motionArea" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="currentColor" stopOpacity="0.28" />
            <stop offset="100%" stopColor="currentColor" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
          const gy = pad.top + fraction * (height - pad.top - pad.bottom);
          const label = stats.max - fraction * range;
          return <g key={fraction}><line x1={pad.left} x2={width - pad.right} y1={gy} y2={gy} className="motion-grid" /><text x={pad.left - 10} y={gy + 4} textAnchor="end" className="motion-axis-label">{fmt(label, 0)}°</text></g>;
        })}
        <polygon points={area} className="motion-area" />
        <polyline points={points} fill="none" className="motion-line" />
        {values.length > 0 && <>
          <line x1={x(stats.maxIndex)} x2={x(stats.maxIndex)} y1={pad.top} y2={height - pad.bottom} className="motion-event-line peak" />
          <circle cx={x(stats.maxIndex)} cy={y(stats.max)} r="6" className="motion-event-dot peak" />
          <text x={x(stats.maxIndex)} y={Math.max(pad.top + 12, y(stats.max) - 13)} textAnchor="middle" className="motion-event-label">PICO {fmt(stats.max)}°</text>
        </>}
        <line x1={currentX} x2={currentX} y1={pad.top} y2={height - pad.bottom} className="motion-cursor" />
        <circle cx={currentX} cy={currentY} r="7" className="motion-current-dot" />
        <g className="motion-time-labels">
          <text x={pad.left} y={height - 12}>INÍCIO</text>
          <text x={width - pad.right} y={height - 12} textAnchor="end">{duration == null ? "FIM" : `${fmt(duration, 2)} s`}</text>
        </g>
      </svg>
      <div className="motion-chart-caption"><span className="markerless-pulse" /> Arraste ou clique na curva para sincronizar o vídeo ao frame correspondente.</div>
    </div>
  );
}
