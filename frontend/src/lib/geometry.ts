import type { Point2D } from "../types";

export function clamp(value: number, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

export function geometricAngleDeg(a: Point2D, b: Point2D, c: Point2D) {
  const v1x = a.x - b.x;
  const v1y = a.y - b.y;
  const v2x = c.x - b.x;
  const v2y = c.y - b.y;
  const n1 = Math.hypot(v1x, v1y);
  const n2 = Math.hypot(v2x, v2y);
  if (n1 < 1e-9 || n2 < 1e-9) return null;
  const cosine = clamp((v1x * v2x + v1y * v2y) / (n1 * n2), -1, 1);
  return (Math.acos(cosine) * 180) / Math.PI;
}

export function flexionClinicalAngleDeg(a: Point2D, b: Point2D, c: Point2D) {
  const geometric = geometricAngleDeg(a, b, c);
  if (geometric == null) return null;
  return clamp(180 - geometric, 0, 180);
}

export function ankleDorsiflexionClinicalAngleDeg(a: Point2D, b: Point2D, c: Point2D) {
  const geometric = geometricAngleDeg(a, b, c);
  if (geometric == null) return null;
  return clamp(90 - geometric, 0, 90);
}

export function clinicalAngleDeg(
  convention: string | null | undefined,
  a: Point2D,
  b: Point2D,
  c: Point2D,
) {
  if (convention === "ankle_dorsiflexion_from_internal_angle") {
    return ankleDorsiflexionClinicalAngleDeg(a, b, c);
  }
  if (convention === "direct_internal_angle") {
    return geometricAngleDeg(a, b, c);
  }
  return flexionClinicalAngleDeg(a, b, c);
}

// Aliases para compatibilidade com versões anteriores.
export const elbowClinicalAngleDeg = flexionClinicalAngleDeg;

export function signedErrorDeg(automatic: number | null | undefined, reference: number | null | undefined) {
  if (automatic == null || reference == null) return null;
  return automatic - reference;
}
