import type { Point2D } from "../types";

export type BodySide = "direito" | "esquerdo" | string;

export interface AnatomicalChain {
  landmarkIndices: number[];
  segments: Array<[number, number]>;
  vertexIndex: number;
  labels: Record<number, string>;
}

const SIDE = {
  esquerdo: { shoulder: 11, elbow: 13, wrist: 15, pinky: 17, index: 19, thumb: 21, hip: 23, knee: 25, ankle: 27, heel: 29, foot: 31 },
  direito: { shoulder: 12, elbow: 14, wrist: 16, pinky: 18, index: 20, thumb: 22, hip: 24, knee: 26, ankle: 28, heel: 30, foot: 32 },
} as const;

function sideMap(side: BodySide) {
  return side.toLowerCase().startsWith("esq") ? SIDE.esquerdo : SIDE.direito;
}

export function anatomicalChainFor(movement: string, side: BodySide): AnatomicalChain {
  const s = sideMap(side);
  const normalized = movement.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();

  if (normalized.includes("ombro") || normalized.includes("cotovelo")) {
    const vertexIndex = normalized.includes("cotovelo") ? s.elbow : s.shoulder;
    return {
      landmarkIndices: [s.shoulder, s.elbow, s.wrist, s.thumb, s.index, s.pinky],
      segments: [
        [s.shoulder, s.elbow],
        [s.elbow, s.wrist],
        [s.wrist, s.thumb],
        [s.wrist, s.index],
        [s.wrist, s.pinky],
        [s.pinky, s.index],
      ],
      vertexIndex,
      labels: { [s.shoulder]: "OMBRO", [s.elbow]: "COT", [s.wrist]: "PUNHO" },
    };
  }

  if (normalized.includes("quadril")) {
    return {
      landmarkIndices: [s.hip, s.knee, s.ankle, s.heel, s.foot],
      segments: [[s.hip, s.knee], [s.knee, s.ankle], [s.ankle, s.heel], [s.heel, s.foot], [s.ankle, s.foot]],
      vertexIndex: s.hip,
      labels: { [s.hip]: "QUADRIL", [s.knee]: "JOELHO", [s.ankle]: "TORNOZELO" },
    };
  }

  if (normalized.includes("joelho")) {
    return {
      landmarkIndices: [s.hip, s.knee, s.ankle, s.heel, s.foot],
      segments: [[s.hip, s.knee], [s.knee, s.ankle], [s.ankle, s.heel], [s.heel, s.foot], [s.ankle, s.foot]],
      vertexIndex: s.knee,
      labels: { [s.hip]: "QUADRIL", [s.knee]: "JOELHO", [s.ankle]: "TORNOZELO" },
    };
  }

  return {
    landmarkIndices: [s.knee, s.ankle, s.heel, s.foot],
    segments: [[s.knee, s.ankle], [s.ankle, s.heel], [s.heel, s.foot], [s.ankle, s.foot]],
    vertexIndex: s.ankle,
    labels: { [s.knee]: "JOELHO", [s.ankle]: "TORNOZELO", [s.foot]: "PÉ" },
  };
}

export function safeLandmark(landmarks: Array<Point2D | null> | null | undefined, index: number) {
  const point = landmarks?.[index] ?? null;
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) return null;
  return point;
}
