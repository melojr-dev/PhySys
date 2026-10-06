from __future__ import annotations

import math
import statistics
from typing import Iterable, Sequence


def _finite_pairs(pairs: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in pairs:
        aa, bb = float(a), float(b)
        if math.isfinite(aa) and math.isfinite(bb):
            out.append((aa, bb))
    return out


def pearson_r(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom <= 1e-12:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def concordance_correlation_coefficient(pairs: Sequence[tuple[float, float]]) -> float | None:
    """Lin's concordance correlation coefficient (CCC)."""
    if len(pairs) < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    n = len(pairs)
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    cov = sum((x - mx) * (y - my) for x, y in pairs) / n
    denom = vx + vy + (mx - my) ** 2
    if denom <= 1e-12:
        return 1.0 if all(abs(x - y) <= 1e-12 for x, y in pairs) else None
    return (2.0 * cov) / denom


def icc_a1(pairs: Sequence[tuple[float, float]]) -> float | None:
    """ICC(A,1): two-way absolute-agreement, single-measure ICC.

    Rows are measurement occasions/targets and the two columns are methods
    (PhysioAI and reference). This statistic is intentionally not called
    "accuracy"; it quantifies absolute agreement between methods.
    """
    n = len(pairs)
    k = 2
    if n < 2:
        return None

    matrix = [[float(a), float(b)] for a, b in pairs]
    grand = sum(sum(row) for row in matrix) / (n * k)
    row_means = [sum(row) / k for row in matrix]
    col_means = [sum(matrix[i][j] for i in range(n)) / n for j in range(k)]

    ss_rows = k * sum((m - grand) ** 2 for m in row_means)
    ss_cols = n * sum((m - grand) ** 2 for m in col_means)
    ss_error = 0.0
    for i in range(n):
        for j in range(k):
            residual = matrix[i][j] - row_means[i] - col_means[j] + grand
            ss_error += residual * residual

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_error + (k * (ms_cols - ms_error) / n)
    if abs(denom) <= 1e-12:
        return 1.0 if all(abs(a - b) <= 1e-12 for a, b in pairs) else None
    return (ms_rows - ms_error) / denom


def agreement_metrics(pairs: Iterable[tuple[float, float]]) -> dict[str, float | int | None]:
    """Descriptive error + agreement metrics for automatic vs comparator.

    Signed error convention: automatic - comparator.
    Bland-Altman limits use the sample SD of signed differences.
    """
    clean = _finite_pairs(pairs)
    n = len(clean)
    empty: dict[str, float | int | None] = {
        "n": 0,
        "mae_deg": None,
        "median_ae_deg": None,
        "rmse_deg": None,
        "bias_deg": None,
        "sd_error_deg": None,
        "loa_lower_deg": None,
        "loa_upper_deg": None,
        "within_2_deg_pct": None,
        "within_5_deg_pct": None,
        "pearson_r": None,
        "ccc": None,
        "icc_a1": None,
        "mean_automatic_deg": None,
        "mean_comparator_deg": None,
    }
    if not clean:
        return empty

    errors = [a - b for a, b in clean]
    abs_errors = [abs(v) for v in errors]
    bias = statistics.fmean(errors)
    sd = statistics.stdev(errors) if n >= 2 else 0.0
    r = pearson_r(clean)
    ccc = concordance_correlation_coefficient(clean)
    icc = icc_a1(clean)

    return {
        "n": n,
        "mae_deg": round(statistics.fmean(abs_errors), 4),
        "median_ae_deg": round(float(statistics.median(abs_errors)), 4),
        "rmse_deg": round(math.sqrt(statistics.fmean(v * v for v in errors)), 4),
        "bias_deg": round(bias, 4),
        "sd_error_deg": round(sd, 4),
        "loa_lower_deg": round(bias - 1.96 * sd, 4),
        "loa_upper_deg": round(bias + 1.96 * sd, 4),
        "within_2_deg_pct": round(sum(v <= 2.0 for v in abs_errors) / n * 100.0, 2),
        "within_5_deg_pct": round(sum(v <= 5.0 for v in abs_errors) / n * 100.0, 2),
        "pearson_r": None if r is None else round(r, 6),
        "ccc": None if ccc is None else round(ccc, 6),
        "icc_a1": None if icc is None else round(icc, 6),
        "mean_automatic_deg": round(statistics.fmean(a for a, _ in clean), 4),
        "mean_comparator_deg": round(statistics.fmean(b for _, b in clean), 4),
    }


def bland_altman_points(pairs: Iterable[tuple[float, float]]) -> list[dict[str, float]]:
    clean = _finite_pairs(pairs)
    return [
        {
            "mean_deg": round((automatic + comparator) / 2.0, 4),
            "difference_deg": round(automatic - comparator, 4),
        }
        for automatic, comparator in clean
    ]
