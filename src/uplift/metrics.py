"""Uplift evaluation.

The point of this module: a model with AUC 0.85 can have zero uplift power.
Ranking by predicted conversion is not the same as ranking by predicted
*incremental* conversion, and Qini is what exposes the difference.
"""

from __future__ import annotations

import numpy as np


def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative incremental conversions, ordered by descending score.

    Incremental conversions at rank k =
        conversions in treated - conversions in control * (n_treated / n_control)
    The second term removes the part that would have happened anyway.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    y, t = y[order], t[order]

    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * (1 - t))

    with np.errstate(divide="ignore", invalid="ignore"):
        adjusted = y_t - y_c * (n_t / np.maximum(n_c, 1))
    adjusted = np.nan_to_num(adjusted)
    frac = np.arange(1, len(y) + 1) / len(y)
    return frac, adjusted


def qini_coefficient(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    """Area between the Qini curve and the random-targeting diagonal, normalised.

    1.0 = as good as knowing each member's true effect, 0.0 = random,
    negative = worse than random. Subtracting the diagonal matters: without it
    even a reversed ranking scores positive, because the curve itself stays
    above zero whenever the campaign has any total effect.
    """
    frac, adjusted = qini_curve(y, t, score)
    model_area = float(np.trapezoid(adjusted, frac))
    total = float(adjusted[-1])
    random_area = 0.5 * total  # diagonal from (0,0) to (1,total)

    oracle_score = np.asarray(y) * (2 * np.asarray(t) - 1)
    _, adj_oracle = qini_curve(y, t, oracle_score)
    oracle_area = float(np.trapezoid(adj_oracle, frac))

    denom = oracle_area - random_area
    if denom == 0:
        return 0.0
    return round((model_area - random_area) / denom, 4)


def auuc(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    """Area under the uplift curve, scaled by population size."""
    frac, adjusted = qini_curve(y, t, score)
    return round(float(np.trapezoid(adjusted, frac)), 4)


def uplift_at_k(y: np.ndarray, t: np.ndarray, score: np.ndarray, k: float = 0.1) -> float:
    """Observed treated-minus-control rate among the top-k scored members."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n = max(1, int(len(y) * k))
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")[:n]
    yt, tt = y[order], t[order]
    p1 = yt[tt == 1].mean() if (tt == 1).any() else 0.0
    p0 = yt[tt == 0].mean() if (tt == 0).any() else 0.0
    return round(float(p1 - p0), 4)


def decile_table(y: np.ndarray, t: np.ndarray, score: np.ndarray, buckets: int = 5) -> list[dict]:
    """Score-sorted buckets with observed uplift, so the curve can be checked."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    chunks = np.array_split(order, buckets)
    out = []
    for i, idx in enumerate(chunks, 1):
        yt, tt = y[idx], t[idx]
        p1 = float(yt[tt == 1].mean()) if (tt == 1).any() else 0.0
        p0 = float(yt[tt == 0].mean()) if (tt == 0).any() else 0.0
        out.append(
            {
                "bucket": i,
                "n": len(idx),
                "treated_rate": round(p1, 4),
                "control_rate": round(p0, 4),
                "uplift": round(p1 - p0, 4),
            }
        )
    return out
