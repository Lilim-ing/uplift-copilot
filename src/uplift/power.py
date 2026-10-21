"""Experiment sizing for uplift tests."""

from __future__ import annotations

import math
from statistics import NormalDist


def sample_size(baseline: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> dict:
    p1 = float(baseline)
    p2 = p1 + float(mde_abs)
    if not (0 < p1 < 1) or not (0 < p2 < 1):
        raise ValueError("baseline and baseline+mde must fall in (0,1)")
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    p_bar = (p1 + p2) / 2
    numerator = (
        z_a * math.sqrt(2 * p_bar * (1 - p_bar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    per_group = math.ceil(numerator / mde_abs**2)
    return {
        "baseline": round(p1, 4),
        "target": round(p2, 4),
        "mde_abs": round(mde_abs, 4),
        "mde_rel": round(mde_abs / p1, 4),
        "per_group": per_group,
        "total": per_group * 2,
        "alpha": alpha,
        "power": power,
    }


def power_given_n(n_per_group: int, baseline: float, mde_abs: float, alpha: float = 0.05) -> float:
    p1 = float(baseline)
    p2 = p1 + float(mde_abs)
    se = math.sqrt(p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group)
    if se == 0:
        return 1.0
    z = abs(p2 - p1) / se - NormalDist().inv_cdf(1 - alpha / 2)
    return round(NormalDist().cdf(z), 4)


def minimum_detectable_effect(n_per_group: int, baseline: float, power: float = 0.8, alpha: float = 0.05) -> float:
    """Smallest absolute lift this sample size could detect, by bisection."""
    lo, hi = 1e-5, 1 - baseline - 1e-5
    for _ in range(60):
        mid = (lo + hi) / 2
        if power_given_n(n_per_group, baseline, mid, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return round(hi, 4)
