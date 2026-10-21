"""Budget-constrained targeting.

Two constraints make this different from "sort by uplift":
  1. a fixed budget, so expensive members are not automatically better
  2. a fixed touch cost, so the same uplift buys different ROI per member
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Allocation:
    chosen: np.ndarray
    spend: float
    budget: float
    expected_uplift: float
    n_chosen: int

    @property
    def utilisation(self) -> float:
        return round(self.spend / self.budget, 4) if self.budget else 0.0


def allocate_by_ratio(uplift: np.ndarray, cost: np.ndarray, budget: float) -> Allocation:
    """Greedy by uplift-per-yuan, restricted to positive-uplift members.

    Members with negative predicted uplift are never worth touching: paying to
    lower their conversion is worse than doing nothing, so they are excluded
    before the budget is spent.
    """
    uplift = np.asarray(uplift, dtype=float)
    cost = np.asarray(cost, dtype=float)
    candidates = np.where(uplift > 0)[0]
    if len(candidates) == 0:
        return Allocation(np.array([], dtype=int), 0.0, budget, 0.0, 0)

    score = uplift[candidates] / np.maximum(cost[candidates], 1e-9)
    order = candidates[np.argsort(-score, kind="mergesort")]

    cum = np.cumsum(cost[order])
    k = int(np.searchsorted(cum, budget, side="right"))
    chosen = order[:k]
    return Allocation(
        chosen=chosen,
        spend=round(float(cost[chosen].sum()), 2),
        budget=budget,
        expected_uplift=round(float(uplift[chosen].sum()), 4),
        n_chosen=int(len(chosen)),
    )


def allocate_by_threshold(uplift: np.ndarray, cost: np.ndarray, budget: float) -> Allocation:
    """Lagrangian form: keep members with uplift - lambda*cost > 0, bisect lambda.

    Higher lambda is stricter (fewer, higher-uplift members), so a spend over
    budget means lambda must increase.
    """
    uplift = np.asarray(uplift, dtype=float)
    cost = np.asarray(cost, dtype=float)
    max_ratio = float((uplift / np.maximum(cost, 1e-9)).max())
    lo, hi = 0.0, max(max_ratio, 1e-9)
    best = np.where(uplift > 0)[0]
    for _ in range(80):
        lam = (lo + hi) / 2
        mask = uplift - lam * cost > 0
        spend = float(cost[mask].sum())
        if spend > budget:
            lo = lam
        else:
            hi = lam
            best = np.where(mask)[0]
    return Allocation(
        chosen=best,
        spend=round(float(cost[best].sum()), 2),
        budget=budget,
        expected_uplift=round(float(uplift[best].sum()), 4),
        n_chosen=int(len(best)),
    )


def _observed_diff(
    y: np.ndarray, t: np.ndarray, mask: np.ndarray, n_resamples: int = 1000, seed: int = 42
) -> tuple[float | None, float | None, float | None]:
    """Observed treated-minus-control rate with a percentile bootstrap CI.

    A point estimate on a few hundred selected members is too noisy to rank
    policies on its own; the interval makes that visible.
    """
    if y is None or t is None or not mask.any():
        return None, None, None
    yy, tt = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    sel_t = mask & (tt == 1)
    sel_c = mask & (tt == 0)
    if not sel_t.any() or not sel_c.any():
        return None, None, None

    yt, yc = yy[sel_t], yy[sel_c]
    point = float(yt.mean() - yc.mean())
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        diffs[i] = rng.choice(yt, len(yt), replace=True).mean() - rng.choice(yc, len(yc), replace=True).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(point, 4), round(float(lo), 4), round(float(hi), 4)


def compare_policies(
    uplift: np.ndarray,
    cost: np.ndarray,
    budget: float,
    baseline_score: np.ndarray | None = None,
    segment_score: np.ndarray | None = None,
    y: np.ndarray | None = None,
    t: np.ndarray | None = None,
) -> list[dict]:
    """Policies on the same budget, with predicted AND observed uplift.

    Predicted uplift is what the model believes; observed uplift is what
    actually happened to the selected members in the holdout set. Reporting both
    is the only way to tell a real targeting gain from a modelling artefact.
    """
    uplift = np.asarray(uplift, dtype=float)
    cost = np.asarray(cost, dtype=float)
    n = len(uplift)

    def observed(mask: np.ndarray) -> tuple[float | None, float | None, float | None]:
        return _observed_diff(y, t, mask)

    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    cum_rand = np.cumsum(cost[perm])
    k_rand = int(np.searchsorted(cum_rand, budget, side="right"))
    rand_mask = np.zeros(n, dtype=bool)
    rand_mask[perm[:k_rand]] = True

    ratio = allocate_by_ratio(uplift, cost, budget)
    threshold = allocate_by_threshold(uplift, cost, budget)

    entries = [
        ("random", rand_mask),
        ("uplift_per_yuan", np.isin(np.arange(n), ratio.chosen)),
        ("uplift_threshold", np.isin(np.arange(n), threshold.chosen)),
    ]

    if baseline_score is not None:
        order = np.argsort(-np.asarray(baseline_score, dtype=float), kind="mergesort")
        cum = np.cumsum(cost[order])
        k = int(np.searchsorted(cum, budget, side="right"))
        mask = np.zeros(n, dtype=bool)
        mask[order[:k]] = True
        entries.append(("highest_predicted_conversion", mask))

    if segment_score is not None:
        order = np.argsort(-np.asarray(segment_score, dtype=float), kind="mergesort")
        cum = np.cumsum(cost[order])
        k = int(np.searchsorted(cum, budget, side="right"))
        mask = np.zeros(n, dtype=bool)
        mask[order[:k]] = True
        entries.append(("segment_uplift", mask))

    rows = []
    for name, mask in entries:
        spend = float(cost[mask].sum())
        pred = float(uplift[mask].sum())
        observed_point, observed_lo, observed_hi = observed(mask)
        rows.append(
            {
                "policy": name,
                "n": int(mask.sum()),
                "spend": round(spend, 2),
                "predicted_uplift": round(pred, 4),
                "observed_uplift": observed_point,
                "observed_ci95": [observed_lo, observed_hi],
                "uplift_per_yuan": round(pred / spend, 6) if spend else 0.0,
            }
        )
    return sorted(rows, key=lambda r: -r["predicted_uplift"])
