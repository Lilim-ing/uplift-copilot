import numpy as np
import pytest

from uplift.metrics import auuc, decile_table, qini_coefficient, qini_curve, uplift_at_k
from uplift.policy import allocate_by_ratio, allocate_by_threshold, compare_policies
from uplift.power import minimum_detectable_effect, power_given_n, sample_size


def synthetic(n=4000, seed=7, signal=1.0):
    """Confounded-free synthetic where uplift depends on a single feature."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.integers(0, 2, size=n)
    true_uplift = np.clip(0.30 * signal * (x > 0) - 0.05, 0.0, 0.5)
    p = np.clip(0.08 + true_uplift * t, 0, 0.95)
    y = (rng.random(n) < p).astype(float)
    return x.reshape(-1, 1), t, y, true_uplift


def test_qini_curve_ends_at_total_increment():
    _, t, y, _ = synthetic()
    _, adjusted = qini_curve(y, t, np.zeros(len(y)))
    n_t = t.sum()
    n_c = (1 - t).sum()
    expected = y[t == 1].sum() - y[t == 0].sum() * (n_t / n_c)
    assert adjusted[-1] == pytest.approx(expected, abs=1e-6)


def test_perfect_ranking_scores_higher_than_random():
    _, t, y, true_uplift = synthetic()
    good = qini_coefficient(y, t, true_uplift)
    bad = qini_coefficient(y, t, np.random.default_rng(1).random(len(y)))
    # 0.24 is what "knowing the true uplift" achieves on this noisy synthetic:
    # the oracle also captures realisation noise that no model can see.
    assert good > 0.2
    assert abs(bad) < 0.15
    assert good > bad


def test_reversed_ranking_is_negative():
    _, t, y, true_uplift = synthetic()
    reversed_score = -true_uplift
    assert qini_coefficient(y, t, reversed_score) < 0


def test_uplift_at_k_uses_treated_and_control():
    _, t, y, true_uplift = synthetic()
    top = uplift_at_k(y, t, true_uplift, 0.1)
    full = uplift_at_k(y, t, np.zeros(len(y)), 1.0)
    assert top > full


def test_decile_table_sums_to_population():
    _, t, y, true_uplift = synthetic()
    rows = decile_table(y, t, true_uplift, 5)
    assert sum(r["n"] for r in rows) == len(y)


def test_auuc_is_finite():
    _, t, y, true_uplift = synthetic()
    assert np.isfinite(auuc(y, t, true_uplift))


def test_allocation_respects_budget():
    rng = np.random.default_rng(3)
    uplift = rng.random(1000) * 0.1
    cost = np.full(1000, 8.0)
    alloc = allocate_by_ratio(uplift, cost, 2000)
    assert alloc.spend <= 2000
    assert alloc.n_chosen == 250


def test_allocation_excludes_negative_uplift():
    uplift = np.array([0.05, -0.02, 0.01, -0.03])
    cost = np.full(4, 10.0)
    alloc = allocate_by_ratio(uplift, cost, 1000)
    assert set(alloc.chosen.tolist()) <= {0, 2}


def test_threshold_and_ratio_agree_with_uniform_cost():
    rng = np.random.default_rng(5)
    uplift = rng.random(500) * 0.1
    cost = np.full(500, 5.0)
    a = allocate_by_ratio(uplift, cost, 1000)
    b = allocate_by_threshold(uplift, cost, 1000)
    assert abs(a.n_chosen - b.n_chosen) <= 5


def test_policies_are_comparable_on_observed_uplift():
    rng = np.random.default_rng(11)
    n = 2000
    uplift = rng.random(n) * 0.1
    cost = np.full(n, 10.0)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.1 + uplift * t).astype(float)
    rows = compare_policies(uplift, cost, 3000, y=y, t=t)
    assert any(r["policy"] == "random" for r in rows)
    assert all("observed_uplift" in r for r in rows)


def test_observed_uplift_has_ordered_bootstrap_interval():
    rng = np.random.default_rng(13)
    n = 1200
    uplift = rng.random(n) * 0.1
    cost = np.full(n, 10.0)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.1 + uplift * t).astype(float)
    rows = compare_policies(uplift, cost, 4000, y=y, t=t)
    for row in rows:
        lo, hi = row["observed_ci95"]
        assert lo is not None and hi is not None
        assert lo <= row["observed_uplift"] <= hi


def test_no_conversion_arm_without_a_probability_model():
    rng = np.random.default_rng(17)
    n = 500
    uplift = rng.random(n) * 0.1
    cost = np.full(n, 8.0)
    rows = compare_policies(uplift, cost, 2000, baseline_score=None)
    assert all(r["policy"] != "highest_predicted_conversion" for r in rows)


def test_sample_size_grows_as_mde_shrinks():
    big = sample_size(0.1, 0.02)["per_group"]
    small = sample_size(0.1, 0.05)["per_group"]
    assert big > small


def test_power_and_mde_are_consistent():
    n = sample_size(0.1, 0.03)["per_group"]
    assert power_given_n(n, 0.1, 0.03) >= 0.79
    mde = minimum_detectable_effect(n, 0.1)
    assert 0.02 < mde < 0.04


def test_sample_size_rejects_impossible_target():
    with pytest.raises(ValueError):
        sample_size(0.9, 0.2)
