"""Qini/AUUC metric sanity: perfect ordering beats random beats inverted."""

import numpy as np

from causalift.evaluation.qini import auuc, bootstrap_auuc_ci, decile_table, qini_curve, uplift_at_k


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    tau = np.where(rng.random(n) < 0.3, 0.15, 0.0)  # 30% persuadables
    t = rng.integers(0, 2, n)
    base = 0.10
    y = (rng.random(n) < base + t * tau).astype(int)
    return tau, t, y


def test_perfect_ordering_beats_random_and_inverted():
    tau, t, y = _toy()
    rng = np.random.default_rng(1)
    a_perfect = auuc(tau + rng.normal(0, 1e-6, len(tau)), t, y)
    a_random = auuc(rng.random(len(tau)), t, y)
    a_inverted = auuc(-tau + rng.normal(0, 1e-6, len(tau)), t, y)
    assert a_perfect > a_random > a_inverted


def test_qini_curve_endpoints():
    tau, t, y = _toy()
    fractions, qini = qini_curve(tau, t, y)
    assert fractions[0] == 0.0 and qini[0] == 0.0
    assert fractions[-1] == 1.0
    # At 100% targeting the value equals the overall scaled incremental outcome.
    assert qini[-1] != 0.0


def test_uplift_at_k_positive_for_good_scores():
    tau, t, y = _toy()
    assert (
        uplift_at_k(tau, t, y, 0.3)
        > uplift_at_k(np.random.default_rng(2).random(len(tau)), t, y, 0.3) - 0.05
    )


def test_decile_table_shape_and_monotonic_trend():
    tau, t, y = _toy()
    table = decile_table(tau, t, y)
    assert len(table) == 10
    top_half = table.head(5)["observed_uplift"].mean()
    bottom_half = table.tail(5)["observed_uplift"].mean()
    assert top_half > bottom_half


def test_bootstrap_ci_brackets_point():
    tau, t, y = _toy()
    point, lo, hi = bootstrap_auuc_ci(tau, t, y, n_bootstrap=50)
    assert lo <= point <= hi
    assert lo > 0, "true signal should have CI excluding zero"
