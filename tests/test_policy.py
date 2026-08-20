"""Policy simulator: profit curve shape and cost sensitivity."""

import numpy as np

from causalift.policy.simulator import simulate


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    tau = np.where(rng.random(n) < 0.3, 0.15, 0.0)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.10 + t * tau).astype(int)
    return tau, t, y


def test_best_policy_targets_subset_not_everyone():
    tau, t, y = _toy()
    # With a meaningful contact cost, the optimum should be interior (0 < f < 1).
    result = simulate(tau, t, y, margin=10.0, cost=0.5)
    assert 0.0 < result.best.fraction < 1.0
    assert result.best.profit > 0


def test_zero_cost_pushes_towards_full_targeting():
    tau, t, y = _toy()
    free = simulate(tau, t, y, margin=10.0, cost=0.0)
    costly = simulate(tau, t, y, margin=10.0, cost=2.0)
    assert free.best.fraction >= costly.best.fraction


def test_population_scaling():
    tau, t, y = _toy()
    small = simulate(tau, t, y, margin=10.0, cost=0.5, population=1000)
    large = simulate(tau, t, y, margin=10.0, cost=0.5, population=100_000)
    assert large.best.profit > small.best.profit
    assert large.best.n_contacted > small.best.n_contacted


def test_curve_starts_at_zero_profit():
    tau, t, y = _toy()
    result = simulate(tau, t, y, margin=10.0, cost=0.5)
    assert result.points[0].fraction == 0.0
    assert result.points[0].profit == 0.0
