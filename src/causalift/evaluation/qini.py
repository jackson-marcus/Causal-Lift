"""Uplift evaluation: Qini curve, AUUC, uplift@k, decile diagnostics.

All metrics work from (uplift_score, treatment, outcome) triples on a held-out
set — the fundamental trick being that individual effects are unobservable, so
quality is measured by *sorting*: if high-scored units are truly more
persuadable, the treated-minus-control gap concentrates at the top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def qini_curve(uplift: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, n_points: int = 101):
    """Cumulative incremental outcomes vs fraction targeted (descending score)."""
    order = np.argsort(-uplift)
    t = treatment[order].astype(float)
    y = outcome[order].astype(float)

    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1 - t)
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1 - t))

    n = len(uplift)
    fractions = np.linspace(0, 1, n_points)
    qini = np.zeros(n_points)
    for i, frac in enumerate(fractions[1:], start=1):
        k = max(int(frac * n) - 1, 0)
        nt, nc = cum_t[k], cum_c[k]
        # Incremental outcomes among top-k, control-scaled to the treated count.
        qini[i] = cum_yt[k] - (cum_yc[k] * nt / nc if nc > 0 else 0.0)
    return fractions, qini


def auuc(uplift: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> float:
    """Area under the Qini curve minus the random-targeting diagonal."""
    fractions, qini = qini_curve(uplift, treatment, outcome)
    random_line = fractions * qini[-1]
    return float(np.trapezoid(qini - random_line, fractions))


def uplift_at_k(
    uplift: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, k: float = 0.1
) -> float:
    """Observed treated-vs-control outcome gap inside the top-k fraction."""
    order = np.argsort(-uplift)
    top = order[: max(int(k * len(uplift)), 1)]
    t, y = treatment[top], outcome[top]
    if t.sum() == 0 or (1 - t).sum() == 0:
        return 0.0
    return float(y[t == 1].mean() - y[t == 0].mean())


def decile_table(uplift: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, n_bins: int = 10):
    """Per-decile observed uplift — the calibration workhorse plot."""
    df = pd.DataFrame({"uplift": uplift, "t": treatment, "y": outcome})
    df["decile"] = pd.qcut(df["uplift"].rank(method="first"), n_bins, labels=False)
    rows = []
    for decile, part in df.groupby("decile"):
        treated = part[part.t == 1]
        control = part[part.t == 0]
        rows.append(
            {
                "decile": int(decile) + 1,
                "n": len(part),
                "mean_predicted": float(part["uplift"].mean()),
                "observed_uplift": float(
                    (treated.y.mean() if len(treated) else 0.0)
                    - (control.y.mean() if len(control) else 0.0)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("decile", ascending=False).reset_index(drop=True)


def bootstrap_auuc_ci(
    uplift: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """(point, lo, hi) percentile CI for AUUC."""
    rng = np.random.default_rng(seed)
    n = len(uplift)
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        stats.append(auuc(uplift[idx], treatment[idx], outcome[idx]))
    point = auuc(uplift, treatment, outcome)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
