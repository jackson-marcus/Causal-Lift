"""Targeting-policy simulator: turn uplift scores into a business decision.

Given per-unit uplift scores on a held-out RCT sample, estimate incremental
profit as a function of the fraction of customers targeted:

    profit(f) = margin * incremental_outcomes(top f) - cost * n_contacted(top f)

using the Qini construction (control-scaled incremental outcomes). Output is
the full curve plus the argmax — "contact the top X%, expect $Y".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causalift.evaluation.qini import qini_curve
from causalift.settings import get_config


@dataclass
class PolicyPoint:
    fraction: float
    n_contacted: int
    incremental_outcomes: float
    profit: float


@dataclass
class PolicyResult:
    points: list[PolicyPoint]
    best: PolicyPoint
    margin: float
    cost: float

    def as_dict(self) -> dict:
        return {
            "margin_per_conversion": self.margin,
            "cost_per_contact": self.cost,
            "best": vars(self.best),
            "curve": [vars(p) for p in self.points],
        }


def simulate(
    uplift: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    margin: float | None = None,
    cost: float | None = None,
    population: int | None = None,
) -> PolicyResult:
    cfg = get_config()["policy"]
    margin = cfg["margin_per_conversion"] if margin is None else margin
    cost = cfg["cost_per_contact"] if cost is None else cost
    n = len(uplift) if population is None else population
    scale = n / len(uplift)

    fractions, qini = qini_curve(uplift, treatment, outcome, n_points=51)
    points = []
    for frac, inc in zip(fractions, qini, strict=True):
        contacted = int(frac * n)
        profit = margin * inc * scale - cost * contacted
        points.append(
            PolicyPoint(
                fraction=round(float(frac), 3),
                n_contacted=contacted,
                incremental_outcomes=round(float(inc * scale), 2),
                profit=round(float(profit), 2),
            )
        )
    best = max(points, key=lambda p: p.profit)
    return PolicyResult(points=points, best=best, margin=margin, cost=cost)
