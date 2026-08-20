"""X-learner (Künzel et al. 2019): imputed individual effects + crossed models.

Stage 1: outcome models f1 (treated), f0 (control), as in the T-learner.
Stage 2: impute individual effects
    D1 = y1 - f0(X1)   (treated units: observed minus counterfactual)
    D0 = f1(X0) - y0   (control units: counterfactual minus observed)
and regress tau1 ~ X1 on D1, tau0 ~ X0 on D0.
Stage 3: tau(x) = g(x)·tau0(x) + (1-g(x))·tau1(x). In a randomized experiment
with a balanced design g(x) is the constant propensity.

Usually the strongest meta-learner when arms are imbalanced or effects are
heterogeneous — which is the whole point of uplift modeling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causalift.models.base import make_effect_model, make_outcome_model


class XLearner:
    name = "x_learner"

    def __init__(self, propensity: float | None = None):
        self.model_treated = make_outcome_model()
        self.model_control = make_outcome_model()
        self.tau_treated = make_effect_model()
        self.tau_control = make_effect_model()
        self.propensity = propensity  # None -> estimated from data at fit time

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> XLearner:
        mask = t.astype(bool)
        x1, y1 = x[mask], y[mask]
        x0, y0 = x[~mask], y[~mask]

        self.model_treated.fit(x1, y1)
        self.model_control.fit(x0, y0)

        d1 = y1 - self.model_control.predict_proba(x1)[:, 1]
        d0 = self.model_treated.predict_proba(x0)[:, 1] - y0

        self.tau_treated.fit(x1, d1)
        self.tau_control.fit(x0, d0)

        if self.propensity is None:
            self.propensity = float(mask.mean())
        return self

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        g = self.propensity
        return g * self.tau_control.predict(x) + (1 - g) * self.tau_treated.predict(x)
