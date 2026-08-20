"""T-learner: separate outcome models per arm; uplift = f1(X) - f0(X)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causalift.models.base import make_outcome_model


class TLearner:
    name = "t_learner"

    def __init__(self):
        self.model_treated = make_outcome_model()
        self.model_control = make_outcome_model()

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> TLearner:
        mask = t.astype(bool)
        self.model_treated.fit(x[mask], y[mask])
        self.model_control.fit(x[~mask], y[~mask])
        return self

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        return self.model_treated.predict_proba(x)[:, 1] - self.model_control.predict_proba(x)[:, 1]
