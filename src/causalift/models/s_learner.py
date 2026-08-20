"""S-learner: one model over (X, treatment); uplift = f(X,1) - f(X,0).

Simplest meta-learner. Its weakness — the treatment indicator competes with
every covariate for splits, so small effects can vanish — is exactly why we
compare it against T/X-learners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causalift.models.base import make_outcome_model


class SLearner:
    name = "s_learner"

    def __init__(self):
        self.model = make_outcome_model()

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> SLearner:
        xt = x.copy()
        xt["__treatment__"] = t.astype(int)
        self.model.fit(xt, y)
        return self

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        x1 = x.copy()
        x1["__treatment__"] = 1
        x0 = x.copy()
        x0["__treatment__"] = 0
        return self.model.predict_proba(x1)[:, 1] - self.model.predict_proba(x0)[:, 1]
