"""The uplift-model contract shared by all meta-learners.

An uplift model estimates the conditional average treatment effect (CATE):
    tau(x) = E[Y | X=x, T=1] - E[Y | X=x, T=0]
i.e. how much the outcome probability changes *because of* the treatment,
per individual — not how likely the outcome is.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from causalift.settings import get_config


@runtime_checkable
class UpliftModel(Protocol):
    name: str

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> UpliftModel: ...

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray: ...


def make_outcome_model() -> LGBMClassifier:
    """Binary-outcome learner used inside every meta-learner."""
    params = get_config()["training"]["lgbm"]
    return LGBMClassifier(**params, random_state=get_config()["data"]["random_state"])


def make_effect_model() -> LGBMRegressor:
    """Regressor for pseudo-effects (X-learner second stage)."""
    params = get_config()["training"]["lgbm"]
    return LGBMRegressor(**params, random_state=get_config()["data"]["random_state"])
