"""Meta-learner correctness: recover the known effect structure on synthetic data."""

import numpy as np
import pytest

from causalift.models.base import UpliftModel
from causalift.models.s_learner import SLearner
from causalift.models.t_learner import TLearner
from causalift.models.x_learner import XLearner

LEARNERS = [SLearner, TLearner, XLearner]


@pytest.mark.parametrize("cls", LEARNERS)
def test_satisfies_protocol(cls):
    assert isinstance(cls(), UpliftModel)


@pytest.mark.parametrize("cls", LEARNERS)
def test_recovers_positive_ate_direction(cls, synthetic, features):
    df, tau = synthetic
    model = cls().fit(df[features], df["treatment"].to_numpy(), df["outcome"].to_numpy())
    pred = model.predict_uplift(df[features])
    assert pred.shape == (len(df),)
    # Mean predicted effect should have the same sign as the true ATE.
    assert np.sign(pred.mean()) == np.sign(tau.mean())


@pytest.mark.parametrize("cls", [TLearner, XLearner])
def test_correlates_with_true_tau(cls, synthetic, features):
    df, tau = synthetic
    model = cls().fit(df[features], df["treatment"].to_numpy(), df["outcome"].to_numpy())
    pred = model.predict_uplift(df[features])
    corr = np.corrcoef(pred, tau)[0, 1]
    assert corr > 0.3, f"{cls.__name__} tau correlation too weak: {corr:.3f}"


def test_x_learner_ranks_persuadables_above_sleeping_dogs(synthetic, features):
    df, tau = synthetic
    model = XLearner().fit(df[features], df["treatment"].to_numpy(), df["outcome"].to_numpy())
    pred = model.predict_uplift(df[features])
    persuadable = pred[tau > 0.05].mean()
    sleeping_dog = pred[tau < -0.02].mean()
    assert persuadable > sleeping_dog
