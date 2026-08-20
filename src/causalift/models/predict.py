"""Serving-side loader for the persisted champion uplift model."""

from __future__ import annotations

import functools
import pickle

import pandas as pd

from causalift.settings import get_config, resolve_path


@functools.lru_cache(maxsize=1)
def load_champion() -> dict:
    path = resolve_path(get_config()["data"]["artifacts_dir"]) / "model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No champion at {path}. Run `python -m causalift.models.train --save-best` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def score_frame(df: pd.DataFrame) -> pd.Series:
    bundle = load_champion()
    features = bundle["features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    uplift = bundle["model"].predict_uplift(df[features])
    return pd.Series(uplift, index=df.index, name="uplift")
