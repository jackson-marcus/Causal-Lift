"""Train and compare S/T/X meta-learners with MLflow tracking.

Uplift scores for evaluation are produced out-of-fold (cross-fitting): the
model scoring a unit never saw it in training, which keeps the Qini/AUUC
estimates honest.

Usage:
    python -m causalift.models.train [--data processed|synthetic] [--save-best]
"""

from __future__ import annotations

import argparse
import logging
import pickle

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from causalift.data.prepare import feature_columns
from causalift.evaluation.qini import bootstrap_auuc_ci, decile_table, uplift_at_k
from causalift.models.s_learner import SLearner
from causalift.models.t_learner import TLearner
from causalift.models.x_learner import XLearner
from causalift.settings import get_config, get_settings, resolve_path

logger = logging.getLogger(__name__)

LEARNERS = {"s_learner": SLearner, "t_learner": TLearner, "x_learner": XLearner}


def oof_uplift(learner_cls, df: pd.DataFrame, features: list[str], n_splits: int) -> np.ndarray:
    """Out-of-fold uplift scores via cross-fitting."""
    scores = np.zeros(len(df))
    strata = df["treatment"].astype(str) + df["outcome"].astype(str)
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=get_config()["data"]["random_state"]
    )
    for train_idx, score_idx in cv.split(df, strata):
        fold = df.iloc[train_idx]
        model = learner_cls().fit(
            fold[features], fold["treatment"].to_numpy(), fold["outcome"].to_numpy()
        )
        scores[score_idx] = model.predict_uplift(df.iloc[score_idx][features])
    return scores


def train_all(df: pd.DataFrame, true_tau: np.ndarray | None = None) -> dict[str, dict]:
    cfg = get_config()
    features = [c for c in feature_columns(df) if c != "true_tau"]
    t = df["treatment"].to_numpy()
    y = df["outcome"].to_numpy()

    results: dict[str, dict] = {}
    for name, cls in LEARNERS.items():
        with mlflow.start_run(run_name=name):
            uplift = oof_uplift(cls, df, features, cfg["training"]["n_splits"])
            point, lo, hi = bootstrap_auuc_ci(
                uplift, t, y, n_bootstrap=cfg["evaluation"]["n_bootstrap"]
            )
            metrics = {
                "auuc": point,
                "auuc_ci_lo": lo,
                "auuc_ci_hi": hi,
                "uplift_at_10pct": uplift_at_k(uplift, t, y, 0.10),
                "uplift_at_30pct": uplift_at_k(uplift, t, y, 0.30),
            }
            if true_tau is not None:
                metrics["tau_correlation"] = float(np.corrcoef(uplift, true_tau)[0, 1])
            mlflow.log_params({"learner": name, "n_rows": len(df), "n_features": len(features)})
            mlflow.log_metrics(metrics)
            deciles = decile_table(uplift, t, y)
            logger.info(
                "%s: AUUC=%.2f [%.2f, %.2f] uplift@10%%=%.4f%s",
                name,
                point,
                lo,
                hi,
                metrics["uplift_at_10pct"],
                f" tau_corr={metrics.get('tau_correlation', float('nan')):.3f}"
                if true_tau is not None
                else "",
            )
            results[name] = {"metrics": metrics, "uplift": uplift, "deciles": deciles}
    return results


def save_best(df: pd.DataFrame, results: dict[str, dict]) -> str:
    """Refit the AUUC-best learner on all rows and persist it for serving."""
    best = max(results, key=lambda k: results[k]["metrics"]["auuc"])
    features = [c for c in feature_columns(df) if c != "true_tau"]
    model = LEARNERS[best]().fit(df[features], df["treatment"].to_numpy(), df["outcome"].to_numpy())
    artifacts = resolve_path(get_config()["data"]["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    with open(artifacts / "model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": features, "name": best}, f)
    results[best]["deciles"].to_parquet(artifacts / "deciles.parquet", index=False)
    logger.info("Saved champion %s -> %s", best, artifacts / "model.pkl")
    return best


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="processed", choices=["processed", "synthetic"])
    parser.add_argument("--save-best", action="store_true")
    args = parser.parse_args()

    cfg = get_config()
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(cfg["training"]["experiment_name"])

    processed = resolve_path(cfg["data"]["processed_dir"])
    if args.data == "synthetic":
        df = pd.read_parquet(processed / "synthetic.parquet")
        true_tau = df["true_tau"].to_numpy()
    else:
        df = pd.read_parquet(processed / "train.parquet")
        true_tau = None

    results = train_all(df, true_tau)
    if args.save_best:
        save_best(df, results)


if __name__ == "__main__":
    main()
