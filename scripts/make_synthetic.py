"""Known-truth synthetic uplift data: heterogeneous CATE with sleeping dogs.

Generates an RCT where the true individual treatment effect tau(x) is known:
- "persuadables": strong positive effect (recent, engaged customers)
- neutral mass: near-zero effect
- "sleeping dogs": negative effect (treatment annoys them)

Because tau is returned, tests can assert that meta-learners actually recover
the effect structure — impossible with real data where tau is never observed.

Usage:
    uv run python scripts/make_synthetic.py [--rows 40000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causalift.settings import get_config, resolve_path


def generate(n: int, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (frame with treatment/outcome/features, true tau per row)."""
    rng = np.random.default_rng(seed)

    recency = rng.exponential(6, n).clip(0, 24)  # months since last purchase
    history = rng.lognormal(4.5, 1.0, n).round(2)  # past spend
    mens = rng.integers(0, 2, n)
    womens = rng.integers(0, 2, n)
    newbie = rng.integers(0, 2, n)
    engagement = np.clip(rng.normal(0.5, 0.2, n) - recency / 60, 0, 1)

    # True heterogeneous effect on outcome probability.
    tau = (
        0.08 * (recency < 4) * (engagement > 0.45)  # persuadables
        - 0.05 * ((recency > 12) & (newbie == 0))  # sleeping dogs
        + 0.02 * womens
        + rng.normal(0, 0.005, n)  # mild idiosyncrasy
    )

    base = np.clip(
        0.06
        + 0.10 * engagement
        + 0.02 * (history > 200)
        - 0.002 * recency
        + rng.normal(0, 0.01, n),
        0.005,
        0.6,
    )

    treatment = rng.integers(0, 2, n)  # 50/50 RCT
    p = np.clip(base + treatment * tau, 0.001, 0.95)
    outcome = (rng.random(n) < p).astype(int)

    df = pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": outcome,
            "recency": recency.round(2),
            "history": history,
            "mens": mens,
            "womens": womens,
            "newbie": newbie,
            "engagement": engagement.round(4),
        }
    )
    return df, tau


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df, tau = generate(args.rows, args.seed)
    out_dir = resolve_path(get_config()["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df["true_tau"] = tau  # kept in the synthetic file only, for benchmarking
    path = out_dir / "synthetic.parquet"
    df.to_parquet(path, index=False)
    ate = float(
        df.loc[df.treatment == 1, "outcome"].mean() - df.loc[df.treatment == 0, "outcome"].mean()
    )
    print(f"Wrote {len(df):,} rows -> {path}")
    print(f"True ATE={tau.mean():+.4f}, empirical diff-in-means={ate:+.4f}")


if __name__ == "__main__":
    main()
