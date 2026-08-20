"""Shared fixtures: small known-truth synthetic uplift data."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_synthetic import generate


@pytest.fixture(scope="session")
def synthetic():
    """(df, true_tau) — CATE estimation is data-hungry, so the fixture is
    sized where tau recovery is reliably strong (measured: X-learner corr
    ~0.18 at 6k rows vs ~0.50 at 15k)."""
    return generate(15_000, seed=7)


@pytest.fixture(scope="session")
def features(synthetic):
    df, _ = synthetic
    return [c for c in df.columns if c not in ("treatment", "outcome")]
