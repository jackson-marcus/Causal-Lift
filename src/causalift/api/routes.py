"""API routes: /score, /policy/simulate, /health, /model-info."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from causalift.models import predict
from causalift.policy.simulator import simulate
from causalift.settings import get_config, resolve_path

logger = logging.getLogger(__name__)
router = APIRouter()


class Customer(BaseModel):
    model_config = {"extra": "allow"}

    recency: float = Field(ge=0, description="Months since last purchase")
    history: float = Field(ge=0, description="Historical spend")
    mens: int = Field(ge=0, le=1, default=0)
    womens: int = Field(ge=0, le=1, default=0)
    newbie: int = Field(ge=0, le=1, default=0)


class ScoreRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=10_000)


class ScoreResponse(BaseModel):
    uplift: list[float]
    model_name: str
    recommendation: list[str]


class PolicyRequest(BaseModel):
    margin_per_conversion: float = Field(gt=0, default=25.0)
    cost_per_contact: float = Field(ge=0, default=0.10)
    population: int | None = Field(default=None, ge=1)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/model-info")
def model_info() -> dict:
    try:
        bundle = predict.load_champion()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "model_name": bundle["name"],
        "n_features": len(bundle["features"]),
        "features": bundle["features"],
    }


@router.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    df = pd.DataFrame([c.model_dump() for c in request.customers])
    try:
        bundle = predict.load_champion()
        # Fill one-hot columns the model expects but the payload omits.
        for col in bundle["features"]:
            if col not in df.columns:
                df[col] = 0
        uplift = predict.score_frame(df)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    def recommend(u: float) -> str:
        if u > 0.01:
            return "target"
        if u < -0.01:
            return "avoid (sleeping dog)"
        return "indifferent"

    return ScoreResponse(
        uplift=[round(float(u), 5) for u in uplift],
        model_name=bundle["name"],
        recommendation=[recommend(u) for u in uplift],
    )


@router.post("/policy/simulate")
def policy_simulate(request: PolicyRequest) -> dict:
    """Simulate targeting policies on the held-out test split."""
    test_path = resolve_path(get_config()["data"]["processed_dir"]) / "test.parquet"
    if not test_path.exists():
        raise HTTPException(
            status_code=503, detail="No test split; run data prepare + train first."
        )
    df = pd.read_parquet(test_path)
    try:
        uplift = predict.score_frame(df).to_numpy()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result = simulate(
        uplift,
        df["treatment"].to_numpy(),
        df["outcome"].to_numpy(),
        margin=request.margin_per_conversion,
        cost=request.cost_per_contact,
        population=request.population,
    )
    out = result.as_dict()
    out["n_eval_rows"] = len(df)
    out["curve"] = out["curve"][::5]  # thin the payload
    if not np.isfinite(out["best"]["profit"]):
        raise HTTPException(status_code=500, detail="Simulation produced non-finite profit")
    return out
