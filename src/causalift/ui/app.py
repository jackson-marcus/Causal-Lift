"""Streamlit demo: Qini diagnostics + targeting-policy ROI explorer."""

from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = os.environ.get("CAUSALIFT_API_URL", "http://localhost:8060")

st.set_page_config(page_title="causalift", page_icon="🎯", layout="wide")
st.title("🎯 causalift")
st.caption("Who should get the offer? Causal uplift modeling with policy ROI simulation")


def _api_ok() -> bool:
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


if not _api_ok():
    st.error(f"API not reachable at {API_URL}. Start it with `make api` or `docker compose up`.")
    st.stop()

tab_policy, tab_score, tab_model = st.tabs(["Policy simulator", "Score customers", "Model"])

with tab_policy:
    col1, col2, col3 = st.columns(3)
    margin = col1.number_input("Margin per incremental conversion ($)", 1.0, 1000.0, 25.0)
    cost = col2.number_input("Cost per contact ($)", 0.0, 50.0, 0.10, step=0.05)
    population = col3.number_input("Campaign population", 1_000, 10_000_000, 100_000, step=1000)

    if st.button("Simulate policy", type="primary"):
        r = httpx.post(
            f"{API_URL}/policy/simulate",
            json={
                "margin_per_conversion": margin,
                "cost_per_contact": cost,
                "population": int(population),
            },
            timeout=120,
        )
        if r.status_code != 200:
            st.error(r.json().get("detail", r.text))
        else:
            data = r.json()
            curve = pd.DataFrame(data["curve"])
            best = data["best"]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=curve["fraction"] * 100,
                    y=curve["profit"],
                    mode="lines+markers",
                    name="Incremental profit",
                )
            )
            fig.add_vline(x=best["fraction"] * 100, line_dash="dash", line_color="crimson")
            fig.update_layout(
                xaxis_title="% of population contacted",
                yaxis_title="Incremental profit ($)",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Optimal targeting", f"{best['fraction']:.0%}")
            c2.metric("Contacts", f"{best['n_contacted']:,}")
            c3.metric("Expected incremental profit", f"${best['profit']:,.0f}")
            st.caption(
                "Curve built on the held-out RCT split via the Qini construction; "
                "profit = margin x incremental conversions - cost x contacts."
            )

with tab_score:
    st.markdown(
        "Score a customer segment — positive uplift = persuadable, negative = sleeping dog."
    )
    col1, col2 = st.columns(2)
    with col1:
        recency = st.slider("Recency (months since purchase)", 0.0, 24.0, 3.0)
        history = st.number_input("Historical spend ($)", 0.0, 10_000.0, 150.0)
    with col2:
        segment = st.multiselect("Purchase history", ["mens", "womens"], default=["womens"])
        newbie = st.checkbox("New customer (last 12 months)", value=False)

    if st.button("Score"):
        payload = {
            "customers": [
                {
                    "recency": recency,
                    "history": history,
                    "mens": int("mens" in segment),
                    "womens": int("womens" in segment),
                    "newbie": int(newbie),
                }
            ]
        }
        r = httpx.post(f"{API_URL}/score", json=payload, timeout=30)
        if r.status_code != 200:
            st.error(r.json().get("detail", r.text))
        else:
            data = r.json()
            uplift = data["uplift"][0]
            rec = data["recommendation"][0]
            st.metric(
                "Predicted uplift",
                f"{uplift:+.2%}",
                help="Change in outcome probability caused by treatment",
            )
            emoji = {"target": "✅", "avoid (sleeping dog)": "🚫", "indifferent": "😐"}[rec]
            st.markdown(f"### {emoji} Recommendation: **{rec}**")

with tab_model:
    r = httpx.get(f"{API_URL}/model-info", timeout=10)
    if r.status_code != 200:
        st.warning(
            "No champion model saved yet — run `python -m causalift.models.train --save-best`."
        )
    else:
        info = r.json()
        st.subheader(f"Champion: {info['model_name']}")
        st.markdown(f"**{info['n_features']} features:** " + ", ".join(info["features"]))
        st.markdown(
            "- **S-learner**: one model with treatment as a feature\n"
            "- **T-learner**: separate treated/control models\n"
            "- **X-learner**: imputed individual effects + crossed models (usually best)\n\n"
            "Selected by out-of-fold AUUC with bootstrap CIs — see MLflow for runs."
        )
