# CausaliFT — Causal Uplift Modeling & Campaign Optimization

<div align="center">

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-blue.svg?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

> **Production-grade causal machine learning platform implementing S, T, and X meta-learners to estimate Individual Treatment Effects (ITE/CATE) and optimize marketing campaign targeting policies.**

---

## 📖 Executive Summary & Value Proposition

**`causalift`** is a production-grade, end-to-end machine learning system built with strict engineering discipline, reproducible pipelines, and enterprise MLOps best practices. It bridges the gap between theoretical statistical rigor and high-availability operational microservices.

## 🎯 Core Methodologies & Mathematical Foundations

### 1. Causal Meta-Learners
- **S-Learner (Single Model):** Models response $Y = f(X, W)$ with treatment indicator $W \in \{0, 1\}$, estimating $	au(X) = f(X, 1) - f(X, 0)$.
- **T-Learner (Two Models):** Fits separate LightGBM models $\mu_1(X) = \mathbb{E}[Y \mid X, W=1]$ and $\mu_0(X) = \mathbb{E}[Y \mid X, W=0]$, deriving $	au(X) = \mu_1(X) - \mu_0(X)$.
- **X-Learner (Counterfactual Residuals):** Imputes counterfactuals, fits second-stage propensity-weighted models $D_1 = Y_1 - \hat{\mu}_0(X_1)$ and $D_0 = \hat{\mu}_1(X_0) - Y_0$, and blends predictions:
$$\hat{	au}(X) = e(X)\hat{	au}_0(X) + (1 - e(X))\hat{	au}_1(X)$$

### 2. Uplift Evaluation: Qini Curves & AUUC
- Evaluates uplift models using the cumulative Qini curve:
$$Q(u) = n_{t, 1}(u) - n_{c, 1}(u) \cdot rac{N_t}{N_c}$$
- Computes Area Under the Uplift Curve (AUUC) and normalized Qini scores with 1,000-sample bootstrap confidence intervals.

### 3. Business Policy & ROI Simulation
- Simulates campaign profitability under finite budgets:
$$\mathbb{E}[	ext{Profit}] = \sum_{i \in 	ext{Targeted}} \left( \hat{	au}(X_i) \cdot 	ext{Margin} - 	ext{Cost}_{	ext{treatment}} ight)$$
- Identifies optimal targeting cutoffs to avoid treating *Sleeping Dogs* (users negatively impacted by outreach) and *Sure Things* (users who convert anyway).

## 📊 Architecture & Pipeline

```mermaid
flowchart LR
    RCT[MineThatData RCT / Synthetic Data] --> V[Pandera Feature Validation]
    V --> Meta[Meta-Learner Engine<br/>S / T / X Learners]
    Meta --> Eval[Qini & AUUC Evaluator<br/>Bootstrap 95% CI]
    Eval --> Pol[Targeting Policy Simulator<br/>ROI & Budget Optimizer]
    Pol --> API[FastAPI :8020] --> UI[Streamlit Uplift Studio :8521]
```

## 🛠️ Tech Stack & Engineering Standards
- **Core ML:** Python 3.12, LightGBM 4.3+, Scikit-Learn 1.5+, Pandera
- **Serving & UI:** FastAPI, Streamlit, MLflow
- **Testing:** Pytest (100% test coverage on learners, policy, and Qini evaluation)


---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites & Environment Setup
Using **[uv](https://docs.astral.sh/uv/)** for lightning-fast, reproducible dependency resolution:

```bash
# Clone the repository
git clone https://github.com/jackson-marcus/causalift.git
cd causalift

# Install dependencies and pre-commit hooks
uv sync --group dev
```

### 2. Run Test Suite & Code Quality Checks
```bash
# Run unit & integration tests with coverage
uv run pytest --cov

# Run ruff linter and formatting checks
uv run ruff check .
uv run ruff format --check .
```

### 3. Launch Services Locally
```bash
# Start FastAPI REST API (listening on port :8020)
make api
# Or: uv run uvicorn causalift.api.main:app --reload --port 8020

# Start interactive Streamlit dashboard (listening on port :8521)
make ui

# Launch local MLflow Experiment Tracking UI (listening on port :5002)
make mlflow
```

### 4. Run with Docker Compose
```bash
# Spin up the complete microservice stack
docker compose up --build
```

---

## 📂 Repository Layout

```
causalift/
├── .github/workflows/ci.yml       # GitHub Actions CI pipeline (lint, test, build)
├── configs/                      # Configuration files and hyperparameters
├── data/                         # Data directory (raw, interim, processed)
├── scripts/                      # Data generators and operational scripts
├── src/causalift/               # Core Python package
│   ├── api/                      # FastAPI routes, schemas, and endpoints
│   ├── models/                   # Statistical models, ML algorithms, and estimators
│   ├── ui/                       # Streamlit interactive application
│   └── settings.py               # Centralized configuration & environment loader
├── tests/                        # Comprehensive Pytest suite
├── docker-compose.yml            # Multi-service container orchestration
├── Dockerfile                    # Container definition for API service
├── Makefile                      # Standardized project tasks
└── pyproject.toml                # Pinned dependencies and tool configs
```

---

## 👤 Author & Contact

**Jackson Marcus**
- **Email:** [jackson.marcus.work@gmail.com](mailto:jackson.marcus.work@gmail.com)
- **Upwork:** [Jackson Marcus on Upwork](https://www.upwork.com/freelancers/~012235717501ad9c7b)
- **GitHub:** [@jackson-marcus](https://github.com/jackson-marcus)

*Available for machine learning engineering, MLOps, data science, and AI system architecture consulting and contract engagements.*

