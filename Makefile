.PHONY: install lint format test data synthetic train api ui mlflow docker-up docker-down

install:
	uv sync --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest --cov

data:
	uv run python -m causalift.data.download
	uv run python -m causalift.data.prepare

synthetic:
	uv run python scripts/make_synthetic.py

train:
	uv run python -m causalift.models.train --save-best

train-synthetic:
	uv run python -m causalift.models.train --data synthetic

api:
	uv run uvicorn causalift.api.main:app --reload --port 8060

ui:
	CAUSALIFT_API_URL=http://localhost:8060 uv run streamlit run src/causalift/ui/app.py --server.port 8561

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5006

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
