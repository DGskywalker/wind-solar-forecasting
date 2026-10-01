.PHONY: help install data features train evaluate market report test lint docker-build docker-up clean all

PYTHON ?= .venv/bin/python
UV ?= uv

help:
	@echo "Available commands:"
	@echo "  make install      - Install python dependencies with uv in .venv"
	@echo "  make data         - Ingest weather & market data (or generate synthetic baseline)"
	@echo "  make features     - Run physical models and construct feature store"
	@echo "  make train        - Train model ladder (Baselines, GBDT, NGBoost, TFT, Ensemble)"
	@echo "  make evaluate     - Compute CRPS, Pinball, Winkler, reliability diagrams & DM tests"
	@echo "  make market       - Run price impact econometrics (FE, 2SLS, DML) and trading backtest"
	@echo "  make report       - Execute notebooks and compile research report with figures"
	@echo "  make test         - Run pytest suite including strict temporal leakage checks"
	@echo "  make lint         - Run ruff linter and mypy type checks"
	@echo "  make docker-build - Build Docker container images"
	@echo "  make docker-up    - Run Postgres, MLflow, API, and Dashboard via Docker Compose"
	@echo "  make all          - Run complete end-to-end research pipeline"

install:
	@echo "==> Setting up environment..."
	$(UV) venv --python 3.11 .venv
	$(UV) pip install --python $(PYTHON) -e ".[dev]"

data:
	@echo "==> Ingesting weather and power market data..."
	$(PYTHON) -m wind_solar_forecast.pipeline.ingest --sample

features:
	@echo "==> Building feature store..."
	$(PYTHON) -m wind_solar_forecast.features.build_features

train:
	@echo "==> Training probabilistic generation model ladder..."
	$(PYTHON) -m wind_solar_forecast.pipeline.train --zone DE_LU --horizon D-1

evaluate:
	@echo "==> Running evaluation, reliability diagnostics, and Diebold-Mariano tests..."
	$(PYTHON) -m wind_solar_forecast.pipeline.evaluate

market:
	@echo "==> Estimating price impact regressions and running trading backtest..."
	$(PYTHON) -m wind_solar_forecast.market.price_impact
	$(PYTHON) -m wind_solar_forecast.market.backtest

report:
	@echo "==> Compiling research report and executing research notebooks..."
	$(PYTHON) -m wind_solar_forecast.pipeline.report

test:
	@echo "==> Running unit, integration, and temporal leakage tests..."
	$(PYTHON) -m pytest tests/ -v --cov=src/wind_solar_forecast

lint:
	@echo "==> Running Ruff and Mypy..."
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m mypy src/

docker-build:
	@echo "==> Building Docker images..."
	docker compose -f docker/docker-compose.yml build

docker-up:
	@echo "==> Launching stack (TimescaleDB, MLflow, API, Dashboard)..."
	docker compose -f docker/docker-compose.yml up -d

clean:
	@echo "==> Cleaning temporary caches..."
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} +

all: data features train evaluate market report test
	@echo "==> Complete end-to-end research pipeline executed successfully."
