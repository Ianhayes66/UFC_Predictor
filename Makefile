PYTHON ?= python3.11
VENV ?= .venv
PIP ?= $(VENV)/bin/pip
PY ?= $(VENV)/bin/python

.DEFAULT_GOAL := help

help:
@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?##"}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'

$(VENV): ## Create virtualenv
@$(PYTHON) -m venv $(VENV)
@$(PIP) install --upgrade pip

setup: $(VENV) ## Install project dependencies and tooling
@$(PIP) install -e .[dev]
@$(VENV)/bin/pre-commit install

fmt: ## Format code using black
@$(VENV)/bin/black src tests

lint: ## Run ruff
@$(VENV)/bin/ruff check src tests

mypy: ## Run mypy static type checks
@$(VENV)/bin/mypy src

check: fmt lint mypy test ## Run full lint + test suite

pytest: ## Run pytest
@$(VENV)/bin/pytest

test: pytest ## Alias for pytest

data: ## Run data build pipeline on sample fixtures
@$(PY) -m ufc_winprob.pipelines.build_dataset

features: ## Build feature set
@$(PY) -m ufc_winprob.pipelines.build_dataset --stage features

train: ## Train models on prepared data
@$(PY) -m ufc_winprob.models.training

predict: ## Generate predictions for upcoming fights
@$(PY) -m ufc_winprob.models.predict

backtest: ## Run backtest simulations
@$(PY) -m ufc_winprob.models.backtest

refresh: ## Execute daily refresh workflow and publish reports
@$(PY) -m ufc_winprob.pipelines.daily_refresh

api: ## Launch FastAPI server
@$(PY) -m ufc_winprob.api.main

ui: ## Launch Streamlit dashboard
@STREAMLIT_ENV=$(VENV) $(VENV)/bin/streamlit run src/ufc_winprob/ui/app.py

jobs: ## Run scheduler service
@$(PY) -m ufc_winprob.jobs.scheduler

demo: ## End-to-end demonstration on synthetic data
@$(PY) src/ufc_winprob/scripts/run_all.py

clean: ## Remove build artifacts
rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache *.egg-info

.PHONY: help setup fmt lint mypy check pytest test data features train predict backtest refresh api ui jobs demo clean
