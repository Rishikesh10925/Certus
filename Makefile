.PHONY: setup data calibrate loso figures paper clean

PYTHON ?= python
DATA_PATH ?=

setup:
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install
	@if command -v dvc >/dev/null 2>&1; then \
		[ -d .dvc ] || dvc init; \
	else \
		echo "dvc not found on PATH -- it is pinned in pyproject.toml; re-run 'make setup' after installing it"; \
	fi

data:
	@if [ -z "$(DATA_PATH)" ]; then \
		echo "Usage: make data DATA_PATH=/path/to/cbri/export"; exit 1; \
	fi
	$(PYTHON) scripts/audit_data.py --path "$(DATA_PATH)"

calibrate:
	@echo "calibrate: not implemented yet -- Phase 1-2, roadmap Part VI Weeks 3-8"

loso:
	@echo "loso: not implemented yet -- Phase 3, roadmap Part VI Weeks 9-12"

figures:
	@echo "figures: not implemented yet -- see roadmap Part V Section 5.6 for the figure list"

paper:
	@echo "paper: not implemented yet -- will run latexmk on paper/main.tex once it exists"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
