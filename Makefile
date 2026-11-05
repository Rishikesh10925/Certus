.PHONY: setup data calibrate e1 e2 e3 loso latency figures paper clean

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
	$(PYTHON) scripts/build_service_calibration.py

# Experiments (need the raw images or the score cache, and a GPU for scoring).
e1:
	$(PYTHON) scripts/run_e1.py

e2:
	$(PYTHON) scripts/run_e2.py

e3:
	$(PYTHON) scripts/run_e3.py

loso:
	$(PYTHON) scripts/run_loso_corrections.py

latency:
	$(PYTHON) scripts/measure_latency.py --device cuda

# Figures F1-F9 from the committed results/ only: works on a fresh clone.
figures:
	$(PYTHON) scripts/make_figures.py

# The paper is local (paper/ is gitignored); needs a LaTeX install.
paper:
	$(PYTHON) scripts/make_figures.py --paper
	cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main \
		&& pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
