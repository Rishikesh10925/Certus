# Certus

Distribution-free risk guarantees for automated visual inspection: a conformal risk-control layer over a fine-tuned vision-language building-defect detector, returning a triage decision plus a formal bound on the missed-defect rate.

> **Status: Phase 0 — Foundation** (roadmap Weeks 1–2). No calibration or scoring logic is implemented yet. This repository currently contains the project skeleton, CI, and the data audit tooling only.

## Headline results

_Not yet available — populated after Phase 2 (Figure F3) and Phase 3 (Figures F4–F6). See `paper/intro.tex` for the three locked contribution claims._

| Metric | Value |
|---|---|
| Guaranteed missed-defect rate (ε) | — |
| Human-review fraction (HRF) | — |
| LOSO coverage gap | — |

## Figure

_Figure F4 (the efficiency curve — the paper's centrepiece) will be embedded here once it exists, Week 10._

## Install

```bash
pip install -e ".[dev]"
pre-commit install
```

DVC is initialized (`.dvc/`) but has no remote configured yet — no data has been pulled into this repo, so there's nothing to push. Once a remote is decided (see `data/README.md`):

```bash
dvc remote add -d storage gdrive://<FOLDER_ID>
```

Weights & Biases is a pinned dependency (`wandb`), used from Week 5 onward once `eval/protocol.py` exists and there are runs worth logging. Nothing to initialize yet beyond having an account and running `wandb login` locally when that lands.

## Reproduction

```bash
make setup
make data DATA_PATH=/path/to/cbri/export   # runs scripts/audit_data.py — read-only
make calibrate   # not implemented yet — Phase 1-2, roadmap Weeks 3-8
make loso        # not implemented yet — Phase 3, roadmap Weeks 9-12
make figures     # not implemented yet — regenerates F1-F9 from cached scores
```

## Citation

_Added once the arXiv preprint is public — roadmap Week 16, Gate 2._

## Project structure

See `configs/`, `src/certus/`, and the module docstrings throughout for what each piece is for and when it gets built. The full technical blueprint (`docs/Certus-Technical-Blueprint.docx`) and week-by-week work plan (`docs/Certus-18-Week-Work-Plan.docx`) are the source of truth for scope, gates, and kill criteria.
