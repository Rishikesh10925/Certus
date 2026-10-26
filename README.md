# Certus

Distribution-free risk guarantees for automated visual inspection: a conformal risk-control layer over a fine-tuned vision-language building-defect detector, returning a triage decision plus a formal, assumption-light bound on the missed-defect rate.

[![CI](https://github.com/Rishikesh10925/Certus/actions/workflows/ci.yml/badge.svg)](https://github.com/Rishikesh10925/Certus/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

> **Status: Phase 2 — Scores & risk control (Week 7 of 18).** Learn-then-Test adds a 95%-confidence version of the guarantee for +4–10 points of human review. The GPT-4V bootstrap labels agree with hand-labelled gold at chance (κ = −0.007); calibrating on them doubles the realised miss rate under the joint guarantee. Conformal risk control holds its missed-defect bound at ε = 0.10–0.01 for all four scores (S1 CLIP, S2 Qwen2-VL, S3 verbal confidence, S4 OWLv2) over 100 seeds; the score decides the human-review cost (7%–100%). Split CP (LAC, APS, RAPS) is validated on real data: all 144 conformal cells of E1 land within ±2 points of nominal coverage over 100 seeds (Figure F2), while the softmax, temperature-scaling and selective-classification baselines miss by up to 20 points. Shift correction is not implemented yet. See [`docs/logbook.md`](docs/logbook.md) for the running record of what's actually been done, week by week.

---

## 1. Problem Statement

Structural-defect detectors already exist for common damage classes — cracks, spalling, efflorescence — and report strong aggregate numbers: 90%+ accuracy, high mAP. None of that is deployable on its own, because it answers a question nobody actually asked. A site engineer signing off on an inspection is personally liable. An insurer pricing a policy needs a loss distribution. A municipal authority approving occupancy needs a documented error bound. All three need the same object: **a bound on the probability that this specific inspection missed something structurally significant.** Aggregate accuracy doesn't provide that. Neither does a softmax probability — it's the output of an exponential normalization, not a calibrated statement about correctness, and it carries no finite-sample guarantee for any single decision.

The practical consequence is that organizations deploy the model and then have a human re-check every image anyway, because nobody can justify trusting any subset of the model's outputs without a guarantee. Automation saves nothing. That's the deployment gap Certus addresses — a statistics problem wearing an engineering costume.

## 2. Why Certus

Conformal risk control, with an asymmetric inspection loss, applied to a vision-language defect detector, evaluated under structured leave-one-site-out (LOSO) distribution shift, reported in terms of **human-review fraction** rather than coverage alone. Each of those pieces exists separately in the conformal-prediction and industrial-inspection literatures. The conjunction does not.

Certus is an applied contribution, not a theoretical one. The conformal machinery it builds on — split conformal prediction, conformal risk control, Learn-then-Test — is mature and taken as given; no new coverage theorem is being proved. The empirical and engineering content is the real work: a systematic comparison of nonconformity scores for this problem, the efficiency curve that tells a deployer what a given risk budget actually costs in human review hours, and a measured account of how badly the guarantee degrades under the site-level shift that inspection systems actually face in the field — plus how much of it three different correction methods recover.

## 3. How the System Works

Two loops — offline calibration and online triage — joined by a random audit stream that keeps ongoing recalibration statistically valid instead of silently biased by the triage policy's own decisions.


The frozen VLM never trains on the critical path — it's scored once per calibration image (cached and reused across every experiment) and once per online image (a single, sub-millisecond forward pass). The calibration split is treated as physically inaccessible to anything that fits a score: no function whose name begins with `fit_` is allowed to see calibration data, which is the discipline that rules out the most common conformal-prediction bug before it can compile.

## 4. Research Objectives

- **O1** — Implement the corrected split-conformal quantile and validate empirical coverage against nominal coverage on exchangeable synthetic and real data (±2 points, 100 seeds).
- **O2** — Implement and compare at least five nonconformity scores on identical splits.
- **O3** — Implement conformal risk control and Learn-then-Test; plot the efficiency curve (human-review fraction vs. guaranteed miss rate) across at least four risk targets.
- **O4** — Run a leave-one-site-out protocol and demonstrate a measurable coverage gap under realistic distribution shift.
- **O5** — Implement and evaluate at least three shift-correction methods (Mondrian, weighted, localized CP), reporting recovered coverage and the efficiency cost of recovery.
- **O6** — Hand-label a gold subset, calibrate exclusively on it, and quantify how much the guarantee degrades when calibrated on machine-bootstrapped labels instead.
- **O7** — Build a FastAPI triage service and a minimal console returning a risk statement per image in under 400 ms at p95.
- **O8** — Publish an arXiv preprint with a fully reproducible repository and release the calibration benchmark with a Zenodo DOI.

### The three contribution claims

Locked in Week 1 (verbatim in [`paper/intro.tex`](paper/intro.tex); not to be changed without deliberately reopening scope):

1. **C1** — A systematic comparison of nonconformity scores for VLM-based defect detection, establishing which score family yields the tightest risk-controlled triage.
2. **C2** — The efficiency curve for risk-controlled inspection: the human-review fraction required to hold missed defects below ε, across two orders of magnitude of ε. This is the figure industry cares about.
3. **C3** — A quantified failure of exchangeability under site-level shift, and a measured comparison of three shift-correction strategies in terms of both recovered coverage and lost efficiency.

## 5. Key Methods

| Method | What it does | Status |
|---|---|---|
| **Split conformal prediction** | Corrected quantile `⌈(n+1)(1−α)⌉/n` on a held-out calibration set; LAC / APS / RAPS nonconformity scores | Not implemented — Week 3–4 |
| **Conformal Risk Control (CRC)** | Extends coverage to any monotone loss, giving `E[L] ≤ ε`; the primary method, since the missed-defect loss is asymmetric | Not implemented — Week 6 |
| **Learn-then-Test (LTT)** | Risk control via Hoeffding–Bentkus p-values and fixed-sequence FWER control, giving `P(R ≤ ε) ≥ 1−δ` — the stronger, regulator-grade statement | Not implemented — Week 7 |
| **Mondrian CP** | Group-conditional calibration, one threshold per site cluster | Not implemented — Week 11 |
| **Weighted CP** | Domain-classifier likelihood-ratio weighting with effective-sample-size diagnostics | Not implemented — Week 11 |
| **Localized CP** | Latent-similarity weighting with a nonlinear transform | Not implemented — Week 12 |

Every guarantee here rests on **exchangeability** — the assumption that calibration and test points are drawn in a way invariant to reordering. Site-level distribution shift is exactly the condition that breaks it, which is what the shift-correction methods above exist to measure and partially recover.

## 6. Experimental Setup and Evaluation Metrics

| Protocol | Purpose |
|---|---|
| **E1 — Validity** | 100-seed coverage check on exchangeable data; must land on the nominal diagonal within ±2 points before anything else proceeds |
| **E2 — Score comparison** | All seven nonconformity scores, 100 seeds, identical splits |
| **E3 — Efficiency curve** | Sweep ε across ≥4 risk targets, human-review fraction with bootstrap CIs |
| **E4 — LOSO shift** | Calibrate on all sites but one, evaluate on the held-out site, rotate; then repeat under each shift-correction method |

| Category | Metric | Definition |
|---|---|---|
| Validity | Empirical marginal coverage | Mean ± SD over 100 seeds against the nominal 1−α line |
| Risk | Realised miss rate | Empirical missed-defect rate at λ̂, must sit at or below ε |
| Efficiency | Human-review fraction (HRF) | (FLAG + ESCALATE) / N, bootstrap 95% CI |
| Conditional quality | SSCV, class-conditional coverage | How badly the marginal guarantee hides conditional failure |
| Shift | Coverage gap under LOSO | Nominal minus empirical coverage on the held-out site |
| System | Latency (p50/p95/p99) | End-to-end triage endpoint response time |

Compared against eleven baselines spanning raw softmax thresholding, temperature scaling, selective classification, deep-ensemble/MC-dropout confidence, standard conformal prediction (LAC/APS/RAPS), and verbalized VLM confidence — see the technical blueprint for the full table.

### Results

*Not yet available.* Populated after Phase 2 (Figure F3, score comparison) and Phase 3 (Figures F4–F6, efficiency curve and LOSO shift study). This table is the target shape for the paper's headline claims — filled in as real numbers land, never with placeholders that look like results.

| Metric | Value | Source |
|---|---|---|
| Guaranteed missed-defect rate (ε) | — | Figure F4 |
| Human-review fraction (HRF) at that ε | — | Figure F4 |
| LOSO coverage gap (uncorrected) | — | Figure F5 |
| Best recovered coverage (shift-corrected) | — | Figure F6 |

## 7. Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| Core language | Python, PyTorch | Frozen VLM inference only — no training on the critical path |
| Experiment config | Hydra | Every experiment is a config, never a CLI flag |
| Database | SQLite | Sufficient at this scale — no concurrency, no need for Postgres |
| Version control | Git | Small, focused, conventional-commit-prefixed commits directly to `main` |
| Data & model versioning | DVC | Git holds pointers only (`data/raw/cbri.dvc`); no data blobs in version control |
| Experiment tracking | Weights & Biases | Planned from Week 5, once the evaluation harness exists and there are runs worth logging |
| Service | FastAPI | `/v1/inspect`, `/v1/calibration`, `/v1/feedback`, `/v1/audit`, `/v1/metrics` |
| Console | Streamlit (React is an optional stretch goal, not the baseline plan) | One screen, one slider |
| Containerization | Docker | Planned for the Week 14 service demo; not yet implemented |
| CI | GitHub Actions — ruff, mypy, pytest, nbstripout | The validity property test runs on every push once it exists |
| Release | arXiv (cs.CV, cross-list stat.ML), Zenodo DOI | Both the preprint and the calibration benchmark are meant to be citable |

## 8. Repository Structure

```
certus/
├── configs/            # Hydra configs: data, score, conformal, experiment
├── data/                # DVC pointers only -- no raw data in git (see data/README.md)
├── docs/                # Technical blueprint, work plan, logbook, audit reports
├── notebooks/           # Exploration only -- never on the critical path
├── paper/                # LaTeX source; intro.tex holds the three locked contribution claims
├── scripts/              # audit_data.py and other one-off tooling
├── service/              # FastAPI inspection service -- Week 14, not yet built
├── src/certus/
│   ├── conformal/         # quantile, split_cp, crc, ltt, mondrian, weighted, localized
│   ├── data/               # loaders, site/camera extraction, split management
│   ├── eval/                # coverage, efficiency, the 100-seed evaluation protocol
│   ├── models/               # VLM wrapper, classification heads
│   ├── scores/                # nonconformity score registry (S1-S7)
│   ├── triage/                 # policy + economics (HRF, cost break-even)
│   └── viz/                     # figure generation (F1-F9)
├── tests/                # pytest suite -- the validity property test is the load-bearing one
└── ui/                    # Streamlit/React triage console -- Week 14, not yet built
```

## 9. Installation / Setup

```bash
git clone https://github.com/Rishikesh10925/Certus.git
cd Certus
pip install -e ".[dev]"
pre-commit install
```

DVC is initialized (`.dvc/`) but has no remote configured yet — nothing to `dvc pull` until that's decided (see [`data/README.md`](data/README.md)). Weights & Biases has no active use yet; `wandb login` will be needed starting Week 5.

## 10. How to Run Experiments

```bash
make setup                                  # install deps, pre-commit, dvc init
make data DATA_PATH=/path/to/cbri/export    # runs scripts/audit_data.py -- read-only, no side effects
make calibrate                              # not implemented yet -- Phase 1-2, Weeks 3-8
make loso                                   # not implemented yet -- Phase 3, Weeks 9-12
make figures                                # not implemented yet -- regenerates F1-F9 from cached scores
```

Every experiment is meant to run as a Hydra config composition (e.g. `configs/experiment/e1_validity.yaml`), not a hand-written CLI invocation, once the corresponding pipeline stage exists.

## 11. Reproducibility

- **Data**: DVC-tracked, content-hashed; the train/cal/test split manifest gets frozen and hashed in Week 2 so every downstream figure traces back to an exact split.
- **Config provenance**: every run is expected to log its fully resolved Hydra config, split hash, model version, score id, and seed (Weights & Biases, from Week 5).
- **Fresh-clone check**: Week 16 includes an explicit reproducibility gate — clone into a fresh container, run `make figures`, confirm every figure regenerates from cached scores with no manual steps.
- **Calibration benchmark release**: mints a Zenodo DOI at Week 16, independent of the arXiv preprint, so the benchmark itself is citable.

## 12. FastAPI + Docker + React Demo

*Planned for Week 14; not yet implemented.* The service contract as specified:

| Endpoint | Purpose |
|---|---|
| `POST /v1/inspect` | Multipart image + site_id → triage decision with an attached risk statement |
| `GET /v1/calibration/{cluster}` | Current λ̂ and q̂ for a site cluster |
| `POST /v1/feedback` | Human-supplied ground truth (does not trigger recalibration — that's a deliberate weekly batch job) |
| `GET /v1/audit/next` | Next randomly sampled image awaiting a label |
| `GET /v1/metrics` | Prometheus-format metrics |

Target: p95 latency under 400ms per image. The console is specified as a single Streamlit screen — one slider for the guaranteed maximum missed-defect rate, a live-repartitioning thumbnail grid, an HRF/projected-hours panel, and a plain-English risk statement footer (React is an acceptable but non-default substitute). Docker packaging for the demo is planned but has no implementation or Dockerfile yet.

## 13. Current Project Status / Roadmap

| Weeks | Phase | Milestone / Gate | Status |
|---|---|---|---|
| 1–2 | Foundation — data audit, shift verification, scope lock | Hard gate at Week 2: n_pos and shift viability | **Week 1 done** — see [`docs/logbook.md`](docs/logbook.md); Week 2 next |
| 3–5 | Conformal core — quantile, property test, split CP | Gate at Week 5: Figure F2 must land on the diagonal | Not started |
| 6–8 | Scores & risk control — CRC, LTT, label-noise study | Gate 1 at Week 8: Figure F3 exists | Not started |
| 9–12 | Shift research — efficiency curve, LOSO, corrections | Gate at Week 12: pivot check on the shift result | Not started |
| 13–14 | Product & remaining figures — service, console, F6–F9 | — | Not started |
| 15–16 | Paper & release — write, cut, ship | Gate 2 at Week 16: arXiv preprint public | Not started |
| 17–18 | Buffer & outreach — video, applications, submissions | Project closed | Not started |

Full week-by-week detail lives in [`docs/Certus-18-Week-Work-Plan.docx`](docs/Certus-18-Week-Work-Plan.docx); the full specification is [`docs/Certus-Technical-Blueprint.docx`](docs/Certus-Technical-Blueprint.docx). This section is the one meant to be edited every week — everything above it should barely need to change.

## 14. Limitations

Explicitly out of scope, by design, not by oversight:

- Improving the underlying VLM detector's accuracy — Certus is model-agnostic and does not retrain the backbone.
- New conformal theory — applying existing theory rigorously is the contribution, not extending it.
- Defect localization or segmentation-level risk bounds.
- Multi-modal fusion (thermal, LiDAR, acoustic), federated/privacy-preserving calibration.
- A production-grade product — no authentication, tenancy, or billing.
- Real-time video inspection.

Two honest risks worth stating up front rather than discovering late: the existing labels were bootstrapped with GPT-4V, so any guarantee calibrated against them is a guarantee about agreement with GPT-4V, not about missed defects, unless calibration is done exclusively on a hand-labelled gold subset (the plan, per O6) — bootstrapped labels are training data, never calibration data. And it is a live possibility that the leave-one-site-out study finds no meaningful shift at all; if the domain-classifier AUC comes back low, the pre-registered plan is to say so plainly rather than force a shift narrative the data doesn't support.

This is also a solo, 18-week capstone project (~12–14 hrs/week alongside other commitments) — scope is deliberately fenced to stay achievable in that window, and the gates above exist specifically to catch scope drift early rather than at Week 16.

## 15. Citation

Not yet available. An arXiv preprint and citation entry will be added here once it's public — targeted for Week 16 (Gate 2).

```bibtex
% To be added once the arXiv preprint is live.
```

## 16. License

[MIT](LICENSE)
