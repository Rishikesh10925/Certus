# Certus -- Logbook

One entry per week, written the way the review evidence asks for it: the
number, not just the vibe. See `docs/Certus-18-Week-Work-Plan.docx` for what
each week owes.

## Week 1 -- Data Audit & Literature (2026-08-16)

### n_pos

Ran `scripts/audit_data.py` against the real CBRI export (Roboflow project
`rishikesh-talpalikar/building-defect-decetation-wmfkw`, version 1,
generated with no augmentation and no preprocessing so the count is the raw
dataset, not an inflated one). Full report: `docs/audit/cbri_audit_report_v1.json`.

| | |
|---|---|
| Total images | 1518 |
| **n_pos (defect-positive)** | **982** |
| n_neg (defect-negative) | 536 |

Per-class annotation counts:

| Class | Count |
|---|---|
| crack | 920 |
| vegetation | 552 |
| corrosion | 124 |
| dampness | 23 |
| dampens | 16 |

`dampens` is almost certainly a labelling typo for `dampness` in the source
project rather than a sixth real class -- flagged here, not silently
merged, since collapsing it is a data-cleaning decision that belongs in a
`data:`-prefixed commit of its own, not buried in an audit script.

### Achievable epsilon range (roadmap 4.1, n_pos >~ 10/epsilon)

| epsilon | n_pos needed | verdict |
|---|---|---|
| 0.10 | 100 | OK |
| 0.05 | 200 | OK |
| 0.02 | 500 | OK |
| 0.01 | 1000 | SHORT (982, 18 short) |
| 0.005 | 2000 | SHORT |

So on CBRI alone the honest floor is **epsilon >~ 0.02**, with 0.01 close
enough that pooling in the secondary dataset (Week 2) plausibly closes the
gap -- worth re-checking once that dataset is in hand, not worth assuming
either way yet.

### site_id / camera_id metadata

Confirmed absent, from two independent angles:

1. Roboflow's own schema for this project has no `site_id` or `camera_id`
   field -- checked directly against the API (`GET /{workspace}/{project}`)
   before download.
2. EXIF probe (`scripts/audit_data.py`, 50-image sample): 0 images with GPS
   EXIF, no `Make`/`Model`/`DateTimeOriginal` tags at all. Filenames are
   bare numeric IDs plus a Roboflow-added content hash (`00735_jpg.rf.b66b...`)
   -- no site encoding there either.

Plan (per roadmap 4.1, to execute in Week 2): derive proxy sites via
unsupervised clustering on frozen-VLM embeddings, then run the
domain-classifier diagnostic (AUC > 0.80 gate) before trusting the clusters
for the LOSO study. Recorded in `src/certus/data/sites.py`.

### Repo setup

- `week-01-foundations` branch created; this is the first real commit.
- `.venv` (Python 3.12) created; `ruff`, `ruff format`, `mypy --strict` on
  `src/certus`, and `pytest` all pass locally -- same commands CI runs.
- Added `tests/test_audit_data.py` (4 tests, synthetic COCO/YOLO fixtures)
  because the stub test files scheduled for Week 3 meant `pytest` collected
  zero tests and would have failed CI with exit code 5. The audit script is
  the actual Week 1 deliverable code, so it's the thing that gets tested
  now; the conformal-quantile tests stay stubbed for Week 3 as planned.
- Fixed 5 ruff `E501` line-length violations in `scripts/audit_data.py`
  (pre-existing, unrelated to the audit logic itself).
- `pre-commit run --all-files` passes (ruff, ruff-format, mypy, nbstripout,
  trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files).
- `dvc init` done; `data/raw/cbri` (108 MB, 1518 images) added and tracked
  via `data/raw/cbri.dvc`. No remote configured yet -- that's a deliberate
  Week 2+ decision (`data/README.md`), not an oversight.
  - Had to pin `pathspec==0.11.2` in `pyproject.toml`: dvc==3.44.0 crashes
    on current pathspec releases (`ImportError: cannot import name
    '_DIR_MARK'`) because it uses a private API pathspec since removed.
    Without the pin, `make setup` breaks for anyone who runs it fresh.
  - Fixed `.gitignore`'s DVC section -- the original `!/data/*.dvc`
    negation only un-ignores `.dvc` files directly under `/data/`, not
    nested ones like `data/raw/cbri.dvc`, because `/data/**` blocks git
    from even descending into `data/raw/` to check. Needed
    `!/data/**/` first to re-permit traversal.
- Weights & Biases: dependency pinned, nothing to initialize yet -- first
  real use is Week 5 (`eval/protocol.py`, the 100-seed harness). Login with
  a personal API key happens then.
- Three contribution claims (C1-C3) were already written verbatim into
  `paper/intro.tex` as a locked header comment when this session started;
  confirmed they match the roadmap's thesis statement and Section 2.5
  exactly, left untouched.

### Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction
### and Distribution-Free Uncertainty Quantification" (arXiv:2107.07511)

Read end to end, appendices included. Notes against what Certus actually
needs:

- The whole guarantee rests on **exchangeability**, not i.i.d. -- calibration
  and test points just need to be drawn in a way that's invariant to
  reordering. That's the precise assumption site-level shift breaks, which
  is the paper's own honest-scoping point (Section 9) and the entire
  reason Certus's C3 exists.
- **Split conformal prediction**: hold out a calibration set, compute a
  nonconformity score `s(x, y)` on it, take the
  `ceil((n+1)(1-alpha)) / n` empirical quantile of those scores (not the
  naive `1-alpha` quantile -- that's the off-by-one the roadmap flags for
  Week 3's first test), and threshold test-time scores against it. Marginal
  coverage `P(Y in C(X)) >= 1-alpha` holds in finite samples, no
  distributional assumptions on the model or the data.
- **Score choice determines efficiency, not validity**: LAC (1 - softmax of
  the true class) gives valid but often needlessly large sets; APS
  (cumulative softmax mass until the true class is covered) adapts set size
  to difficulty and is usually tighter; RAPS regularizes APS to stop
  pathologically large sets on noisy softmax tails. This is directly S1
  vs. the APS/RAPS baselines Certus compares in Week 4.
  <br>Overcoverage is expected and bounded above by roughly `1/(n_cal+1)`;
  undercoverage below nominal is the real bug signal -- that asymmetry is
  what Week 5's Figure F2 gate is checking for, not just "close to the
  diagonal."
- **Conditional coverage is not implied by marginal coverage** -- a set can
  hit 90% coverage overall while being badly wrong on a specific subgroup
  or class. The appendices' discussion of this is exactly the SSCV /
  class-conditional coverage work scheduled for Week 13, and it's the
  reason a single marginal LOSO number in Week 10-12 won't be enough on its
  own without the per-class breakdown later.
- The appendices also cover the **calibration-set-size / finite-sample
  correction** in more depth and touch conformal prediction beyond
  classification (regression, few-shot) -- useful background, not
  load-bearing for Certus's classification-shaped triage problem.
- Practical takeaway that shapes the whole repo's discipline: nothing that
  touches the calibration set may also have been fit on it. The blueprint's
  rule that no function whose name starts with `fit_` may see calibration
  data is this paper's central failure mode, encoded as a naming
  convention instead of trusted to memory.

Angelopoulos et al., "Conformal Risk Control" (ICLR 2024) is scheduled for
Week 2, not read yet.
