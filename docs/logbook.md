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

Read end to end, appendices included. Notes below were written from memory
first, then checked against the live text (ar5iv.labs.arxiv.org/html/2107.07511,
2026-08-16) and corrected in two places where memory was imprecise --
flagged inline as **(corrected)**.

Full structure, confirmed against the live text: Section 1 Conformal
Prediction, 2 Examples of Conformal Procedures, 3 Evaluating Conformal
Prediction, 4 Extensions (includes 4.1 Group-Balanced CP and 4.4 Outlier
Detection), 5 Worked Examples, 6 Full Conformal Prediction; Appendix A
Distribution-Free Control of General Risks (includes a Learn-then-Test
procedure), B Examples of Distribution-Free Risk Control (multilabel FDR
control, simultaneous OOD detection), C Concentration Properties of
Empirical Coverage, D Theorem and Proof of the coverage guarantee.

- **Split conformal prediction**: hold out a calibration set, compute a
  nonconformity score `s(x, y)` on it, take the
  `ceil((n+1)(1-alpha)) / n` empirical quantile of those scores (not the
  naive `1-alpha` quantile -- that's the off-by-one the roadmap flags for
  Week 3's first test), and threshold test-time scores against it. Confirmed
  verbatim in the text as Equation 1:
  `1-alpha <= P(Y_test in C(X_test)) <= 1-alpha + 1/(n+1)` -- so the
  overcoverage bound isn't "roughly" 1/(n+1), it's the exact stated upper
  bound. Undercoverage below nominal is the real bug signal; that asymmetry
  is what Week 5's Figure F2 gate is checking for, not just "close to the
  diagonal."
- **(corrected) i.i.d. vs. exchangeability**: the paper's own Theorem 1 is
  stated for the case where calibration and test points are i.i.d., not the
  weaker exchangeability condition -- "exchangeable" appears exactly once
  in the whole text, in Section 4.4's outlier-detection permutation test,
  not as the headline assumption. The "exchangeability is THE assumption
  conformal prediction rests on, and it's what site shift breaks" framing
  is accurate to the broader literature -- it's literally how the Certus
  blueprint's own glossary defines it, and it's the explicit framing of two
  papers later on the reading list (Tibshirani et al., "Conformal
  Prediction Under Covariate Shift"; Barber et al., "Conformal Prediction
  Beyond Exchangeability") -- but it's not how this particular tutorial
  states its own core theorem. Previously misattributed a "Section 9
  honest-scoping point" to this paper; that section number belongs to the
  Certus blueprint, not to Angelopoulos & Bates. My mistake, corrected here
  rather than silently.
- **Score choice determines efficiency, not validity**: LAC (1 - softmax of
  the true class) gives valid but often needlessly large sets; APS
  (cumulative softmax mass until the true class is covered) adapts set size
  to difficulty and is usually tighter. This is directly S1 vs. the
  APS baseline Certus compares in Week 4.
  <br>**(corrected) RAPS**: the tutorial cites Angelopoulos et al. 2020 for
  RAPS ("significant practical improvements, which we implemented here")
  but doesn't itself explain the regularization mechanism -- that detail
  (a penalty added past the top-k classes to stop pathologically large sets
  on noisy softmax tails) comes from the cited paper, not from this
  tutorial's own exposition. Worth reading that paper directly before
  implementing S1's APS/RAPS baselines in Week 4, rather than relying on
  this tutorial's one-line mention.
- **Conditional coverage is not implied by marginal coverage** -- a set can
  hit 90% coverage overall while being badly wrong on a specific subgroup
  or class. This is exactly the SSCV / class-conditional coverage work
  scheduled for Week 13, and it's the reason a single marginal LOSO number
  in Week 10-12 won't be enough on its own without the per-class breakdown
  later.
- Section 4.1, "Group-Balanced Conformal Prediction," is the direct
  precursor to the Mondrian CP (`mondrian.py`) Certus implements in Week
  11 -- one threshold per group instead of one global threshold, which is
  exactly the per-site-cluster calibration the shift-correction arm needs.
- Appendix A's Learn-then-Test procedure previews the separate LTT paper
  (Angelopoulos et al., "Learn Then Test") on the Week 7 reading list --
  useful to have seen the simpler version here first.
- Practical takeaway that shapes the whole repo's discipline: nothing that
  touches the calibration set may also have been fit on it. Step 1 of the
  algorithm ("identify a heuristic uncertainty notion using the
  **pre-trained** model") assumes the model is already frozen before
  calibration ever starts. The blueprint's rule that no function whose name
  starts with `fit_` may see calibration data encodes exactly that failure
  mode as a naming convention instead of trusting it to memory.

Angelopoulos et al., "Conformal Risk Control" (ICLR 2024) is scheduled for
Week 2, not read yet.

## Week 2 -- Shift Verification & Scope Lock (2026-08-17)

### Angelopoulos, Bates, Fisch, Lei, Schuster, "Conformal Risk Control"
### (arXiv:2208.02814, ICLR 2024)

Fetched and read directly from ar5iv.labs.arxiv.org/html/2208.02814 rather
than from memory, given the correction needed last week. One thing memory
had wrong immediately: the author list. This is Angelopoulos, Bates,
**Fisch, Lei, and Schuster** -- not Candes and Jordan, who I'd have guessed
if I hadn't checked.

- **The algorithm**: `lambda_hat = inf{lambda : (n/(n+1)) * R_hat_n(lambda)
  + B/(n+1) <= alpha}`, where `R_hat_n(lambda)` is the empirical average of
  a per-example loss `L_i(lambda)` over the calibration set, and `B` is a
  bound on the loss. If no `lambda` satisfies the constraint, fall back to
  `lambda_max`. This is `crc.py`'s entire job.
- **The guarantee (Theorem 1)**: `E[L_{n+1}(lambda_hat)] <= alpha`. Theorem
  2 adds a matching lower bound, `E[L_{n+1}(lambda_hat)] >= alpha -
  2B/(n+1)` for i.i.d. continuous losses -- so the guarantee is tight up to
  `O(1/n)`, not just an inequality that could be loose in practice.
- **Monotonicity is the load-bearing assumption**: `L_i(lambda)` must be
  non-increasing in `lambda` (more conservative threshold -> loss can only
  go down). The proof leans on this through exchangeability and order
  statistics; Proposition 2 shows control can fail outright for non-monotone
  losses. This is exactly why the roadmap calls out "unit-test monotonicity
  of L(lambda) explicitly" for Week 6 -- it's not a nice-to-have property,
  it's the one thing standing between this working and silently not.
- **Relation to split CP**: Section 2.3 shows plain conformal prediction
  (miscoverage loss, `L = 1{Y not in C(X)}`) is the special case where this
  formula reduces to the ordinary conformal quantile. CRC is the
  generalization to any monotone loss -- concretely, the asymmetric
  missed-defect loss Certus actually cares about, not just 0/1 coverage.
- **Extensions (Section 4)**, relevant to later weeks in order: 4.1
  distribution shift via importance weighting (Propositions 3-4) --
  directly relevant to `weighted.py`, Week 11; 4.2 quantile risk control
  (Proposition 5); 4.3 multiple risk control over an indexed family
  (Proposition 6); 4.4 adversarial risk via a supremum over perturbations;
  4.5 U-risk control for paired losses. Appendix A covers non-monotone
  losses via asymptotic monotonization, for whenever monotonicity itself
  can't be guaranteed by construction.

### Shift diagnostic, secondary dataset, and split freeze

In progress -- see below once the domain-classifier AUC, secondary dataset
download, and manifest hash are in hand.
