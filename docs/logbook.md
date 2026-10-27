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

### Secondary dataset

Picked SDNET2018 (Utah State University, doi.org/10.15142/T3TD19) --
concrete crack images from decks, pavements, and walls, 256x256, a
completely different acquisition source from CBRI/Roboflow (different
camera setup, different subject framing -- close-up concrete surfaces
rather than whole-building photos). Downloaded directly from USU's
DigitalCommons (503.8 MB); the connection truncated twice partway through
(curl error 18, "end of response ... bytes missing") before a `curl -C -`
resume completed it. Verified: `unzip -t` reported no errors, and the
extracted counts match the published dataset description exactly --

| | Cracked | Uncracked | Total |
|---|---|---|---|
| Decks (D) | 2025 | 11595 | 13620 |
| Pavements (P) | 2608 | 21726 | 24334 |
| Walls (W) | 3851 | 14287 | 18138 |
| **Total** | **8484** | **47608** | **56092** |

One thing not in the public dataset description, found by inspecting the
filenames directly: every image is named `<structure_id>-<n>.jpg` (e.g.
`7001-115.jpg`), and `structure_id` is a genuine, consistent physical
structure identifier -- 54 distinct decks, 104 pavements, 72 walls, 230
total, confirmed by checking that every one of the 56,092 filenames
matches the pattern and that structure IDs don't collide across images.
This is real site_id, not a proxy -- see `load_sdnet2018_images` in
`src/certus/data/loaders.py`. DVC-tracked at `data/raw/sdnet2018.dvc`.

### Domain-classifier shift diagnostic

Two separate runs (`scripts/shift_diagnostic.py`), both on frozen CLIP
ViT-B/32 embeddings (`openai/clip-vit-base-patch32` -- there's no
fine-tuned Certus VLM checkpoint yet, see `src/certus/models/vlm.py`),
full results in `docs/audit/week2_shift_diagnostic.json`.

**1. Within-CBRI proxy sites** (KMeans on CBRI's own 1518 embeddings, k
chosen by silhouette score over k=3..7, then a logistic regression asked
to tell the resulting clusters apart):

- Best k = 4, but the silhouette score at every k tested was ~0.10-0.11.
  By the usual reading of that statistic (Kaufman & Rousseeuw: below ~0.25
  means no substantial structure found), CBRI's own feature space doesn't
  contain natural, well-separated groups -- consistent with Week 1's
  finding that there's no real site metadata, extending it to "and no
  strong latent site structure in the visual features either, at least
  not one KMeans finds."
- Despite that, the domain classifier scored **AUC = 0.9988** on those
  same clusters -- which reads as a contradiction until you remember what
  it's actually testing. A linear classifier separating KMeans's own
  output in a 512-dimensional embedding space with ~1500 points is close
  to guaranteed regardless of whether the clusters mean anything: KMeans
  partitions are close to linearly separable by construction, and 512
  dimensions over ~1500 points is exactly the regime where that bites
  hardest. **This AUC is not evidence the proxy sites are real.**
  Reporting it without the silhouette context would have been a
  methodologically indefensible way to wave the Week 2 gate through --
  flagging it here instead. Conclusion: CBRI-internal clustering is not
  used as a LOSO site definition.

**2. CBRI vs. SDNET2018** (1518 CBRI embeddings vs. 1500 randomly sampled
SDNET2018 embeddings, binary domain classifier): **AUC = 0.9996**. This
one is not circular -- CBRI and SDNET2018 are independently collected by
different people, with different cameras, at different times, of
different subjects (whole buildings vs. close-up concrete surfaces). A
near-perfect AUC here is exactly what "genuine distribution shift" should
look like, and it's the real result the Week 2 gate needed.

### Week 2 hard gate: MET

The gate requires 400+ defect-positive images across 3+ distinguishable
sites, or an explicit pivot. Both halves are satisfied, but not the way
the roadmap's default plan assumed:

- **Defect-positive count**: 982 (CBRI) + 8484 (SDNET2018) = 9466, or 982
  alone -- either way, far past 400. Never actually in doubt; Week 1
  already showed CBRI alone clears this.
- **3+ distinguishable sites**: not from CBRI-internal proxy clusters
  (shown above to be unreliable) -- from SDNET2018's 230 genuine
  structure IDs (or, coarsely, its 3 real structure-type categories:
  deck/pavement/wall) plus CBRI as a confirmed-distinguishable fourth
  source (AUC 0.9996 against SDNET2018). That's comfortably past 3, on
  real metadata for SDNET2018's side and a verified diagnostic for the
  CBRI-vs-secondary split.

**Decision, recorded explicitly rather than deferred**: the LOSO site
taxonomy for later weeks will be built from SDNET2018's real structure IDs
and the CBRI/SDNET2018 dataset boundary, not from clustering CBRI on its
own. If within-CBRI shift specifically turns out to matter later, that's a
question for the Week 2.4.4 domain-classifier monitoring going forward,
revisited with better evidence than a k=4 KMeans call -- not resolved by
fiat here.

### Achievable epsilon range, updated

Unchanged from Week 1 for the primary calibration engine: CBRI n_pos = 982
still governs it, since SDNET2018 is a shift-study dataset, not part of
the primary detector's calibration pool (different classes -- crack only,
vs. CBRI's crack/vegetation/corrosion/dampness -- and a different visual
domain entirely). epsilon >~ 0.02 is still the honest floor; 0.01 is still
18 images short (982 vs. 1000 needed). SDNET2018 does not change this
number, and pooling the two for calibration would not be a defensible
move given how different their label schemas and domains are -- it
supplies additional *sites* for the shift study, not additional
*calibration* n_pos for the primary engine.

### Split manifest

Frozen and content-hashed at `data/manifest/splits_v1.csv`
(`scripts/freeze_splits.py`) -- CBRI keeps Roboflow's train/valid/test
(valid renamed to cal), SDNET2018 gets a fresh deterministic 70/15/15
stratified split (seed 0) since it ships none.

- **57,610 rows** (1518 CBRI + 56,092 SDNET2018), **9466 defect-positive**
  (982 + 8484).
- **Manifest hash**:
  `aeaa352bc406420f5a685578ed26bd346332fa9b241b54e5e41407056612aeb7` --
  sha256 over the sorted per-file content hashes. This is the number that
  makes the split tamper-evident: any changed pixel in any tracked image,
  or any re-run of the freeze script with a different seed or ratio,
  changes this hash. It does not change on its own between runs (verified:
  re-ran `freeze_splits.py` twice, identical hash both times).
- Committed directly to git (not DVC) at 6.9 MB -- it's provenance data
  meant to be diffed and read, not a raw data blob; `.pre-commit-config.yaml`'s
  large-file check now excludes `data/manifest/` explicitly for exactly
  this reason rather than raising the limit globally.

## Week 3 -- The Quantile, Tested First (2026-10-19)

### Corrected quantile

`src/certus/conformal/quantile.py`: `conformal_quantile(scores, alpha)`
returns the exact k-th order statistic, k = ceil((n+1)(1-alpha)), and +inf
when k > n. That last case is real, not an edge case: at alpha = 0.01 with
n_cal = 50, k = 51 > 50, so no finite threshold certifies 99% and every
set is the full label set. Deliberately not `np.quantile`. Its default
linear interpolation lands between two order statistics, below the k-th,
which is precisely the under-coverage bug. `tests/test_quantile.py` pins
hand-computed k values, including the exact-integer cases (n=9 and n=19
at alpha=0.1) where a sloppy ceil would round up.

### Split CP with LAC

`src/certus/conformal/split_cp.py`: `lac_scores` (1 - p_true),
`calibrate_lac`, `predict_sets_lac`. Calibration functions are named
`calibrate_*`, never `fit_*`, per the blueprint's naming rule. Set
membership is tested on the score (`1 - p <= q_hat`), not rearranged to
`p >= 1 - q_hat`, so the comparison is bit-identical to calibration and
the boundary point always covers itself. APS and RAPS are Week 4.

### Validity property test

`tests/test_validity.py`: 500 trials per configuration, alpha in
{0.01, 0.05, 0.10, 0.20} x n_cal in {50, 200, 1000} (the e1_validity
grid), 500 test points per trial, K=5 synthetic classes with a
deliberately miscalibrated, noisy model. Pass condition is asymmetric:
mean coverage >= 1-alpha - 4 SE, and <= 1-alpha + 1/(n_cal+1) + 4 SE.
It already runs on every push because CI runs `pytest`, so the workflow
needed no change. The whole suite takes about 7 s locally.

| alpha | n_cal | mean coverage | SE | theory E[cov] = k/(n+1) |
|---|---|---|---|---|
| 0.01 | 50 | 1.0000 | 0.0000 | 1.0000 (k > n, full sets) |
| 0.01 | 200 | 0.9903 | 0.0004 | 0.9900 |
| 0.01 | 1000 | 0.9899 | 0.0002 | 0.9900 |
| 0.05 | 50 | 0.9604 | 0.0013 | 0.9608 |
| 0.05 | 200 | 0.9493 | 0.0008 | 0.9502 |
| 0.05 | 1000 | 0.9492 | 0.0005 | 0.9500 |
| 0.10 | 50 | 0.9002 | 0.0020 | 0.9020 |
| 0.10 | 200 | 0.9015 | 0.0011 | 0.9005 |
| 0.10 | 1000 | 0.8992 | 0.0008 | 0.9001 |
| 0.20 | 50 | 0.8005 | 0.0026 | 0.8039 |
| 0.20 | 200 | 0.8033 | 0.0015 | 0.8010 |
| 0.20 | 1000 | 0.8008 | 0.0010 | 0.8002 |

Every row is within about 2 SE of the exact theoretical expectation, so
nothing here is luck at the 4 SE tolerance. The lowest rows, like
0.9492 at alpha=0.05 and n=1000, sit just under nominal by Monte-Carlo
noise (0.8 SE below the 0.9500 expectation), not by a bias.

The test also tests itself. `test_validity_check_catches_the_naive_quantile`
swaps in `np.quantile(scores, 1-alpha)` at n_cal=20, alpha=0.1. Coverage
falls to **0.854** against a nominal 0.90, and the check flags it as
UNDER-coverage. A property test that couldn't catch the off-by-one would
be decoration.

### Gold labelling

`scripts/gold_labels.py` writes and validates `data/gold/gold_labels_v1.csv`,
which is committed in git like the manifest (the `.gitignore` exception
was widened to include it). It holds 175 rows in a fixed seeded order:
CBRI's 125 cal images first, then its 50 test images. It never includes
train, because the detector was fit on those. The sheet leaves out the
GPT-4V bootstrap classes on purpose, so the gold labels can't be anchored
to them, which would inflate the Week 7 kappa. `status` re-checks every
row's content hash against the manifest. `scripts/label_gold.py` is a
small Tk window for the actual labelling: one image at a time, class
checkboxes, every save written straight to the sheet.

**Gold-label count: 71 / 60 target** (rows 1-71, all from the cal split,
labeller RT, `status` reports no errors). 70 defect-positive, 1 negative.
Class ticks: crack 45, vegetation 19, corrosion 7, dampness 4.

**First label-noise finding.** These 71 were compared against the
Roboflow/GPT-4V labels only after labelling was finished:

| | count |
|---|---|
| gold and bootstrap agree on defect yes/no | 37 |
| gold says defect, bootstrap has no annotation | 34 |
| gold says no defect, bootstrap says defect | 0 |

The disagreement runs entirely one way. Bootstrap "negatives" are mostly
images that were never annotated, not images that show no defect. Row 2
is a clear case: its source filename is German for "young saplings
sprouting from a concrete wall", gold says vegetation, and the bootstrap
has no annotation. Consequences:

- Week 1's n_neg = 536 is very likely inflated, and n_pos = 982 is an
  undercount. Both came from bootstrap annotations, not from looking at
  the images.
- Calibrating on bootstrap labels would score real defects as negatives,
  which corrupts the missed-defect guarantee without any visible sign.
  This is the concrete case for calibrating on gold only.
- It is the opening data point for the Week 7 kappa / label-noise study.
  With only 1 gold negative so far, kappa on this sample isn't meaningful
  yet. A 48% one-way miss rate on the bootstrap's negatives is the number
  to carry forward.

**Problem found, not resolved:** CBRI's non-train pool is only 175 images
(125 cal + 50 test). The blueprint's gold target is 300-500, and using
only the cal split as the calibration anchor caps n_cal at 125. That's
far below the 982-n_pos figure the epsilon-range analysis assumed. Either
some CBRI train images move to gold/cal (this needs a v2 manifest, and
the detector must not have seen them), or the epsilon floor gets
recomputed on the real gold pool. This decision has to be made before
Week 5's real-data validity run.

## Week 4 -- Baselines and Scores S1-S2 (2026-10-20)

### APS and RAPS

`src/certus/conformal/split_cp.py` now has all three split-CP baselines
(B5-B7) behind one shape: build an (n, n_classes) score matrix, calibrate
on the true-label column, include a class iff score <= q_hat.
`aps_score_matrix` / `raps_score_matrix` take an optional per-row uniform
`u` for the randomised versions. RAPS uses lam=0.1, k_reg=1, fixed in
`configs/conformal/split.yaml` and not tuned, because tuning them on the
calibration set would void the guarantee and there is no tuning split yet.

The validity property test now covers four methods (LAC, deterministic
APS, randomised APS, randomised RAPS) x the same 12 (alpha, n_cal) cells,
500 trials each. All 48 pass. One wrinkle the test caught: deterministic
APS gives the lowest-ranked class a score of exactly 1.0 on every row.
Those tied scores make it over-cover by more than 1/(n+1) at alpha=0.01.
That's expected behaviour, not a bug, so for deterministic APS only the
lower bound (the actual guarantee) is checked, and the reason is written
in the test.

### There is no fine-tuned detector

The blueprint assumes an "existing fine-tuned VLM defect detector". It
doesn't exist. The source Roboflow Universe project
(universe.roboflow.com/cbri-pdll5/building-defect-decetation) shows
**Models: 0**, and a search of the local drives found only checkpoints
from unrelated projects. Every score is therefore built on off-the-shelf
models. That's consistent with the blueprint's own "model-agnostic"
framing, but the paper must say it plainly.

### S1 -- classifier logit (zero-shot CLIP)

First attempt: a logistic-regression head on frozen CLIP ViT-B/32
features, fit on CBRI train with the bootstrap labels. **5-fold CV AUC
0.507**, which is chance. That's exactly what the Week 3 gold comparison
predicted: the bootstrap "negatives" are mostly unannotated defects, so
those labels carry almost no signal to learn from.

Replaced with zero-shot CLIP. p(defect) is the softmax mass on four defect
prompts (crack / vegetation / corrosion / dampness) against two clean-wall
prompts, at CLIP's trained logit scale, and S1 = 1 - p(defect). No
training labels are used, so nothing can leak calibration data. The prompt
strings are in `src/certus/scores/s1.py` and are hashed into the model
version (`clip-vit-b32+zeroshot:379df781c421`).

### S2 -- generative log-probability (Qwen2-VL-2B)

Qwen2-VL-2B-Instruct, bf16 on the RTX 5050 (4.9 GB peak VRAM, ~0.45 s per
image). S2 = -log p("Yes") as the answer to a fixed question
(`src/certus/scores/s2.py`), teacher-forced rather than parsed from a free
generation, so it's deterministic. The config's original wording, "mean
token log-prob of the generated assertion", was tried first with the full
sentence "Yes, there is a structural defect." and scored AUC 0.58 on 80
SDNET2018 cal images. The later tokens are near-certain whatever the image
shows, and averaging them in drowns the one token that carries
information. "Yes" alone scored 0.82 on the same 80. That choice was made
on those 80 images, so the quality number below comes from different ones.

Needed for this: torch 2.2.1 (CPU) -> 2.7.1+cu128 (RTX 50-series needs
CUDA 12.8), and transformers 4.38.2 -> 4.51.3 (Qwen2-VL support). mypy
1.8 crashes on transformers 4.51's internals, so transformers is set to
`follow_imports = "skip"` in pyproject.toml. All 105 tests pass on the
new stack.

### Score quality on labels we can trust

CBRI can't measure this: its bootstrap labels are the problem, and gold is
70 positives to 1 negative so far. `scripts/score_sanity.py` uses
SDNET2018's human cracked/uncracked labels: a balanced 500/500 sample from
the **train** split (disjoint from the 80 cal images used for the S2
choice).

| score | AUC | deck | pavement | wall |
|---|---|---|---|---|
| S1 zero-shot CLIP | 0.764 | 0.801 | 0.864 | 0.706 |
| S2 Qwen2-VL "Yes" | 0.775 | 0.791 | 0.843 | 0.773 |

Both scores carry real signal with no task training at all, and neither
is good. That's fine for conformal validity, which holds for any score,
but it will show up as large sets and a high human-review fraction. Walls,
the subject closest to CBRI's, are S1's weakest category.

### Score cache

`src/certus/data/store.py`: SQLite `scores` table keyed by (dataset,
relpath, score_id, model_version), at `data/cache/certus.sqlite` (local
cache, regenerable, not in git). Populated:

| dataset | score | model_version | rows |
|---|---|---|---|
| cbri | s1 | clip-vit-b32+zeroshot:379df781c421 | 1518 |
| cbri | s2 | qwen2-vl-2b+assert:5c1c607f66ce | 1518 |
| sdnet2018 | s1 | clip-vit-b32+zeroshot:379df781c421 | 1000 |
| sdnet2018 | s2 | qwen2-vl-2b+assert:5c1c607f66ce | 1000 |
| cbri | s1 | clip-vit-b32+logreg:3e9f71733d6f | 1518 (superseded, kept for the record) |

Sample (S1 / S2 for `valid/000140_jpg.rf.6fcc...`): 0.105 / 0.917.
`read_scores` refuses to return a partial vector, because silently
dropping rows would shrink n_cal.

### Three split-CP variants on real data

`scripts/run_split_cp.py`: S1 as a 2-class probability, calibrated on a
balanced 1000-image SDNET2018 cal sample, evaluated on 1000 test images,
alpha = 0.1:

| method | q_hat | coverage | mean set size | singleton fraction |
|---|---|---|---|---|
| LAC | 0.735 | 0.914 | 1.50 | 0.50 |
| APS (randomised) | 0.927 | 0.922 | 1.66 | 0.34 |
| RAPS (randomised) | 1.027 | 0.925 | 1.66 | 0.34 |

All three hold coverage >= 0.90 on real, human-labelled images. With a
weak 2-class score, half the sets are {crack, no crack}, which in triage
terms means "send to a human". Better scores are where efficiency comes
from, and that's what E2 (Week 8) is for.

### Gold labelling

**71 / 130** (unchanged since Week 3). The 59 more images needed this week
are hand work: `python scripts/label_gold.py --labeller RT`.

### Open decisions carried forward

- Gold pool size (from Week 3): CBRI has only 175 non-train images, and
  the blueprint wants 300-500.
- The primary calibration engine was meant to sit on CBRI. With CBRI's
  labels this noisy, SDNET2018 (human labels, 56k images) may need to
  carry more of the headline numbers than planned.

### Week 4 close-out: gold labelling complete (2026-10-25)

**Gold-label count: 175 / 175**, the entire CBRI non-train pool (125 cal
+ 50 test), labeller RT, `status` reports no errors. There is nothing left
in CBRI to hand-label without touching train, so this is the ceiling, not
a milestone on the way to the blueprint's 300.

| | count |
|---|---|
| gold defect-positive | 172 (98.3%) |
| gold defect-negative | 3 |
| bootstrap said negative | 74, of which gold says defect: **73** |
| gold says negative, bootstrap says defect | 2 (both in test) |

Class ticks: crack 104, vegetation 50, corrosion 17, dampness 8.

What this settles:
- The Week 3 finding holds at full size. Bootstrap "negatives" are
  unannotated defect images 73 times out of 74.
- **CBRI is effectively an all-defect dataset.** It can measure missed
  defects, but it has almost no clean images to auto-clear, so the triage
  efficiency question (human-review fraction) can't be answered on CBRI.
- Caveat: ~100 of these 175 labels were entered in a few minutes. Before
  any headline number rests on them, a blind re-label of a random 20 is
  worth doing to measure intra-labeller agreement.

## Week 5 -- Validity on Real Data (2026-10-25)

### Evaluation harness

`src/certus/eval/protocol.py`: `run_protocol` draws a fresh random cal/test
partition of a labelled pool per seed (seed = (base_seed, i), so any seed
replays alone), runs one prediction-set method, and records coverage, mean
set size, and singleton fraction. `summarize` gives the mean, SD, and a 95%
percentile-bootstrap CI over seeds. Six methods share one interface: split
CP (LAC, APS, RAPS) and baselines B1-B3.

### Baselines B1-B3

`src/certus/baselines.py`, all as prediction-set methods so they sit on
F2's axes:

- **B1 raw softmax:** take classes until raw mass >= 1-alpha.
- **B2 temperature scaling:** NLL-fit temperature, then B1.
- **B3 selective classification:** answer the top class above a
  max-prob threshold, abstain (full set) below it. The threshold is the
  most permissive one whose *empirical* miscoverage on the tuning rows is
  <= alpha, which is plug-in with no +1 correction.

B2/B3 fit on the same rows conformal calibrates on (same data budget),
passed to the fit_* guard as "tune".

### E1 and Figure F2

`scripts/run_e1.py` (`make e1`), 100 seeds per cell, alpha in {0.01, 0.05,
0.10, 0.20} x n_cal in {50, 200, 1000} x scores {S1, S2} x two pools:

- **SDNET2018:** a seeded 3000-image sample of its cal split, human labels,
  457 cracked.
- **CBRI:** all 1518 images, bootstrap labels, 982 positive. Validity is
  a property of the procedure, so it must hold against *any* fixed labels,
  including noisy ones. This pool tests correctness, not detection.

Provenance (manifest hash, score model versions, pool sizes, n_pos) is in
`results/e1/e1_validity.json`. Per-seed rows are in
`results/e1/e1_validity_seeds.csv.gz`. Figures:
`paper/figures/F2_validity_s1.png`, `F2_validity_s2.png`.

### Week 5 gate: PASSED

**144 / 144 conformal cells within +/-2 points of nominal** (3 methods x
4 alphas x 3 n_cal x 2 scores x 2 pools). None of the four usual failure
causes (off-by-one, sign error, split leakage, calibration-set fitting)
needed hunting.

| method | worst under-coverage | worst over-coverage | cells outside +/-2 pts |
|---|---|---|---|
| Split CP LAC | -0.49 | +1.00 | **0 / 48** |
| Split CP APS | -0.58 | +1.40 | **0 / 48** |
| Split CP RAPS | -0.58 | +1.29 | **0 / 48** |
| B1 raw softmax | -7.74 | +17.13 | 39 / 48 |
| B2 temperature | +0.40 | +20.00 | 36 / 48 |
| B3 selective | -1.67 | +8.01 | 6 / 48 |

(points; mean coverage minus 1-alpha)

At alpha = 0.10, n_cal = 1000:

| pool | score | LAC coverage | LAC set size | B1 coverage | B2 coverage |
|---|---|---|---|---|---|
| SDNET2018 | S1 | 0.899 +/- 0.012 | 1.20 | 0.994 | 0.976 |
| SDNET2018 | S2 | 0.899 +/- 0.012 | 1.07 | 0.992 | 0.958 |
| CBRI | S1 | 0.902 +/- 0.016 | 1.74 | **0.824** | 1.000 |
| CBRI | S2 | 0.899 +/- 0.017 | 1.83 | 0.994 | 1.000 |

What the baselines show:
- The raw softmax (B1) is wrong in *both* directions depending on the
  data. On SDNET2018 it over-covers (+17 pts at worst) with near-full sets.
  On CBRI with S1 it under-covers by up to 7.7 pts. Nobody using it could
  know which way it will fail.
- Temperature scaling (B2) doesn't fix that, because it only moves the
  error.
- B3, the plug-in threshold, comes close at n_cal=1000 (it's split CP
  without the +1), but drifts at small n: 6 cells out of tolerance,
  worst -1.67 pts.
- Split CP is the only method on the diagonal everywhere, which is the
  entire point of F2.
- S2 gives tighter sets than S1 on SDNET2018 (1.07 vs 1.20 at alpha =
  0.1), the first hint for the E2 score comparison.

### Not done this week

- Weights & Biases: still not initialised (it needs your personal API
  key). Runs are logged as JSON + per-seed CSV with full provenance
  instead, which covers what the blueprint asks every run to log.
- Gold labelling to 200: impossible on CBRI (the ceiling is 175, see
  above).

### Decision needed: primary dataset

Recommended: **SDNET2018 becomes the primary calibration dataset** (human
labels, 56k images, 15% cracked, so there's a real auto-clear question),
and **CBRI becomes a held-out target site** for the shift study, using the
175 gold labels as its trustworthy evaluation set. Recorded as a
recommendation only, until the decision is made.

## Week 6 -- Conformal Risk Control (2026-10-26)

### CRC

`src/certus/conformal/crc.py`. With the shared orientation (high score =
little defect evidence), an image is AUTO_CLEARed iff s(x) > lambda, and
the missed-defect loss L_i(lambda) = 1{y_i = defect and s_i > lambda} is
non-increasing in lambda. lambda_hat = inf{lambda : n/(n+1) R_hat(lambda) +
1/(n+1) <= epsilon}, searched over the calibration scores plus +inf (clear
nothing). Two guarantees:

- **joint:** E[1{defect and cleared}] <= epsilon, calibrated on all rows.
- **conditional:** P(cleared | defect) <= epsilon, calibrated on the
  defect rows only. This is the "at most epsilon of real defects are
  missed" statement the blueprint's pitch actually makes.

Monotonicity is tested explicitly, as the plan asks:
- `test_missed_defect_loss_is_non_increasing_in_lambda` checks every
  per-example L_i, not just the mean.
- `calibrate_crc` runs `check_monotone` on every call.
- `test_check_monotone_catches_a_sign_error` proves the flipped rule
  (clear when s < lambda) is rejected rather than silently returning
  garbage.

A 400-trial synthetic test confirms risk <= epsilon for both guarantees.

### S3 and S4

- **S3, verbalized confidence** (`src/certus/scores/s3.py`): Qwen2-VL-2B
  is asked for a 0-100 confidence, and S3 = 1 - c/100. All 3000 answers
  parsed. But the model effectively answers only **0 (1519 times) or 50
  (1478 times)**, plus 3 others. It's a two-valued score, and the "50"
  bucket holds 308 of the 457 cracked images. Sanity AUC on 1000 SDNET
  train images: **0.628**. That's the poorly calibrated practitioner
  baseline (B11) the blueprint predicted, in its most extreme form.
- **S4, detector objectness** (`src/certus/scores/s4.py`): OWLv2
  zero-shot, one query per defect class, S4 = 1 - max box confidence.
  Sanity AUC: **0.521** (deck 0.513, pavement 0.582, wall 0.498), which
  is chance. On 256 px close-ups that are *all* concrete texture, an
  object detector has no object to find. This matches the blueprint's
  warning that detector scores are weak on diffuse defects, here in its
  strongest form.

Two engineering fixes were needed before S4 would run at all, both
committed:
1. fp16. fp32 at batch 8 needed 11.2 GB on the 8 GB card and spilled to
   system memory (~11 s/image).
2. The stock Owlv2ImageProcessor resizes with scipy on the CPU, which
   left the GPU idle ~90% of the time (confirmed with a py-spy stack
   dump). The same steps (pad with 0.5, bilinear resize to 960, clip,
   normalise) now run in torch on the GPU. Max |difference| vs the stock
   pipeline on 32 images: 0.006, correlation 0.999.

### CRC at epsilon in {0.10, 0.05, 0.02, 0.01}

`scripts/run_crc.py`, SDNET2018 E1 pool (3000 images, 457 cracked), n_cal
= 2000 (~305 cracked calibrate the conditional guarantee), 100 seeds.
Full table in `results/crc/crc_week6.json`. **All 32 cells valid: mean
realised risk <= epsilon in every one.**

Human-review fraction (HRF, share of images *not* auto-cleared), mean over
100 seeds:

| score (AUC) | joint 0.10 | 0.05 | 0.02 | 0.01 | conditional 0.10 | 0.05 | 0.02 | 0.01 |
|---|---|---|---|---|---|---|---|---|
| S1 CLIP (0.76) | 0.09 | 0.35 | 0.67 | 0.81 | 0.73 | 0.86 | 0.95 | 0.98 |
| S2 Qwen "Yes" (0.78) | **0.07** | 0.38 | 0.72 | 0.84 | 0.78 | 0.88 | 0.96 | 0.99 |
| S3 verbal (0.63) | 0.50 | 0.66 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| S4 OWLv2 (0.52) | 0.30 | 0.60 | 0.83 | 0.90 | 0.86 | 0.92 | 0.97 | 0.98 |

Realised risk vs target, joint guarantee (mean +/- SD):

| score | 0.10 | 0.05 | 0.02 | 0.01 |
|---|---|---|---|---|
| S1 | 0.1009 +/- 0.0124 | 0.0504 +/- 0.0079 | 0.0198 +/- 0.0050 | 0.0099 +/- 0.0036 |
| S2 | 0.1012 +/- 0.0114 | 0.0518 +/- 0.0086 | 0.0205 +/- 0.0053 | 0.0099 +/- 0.0037 |
| S4 | 0.1009 +/- 0.0117 | 0.0513 +/- 0.0085 | 0.0207 +/- 0.0053 | 0.0104 +/- 0.0040 |

The few means a hair over target (e.g. S2 at 0.05: 0.0518) are within 2
SE of epsilon: CRC bounds the *expected* risk, and is tight to O(1/n) by
Theorem 2. "valid" in the JSON = mean <= epsilon + 3 SE.

What it shows:
- **The guarantee holds for every score, good or useless.** That's the
  point of distribution-free risk control.
- **What the score changes is the price.** At joint epsilon = 0.10, S2
  sends 7% of images to a human and S4 sends 30%. Same guarantee, 4x the
  labour.
- **S3's ties are fatal.** Its two values mean CRC can either clear the
  whole "c = 0" bucket (146 cracked images inside) or nothing. At epsilon
  <= 0.02, or under the conditional guarantee at any level, it clears
  nothing, so HRF = 1.00. Valid, and worthless. That's a concrete,
  quotable reason why verbalized confidence fails as a risk-control
  score.
- **The conditional guarantee is expensive with off-the-shelf scores.**
  Missing at most 5% of real cracks costs 86-92% human review. This is
  the headline efficiency problem for E3 (Week 9): better scores (E2,
  Week 8), not better conformal machinery, are where the gains are.

### Gold labelling

The plan's "300 total" can't happen: CBRI's non-train pool is 175, and all
175 have been labelled since Week 4's close-out. Logged, not deferred.

### Still open

- Primary dataset: Weeks 5-6 run on SDNET2018 per the Week 5
  recommendation. Not yet formally confirmed.

## Week 7 -- Learn-then-Test & Label Noise (2026-10-26)

### Learn-then-Test

`src/certus/conformal/ltt.py`. It uses the same loss and decision rule as
CRC, but a different promise: P(R(lambda_hat) <= epsilon) >= 1 - delta.
H_lambda: R > epsilon is tested with the Hoeffding-Bentkus p-value. Fixed-
sequence testing walks from +inf (clear nothing) towards clearing more and
stops at the first non-rejection. Monotone risk makes that order natural,
so delta needs no splitting.

Tests (`tests/test_ltt.py`):
- The p-value behaves as expected.
- The vectorised and scalar p-values agree.
- LTT is never less conservative than CRC on the same calibration set.
- The PAC guarantee holds over 300 synthetic trials, measured against a
  200k-sample "true" risk.

### CRC vs LTT (delta = 0.05)

SDNET2018 pool, n_cal = 2000, 100 seeds. "Exceed" is the share of
deployments whose test-set risk came out above epsilon.

| S2, joint | CRC risk | CRC exceed | CRC HRF | LTT risk | LTT exceed | LTT HRF |
|---|---|---|---|---|---|---|
| eps 0.10 | 0.101 | 0.53 | 0.07 | 0.087 | 0.09 | 0.11 |
| eps 0.05 | 0.052 | 0.57 | 0.38 | 0.041 | 0.13 | 0.48 |
| eps 0.02 | 0.021 | 0.47 | 0.72 | 0.014 | 0.08 | 0.79 |
| eps 0.01 | 0.010 | 0.42 | 0.84 | 0.005 | 0.05 | 0.90 |

S1 shows the same pattern: CRC exceeds in 38-50% of deployments, LTT in
4-12%.

**The price of the PAC guarantee is +4 to +10 points of human review**
(joint). That's what turns "on average across deployments" into "for
this deployment, with 95% confidence". Under the conditional guarantee
the cost is +5 to +8 points, on top of an already 73-98% HRF.

LTT's exceed fraction sits a little above delta in some cells (0.13 at
S2/0.05). The guarantee is on the *true* risk, while "exceed" here is
measured on a finite 1000-image test set, so test-set noise adds
apparent violations. The synthetic test with a 200k-sample true risk
stays within delta.

### Label noise

Code: `src/certus/eval/label_noise.py`, `scripts/run_week7.py`. Results:
`results/week7/week7.json`. Draft paper section: `paper/label_noise.tex`
(the week's deliverable).

**kappa (GPT-4V bootstrap vs gold, CBRI, n = 175):**

| label | kappa |
|---|---|
| defect / no defect | **-0.007** |
| crack | 0.160 |
| vegetation | 0.287 |
| corrosion | -0.019 |
| dampness | 0.161 |

That's chance-level agreement on the label that matters, far below the
risk register's 0.6 line. The failure is one-sided: the bootstrap misses
42% of gold defects (73/172) and almost never invents one (2/175).

The missed defects are *not* the subtle ones. Their S1/S2 scores match
the annotated defects (one-sided Mann-Whitney p = 0.71 / 0.90), so the
misses look random. That turns out to decide everything.

**Guarantee degradation, stated plainly:**

1. **CBRI, real labels.** Calibrate on GPT-4V labels, measure against
   gold:
   - **Joint guarantee: realised miss rate 1.6-2.0x the promised bound.**
     At epsilon = 0.05 it's 0.093 (S1) and **0.101 (S2)**, exceeded in
     86-92% of deployments. Calibrated on gold it's 0.052.
   - **Conditional guarantee: survives** (0.042 / 0.052 at epsilon =
     0.05), because the misses are random.
2. **SDNET2018, controlled** (hide fraction h of defects in calibration
   labels only):
   - **Random hiding at CBRI's h = 0.42:** joint risk 1.7-2.0x epsilon,
     conditional risk still <= epsilon. This matches CBRI.
   - **Worst-case hiding** (the most clean-looking defects first): h = 0.10
     already gives a conditional miss rate of 0.14 at epsilon = 0.05
     (2.9x). At h = 0.42 it's 0.45, so **nearly half of real defects are
     auto-cleared under a nominal 5% guarantee**.

Why: a hidden defect looks free to clear, which inflates joint risk by
~1/(1-h). The conditional guarantee only needs the labelled defects to be
an exchangeable sample of all defects. Random misses keep that true;
correlated misses destroy it. **Without a gold set there is no way to
tell which case you're in.** That's the paper's argument for the gold
set, now with numbers.

### Still open

- Primary dataset (SDNET2018 + CBRI as target site): Weeks 5-7 run that
  way. Still not formally confirmed.
- The intra-labeller check on the gold set (re-label 20 blind) is still
  worth doing, now that the gold set carries a paper section.

## Week 8 -- Score Comparison (2026-10-27)

### S5, S6, S7

- **S5, MC-dropout** (`src/certus/scores/s5.py`). CLIP has no dropout
  layers, so 20 passes of dropout (p = 0.1) go on the image embedding
  before S1's zero-shot head. Dispersion alone has no direction, and
  "the model is unsure" can't justify auto-clearing. So S5 is a lower
  confidence bound on cleanliness, mean(1-p) - std(1-p). Masks are
  seeded per image, so a score never depends on its batch (tested).
- **S6, feature typicality** (`s6.py`). Negative Mahalanobis distance
  (Ledoit-Wolf) from the CLIP embedding to 3000 *defect-free* SDNET2018
  train images, excluding the tuning set. Defects and unfamiliar scenes
  both score low. That's the shift-aware property the blueprint wants
  for the LOSO weeks.
- **S7, learned combination** (`s7.py`). Logistic regression over
  standardised S1-S6, fit only on the fixed tuning set
  (`certus.data.splits.tune_records`: 500+500 SDNET2018 *train* images,
  the same 1000 that score_sanity reports AUCs on). It goes through the
  fit_* guard and refuses to score its own tuning rows (tested).
  Coefficients x1000: S1 -1397, S2 -1494, S3 +448, S4 +488, S5 +520,
  S6 +221. S1 and S2 carry it. The positive weights on S3-S6 are mostly
  corrections between correlated scores (S5 is nearly S1).

### E2 -- all seven scores, 100 seeds

`scripts/run_e2.py`, SDNET2018 pool (3000, 457 cracked), CRC, n_cal = 2000.
Results in `results/e2/e2_scores.json`, figure in
`paper/figures/F3_scores.png`. Human-review fraction (mean over seeds):

| score | AUC (pool) | joint 0.10 | joint 0.05 | joint 0.02 | joint 0.01 | cond 0.10 | cond 0.05 | cond 0.02 |
|---|---|---|---|---|---|---|---|---|
| S7 combination | 0.793 | **0.07** | **0.30** | **0.63** | **0.78** | **0.71** | **0.83** | **0.91** |
| S1 CLIP | 0.755 | 0.09 | 0.35 | 0.67 | 0.81 | 0.73 | 0.85 | 0.95 |
| S5 MC-dropout | 0.748 | 0.09 | 0.37 | 0.67 | 0.80 | 0.73 | **0.83** | 0.95 |
| S2 Qwen2-VL | 0.747 | **0.07** | 0.38 | 0.72 | 0.84 | 0.78 | 0.88 | 0.96 |
| S6 typicality | 0.649 | 0.16 | 0.52 | 0.78 | 0.89 | 0.83 | 0.91 | 0.95 |
| S3 verbal | 0.612 | 0.50 | 0.66 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| S4 OWLv2 | 0.549 | 0.30 | 0.60 | 0.83 | 0.90 | 0.86 | 0.92 | 0.97 |

The guarantee holds for all 56 cells (worst mean risk / epsilon = 1.04,
within Monte-Carlo error of a bound on the expectation).

What it says (contribution C1):
- **The learned combination wins.** S7 is best or tied-best in every
  cell. It saves 5 points of review vs the best single score at joint
  0.05 (30% vs 35%), and 4 points at conditional 0.02 (91% vs 95%).
  That's a real but modest gain from stacking off-the-shelf scores.
- **Two families, a clear gap.** The "does the model think it's a
  defect" scores (S1, S2, S5, S7: AUC 0.75-0.79) need 30-38% review at
  joint 0.05. Everything else (typicality, verbal confidence, detector
  boxes: AUC 0.55-0.65) needs 52-66%.
- **AUC predicts the ranking almost perfectly.** The one exception is S3:
  its ties make it worse than its AUC suggests (it clears nothing at
  epsilon <= 0.02).
- **S5's dropout bound adds nothing over S1.** The same signal, slightly
  noisier. MC-dropout on a frozen embedding isn't real epistemic
  uncertainty. A head with genuine dropout might differ, but there's no
  trained head to put it on.
- **The conditional guarantee stays expensive with every score.** Even S7
  needs 71% review to miss at most 10% of cracks. Off-the-shelf models
  are the bottleneck, not the conformal machinery.

### Paper: Method and Setup drafted

`paper/method.tex` (~950 words) and `paper/setup.tex` (~670 words), about
three pages, matching the plan. Method covers the triage rule, the loss,
both guarantees, CRC, LTT, split CP, all seven scores, and the "three
splits" enforcement. Setup covers data, provenance, the tuning set, the
protocol, baselines, metrics and compute.

One disclosure found while writing it, now stated in the paper: the
Week 4 choice of S2's assertion ("Yes" vs the full sentence) was made on
80 cal-split images, and 31 of them fall inside the 3000-image evaluation
pool (1%). It was a single binary choice, so the effect is negligible,
but "disjoint" would have been false.

### Gate 1

| criterion | status |
|---|---|
| Figure F3 exists | yes: `paper/figures/F3_scores.png` |
| Method and Setup sections written | yes: drafted, ~3 pages |
| Half the paper is real | intro claims (Week 1) + Method + Setup + label-noise section + F2 + F3. Results for C1 are in hand; C2 (efficiency curve, Week 9) and C3 (shift, Weeks 10-13) are not yet |

The plan's gate evidence asks for the drafted sections to be "read aloud
or shared for a sanity check". **That part is the user's.**

### Gate 1 sanity check (2026-10-27)

Every factual claim in Method and Setup was checked against the code and
`results/`. Seven fixes:

- **Setup, B3**: said it fits on "tuning rows"; it fits on the calibration
  rows (`protocol._b3`), the same budget as every other method.
- **Method, LTT motivation**: "roughly half of deployments" replaced by the
  measured exceedance, 31-57% of seeds.
- **Method, CRC fallback**: `<=` was wrong at the boundary; at
  eps = 1/(n+1), a finite threshold can still pass.
- **Method, split CP**: coverage 0.854 replaced by the test's expected
  value, ~0.86 (18.1/21).
- **Method, S5**: "S1's head" changed to "prompt comparison"; S1 has no
  trained head.
- **Method, three splits**: rewrote the sentence, which read as if S6
  used the tuning set.
- **Method, opening**: now states the exchangeability assumption up front,
  rather than implying the bound holds unconditionally.

Confirmed correct: the SDNET2018 counts, the 175 gold labels (172
positive), blind labelling (the labeller shows only the image), and
n_cal = 2000 leaving ~300 defects. A human read-aloud by the user is still
welcome but no longer blocks Gate 1.
