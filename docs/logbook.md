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
