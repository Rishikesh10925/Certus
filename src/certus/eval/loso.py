"""Leave-one-site-out (LOSO) harness, the E4 shift protocol (roadmap 5.2,
contribution C3). Calibrate on every source site but one, deploy on the
held-out site, and measure how far realised coverage or risk lands from
its nominal level.

Sites are whatever grouping the caller passes: here SDNET2018's three
structure types (deck, pavement, wall), plus CBRI as a target that is
never calibrated on. Each fold is paired with an in-site reference that
calibrates and tests inside the held-out site. That reference is
exchangeable by construction, so the gap between the two is the shift
alone, not small-sample noise.

Each seed draws a fresh calibration sample of n_cal source rows; the test
set is the whole held-out site (or, for the reference, the other half of
it).

Three ways to set the thresholds (`method`), all evaluated identically:
- vanilla: one split-CP q_hat and one CRC lambda_hat from the source draw.
- mondrian: one per k-means cluster of the VLM features (conformal/mondrian.py).
- weighted: one per test image, likelihood-ratio weighted
  (conformal/weighted.py). The domain classifier does not depend on the
  score or the level, so the caller fits it once per fold with
  `fold_weights` and passes the result in.
- localized: one per test image, weighted by latent similarity to each
  calibration row (conformal/localized.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from certus.conformal.crc import calibrate_crc
from certus.conformal.localized import localized_quantile, row_ess, similarity_kernel
from certus.conformal.mondrian import (
    FeatureClusters,
    calibrate_mondrian_crc,
    calibrate_mondrian_lac,
)
from certus.conformal.split_cp import calibrate_lac, lac_score_matrix, lac_scores
from certus.conformal.weighted import (
    DomainWeights,
    domain_weights,
    effective_sample_size,
    weighted_quantile,
)
from certus.eval.coverage import marginal_coverage

LOSO_METHODS = ("vanilla", "mondrian", "weighted", "localized")


@dataclass(frozen=True)
class LosoFold:
    site: str
    source: NDArray[np.int64]  # rows calibration may draw from
    target: NDArray[np.int64]  # the held-out site's rows


@dataclass(frozen=True)
class LosoSeedResult:
    seed: int
    coverage: float  # split CP marginal coverage on the target rows
    miss_rate: float  # CRC conditional: P(cleared | defect) on the target rows
    hrf: float  # CRC conditional: share of target rows not cleared
    ess_frac: float = 1.0  # weighted, localized: ESS / n of the defect rows CRC uses
    domain_auc: float = float("nan")  # weighted: source-vs-target separability
    # Per test row, for the conditional summaries (summarize_conditional).
    test: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64), repr=False)
    covered: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, bool), repr=False)
    set_size: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64), repr=False)
    cleared: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, bool), repr=False)


def loso_folds(sites: NDArray[np.str_], source_sites: tuple[str, ...]) -> list[LosoFold]:
    """One fold per distinct site. Calibration draws only from
    `source_sites` minus the held-out one, so a target-only site (CBRI)
    is held out but never calibrated on."""
    s = np.asarray(sites)
    unknown = set(source_sites) - set(s.tolist())
    if unknown:
        raise ValueError(f"source sites {sorted(unknown)} have no rows")
    folds = []
    for site in sorted(set(s.tolist())):
        source = np.flatnonzero(np.isin(s, [x for x in source_sites if x != site]))
        folds.append(LosoFold(site=site, source=source, target=np.flatnonzero(s == site)))
    return folds


def _draw_cal(fold: LosoFold, n_cal: int, base_seed: int, s: int) -> NDArray[np.int64]:
    if not 0 < n_cal <= fold.source.size:
        raise ValueError(f"n_cal={n_cal} needs 1..{fold.source.size} source rows")
    rng = np.random.default_rng([base_seed, s])
    cal: NDArray[np.int64] = rng.choice(fold.source, size=n_cal, replace=False)
    return cal


def _evaluate(
    probs: NDArray[np.float64],
    scores: NDArray[np.float64],
    labels: NDArray[np.bool_],
    test: NDArray[np.int64],
    q: float | NDArray[np.float64],
    lam: float | NDArray[np.float64],
    seed: int,
    ess_frac: float = 1.0,
    domain_auc: float = float("nan"),
) -> LosoSeedResult:
    """Split-CP sets and CRC decisions on the test rows, for thresholds that
    are either one number or one per test row."""
    sets = lac_score_matrix(probs[test]) <= np.reshape(q, (-1, 1))
    cleared = scores[test] > np.asarray(lam)
    defects = labels[test]
    y = defects.astype(int)
    return LosoSeedResult(
        seed=seed,
        coverage=marginal_coverage(sets, y),
        miss_rate=float(cleared[defects].mean()) if defects.any() else 0.0,
        hrf=float(1.0 - cleared.mean()),
        ess_frac=ess_frac,
        domain_auc=domain_auc,
        test=test,
        covered=sets[np.arange(test.size), y],
        set_size=sets.sum(axis=1),
        cleared=cleared,
    )


def _vanilla(
    probs: NDArray[np.float64],
    scores: NDArray[np.float64],
    labels: NDArray[np.bool_],
    cal: NDArray[np.int64],
    test: NDArray[np.int64],
    alpha: float,
    epsilon: float,
    seed: int,
) -> LosoSeedResult:
    q = calibrate_lac(probs[cal], labels[cal].astype(int), alpha)
    cal_pos = cal[labels[cal]]
    lam = calibrate_crc(scores[cal_pos], labels[cal_pos], epsilon)
    return _evaluate(probs, scores, labels, test, q, lam, seed)


def fold_weights(
    fold: LosoFold,
    features: NDArray[np.float32],
    n_cal: int,
    n_seeds: int = 100,
    base_seed: int = 0,
) -> list[DomainWeights]:
    """Domain-classifier weights for each seed's calibration draw (the same
    draws `run_loso` makes). The source side trains on the source rows
    outside that draw, so no calibration row weights itself."""
    out = []
    for s in range(n_seeds):
        cal = _draw_cal(fold, n_cal, base_seed, s)
        train = np.setdiff1d(fold.source, cal)
        out.append(domain_weights(features[train], features[cal], features[fold.target], seed=s))
    return out


def run_loso(
    fold: LosoFold,
    probs: NDArray[np.float64],
    scores: NDArray[np.float64],
    labels: NDArray[np.bool_],
    alpha: float,
    epsilon: float,
    n_cal: int,
    n_seeds: int = 100,
    base_seed: int = 0,
    method: str = "vanilla",
    features: NDArray[np.float32] | None = None,
    n_clusters: int = 4,
    weights: list[DomainWeights] | None = None,
    k_frac: float = 0.5,
) -> list[LosoSeedResult]:
    """Calibrate on n_cal source rows, test on every target row.
    `mondrian` and `localized` need `features`; `weighted` needs `weights`
    from `fold_weights` with the same n_cal, n_seeds and base_seed."""
    if method not in LOSO_METHODS:
        raise ValueError(f"method must be one of {LOSO_METHODS}, got {method!r}")
    groups = np.full(labels.size, -1, dtype=np.intp)
    if method in ("mondrian", "localized") and features is None:
        raise ValueError(f"{method} needs features")
    if method == "mondrian":
        assert features is not None
        clusters = FeatureClusters(features[fold.source], n_clusters, seed=base_seed)
        for rows in (fold.source, fold.target):
            groups[rows] = clusters.assign(features[rows])
    if method == "weighted" and (weights is None or len(weights) != n_seeds):
        raise ValueError("weighted needs one DomainWeights per seed (see fold_weights)")
    test = fold.target
    y = labels.astype(int)
    out = []
    for s in range(n_seeds):
        cal = _draw_cal(fold, n_cal, base_seed, s)
        if method == "vanilla":
            out.append(_vanilla(probs, scores, labels, cal, test, alpha, epsilon, s))
        elif method == "mondrian":
            g = groups[cal]
            q_g = calibrate_mondrian_lac(probs[cal], y[cal], g, alpha, n_clusters)
            lam_g = calibrate_mondrian_crc(scores[cal], labels[cal], g, epsilon, n_clusters)
            t = groups[test]
            out.append(_evaluate(probs, scores, labels, test, q_g[t], lam_g[t], s))
        elif method == "localized":
            assert features is not None
            pos = cal[labels[cal]]
            k_all = similarity_kernel(features[cal], features[test], k_frac)
            k_pos = similarity_kernel(features[pos], features[test], k_frac)
            q_t = localized_quantile(lac_scores(probs[cal], y[cal]), k_all, alpha)
            lam_t = localized_quantile(scores[pos], k_pos, epsilon)
            ess = float(row_ess(k_pos).mean()) / pos.size
            out.append(_evaluate(probs, scores, labels, test, q_t, lam_t, s, ess))
        else:
            assert weights is not None
            dw = weights[s]
            pos = labels[cal]
            q_t = weighted_quantile(lac_scores(probs[cal], y[cal]), dw.cal, dw.test, alpha)
            lam_t = weighted_quantile(scores[cal][pos], dw.cal[pos], dw.test, epsilon)
            ess = effective_sample_size(dw.cal[pos]) / int(pos.sum())
            out.append(_evaluate(probs, scores, labels, test, q_t, lam_t, s, ess, dw.auc))
    return out


def run_in_site(
    fold: LosoFold,
    probs: NDArray[np.float64],
    scores: NDArray[np.float64],
    labels: NDArray[np.bool_],
    alpha: float,
    epsilon: float,
    n_seeds: int = 100,
    base_seed: int = 0,
) -> list[LosoSeedResult]:
    """The exchangeable reference: half the target site calibrates, the
    other half tests (vanilla thresholds)."""
    out = []
    for s in range(n_seeds):
        rng = np.random.default_rng([base_seed, s])
        perm = rng.permutation(fold.target)
        half = perm.size // 2
        out.append(_vanilla(probs, scores, labels, perm[:half], perm[half:], alpha, epsilon, s))
    return out


def summarize_loso(results: list[LosoSeedResult], alpha: float, epsilon: float) -> dict[str, float]:
    """Means over seeds, the gaps to nominal, and the share of seeds that
    break each guarantee."""
    cov = np.array([r.coverage for r in results])
    miss = np.array([r.miss_rate for r in results])
    hrf = np.array([r.hrf for r in results])
    ess = np.array([r.ess_frac for r in results])
    auc = np.array([r.domain_auc for r in results])
    return {
        "coverage_mean": float(cov.mean()),
        "coverage_gap": float(cov.mean() - (1.0 - alpha)),
        "coverage_below_frac": float((cov < 1.0 - alpha).mean()),
        "miss_rate_mean": float(miss.mean()),
        "miss_rate_ratio": float(miss.mean() / epsilon),
        "miss_ratio_p05": float(np.percentile(miss, 5) / epsilon),
        "miss_ratio_p95": float(np.percentile(miss, 95) / epsilon),
        "miss_exceed_frac": float((miss > epsilon).mean()),
        "hrf_mean": float(hrf.mean()),
        "ess_frac_mean": float(ess.mean()),
        "ess_frac_min": float(ess.min()),
        "domain_auc_mean": float(auc.mean()),
        "n_seeds": float(cov.size),
    }


SSCV_MIN_ROWS = 10


def sscv(
    covered: NDArray[np.bool_],
    set_size: NDArray[np.int64],
    alpha: float,
    min_rows: int = SSCV_MIN_ROWS,
) -> float:
    """Size-stratified coverage violation (Angelopoulos et al. 2021): the
    worst |coverage - (1 - alpha)| over the set sizes held by at least
    `min_rows` test rows. A method can hit 1 - alpha on average by mixing
    over-covered large sets with under-covered small ones; SSCV is how far
    apart they are. With two classes a size-2 set always covers and a
    size-0 set never does, so without the floor one stray empty set in a
    seed would score 1 - alpha; nan if no size has enough rows."""
    sizes, counts = np.unique(set_size, return_counts=True)
    gaps = [
        abs(covered[set_size == k].mean() - (1.0 - alpha))
        for k, c in zip(sizes, counts, strict=True)
        if c >= min_rows
    ]
    return float(max(gaps)) if gaps else float("nan")


def summarize_conditional(
    results: list[LosoSeedResult],
    labels: NDArray[np.bool_],
    alpha: float,
    epsilon: float,
    strata: dict[str, NDArray[np.bool_]] | None = None,
) -> dict[str, float]:
    """What the marginal numbers hide: split-CP coverage per class and per
    set size, SSCV, and, for each named stratum of rows (e.g. a defect
    type), the coverage and CRC miss rate on the defects inside it. Means
    over seeds; a stratum with no test rows in a seed is skipped there."""
    out: dict[str, list[float]] = {}

    def add(key: str, value: float) -> None:
        out.setdefault(key, []).append(value)

    for r in results:
        y = labels[r.test]
        for name, mask in (("clean", ~y), ("defect", y)):
            if mask.any():
                add(f"coverage_{name}", float(r.covered[mask].mean()))
        add("sscv", sscv(r.covered, r.set_size, alpha))
        for k in range(3):
            at_k = r.set_size == k
            add(f"size{k}_frac", float(at_k.mean()))
            if at_k.any():
                add(f"size{k}_coverage", float(r.covered[at_k].mean()))
        for name, mask in (strata or {}).items():
            m = mask[r.test] & y
            if m.any():
                add(f"{name}_n", float(m.sum()))
                add(f"{name}_coverage", float(r.covered[m].mean()))
                add(f"{name}_miss_ratio", float(r.cleared[m].mean()) / epsilon)
    summary = {k: float(np.nanmean(v)) for k, v in out.items()}
    for name in ("clean", "defect"):
        if f"coverage_{name}" in summary:
            summary[f"coverage_{name}_gap"] = summary[f"coverage_{name}"] - (1.0 - alpha)
    summary["sscv_p95"] = float(np.nanpercentile(out["sscv"], 95))
    return summary
