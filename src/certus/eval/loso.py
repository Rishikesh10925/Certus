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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from certus.conformal.crc import auto_clear, calibrate_crc
from certus.eval.coverage import marginal_coverage
from certus.eval.protocol import METHODS


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


def _one(
    probs: NDArray[np.float64],
    scores: NDArray[np.float64],
    labels: NDArray[np.bool_],
    cal: NDArray[np.int64],
    test: NDArray[np.int64],
    alpha: float,
    epsilon: float,
    rng: np.random.Generator,
    seed: int,
) -> LosoSeedResult:
    y = labels.astype(int)
    sets = METHODS["lac"](probs[cal], y[cal], probs[test], alpha, rng)
    cal_pos = cal[labels[cal]]
    lam = calibrate_crc(scores[cal_pos], labels[cal_pos], epsilon)
    cleared = auto_clear(scores[test], lam)
    defects = labels[test]
    return LosoSeedResult(
        seed=seed,
        coverage=marginal_coverage(sets, y[test]),
        miss_rate=float(cleared[defects].mean()) if defects.any() else 0.0,
        hrf=float(1.0 - cleared.mean()),
    )


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
) -> list[LosoSeedResult]:
    """Calibrate on n_cal source rows, test on every target row."""
    if not 0 < n_cal <= fold.source.size:
        raise ValueError(f"n_cal={n_cal} needs 1..{fold.source.size} source rows")
    out = []
    for s in range(n_seeds):
        rng = np.random.default_rng([base_seed, s])
        cal = rng.choice(fold.source, size=n_cal, replace=False)
        out.append(_one(probs, scores, labels, cal, fold.target, alpha, epsilon, rng, s))
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
    other half tests."""
    out = []
    for s in range(n_seeds):
        rng = np.random.default_rng([base_seed, s])
        perm = rng.permutation(fold.target)
        half = perm.size // 2
        out.append(_one(probs, scores, labels, perm[:half], perm[half:], alpha, epsilon, rng, s))
    return out


def summarize_loso(results: list[LosoSeedResult], alpha: float, epsilon: float) -> dict[str, float]:
    """Means over seeds, the gaps to nominal, and the share of seeds that
    break each guarantee."""
    cov = np.array([r.coverage for r in results])
    miss = np.array([r.miss_rate for r in results])
    hrf = np.array([r.hrf for r in results])
    return {
        "coverage_mean": float(cov.mean()),
        "coverage_gap": float(cov.mean() - (1.0 - alpha)),
        "coverage_below_frac": float((cov < 1.0 - alpha).mean()),
        "miss_rate_mean": float(miss.mean()),
        "miss_rate_ratio": float(miss.mean() / epsilon),
        "miss_exceed_frac": float((miss > epsilon).mean()),
        "hrf_mean": float(hrf.mean()),
        "n_seeds": float(cov.size),
    }
