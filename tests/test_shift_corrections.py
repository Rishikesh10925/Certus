"""Covers conformal/mondrian.py and conformal/weighted.py, and their LOSO
wiring: equal weights reduce exactly to the unweighted code, true
likelihood ratios restore validity under covariate shift, and on a
synthetic shifted site both corrections recover what vanilla loses."""

from __future__ import annotations

import math

import numpy as np
import pytest

from certus.conformal.crc import calibrate_crc
from certus.conformal.mondrian import FeatureClusters, calibrate_mondrian_crc
from certus.conformal.quantile import conformal_quantile
from certus.conformal.weighted import (
    domain_weights,
    effective_sample_size,
    weighted_quantile,
)
from certus.eval.loso import fold_weights, loso_folds, run_loso, summarize_loso


@pytest.mark.parametrize("n", [9, 10, 99, 250])
@pytest.mark.parametrize("alpha", [0.1, 0.05, 0.2])
def test_equal_weights_match_conformal_quantile(n: int, alpha: float) -> None:
    s = np.random.default_rng(n).normal(size=n)
    got = weighted_quantile(s, np.ones(n), np.ones(3), alpha)
    assert np.all(got == conformal_quantile(s, alpha))


@pytest.mark.parametrize("eps", [0.1, 0.05, 0.02])
def test_equal_weights_match_conditional_crc(eps: float) -> None:
    s = np.round(np.random.default_rng(0).random(120), 2)  # ties on purpose
    y = np.ones(s.size, dtype=bool)
    got = weighted_quantile(s, np.ones(s.size), np.ones(1), eps)[0]
    assert got == calibrate_crc(s, y, eps)


def test_weighted_quantile_is_inf_when_test_weight_dominates() -> None:
    s = np.arange(10.0)
    assert weighted_quantile(s, np.ones(10), np.array([100.0]), 0.1)[0] == math.inf


def test_effective_sample_size() -> None:
    assert effective_sample_size(np.ones(50)) == pytest.approx(50)
    assert effective_sample_size(np.r_[1e6, np.ones(49)]) == pytest.approx(1.0, abs=1e-3)


def test_true_weights_restore_coverage_under_covariate_shift() -> None:
    """x ~ N(0,1) at calibration, N(1,1) at test; the score's noise grows
    with x, so unweighted CP under-covers and true-ratio weighting fixes it."""
    rng = np.random.default_rng(0)
    alpha, trials = 0.1, 400
    plain, weighted = [], []
    for _ in range(trials):
        xc, xt = rng.normal(0, 1, 300), rng.normal(1, 1, 200)
        sc = np.abs(rng.normal(0, np.exp(xc)))
        st = np.abs(rng.normal(0, np.exp(xt)))
        wc, wt = np.exp(xc - 0.5), np.exp(xt - 0.5)  # N(1,1) / N(0,1)
        plain.append((st <= conformal_quantile(sc, alpha)).mean())
        weighted.append((st <= weighted_quantile(sc, wc, wt, alpha)).mean())
    assert np.mean(plain) < 1 - alpha - 0.03
    assert np.mean(weighted) >= 1 - alpha - 0.01


def test_domain_classifier_auc_tracks_the_shift() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=(1, 16))
    src = base + rng.normal(size=(600, 16))
    same = domain_weights(src[:400], src[400:500], base + rng.normal(size=(200, 16)))
    moved = domain_weights(src[:400], src[400:500], base + 1.5 + rng.normal(size=(200, 16)))
    assert abs(same.auc - 0.5) < 0.12
    assert moved.auc > 0.9
    assert effective_sample_size(moved.cal) < effective_sample_size(same.cal)


def test_mondrian_small_group_goes_to_humans() -> None:
    s = np.r_[np.linspace(0, 1, 100), [0.5, 0.6]]
    y = np.ones(s.size, dtype=bool)
    g = np.r_[np.zeros(100, dtype=int), [1, 1]]
    lam = calibrate_mondrian_crc(s, y, g, 0.05, n_groups=3)
    assert np.isfinite(lam[0]) and lam[1] == math.inf and lam[2] == math.inf


def test_feature_clusters_assign_new_rows_to_nearest() -> None:
    a, b = np.eye(8)[0], np.eye(8)[1]
    rng = np.random.default_rng(0)
    x = np.vstack([a + 0.05 * rng.normal(size=(50, 8)), b + 0.05 * rng.normal(size=(50, 8))])
    cl = FeatureClusters(x, 2)
    got = cl.assign(np.vstack([a, b]))
    assert got[0] != got[1]
    assert got[0] == cl.assign(x[:1])[0]


def _shifted_sites(n: int = 3000) -> tuple[np.ndarray, ...]:
    """Two regions in feature space. In the easy one defects score far from
    clean; in the hard one they overlap. P(Y|X) and the score given X are
    the same everywhere -- pure covariate shift. Sources a, b are mostly
    easy; target c is mostly hard, so vanilla CRC under-calls c's misses."""
    rng = np.random.default_rng(0)
    sites = np.repeat(np.array(["a", "b", "c"]), n // 3)
    hard = rng.random(n) < np.where(sites == "c", 0.7, 0.15)
    centre = np.where(hard[:, None], np.eye(16)[1] * 8, np.eye(16)[0] * 8)
    feats = (centre + rng.normal(size=(n, 16))).astype(np.float32)
    y = rng.random(n) < 0.3
    mu_def = np.where(hard, 0.55, 0.2)
    s = np.where(y, rng.normal(mu_def, 0.1), rng.normal(0.75, 0.1))
    p = np.clip(1.0 - s, 0.0, 1.0)
    return sites, y, s, np.stack([1.0 - p, p], axis=1), feats


def test_corrections_recover_a_covariate_shifted_site() -> None:
    sites, y, s, probs, feats = _shifted_sites()
    fold = next(f for f in loso_folds(sites, ("a", "b", "c")) if f.site == "c")
    args = (fold, probs, s, y, 0.1, 0.1, 1200)
    vanilla = summarize_loso(run_loso(*args, n_seeds=20), 0.1, 0.1)
    mondrian = summarize_loso(
        run_loso(*args, n_seeds=20, method="mondrian", features=feats, n_clusters=2), 0.1, 0.1
    )
    w = fold_weights(fold, feats, 1200, n_seeds=20)
    weighted = summarize_loso(run_loso(*args, n_seeds=20, method="weighted", weights=w), 0.1, 0.1)
    # Seeds share one finite source pool and one target set, so their
    # sampling error does not average away: allow ~0.2x of slack.
    assert vanilla["miss_rate_ratio"] > 3.0
    assert mondrian["miss_rate_ratio"] < 1.3
    assert weighted["miss_rate_ratio"] < 1.3
    assert weighted["domain_auc_mean"] > 0.75
    assert weighted["ess_frac_mean"] < 1.0


def test_run_loso_checks_method_inputs() -> None:
    sites, y, s, probs, feats = _shifted_sites(300)
    fold = loso_folds(sites, ("a", "b", "c"))[0]
    with pytest.raises(ValueError, match="method"):
        run_loso(fold, probs, s, y, 0.1, 0.1, 100, method="nope")
    with pytest.raises(ValueError, match="features"):
        run_loso(fold, probs, s, y, 0.1, 0.1, 100, method="mondrian")
    with pytest.raises(ValueError, match="DomainWeights"):
        run_loso(fold, probs, s, y, 0.1, 0.1, 100, method="weighted")
