"""Covers eval/loso.py: the fold generator, and that LOSO stays valid when
sites are exchangeable but breaks when one site is shifted."""

from __future__ import annotations

import numpy as np
import pytest

from certus.eval.loso import (
    loso_folds,
    run_in_site,
    run_loso,
    sscv,
    summarize_conditional,
    summarize_loso,
)


def _sites(shift: float, n: int = 900) -> tuple[np.ndarray, ...]:
    """Three equal sites; site "c" scores are moved by `shift` toward clean,
    so its defects look cleaner than calibration expects."""
    rng = np.random.default_rng(0)
    sites = np.repeat(np.array(["a", "b", "c"]), n // 3)
    y = rng.random(n) < 0.3
    s = np.where(y, rng.normal(0.3, 0.15, n), rng.normal(0.7, 0.15, n))
    s = s + np.where(sites == "c", shift, 0.0)
    p = np.clip(1.0 - s, 0.0, 1.0)
    probs = np.stack([1.0 - p, p], axis=1)
    return sites, y, s, probs


def test_folds_hold_out_each_site_and_never_calibrate_on_it() -> None:
    sites = np.array(["a", "a", "b", "c", "t"])
    folds = {f.site: f for f in loso_folds(sites, ("a", "b", "c"))}
    assert sorted(folds) == ["a", "b", "c", "t"]
    for site, f in folds.items():
        assert set(f.target.tolist()) == set(np.flatnonzero(sites == site).tolist())
        assert not set(f.source.tolist()) & set(f.target.tolist())
        assert 4 not in f.source  # target-only site never calibrates
    assert folds["t"].source.tolist() == [0, 1, 2, 3]


def test_folds_reject_unknown_source_sites() -> None:
    with pytest.raises(ValueError, match="no rows"):
        loso_folds(np.array(["a", "b"]), ("a", "z"))


def test_exchangeable_sites_stay_valid() -> None:
    sites, y, s, probs = _sites(shift=0.0)
    fold = next(f for f in loso_folds(sites, ("a", "b", "c")) if f.site == "c")
    summ = summarize_loso(run_loso(fold, probs, s, y, 0.1, 0.1, 400, n_seeds=40), 0.1, 0.1)
    assert -0.03 < summ["coverage_gap"] < 0.06  # validity is one-sided
    assert summ["miss_rate_mean"] < 0.13


def test_shifted_site_breaks_loso_but_not_the_in_site_reference() -> None:
    """Shifting every score toward clean hurts defects and helps clean
    images, so marginal coverage barely moves while the miss rate among
    defects multiplies -- marginal coverage hides the failure that matters."""
    sites, y, s, probs = _sites(shift=0.15)
    fold = next(f for f in loso_folds(sites, ("a", "b", "c")) if f.site == "c")
    loso = summarize_loso(run_loso(fold, probs, s, y, 0.1, 0.1, 400, n_seeds=40), 0.1, 0.1)
    ref = summarize_loso(run_in_site(fold, probs, s, y, 0.1, 0.1, n_seeds=40), 0.1, 0.1)
    assert loso["miss_rate_ratio"] > 2.0
    assert abs(loso["coverage_gap"]) < 0.05
    assert ref["miss_rate_ratio"] < 1.4


def test_run_loso_rejects_too_large_n_cal() -> None:
    sites, y, s, probs = _sites(shift=0.0)
    fold = loso_folds(sites, ("a", "b", "c"))[0]
    with pytest.raises(ValueError, match="n_cal"):
        run_loso(fold, probs, s, y, 0.1, 0.1, 10_000)


def test_sscv_finds_the_under_covered_size() -> None:
    # 50 size-1 sets at 70% coverage, 50 size-2 sets (always cover): the
    # marginal 85% looks close to 0.9, the size-1 stratum is 20 points short.
    covered = np.r_[np.arange(50) < 35, np.ones(50, bool)]
    size = np.r_[np.ones(50, np.int64), np.full(50, 2)]
    assert sscv(covered, size, 0.1) == pytest.approx(0.2)


def test_sscv_ignores_sizes_below_the_row_floor() -> None:
    covered = np.r_[np.zeros(3, bool), np.ones(97, bool)]
    size = np.r_[np.zeros(3, np.int64), np.full(97, 2)]
    assert sscv(covered, size, 0.1) == pytest.approx(0.1)
    assert sscv(covered, size, 0.1, min_rows=1) == pytest.approx(0.9)
    assert np.isnan(sscv(covered[:3], size[:3], 0.1))


def test_conditional_summary_splits_classes_and_strata() -> None:
    sites, y, s, probs = _sites(0.2)
    (fold,) = (f for f in loso_folds(sites, ("a", "b", "c")) if f.site == "c")
    res = run_loso(fold, probs, s, y, 0.1, 0.1, 400, n_seeds=10)
    half = np.zeros(y.size, bool)
    half[::2] = True
    c = summarize_conditional(res, y, 0.1, 0.1, strata={"half": half})
    seed0 = res[0]
    yt = y[seed0.test]
    assert seed0.covered.size == seed0.test.size == fold.target.size
    marg = np.mean([r.coverage for r in res])
    share = yt.mean()
    assert c["coverage_defect"] * share + c["coverage_clean"] * (1 - share) == pytest.approx(marg)
    assert sum(c[f"size{k}_frac"] for k in range(3)) == pytest.approx(1.0)
    miss = np.mean([r.miss_rate for r in res]) / 0.1
    assert 0.5 * miss < c["half_miss_ratio"] < 1.5 * miss + 0.5
