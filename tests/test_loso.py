"""Covers eval/loso.py: the fold generator, and that LOSO stays valid when
sites are exchangeable but breaks when one site is shifted."""

from __future__ import annotations

import numpy as np
import pytest

from certus.eval.loso import loso_folds, run_in_site, run_loso, summarize_loso


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
