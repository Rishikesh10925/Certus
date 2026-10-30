"""Covers eval/protocol.py (the seed harness) and baselines.py (B1-B3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from certus.baselines import (
    fit_selective_threshold,
    fit_temperature,
    naive_cumulative_sets,
    selective_sets,
    temperature_scale,
)
from certus.eval.protocol import CONFORMAL, METHODS, run_protocol, summarize


def _pool(
    n: int = 2000, k: int = 3, seed: int = 0, sharpen: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Exchangeable pool with an overconfident model (probs =
    softmax(sharpen * z) while labels follow softmax(z))."""
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=1.5, size=(n, k))
    true_p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    labels = (rng.random((n, 1)) > true_p.cumsum(axis=1)).sum(axis=1).clip(max=k - 1)
    model_p = np.exp(sharpen * z) / np.exp(sharpen * z).sum(axis=1, keepdims=True)
    return model_p, labels.astype(np.intp)


def test_naive_cumulative_sets_stop_once_mass_reached() -> None:
    # row 0: 0.5 + 0.3 = 0.8 < 0.9, so the third class is still needed;
    # row 1: 0.95 alone already reaches 0.9
    probs = np.array([[0.5, 0.3, 0.2], [0.95, 0.04, 0.01]])
    sets = naive_cumulative_sets(probs, alpha=0.1)
    assert sets.tolist() == [[True, True, True], [True, False, False]]
    assert naive_cumulative_sets(probs, alpha=0.25)[0].tolist() == [True, True, False]


def test_fit_temperature_undoes_overconfidence() -> None:
    probs, labels = _pool(5000)
    t = fit_temperature(probs, labels, ["tune"] * len(labels))
    assert 1.6 < t < 2.4  # the model is softmax(2z) of the true softmax(z)
    np.testing.assert_allclose(temperature_scale(probs, 1.0), probs, atol=1e-9)


def test_baseline_fits_refuse_calibration_rows() -> None:
    probs, labels = _pool(10)
    with pytest.raises(ValueError, match="fit_"):
        fit_temperature(probs, labels, ["cal"] * 10)
    with pytest.raises(ValueError, match="fit_"):
        fit_selective_threshold(probs, labels, 0.1, ["test"] * 10)


def test_selective_threshold_meets_alpha_on_its_own_rows() -> None:
    probs, labels = _pool(1000)
    for alpha in (0.05, 0.1, 0.3):
        tau = fit_selective_threshold(probs, labels, alpha, ["tune"] * 1000)
        sets = selective_sets(probs, tau)
        miscoverage = 1 - sets[np.arange(1000), labels].mean()
        assert miscoverage <= alpha + 1e-12
        answered = (sets.sum(axis=1) == 1).mean()
        assert 0 < answered < 1


def test_run_protocol_is_seeded_and_partitions_the_pool() -> None:
    probs, labels = _pool(300)
    a = run_protocol(probs, labels, "lac", 0.1, n_cal=100, n_seeds=5)
    b = run_protocol(probs, labels, "lac", 0.1, n_cal=100, n_seeds=5)
    assert a == b
    assert len({r.coverage for r in a}) > 1  # different partitions per seed
    with pytest.raises(ValueError, match="n_cal"):
        run_protocol(probs, labels, "lac", 0.1, n_cal=300)


@pytest.mark.parametrize("method", CONFORMAL)
def test_conformal_methods_hold_coverage_through_the_harness(method: str) -> None:
    probs, labels = _pool(3000)
    s = summarize(run_protocol(probs, labels, method, 0.1, n_cal=500, n_seeds=40))
    assert s["coverage_mean"] >= 0.9 - 4 * s["coverage_sd"] / np.sqrt(40)
    assert s["coverage_ci_lo"] <= s["coverage_mean"] <= s["coverage_ci_hi"]


def test_raw_softmax_baseline_undercovers_an_overconfident_model() -> None:
    """The point of B1 in F2: trusting raw softmax mass misses nominal."""
    probs, labels = _pool(3000, sharpen=4.0)
    s = summarize(run_protocol(probs, labels, "b1_softmax", 0.1, n_cal=500, n_seeds=20))
    assert s["coverage_mean"] < 0.85
    assert set(METHODS) >= {"b1_softmax", "b2_temperature", "b3_selective"}


def test_figure_f2_renders(tmp_path: Path) -> None:
    from certus.viz.figures import figure_f2

    rows = [
        {
            "dataset": "d",
            "score": "s1",
            "method": m,
            "alpha": a,
            "n_cal": 50,
            "coverage_mean": 1 - a,
            "coverage_sd": 0.01,
        }
        for m in ("lac", "b1_softmax")
        for a in (0.1, 0.2)
    ]
    out = tmp_path / "f2.png"
    figure_f2(rows, "s1", ["d"], [50], ["lac", "b1_softmax"], out)
    assert out.stat().st_size > 1000


@pytest.mark.parametrize("conditional", [False, True])
def test_crc_protocol_controls_risk_and_reports_hrf(conditional: bool) -> None:
    from certus.eval.protocol import run_crc_protocol, summarize_crc

    rng = np.random.default_rng(3)
    y = rng.random(3000) < 0.2
    s = np.where(y, rng.normal(0.3, 0.2, 3000), rng.normal(0.7, 0.2, 3000))
    res = run_crc_protocol(s, y, epsilon=0.05, n_cal=2000, conditional=conditional, n_seeds=60)
    summ = summarize_crc(res)
    assert summ["risk_mean"] <= 0.05 + 4 * summ["risk_se"]
    assert 0.0 < summ["hrf_mean"] < 1.0
    assert summ["lam_finite_frac"] == 1.0
    assert res == run_crc_protocol(s, y, 0.05, 2000, conditional, n_seeds=60)


def test_figure_f3_renders(tmp_path: Path) -> None:
    from certus.viz.figures import figure_f3

    rows = [
        {"score": s, "guarantee": "joint", "epsilon": 0.05, "hrf_mean": h, "hrf_sd": 0.02}
        for s, h in (("s1", 0.4), ("s7", 0.3))
    ]
    out = tmp_path / "f3.png"
    figure_f3(rows, [("joint", 0.05)], out, {"s1": 0.7, "s7": 0.8})
    assert out.stat().st_size > 1000


def test_crc_test_scores_shift_only_the_test_rows() -> None:
    """Identical test_scores change nothing; lowering them clears fewer."""
    from certus.eval.protocol import run_crc_protocol

    rng = np.random.default_rng(0)
    y = rng.random(1000) < 0.3
    s = np.where(y, rng.normal(0.3, 0.2, 1000), rng.normal(0.7, 0.2, 1000))
    base = run_crc_protocol(s, y, 0.05, 500, False, n_seeds=5)
    same = run_crc_protocol(s, y, 0.05, 500, False, n_seeds=5, test_scores=s)
    low = run_crc_protocol(s, y, 0.05, 500, False, n_seeds=5, test_scores=s - 5.0)
    assert base == same
    assert all(a.lam == b.lam for a, b in zip(base, low, strict=True))
    assert all(r.hrf == 1.0 for r in low)
