"""Covers eval/protocol.py (the seed harness) and baselines.py (B1-B3)."""

from __future__ import annotations

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
