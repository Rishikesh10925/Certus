"""Covers conformal/ltt.py: the Hoeffding-Bentkus p-value, fixed-sequence
testing, and the PAC guarantee itself -- over many fresh calibration draws,
the *true* risk of lambda_hat exceeds epsilon in at most delta of them."""

from __future__ import annotations

import math

import numpy as np
import pytest

from certus.conformal.crc import auto_clear, calibrate_crc
from certus.conformal.ltt import calibrate_ltt, hoeffding_bentkus_pvalue


def _pool(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    y = rng.random(n) < 0.3
    s = np.where(y, rng.normal(0.3, 0.2, n), rng.normal(0.7, 0.2, n))
    return s, y


def test_pvalue_behaves() -> None:
    # zero observed risk on many samples: overwhelming evidence risk < eps
    assert hoeffding_bentkus_pvalue(0.0, 1000, 0.05) < 1e-10
    # observed risk above epsilon: no evidence at all
    assert hoeffding_bentkus_pvalue(0.08, 1000, 0.05) > 0.5
    # more data, same observed risk below epsilon -> smaller p
    assert hoeffding_bentkus_pvalue(0.03, 2000, 0.05) < hoeffding_bentkus_pvalue(0.03, 200, 0.05)
    with pytest.raises(ValueError, match="epsilon"):
        hoeffding_bentkus_pvalue(0.1, 10, 1.0)


def test_ltt_is_at_least_as_conservative_as_crc() -> None:
    """Same loss, stronger promise: LTT never clears more images than CRC
    on the same calibration set (lambda_hat_LTT >= lambda_hat_CRC)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        s, y = _pool(rng, 1000)
        assert calibrate_ltt(s, y, 0.05, 0.1) >= calibrate_crc(s, y, 0.05)


def test_ltt_clears_nothing_when_n_is_too_small() -> None:
    s = np.array([0.2, 0.8, 0.9])
    y = np.array([True, False, False])
    assert calibrate_ltt(s, y, epsilon=0.01, delta=0.05) == math.inf


@pytest.mark.parametrize("epsilon", [0.05, 0.02])
def test_ltt_pac_guarantee_over_trials(epsilon: float) -> None:
    """P(R(lambda_hat) > epsilon) <= delta, with R measured on a huge fresh
    sample standing in for the true risk."""
    delta = 0.1
    rng = np.random.default_rng(42)
    big_s, big_y = _pool(rng, 200_000)
    violations = 0
    trials = 300
    for _ in range(trials):
        s, y = _pool(rng, 1000)
        lam = calibrate_ltt(s, y, epsilon, delta)
        true_risk = (auto_clear(big_s, lam) & big_y).mean()
        violations += true_risk > epsilon
    # allow Monte-Carlo slack: binomial(300, 0.1) sd ~ 5.2
    assert violations / trials <= delta + 3 * math.sqrt(delta * (1 - delta) / trials)


def test_vectorised_pvalues_match_scalar() -> None:
    from certus.conformal.ltt import hoeffding_bentkus_pvalues

    grid = np.array([0.0, 0.001, 0.01, 0.03, 0.05, 0.2, 1.0])
    vec = hoeffding_bentkus_pvalues(grid, 500, 0.05)
    np.testing.assert_allclose(vec, [hoeffding_bentkus_pvalue(r, 500, 0.05) for r in grid])
