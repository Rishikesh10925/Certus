"""Covers conformal/crc.py: the monotonicity of the missed-defect loss
(roadmap: "unit-test monotonicity of L(lambda) explicitly"), the CRC
threshold on hand-checkable inputs, and the risk guarantee itself over
many synthetic exchangeable trials."""

from __future__ import annotations

import math

import numpy as np
import pytest

from certus.conformal.crc import (
    auto_clear,
    calibrate_crc,
    check_monotone,
    missed_defect_loss,
)


def _pool(rng: np.random.Generator, n: int, prevalence: float = 0.3) -> tuple[np.ndarray, ...]:
    """Defects score low (look defect-like), clean images score high, with
    overlap -- an imperfect, exchangeable detector."""
    y = rng.random(n) < prevalence
    s = np.where(y, rng.normal(0.3, 0.2, n), rng.normal(0.7, 0.2, n))
    return s, y


def test_missed_defect_loss_is_non_increasing_in_lambda() -> None:
    rng = np.random.default_rng(0)
    s, y = _pool(rng, 500)
    lambdas = np.linspace(-1, 2, 301)
    per_example = missed_defect_loss(s, y, lambdas)
    assert np.all(np.diff(per_example, axis=1) <= 0)  # every L_i, not just the mean
    check_monotone(per_example.mean(axis=0), lambdas)


def test_check_monotone_catches_a_sign_error() -> None:
    """The flipped rule (clear when s < lambda) makes risk increase in
    lambda -- the bug the docstring warns about must not pass silently."""
    rng = np.random.default_rng(1)
    s, y = _pool(rng, 200)
    lambdas = np.linspace(-1, 2, 50)
    flipped = (y[:, None] & (s[:, None] < lambdas[None, :])).mean(axis=0)
    with pytest.raises(ValueError, match="non-increasing"):
        check_monotone(flipped, lambdas)


def test_loss_counts_only_cleared_defects() -> None:
    loss = missed_defect_loss([0.2, 0.8, 0.9], [True, True, False], [0.5])
    assert loss[:, 0].tolist() == [0.0, 1.0, 0.0]  # clean image cleared costs nothing


def test_calibrate_crc_hand_example() -> None:
    # 9 defects at scores 0.1..0.9, n = 9. Clearing above lambda = 0.8 misses
    # one defect: 9/10 * 1/9 + 1/10 = 0.2.
    s = np.arange(1, 10) / 10
    y = np.ones(9, dtype=bool)
    assert calibrate_crc(s, y, epsilon=0.2) == pytest.approx(0.8)
    # epsilon = 0.1 = B/(n+1) leaves no room for any miss: clear above max.
    assert calibrate_crc(s, y, epsilon=0.1) == pytest.approx(0.9)
    # below B/(n+1), nothing can be certified at all
    assert calibrate_crc(s, y, epsilon=0.09) == math.inf


def test_calibrate_crc_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        calibrate_crc([0.1], [1], 0.0)
    with pytest.raises(ValueError, match="finite"):
        calibrate_crc([0.1, np.nan], [1, 0], 0.1)


@pytest.mark.parametrize("epsilon", [0.10, 0.05, 0.02])
@pytest.mark.parametrize("conditional", [False, True])
def test_crc_controls_risk_over_trials(epsilon: float, conditional: bool) -> None:
    """The guarantee: averaged over fresh exchangeable cal/test draws, the
    realised test risk is <= epsilon (within Monte-Carlo error), and not
    wildly conservative (CRC is tight up to O(1/n), Theorem 2)."""
    rng = np.random.default_rng(int(epsilon * 1000) + 7 * conditional)
    risks = []
    for _ in range(400):
        s, y = _pool(rng, 1500)
        cal_s, cal_y, test_s, test_y = s[:1000], y[:1000], s[1000:], y[1000:]
        if conditional:
            lam = calibrate_crc(cal_s[cal_y], cal_y[cal_y], epsilon)
            cleared = auto_clear(test_s[test_y], lam)
            risks.append(cleared.mean())  # fraction of real defects missed
        else:
            lam = calibrate_crc(cal_s, cal_y, epsilon)
            risks.append((auto_clear(test_s, lam) & test_y).mean())
    risks_arr = np.array(risks)
    se = risks_arr.std(ddof=1) / np.sqrt(risks_arr.size)
    assert risks_arr.mean() <= epsilon + 4 * se
    assert risks_arr.mean() >= epsilon - 0.01 - 4 * se
