"""Covers eval/label_noise.py and the noisy-label / LTT paths through the
risk-control protocol."""

from __future__ import annotations

import math

import numpy as np
import pytest

from certus.eval.label_noise import cohen_kappa, hide_defects
from certus.eval.protocol import run_crc_protocol, summarize_crc


def test_cohen_kappa_known_values() -> None:
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)
    # textbook 2x2: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> kappa 0.4
    a = [1] * 25 + [0] * 25
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert math.isnan(cohen_kappa([1, 1], [1, 1]))


def test_hide_defects_only_flips_positives() -> None:
    rng = np.random.default_rng(0)
    y = np.array([True] * 100 + [False] * 50)
    noisy = hide_defects(y, 0.42, rng)
    assert noisy[100:].sum() == 0  # clean rows untouched
    assert noisy[:100].sum() == 58  # 42 of 100 defects hidden
    assert np.array_equal(hide_defects(y, 0.0, rng), y)
    with pytest.raises(ValueError, match="rate"):
        hide_defects(y, 1.5, rng)


def _pool(n: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9)
    y = rng.random(n) < 0.2
    s = np.where(y, rng.normal(0.3, 0.2, n), rng.normal(0.7, 0.2, n))
    return s, y


def test_hide_defects_informative_hides_the_most_clean_looking() -> None:
    y = np.array([True, True, True, True, False])
    s = np.array([0.1, 0.9, 0.5, 0.7, 0.95])
    noisy = hide_defects(y, 0.5, np.random.default_rng(0), scores=s)
    assert noisy.tolist() == [True, False, True, False, False]


def _summ(s: np.ndarray, y: np.ndarray, conditional: bool, cal_y: np.ndarray | None) -> dict:
    res = run_crc_protocol(s, y, 0.05, 2000, conditional, n_seeds=40, calibration_labels=cal_y)
    return summarize_crc(res, 0.05)


def test_random_hiding_breaks_joint_but_not_conditional() -> None:
    s, y = _pool()
    noisy = hide_defects(y, 0.42, np.random.default_rng(1))
    clean_joint, dirty_joint = _summ(s, y, False, None), _summ(s, y, False, noisy)
    assert clean_joint["risk_mean"] <= 0.05 + 4 * clean_joint["risk_se"]
    assert dirty_joint["risk_mean"] > 0.05 + 4 * dirty_joint["risk_se"]
    dirty_cond = _summ(s, y, True, noisy)
    assert dirty_cond["risk_mean"] <= 0.05 + 4 * dirty_cond["risk_se"]


def test_informative_hiding_breaks_conditional() -> None:
    s, y = _pool()
    noisy = hide_defects(y, 0.42, np.random.default_rng(1), scores=s)
    dirty_cond = _summ(s, y, True, noisy)
    assert dirty_cond["risk_mean"] > 0.05 + 4 * dirty_cond["risk_se"]


def test_ltt_exceeds_epsilon_less_often_than_crc_and_costs_more_review() -> None:
    s, y = _pool()
    crc = summarize_crc(run_crc_protocol(s, y, 0.05, 2000, False, n_seeds=60), 0.05)
    ltt = summarize_crc(
        run_crc_protocol(s, y, 0.05, 2000, False, n_seeds=60, method="ltt", delta=0.05), 0.05
    )
    assert ltt["exceed_frac"] < crc["exceed_frac"]
    assert ltt["hrf_mean"] >= crc["hrf_mean"]
    with pytest.raises(ValueError, match="method"):
        run_crc_protocol(s, y, 0.05, 2000, False, n_seeds=1, method="bogus")
