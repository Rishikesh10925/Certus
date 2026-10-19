"""Covers conformal/quantile.py -- the ceil((n+1)(1-alpha))/n off-by-one
(roadmap Section 3.2.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from certus.conformal.quantile import conformal_quantile, conformal_quantile_level, conformal_rank


@pytest.mark.parametrize(
    ("n", "alpha", "k"),
    [
        (9, 0.1, 9),  # (n+1)(1-alpha) = 9 exactly -- must not round up to 10
        (10, 0.1, 10),  # 9.9 -> 10
        (19, 0.1, 18),  # 18 exactly
        (20, 0.1, 19),  # 18.9 -> 19
        (99, 0.05, 95),
        (999, 0.01, 990),
        (5, 0.1, 6),  # 5.4 -> 6 > n: too few points to certify 90%
    ],
)
def test_conformal_rank_matches_hand_computed_values(n: int, alpha: float, k: int) -> None:
    assert conformal_rank(n, alpha) == k
    assert conformal_quantile_level(n, alpha) == pytest.approx(k / n)


def test_conformal_quantile_returns_kth_order_statistic() -> None:
    scores = np.arange(1.0, 21.0)  # 1..20
    # k = ceil(21 * 0.9) = 19 -> the 19th smallest score, 19.0.
    assert conformal_quantile(scores, alpha=0.1) == 19.0


def test_conformal_quantile_is_not_the_naive_quantile() -> None:
    """The exact bug this module exists to prevent: np.quantile at 1-alpha
    interpolates to 18.1 here, below the 19th order statistic, and
    under-covers."""
    scores = np.arange(1.0, 21.0)
    naive = float(np.quantile(scores, 0.9))
    assert naive < conformal_quantile(scores, alpha=0.1)


def test_conformal_quantile_is_permutation_invariant() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=101)
    assert conformal_quantile(scores, 0.1) == conformal_quantile(rng.permutation(scores), 0.1)


def test_conformal_quantile_is_infinite_when_n_too_small() -> None:
    assert conformal_quantile([0.1, 0.2, 0.3, 0.4, 0.5], alpha=0.1) == math.inf


def test_conformal_quantile_returns_a_python_float() -> None:
    q = conformal_quantile([0.3, 0.1, 0.2], alpha=0.5)
    assert isinstance(q, float)
    assert not isinstance(q, np.floating)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_conformal_quantile_rejects_bad_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        conformal_quantile([0.1, 0.2], alpha)


@pytest.mark.parametrize(
    "scores",
    [np.array([]), np.ones((3, 2)), np.array([0.1, np.nan]), np.array([0.1, np.inf])],
)
def test_conformal_quantile_rejects_bad_scores(scores: np.ndarray) -> None:
    with pytest.raises(ValueError):
        conformal_quantile(scores, 0.1)
