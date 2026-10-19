"""THE corrected conformal quantile: ceil((n+1)(1-alpha)) / n, not 1-alpha.
This is the single most common conformal implementation bug in the wild
(roadmap Section 3.2.1) -- with small n the difference is the gap between a
valid and an invalid guarantee. Every other conformal module calls through
this one function; it is covered by tests/test_quantile.py and must never
be reimplemented inline elsewhere.

Implemented as an exact order statistic rather than via `np.quantile`:
`np.quantile`'s default linear interpolation lands *between* two
calibration scores, and anything below the k-th order statistic voids the
guarantee. When k = ceil((n+1)(1-alpha)) exceeds n the calibration set is
too small to certify 1-alpha at all, and the honest answer is an infinite
threshold (every label in every set), not the largest score.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def conformal_rank(n: int, alpha: float) -> int:
    """k = ceil((n+1)(1-alpha)) -- the 1-indexed order statistic of n
    calibration scores that serves as the conformal threshold. May exceed
    n, in which case no finite threshold is valid."""
    _check_alpha(alpha)
    if n < 1:
        raise ValueError(f"need at least one calibration score, got n={n}")
    # Defensive: if float error ever lands an exact integer product a hair
    # above itself, a bare ceil would jump to the next order statistic. The
    # real fractional parts here are multiples of 1/100-ish, far above 1e-9.
    return math.ceil((n + 1) * (1.0 - alpha) - 1e-9)


def conformal_quantile_level(n: int, alpha: float) -> float:
    """ceil((n+1)(1-alpha)) / n. Values above 1 mean the calibration set is
    too small for this alpha."""
    return conformal_rank(n, alpha) / n


def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """The split-conformal threshold q_hat: the k-th smallest calibration
    score, k = ceil((n+1)(1-alpha)), or +inf if k > n.

    Guarantee (exchangeable scores): P(s_test <= q_hat) >= 1 - alpha, and,
    absent ties, <= 1 - alpha + 1/(n+1).
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {s.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    k = conformal_rank(s.size, alpha)
    if k > s.size:
        return math.inf
    return float(np.partition(s, k - 1)[k - 1])
