"""Marginal coverage, class-conditional coverage, and size-stratified
coverage violation (SSCV) -- measures of how well (or how deceptively) a
conformal method's guarantee actually holds, including where a marginal
guarantee hides conditional failure. See roadmap Section 5.3.

Week 3 scope: marginal coverage only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def marginal_coverage(sets: ArrayLike, labels: ArrayLike) -> float:
    """Fraction of rows whose prediction set (boolean (n, n_classes) mask)
    contains the true label."""
    s = np.asarray(sets, dtype=bool)
    y = np.asarray(labels, dtype=np.intp)
    if s.ndim != 2 or y.shape != (s.shape[0],):
        raise ValueError(f"shape mismatch: sets {s.shape}, labels {y.shape}")
    return float(s[np.arange(s.shape[0]), y].mean())
