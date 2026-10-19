"""Split conformal prediction: LAC (Sadinle et al.), APS (Romano et al.),
and RAPS (Angelopoulos et al.) -- the three non-risk-control conformal
baselines (B5-B7). Guarantee: P(Y_test in C(X_test)) >= 1 - alpha, for
exchangeable data, any model, any n. See roadmap Section 3.2.1.

Week 3 scope: LAC only. APS and RAPS land in Week 4.

Naming: calibration entry points are `calibrate_*`, never `fit_*` -- the
blueprint reserves `fit_` for functions that must not see calibration data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from certus.conformal.quantile import conformal_quantile


def _check_probs(probs: ArrayLike) -> NDArray[np.float64]:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"probs must be 2-D (n, n_classes), got shape {p.shape}")
    return p


def lac_scores(probs: ArrayLike, labels: ArrayLike) -> NDArray[np.float64]:
    """LAC nonconformity score: 1 - p(true class)."""
    p = _check_probs(probs)
    y = np.asarray(labels, dtype=np.intp)
    if y.shape != (p.shape[0],):
        raise ValueError(f"labels must have shape ({p.shape[0]},), got {y.shape}")
    scores: NDArray[np.float64] = 1.0 - p[np.arange(p.shape[0]), y]
    return scores


def calibrate_lac(cal_probs: ArrayLike, cal_labels: ArrayLike, alpha: float) -> float:
    """q_hat for LAC from a calibration set. May be +inf if the set is too
    small for alpha (see conformal_quantile)."""
    return conformal_quantile(lac_scores(cal_probs, cal_labels), alpha)


def predict_sets_lac(probs: ArrayLike, q_hat: float) -> NDArray[np.bool_]:
    """Boolean (n, n_classes) mask: class c is in the set iff its LAC score
    1 - p_c <= q_hat. Computed on the score, not as p_c >= 1 - q_hat, so the
    comparison is bit-identical to the one used at calibration."""
    return (1.0 - _check_probs(probs)) <= q_hat
