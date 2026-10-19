"""Split conformal prediction: LAC (Sadinle et al.), APS (Romano et al.),
and RAPS (Angelopoulos et al.) -- the three non-risk-control conformal
baselines (B5-B7). Guarantee: P(Y_test in C(X_test)) >= 1 - alpha, for
exchangeable data, any model, any n. See roadmap Section 3.2.1.

All three share one shape: build an (n, n_classes) score matrix, calibrate
q_hat on the true-label column of the calibration rows, and put class c in
the set iff its score <= q_hat. Calibration and prediction read the same
matrix function, so the comparison is bit-identical on both sides.

APS/RAPS randomisation: pass `u` (one uniform draw per row) for the
randomised scores, which give coverage closest to 1 - alpha; leave it as
None for the deterministic version, which over-covers slightly. Either way
the guarantee holds -- the same `u` convention must be used at
calibration and test time.

Naming: calibration entry points are `calibrate_*`, never `fit_*` -- the
blueprint reserves `fit_` for functions that must not see calibration data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from certus.conformal.quantile import conformal_quantile

FloatMatrix = NDArray[np.float64]


def _check_probs(probs: ArrayLike) -> FloatMatrix:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"probs must be 2-D (n, n_classes), got shape {p.shape}")
    return p


def _check_labels(labels: ArrayLike, n: int) -> NDArray[np.intp]:
    y = np.asarray(labels, dtype=np.intp)
    if y.shape != (n,):
        raise ValueError(f"labels must have shape ({n},), got {y.shape}")
    return y


def lac_score_matrix(probs: ArrayLike) -> FloatMatrix:
    """LAC: 1 - p_c for every class c."""
    return 1.0 - _check_probs(probs)


def raps_score_matrix(
    probs: ArrayLike, lam: float, k_reg: int, u: ArrayLike | None = None
) -> FloatMatrix:
    """RAPS score for every class: probability mass ranked strictly above c,
    plus u * p_c (or all of p_c if u is None), plus lam * max(0, rank_c -
    k_reg) with rank 1-indexed. lam = 0 is plain APS."""
    p = _check_probs(probs)
    n, k = p.shape
    order = np.argsort(-p, axis=1, kind="stable")
    sorted_p = np.take_along_axis(p, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    if u is None:
        sorted_scores = cum
    else:
        uu = np.asarray(u, dtype=np.float64).reshape(n, 1)
        sorted_scores = cum - sorted_p + uu * sorted_p
    ranks = np.arange(1, k + 1)
    sorted_scores = sorted_scores + lam * np.maximum(0, ranks - k_reg)
    scores = np.empty_like(sorted_scores)
    np.put_along_axis(scores, order, sorted_scores, axis=1)
    return scores


def aps_score_matrix(probs: ArrayLike, u: ArrayLike | None = None) -> FloatMatrix:
    """APS: probability mass ranked above c, plus (u times) p_c."""
    return raps_score_matrix(probs, lam=0.0, k_reg=0, u=u)


def calibrate_split(score_matrix: ArrayLike, labels: ArrayLike, alpha: float) -> float:
    """q_hat from a calibration score matrix: the conformal quantile of the
    true-label scores. May be +inf if n is too small for alpha."""
    s = _check_probs(score_matrix)
    y = _check_labels(labels, s.shape[0])
    return conformal_quantile(s[np.arange(s.shape[0]), y], alpha)


def predict_sets(score_matrix: ArrayLike, q_hat: float) -> NDArray[np.bool_]:
    """Boolean (n, n_classes) mask: class c is in the set iff score <= q_hat."""
    return _check_probs(score_matrix) <= q_hat


def lac_scores(probs: ArrayLike, labels: ArrayLike) -> NDArray[np.float64]:
    """LAC nonconformity score of the true class: 1 - p_true."""
    p = _check_probs(probs)
    y = _check_labels(labels, p.shape[0])
    scores: NDArray[np.float64] = 1.0 - p[np.arange(p.shape[0]), y]
    return scores


def calibrate_lac(cal_probs: ArrayLike, cal_labels: ArrayLike, alpha: float) -> float:
    """q_hat for LAC from a calibration set."""
    return calibrate_split(lac_score_matrix(cal_probs), cal_labels, alpha)


def predict_sets_lac(probs: ArrayLike, q_hat: float) -> NDArray[np.bool_]:
    """LAC prediction sets for a fixed q_hat."""
    return predict_sets(lac_score_matrix(probs), q_hat)
