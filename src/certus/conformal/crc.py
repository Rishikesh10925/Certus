"""Conformal risk control (Angelopoulos et al., ICLR 2024) -- the primary
method. Bounds E[L_test(lambda_hat)] <= epsilon for the monotone,
asymmetric missed-defect loss rather than symmetric coverage. Monotonicity
of L(lambda) must be verified explicitly (a non-increasing check) -- a sign
error here silently returns garbage. See roadmap Section 3.2.2.

Decision rule, in the shared score orientation (high score = little
evidence of a defect): AUTO_CLEAR an image iff s(x) > lambda. Raising
lambda clears fewer images, so the missed-defect loss

    L_i(lambda) = 1{ y_i = defect  and  s(x_i) > lambda }

is non-increasing in lambda -- the property CRC needs. lambda = +inf
clears nothing and has zero loss, which is the fallback when no finite
lambda is certifiable.

    lambda_hat = inf{ lambda : n/(n+1) * R_hat_n(lambda) + B/(n+1) <= epsilon }

with B = 1 (the loss is in [0, 1]). The strict ">" makes R_hat
right-continuous, so the infimum is attained at a calibration score.

Two guarantees, depending on which rows are calibrated on:
- joint (default): E[1{defect and cleared}] <= epsilon over all images.
- conditional (`calibrate_crc` on defect-positive rows only): P(cleared |
  defect) <= epsilon, i.e. at most epsilon of real defects are missed.
  Needs n_pos >= 1/epsilon - 1, the 10/epsilon rule's hard floor.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

LOSS_BOUND = 1.0  # B: the missed-defect loss is an indicator


def missed_defect_loss(
    scores: ArrayLike, labels: ArrayLike, lambdas: ArrayLike
) -> NDArray[np.float64]:
    """(n, n_lambdas) matrix of L_i(lambda_j) = 1{y_i = 1 and s_i > lambda_j}."""
    s = np.asarray(scores, dtype=np.float64)[:, None]
    y = np.asarray(labels).astype(bool)[:, None]
    lam = np.asarray(lambdas, dtype=np.float64)[None, :]
    return (y & (s > lam)).astype(np.float64)


def check_monotone(risk: NDArray[np.float64], lambdas: NDArray[np.float64]) -> None:
    """Fails loudly unless the empirical risk is non-increasing in lambda --
    the one property the CRC guarantee rests on (Proposition 2 of the paper
    shows control can fail outright without it)."""
    order = np.argsort(lambdas, kind="stable")
    if np.any(np.diff(risk[order]) > 1e-12):
        raise ValueError("risk is not non-increasing in lambda -- check the loss's sign")


def calibrate_crc(scores: ArrayLike, labels: ArrayLike, epsilon: float) -> float:
    """lambda_hat for the missed-defect loss. +inf if even clearing nothing
    can't be certified at this n (needs epsilon > B/(n+1)) -- then every
    image goes to a human, which is the honest answer."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or s.size == 0 or not np.all(np.isfinite(s)):
        raise ValueError("scores must be a non-empty, finite 1-D array")
    n = s.size
    # R_hat only changes at calibration scores, so those (plus +inf) are
    # the only candidates for the infimum.
    lambdas = np.append(np.unique(s), math.inf)
    risk = missed_defect_loss(s, labels, lambdas).mean(axis=0)
    check_monotone(risk, lambdas)
    ok = n / (n + 1) * risk + LOSS_BOUND / (n + 1) <= epsilon
    if not ok.any():
        return math.inf
    return float(lambdas[np.argmax(ok)])  # lambdas ascend; first True is the inf


def auto_clear(scores: ArrayLike, lam: float) -> NDArray[np.bool_]:
    """True where the image is auto-cleared (s > lambda)."""
    return np.asarray(scores, dtype=np.float64) > lam
