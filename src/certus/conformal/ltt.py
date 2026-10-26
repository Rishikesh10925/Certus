"""Learn-then-Test -- converts risk control into a multiple-hypothesis-
testing problem over a finite lambda grid, using Hoeffding-Bentkus
p-values and fixed-sequence FWER control, giving
P(R(lambda_hat) <= epsilon) >= 1 - delta. The high-probability,
regulator-grade statement. See roadmap Section 3.2.3.

Same loss and decision rule as crc.py (AUTO_CLEAR iff s > lambda,
L = 1{defect and cleared}). The difference is the promise: CRC bounds the
*expected* risk over calibration draws, so about half of all deployments
land a little above epsilon. LTT bounds the risk of *this* deployment with
probability 1 - delta, and pays for it in human review.

For each lambda, H_lambda: R(lambda) > epsilon is tested with the
Hoeffding-Bentkus p-value (Angelopoulos et al., "Learn then Test", Prop.
3 / Bates et al. 2021):

    p_HB = min( exp(-n h(min(R_hat, eps), eps)),  e * P(Binom(n, eps) <= ceil(n R_hat)) )
    h(a, b) = a log(a/b) + (1-a) log((1-a)/(1-b))

Fixed-sequence testing walks the grid from the safest lambda (+inf, clear
nothing) towards clearing more, rejecting while p <= delta and stopping at
the first failure. Because risk is monotone in lambda, that ordering is
the natural one and needs no Bonferroni split of delta.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binom

from certus.conformal.crc import check_monotone, missed_defect_loss


def _kl_bernoulli(a: float, b: float) -> float:
    """h(a, b): KL divergence between Bernoulli(a) and Bernoulli(b)."""
    out = 0.0
    if a > 0:
        out += a * math.log(a / b)
    if a < 1:
        out += (1 - a) * math.log((1 - a) / (1 - b))
    return out


def hoeffding_bentkus_pvalue(risk_hat: float, n: int, epsilon: float) -> float:
    """p-value for H0: true risk > epsilon, from the empirical risk of n
    i.i.d. losses in [0, 1]. Small p = strong evidence the risk is below
    epsilon."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
    hoeffding = math.exp(-n * _kl_bernoulli(min(risk_hat, epsilon), epsilon))
    bentkus = math.e * float(binom.cdf(math.ceil(n * risk_hat - 1e-9), n, epsilon))
    return min(hoeffding, bentkus, 1.0)


def hoeffding_bentkus_pvalues(
    risk_hat: NDArray[np.float64], n: int, epsilon: float
) -> NDArray[np.float64]:
    """Vectorised hoeffding_bentkus_pvalue over a grid of empirical risks."""
    a = np.minimum(risk_hat, epsilon)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(a > 0, a * np.log(a / epsilon), 0.0) + np.where(
            a < 1, (1 - a) * np.log((1 - a) / (1 - epsilon)), 0.0
        )
    hoeffding = np.exp(-n * kl)
    bentkus = math.e * binom.cdf(np.ceil(n * risk_hat - 1e-9), n, epsilon)
    p: NDArray[np.float64] = np.minimum(np.minimum(hoeffding, bentkus), 1.0)
    return p


def calibrate_ltt(scores: ArrayLike, labels: ArrayLike, epsilon: float, delta: float) -> float:
    """lambda_hat with P(R(lambda_hat) <= epsilon) >= 1 - delta. +inf (clear
    nothing) if not even the first step past +inf can be certified."""
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or s.size == 0 or not np.all(np.isfinite(s)):
        raise ValueError("scores must be a non-empty, finite 1-D array")
    n = s.size
    # Safest first: +inf, then descending calibration scores (each step
    # clears more images, so risk can only go up along the sequence).
    lambdas = np.concatenate([[math.inf], np.unique(s)[::-1]])
    risk = missed_defect_loss(s, labels, lambdas).mean(axis=0)
    check_monotone(risk, lambdas)
    rejected = hoeffding_bentkus_pvalues(risk, n, epsilon) <= delta
    # fixed sequence: keep the run of rejections from the start, stop at
    # the first hypothesis that can't be rejected
    n_rejected = int(np.argmin(rejected)) if not rejected.all() else rejected.size
    return math.inf if n_rejected == 0 else float(lambdas[n_rejected - 1])
