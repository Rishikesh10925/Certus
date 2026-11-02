"""Weighted conformal prediction under covariate shift (Tibshirani et al.
2019): reweights calibration points by an estimated likelihood ratio
w(x) = p_test(x) / p_cal(x) from a domain classifier over VLM features.
Valid only when P(Y|X) is stable across the shift. Must log the effective
sample size every run -- it collapses when weights are extreme and is the
diagnostic that flags silent failure. See roadmap Section 3.2.4.

The threshold is per test point: the level-(1 - a) quantile of

    sum_i  w_i / (W + w(x)) * delta(s_i)  +  w(x) / (W + w(x)) * delta(+inf),

W = sum_i w_i. With equal weights this is exactly `conformal_quantile`.

The same quantile is conditional CRC for the missed-defect loss: on
defect-only calibration rows, sum_i w_i 1{s_i > lam} + w(x) B
<= epsilon (W + w(x)) is the level-(1 - epsilon) weighted quantile of the
defect scores. Weights from a marginal domain classifier are right for the
defect-only rows too: with P(Y|X) fixed, p_t(x | defect) / p_s(x | defect)
is w(x) times a constant, and constants cancel once normalised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

P_CLIP = 1e-6  # keeps p / (1 - p) finite when the classifier is certain


def weighted_quantile(
    cal_scores: ArrayLike, cal_weights: ArrayLike, test_weights: ArrayLike, alpha: float
) -> NDArray[np.float64]:
    """Per-test-point threshold: the smallest calibration score whose
    cumulative weight reaches (1 - alpha)(W + w_test), or +inf if none
    does."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = np.asarray(cal_scores, dtype=np.float64)
    w = np.asarray(cal_weights, dtype=np.float64)
    wt = np.asarray(test_weights, dtype=np.float64)
    if s.ndim != 1 or s.shape != w.shape or s.size == 0:
        raise ValueError("cal_scores and cal_weights must be matching non-empty 1-D arrays")
    if np.any(w < 0) or np.any(wt < 0):
        raise ValueError("weights must be non-negative")
    order = np.argsort(s, kind="stable")
    cum = np.cumsum(w[order])
    # Same guard as conformal_rank: an exact product a hair high must not
    # skip to the next order statistic.
    total = cum[-1] + wt
    target = (1.0 - alpha) * total - 1e-9 * total
    k = np.searchsorted(cum, target, side="left")
    out = np.full(wt.shape, np.inf)
    ok = k < s.size
    out[ok] = s[order][k[ok]]
    return out


def effective_sample_size(weights: ArrayLike) -> float:
    """Kish ESS, (sum w)^2 / sum w^2: n for equal weights, near 1 when one
    point carries all the mass."""
    w = np.asarray(weights, dtype=np.float64)
    return float(w.sum() ** 2 / np.square(w).sum())


@dataclass(frozen=True)
class DomainWeights:
    cal: NDArray[np.float64]  # w(x) for each calibration row
    test: NDArray[np.float64]  # w(x) for each test row
    auc: float  # out-of-fold source-vs-target AUC: 0.5 = no detectable shift


def domain_weights(
    source_features: ArrayLike,
    cal_features: ArrayLike,
    test_features: ArrayLike,
    seed: int = 0,
    c: float = 1.0,
) -> DomainWeights:
    """Likelihood-ratio weights from a logistic source-vs-target classifier.

    Cross-fitted so no row is weighted by a classifier that trained on it:
    the source side trains on `source_features` (source rows outside the
    calibration draw), and the test rows are split in two halves, each
    weighted by the classifier trained on the other half. w = p/(1-p)
    rescaled by the class ratio, so it estimates p_test(x) / p_source(x).
    """
    xs = _normalise(source_features)
    xc = _normalise(cal_features)
    xt = _normalise(test_features)
    rng = np.random.default_rng(seed)
    half = rng.permutation(xt.shape[0]) % 2 == 0
    w_cal = np.zeros(xc.shape[0])
    p_cal = np.zeros(xc.shape[0])
    w_test = np.empty(xt.shape[0])
    p_test = np.empty(xt.shape[0])
    for fold in (half, ~half):
        train_t = xt[~fold]
        clf = LogisticRegression(C=c, max_iter=5000).fit(
            np.vstack([xs, train_t]), np.r_[np.zeros(xs.shape[0]), np.ones(train_t.shape[0])]
        )
        ratio = xs.shape[0] / train_t.shape[0]
        pc = clf.predict_proba(xc)[:, 1]
        w_cal += 0.5 * _odds(pc) * ratio
        p_cal += 0.5 * pc
        p_test[fold] = clf.predict_proba(xt[fold])[:, 1]
        w_test[fold] = _odds(p_test[fold]) * ratio
    # Neither classifier trained on the calibration rows, and each test row
    # is scored by the classifier that did not see it.
    y = np.r_[np.zeros(xc.shape[0]), np.ones(xt.shape[0])]
    auc = float(roc_auc_score(y, np.r_[p_cal, p_test]))
    return DomainWeights(cal=w_cal, test=w_test, auc=auc)


def _odds(p: NDArray[np.float64]) -> NDArray[np.float64]:
    q = np.clip(p, P_CLIP, 1.0 - P_CLIP)
    out: NDArray[np.float64] = q / (1.0 - q)
    return out


def _normalise(features: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {x.shape}")
    out: NDArray[np.float64] = x / np.linalg.norm(x, axis=1, keepdims=True)
    return out
