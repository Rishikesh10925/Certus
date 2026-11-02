"""Localized conformal prediction: weights calibration points by similarity
in the VLM's latent space via a nonlinear transform of cosine similarity.
Only attempt once a domain classifier confirms the backbone's
representation genuinely separates sites (t-SNE + AUC check, roadmap
Section 4.4). Sensitive to the similarity transform; the most
implementation-risky of the four shift-correction methods (Section 3.2.4).

The check passed in Week 11: the domain classifier separates the held-out
site from the sources at AUC 0.96-1.00 on CLIP features.

Kernel, for test image x and calibration row i (features L2-normalised):

    d_i(x) = 1 - cos(x, x_i)
    H_i(x) = exp(-(d_i(x) / h(x))^2),   h(x) = d of x's k-th nearest row

with k = ceil(k_frac * n_cal). A fixed bandwidth does not work here: CLIP
distances between images sit in roughly 0.03-0.12, so any h small enough
to localise leaves the calibration mass near zero, and the test point's
own weight H(x, x) = 1 then forces every threshold to +inf. The adaptive
h keeps about k rows in play wherever x lands. Small k_frac localises
harder and has fewer effective rows; k_frac = 1 is close to unweighted.

The threshold is then the weighted quantile with the test point's own
mass, as in conformal/weighted.py, but with one weight vector per test
image. This is the plain localized CP of Guan (2023) without its
randomised correction, and the bandwidth depends on the calibration set,
so validity is approximate, not exact; the LOSO and in-site runs measure
how approximate.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

SELF_WEIGHT = 1.0  # H(x, x): zero distance


def similarity_kernel(
    cal_features: ArrayLike, test_features: ArrayLike, k_frac: float
) -> NDArray[np.float64]:
    """(n_test, n_cal) kernel weights H_i(x) with the adaptive bandwidth."""
    if not 0.0 < k_frac <= 1.0:
        raise ValueError(f"k_frac must be in (0, 1], got {k_frac}")
    xc = _normalise(cal_features)
    xt = _normalise(test_features)
    d = np.clip(1.0 - xt @ xc.T, 0.0, None)
    k = max(1, math.ceil(k_frac * xc.shape[0]))
    h = np.partition(d, k - 1, axis=1)[:, k - 1]
    h = np.maximum(h, 1e-12)  # a duplicate image at distance 0
    kernel: NDArray[np.float64] = np.exp(-np.square(d / h[:, None]))
    return kernel


def localized_quantile(
    cal_scores: ArrayLike, kernel: ArrayLike, alpha: float
) -> NDArray[np.float64]:
    """Per-test-point threshold: the smallest calibration score whose
    cumulative kernel weight reaches (1 - alpha)(sum_i H_i(x) + H(x, x)),
    or +inf if none does. A constant kernel of 1 gives `conformal_quantile`."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = np.asarray(cal_scores, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    if s.ndim != 1 or s.size == 0 or k.ndim != 2 or k.shape[1] != s.size:
        raise ValueError(f"kernel must be (n_test, {s.size}), got {k.shape}")
    order = np.argsort(s, kind="stable")
    cum = np.cumsum(k[:, order], axis=1)
    total = cum[:, -1] + SELF_WEIGHT
    target = (1.0 - alpha) * total - 1e-9 * total  # same guard as conformal_rank
    reached = cum >= target[:, None]
    out = np.full(k.shape[0], np.inf)
    hit = reached.any(axis=1)
    out[hit] = s[order][reached[hit].argmax(axis=1)]
    return out


def row_ess(kernel: ArrayLike) -> NDArray[np.float64]:
    """Kish effective sample size of each test image's calibration weights."""
    k = np.asarray(kernel, dtype=np.float64)
    out: NDArray[np.float64] = k.sum(axis=1) ** 2 / np.square(k).sum(axis=1)
    return out


def _normalise(features: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {x.shape}")
    out: NDArray[np.float64] = x / np.linalg.norm(x, axis=1, keepdims=True)
    return out
