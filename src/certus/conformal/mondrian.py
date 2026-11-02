"""Mondrian / group-conditional conformal prediction: a separate threshold
per site cluster, giving a guarantee conditional on cluster membership.
Data-hungry -- needs roughly 100+ positives per cluster before a small
epsilon is supportable. See roadmap Section 3.2.4.

Under leave-one-site-out the held-out site has no calibration rows, so the
groups cannot be the sites themselves. They are k-means clusters of the
frozen VLM features, fit on the source sites without labels; a test image
joins its nearest cluster and inherits that cluster's threshold. The
guarantee is then per cluster: it transfers to the target only as far as
target images resemble the source images they are clustered with.

A cluster too small for the requested level gets +inf -- every image in it
goes to a human -- which is where the data hunger shows up as HRF.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import KMeans

from certus.conformal.crc import calibrate_crc
from certus.conformal.split_cp import calibrate_lac


class FeatureClusters:
    """k-means over L2-normalised features. Unsupervised, so fitting it on
    rows that later calibrate does not touch any label."""

    def __init__(self, features: ArrayLike, k: int, seed: int = 0) -> None:
        x = _normalise(features)
        if not 1 <= k <= x.shape[0]:
            raise ValueError(f"k={k} needs 1..{x.shape[0]} rows")
        self.k = k
        self._km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)

    def assign(self, features: ArrayLike) -> NDArray[np.intp]:
        groups: NDArray[np.intp] = self._km.predict(_normalise(features)).astype(np.intp)
        return groups


def _normalise(features: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {x.shape}")
    out: NDArray[np.float64] = x / np.linalg.norm(x, axis=1, keepdims=True)
    return out


def calibrate_mondrian_crc(
    scores: ArrayLike, labels: ArrayLike, groups: ArrayLike, epsilon: float, n_groups: int
) -> NDArray[np.float64]:
    """lambda_hat per group, conditional guarantee: CRC on each group's
    defect-positive rows. A group with no defects (or too few for epsilon)
    gets +inf."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    g = np.asarray(groups, dtype=np.intp)
    lam = np.full(n_groups, np.inf)
    for k in range(n_groups):
        pos = (g == k) & y
        if pos.any():
            lam[k] = calibrate_crc(s[pos], y[pos], epsilon)
    return lam


def calibrate_mondrian_lac(
    probs: ArrayLike, labels: ArrayLike, groups: ArrayLike, alpha: float, n_groups: int
) -> NDArray[np.float64]:
    """q_hat per group for LAC split CP; +inf for an empty or too-small group."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.intp)
    g = np.asarray(groups, dtype=np.intp)
    q = np.full(n_groups, np.inf)
    for k in range(n_groups):
        rows = g == k
        if rows.any():
            q[k] = calibrate_lac(p[rows], y[rows], alpha)
    return q
