"""Small trainable heads attached to the frozen VLM backbone: a linear
classification head (feeds S1, roadmap Section 3.3) and an MC-dropout
wrapper for stochastic forward passes (feeds S5). Fit only on the
train/tune split -- never on calibration data.

Week 4 scope: the linear defect head. MC-dropout lands with S5.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from certus.scores.registry import check_fit_splits


def fit_linear_head(
    features: NDArray[np.float32], labels: NDArray[np.bool_], splits: Sequence[str], seed: int = 0
) -> LogisticRegression:
    """Logistic-regression defect head on frozen features. `splits` is the
    split of each row and is checked, so calibration rows can't sneak in."""
    check_fit_splits(splits)
    if len(splits) != features.shape[0] or labels.shape[0] != features.shape[0]:
        raise ValueError("features, labels, and splits must have the same length")
    head = LogisticRegression(max_iter=2000, random_state=seed)
    head.fit(features, labels.astype(int))
    return head


def head_version(head: LogisticRegression) -> str:
    """Short content hash of the fitted weights, for provenance."""
    h = hashlib.sha256(np.ascontiguousarray(head.coef_).tobytes())
    h.update(np.ascontiguousarray(head.intercept_).tobytes())
    return h.hexdigest()[:12]


def predict_defect_prob(
    head: LogisticRegression, features: NDArray[np.float32]
) -> NDArray[np.float64]:
    """p(defect) per row."""
    probs: NDArray[np.float64] = head.predict_proba(features)[:, 1]
    return probs
