"""S6 -- feature-space typicality (roadmap Section 3.3,
configs/score/s6.yaml): Mahalanobis distance from an image's CLIP
embedding to the distribution of *defect-free* training images, with a
Ledoit-Wolf shrunk covariance (512 dims, a few thousand points).

S6 = -distance, so an image that looks like a typical clean surface scores
high (AUTO_CLEAR candidate) and anything unusual -- a defect, or a scene
unlike the training data -- scores low. That second reading is why the
blueprint calls it the shift-aware candidate: on a new site, everything is
atypical, and S6 sends it to a human.

Reference set: REF_SIZE uncracked SDNET2018 images from the *train* split,
excluding the 1000-image tuning set (S7 is fit there), via fit_reference,
which runs the fit_* split guard.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf

from certus.data.loaders import ImageRecord
from certus.data.splits import read_manifest, sample_records, tune_records
from certus.scores import s1
from certus.scores.registry import ScoreContext, ScoreResult, check_fit_splits, register

REF_SIZE = 3000


def reference_records(ctx: ScoreContext) -> list[ImageRecord]:
    tune = {r.relpath for r in tune_records(ctx.manifest)}
    clean_train = [
        r
        for r in read_manifest(ctx.manifest, "sdnet2018")
        if r.split == "train" and not r.defect_positive and r.relpath not in tune
    ]
    return sample_records(clean_train, REF_SIZE, seed=0)


def fit_reference(
    features: NDArray[np.float32], splits: list[str]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean and precision matrix of the reference features."""
    check_fit_splits(splits)
    lw = LedoitWolf().fit(features.astype(np.float64))
    return lw.location_, lw.precision_


def mahalanobis(
    features: NDArray[np.float32], mean: NDArray[np.float64], precision: NDArray[np.float64]
) -> NDArray[np.float64]:
    d = features.astype(np.float64) - mean
    dist: NDArray[np.float64] = np.sqrt(np.einsum("ij,jk,ik->i", d, precision, d))
    return dist


def to_defect_prob(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Monotone map of distance into [0, 1): 1 - exp(-d^2 / (2 * dim))."""
    dim = 512
    probs: NDArray[np.float64] = 1.0 - np.exp(-(values**2) / (2 * dim))
    return probs


@register("s6")
def s6_feature_typicality(records: list[ImageRecord], ctx: ScoreContext) -> ScoreResult:
    ref = reference_records(ctx)
    mean, precision = fit_reference(s1.load_features(ref, ctx), [r.split for r in ref])
    dist = mahalanobis(s1.load_features(records, ctx), mean, precision)
    ref_hash = hashlib.sha256("\n".join(r.relpath for r in ref).encode()).hexdigest()[:12]
    return ScoreResult(values=-dist, model_version=f"{s1.BACKBONE_TAG}+maha:{ref_hash}")
