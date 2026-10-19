"""S1 -- classifier logit (roadmap Section 3.3, configs/score/s1.yaml):
1 - p(defect) from a linear head on frozen pooled visual features.

The head is fit on the `train` rows of whatever records it's handed, using
their bootstrapped (Roboflow/GPT-4V) defect labels -- bootstrap labels are
training data, which is allowed; the blueprint only forbids them for
calibration. Every record, any split, then gets a score.

Features come from FrozenVLM (off-the-shelf CLIP ViT-B/32 until the
fine-tuned checkpoint exists), cached per dataset as relpath -> vector so
the backbone runs once, ever.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from certus.data.loaders import ImageRecord
from certus.models.heads import fit_linear_head, head_version, predict_defect_prob
from certus.scores.registry import ScoreContext, ScoreResult, register

BACKBONE_TAG = "clip-vit-b32"


def _feature_cache_path(cache_dir: Path, dataset: str) -> Path:
    return cache_dir / "features" / f"{dataset}_{BACKBONE_TAG}_by_relpath.npz"


def load_features(records: list[ImageRecord], ctx: ScoreContext) -> NDArray[np.float32]:
    """Frozen features for `records`, in order. Missing ones are computed
    with FrozenVLM and added to the per-dataset cache."""
    out = np.empty((len(records), 0), dtype=np.float32)
    rows: dict[int, NDArray[np.float32]] = {}
    for dataset in sorted({r.dataset for r in records}):
        path = _feature_cache_path(ctx.cache_dir, dataset)
        cache: dict[str, NDArray[np.float32]] = {}
        if path.exists():
            with np.load(path) as z:
                cache = dict(zip(z["relpaths"].tolist(), z["features"], strict=True))
        wanted = [i for i, r in enumerate(records) if r.dataset == dataset]
        missing = sorted({records[i].relpath for i in wanted} - cache.keys())
        if missing:
            from certus.models.vlm import FrozenVLM

            feats = FrozenVLM().image_features([ctx.image_roots[dataset] / m for m in missing])
            cache.update(zip(missing, feats, strict=True))
            path.parent.mkdir(parents=True, exist_ok=True)
            keys = sorted(cache)
            np.savez(path, relpaths=np.array(keys), features=np.stack([cache[k] for k in keys]))
        for i in wanted:
            rows[i] = cache[records[i].relpath]
    if records:
        out = np.stack([rows[i] for i in range(len(records))]).astype(np.float32)
    return out


@register("s1")
def s1_classifier_logit(records: list[ImageRecord], ctx: ScoreContext) -> ScoreResult:
    features = load_features(records, ctx)
    train = np.array([r.split == "train" for r in records])
    if not train.any():
        raise ValueError("S1 needs train rows to fit its head")
    labels = np.array([r.defect_positive for r in records])
    head = fit_linear_head(
        features[train], labels[train], [r.split for r in records if r.split == "train"]
    )
    values = 1.0 - predict_defect_prob(head, features)
    return ScoreResult(values=values, model_version=f"{BACKBONE_TAG}+logreg:{head_version(head)}")
