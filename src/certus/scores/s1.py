"""S1 -- classifier logit (roadmap Section 3.3, configs/score/s1.yaml):
1 - p(defect), where p(defect) is the zero-shot CLIP softmax mass on the
defect prompts against the no-defect prompts.

Why zero-shot and not a trained head (Week 4 finding, docs/logbook.md):
there is no fine-tuned CBRI detector -- the source Roboflow Universe
project has 0 trained models -- and a linear head fit on the bootstrapped
labels scored 5-fold CV AUC 0.51, because most bootstrap "negatives" are
unannotated defect images (Week 3 gold comparison). Zero-shot needs no
training labels at all, so nothing here can leak calibration data either.

The prompt strings are part of the score's definition: they are frozen
here, hashed into the model version, and must be reported verbatim.

Image features come from FrozenVLM (CLIP ViT-B/32), cached per dataset as
relpath -> vector so the backbone runs once, ever.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from certus.data.loaders import ImageRecord
from certus.scores.registry import ScoreContext, ScoreResult, register

BACKBONE_TAG = "clip-vit-b32"
DEFECT_PROMPTS = (
    "a photo of a cracked concrete or brick wall",
    "a photo of plants or moss growing on a building wall",
    "a photo of rusted, corroded metal or exposed rebar on a building",
    "a photo of a wall with damp patches and water stains",
)
CLEAN_PROMPTS = (
    "a photo of a clean, undamaged concrete wall",
    "a photo of a smooth, intact building surface in good condition",
)
CLIP_LOGIT_SCALE = 100.0  # CLIP's learned temperature, fixed at its trained value


def _feature_cache_path(cache_dir: Path, dataset: str) -> Path:
    return cache_dir / "features" / f"{dataset}_{BACKBONE_TAG}_by_relpath.npz"


def _text_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "features" / f"text_{BACKBONE_TAG}_{prompt_hash()}.npy"


def prompt_hash() -> str:
    h = hashlib.sha256("\n".join(DEFECT_PROMPTS + ("--",) + CLEAN_PROMPTS).encode("utf-8"))
    return h.hexdigest()[:12]


def load_features(records: list[ImageRecord], ctx: ScoreContext) -> NDArray[np.float32]:
    """Frozen image features for `records`, in order. Missing ones are
    computed with FrozenVLM and added to the per-dataset cache."""
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
    if not records:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack([rows[i] for i in range(len(records))]).astype(np.float32)


def load_text_features(ctx: ScoreContext) -> NDArray[np.float32]:
    """L2-normalised text embeddings, defect prompts first, then clean."""
    path = _text_cache_path(ctx.cache_dir)
    if path.exists():
        cached: NDArray[np.float32] = np.load(path)
        return cached
    from certus.models.vlm import FrozenVLM

    feats = FrozenVLM().text_features(list(DEFECT_PROMPTS + CLEAN_PROMPTS))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feats)
    return feats


def zero_shot_defect_prob(
    image_features: NDArray[np.float32], text_features: NDArray[np.float32], n_defect: int
) -> NDArray[np.float64]:
    """Softmax over all prompts; p(defect) is the mass on the first
    `n_defect` of them."""
    logits = CLIP_LOGIT_SCALE * image_features.astype(np.float64) @ text_features.T
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    probs: NDArray[np.float64] = e[:, :n_defect].sum(axis=1) / e.sum(axis=1)
    return probs


@register("s1")
def s1_classifier_logit(records: list[ImageRecord], ctx: ScoreContext) -> ScoreResult:
    p_defect = zero_shot_defect_prob(
        load_features(records, ctx), load_text_features(ctx), len(DEFECT_PROMPTS)
    )
    return ScoreResult(
        values=1.0 - p_defect, model_version=f"{BACKBONE_TAG}+zeroshot:{prompt_hash()}"
    )
