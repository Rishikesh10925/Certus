"""S5 -- MC-dropout dispersion (roadmap Section 3.3, configs/score/s5.yaml).

The frozen CLIP backbone has no dropout layers to switch on, so the
stochasticity goes where a head would have it: K passes of dropout
(p = DROPOUT) on the image embedding before S1's zero-shot head. Each pass
gives a p_k(defect). Dispersion alone has no direction -- an
auto-clear rule can't use "the model is unsure" as evidence of "clean" --
so S5 is the lower confidence bound on cleanliness:

    S5 = mean_k(1 - p_k) - std_k(1 - p_k)

High only when the image looks clean *and* the verdict is stable under
dropout. The dropout masks are seeded from each image's relpath, so a
score never depends on which other images were scored with it.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from certus.data.loaders import ImageRecord
from certus.scores import s1
from certus.scores.registry import ScoreContext, ScoreResult, register

K_PASSES = 20
DROPOUT = 0.1


def _rng_for(relpath: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(relpath.encode()).hexdigest()[:16], 16))


def mc_clean_probs(
    features: NDArray[np.float32], text: NDArray[np.float32], relpaths: list[str]
) -> NDArray[np.float64]:
    """(n, K) matrix of 1 - p_k(defect) under K seeded dropout masks."""
    out = np.empty((len(relpaths), K_PASSES))
    for i, (f, rp) in enumerate(zip(features, relpaths, strict=True)):
        masks = _rng_for(rp).random((K_PASSES, f.size)) > DROPOUT
        dropped = masks * f[None, :] / (1.0 - DROPOUT)
        dropped /= np.linalg.norm(dropped, axis=1, keepdims=True)
        out[i] = 1.0 - s1.zero_shot_defect_prob(
            dropped.astype(np.float32), text, len(s1.DEFECT_PROMPTS)
        )
    return out


def to_defect_prob(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Treats the lower bound on 'clean' as 1 - p(defect)."""
    return 1.0 - values


@register("s5")
def s5_mc_dropout(records: list[ImageRecord], ctx: ScoreContext) -> ScoreResult:
    clean = mc_clean_probs(
        s1.load_features(records, ctx), s1.load_text_features(ctx), [r.relpath for r in records]
    )
    values = clean.mean(axis=1) - clean.std(axis=1, ddof=1)
    return ScoreResult(
        values=values,
        model_version=f"{s1.BACKBONE_TAG}+mcdrop{K_PASSES}x{DROPOUT}:{s1.prompt_hash()}",
    )
