"""S2 -- generative log-probability (roadmap Section 3.3,
configs/score/s2.yaml): the negative mean token log-probability that the
generative VLM answers QUESTION with ASSERTION. High means the model is
unwilling to assert a defect -- the same orientation as every other score.

QUESTION and ASSERTION are part of the score's definition (S2 is sensitive
to phrasing): frozen here, hashed into the model version, and reported
verbatim in the paper. Never edit them in place -- add a new score.

Why the assertion is the single word "Yes" (Week 4): on 80 SDNET2018 cal
images, the full sentence "Yes, there is a structural defect." scored AUC
0.58 -- its later tokens are near-certain whatever the image shows, and
averaging them in drowns the one informative token. "Yes" alone scored
0.82. The final S2 quality number is reported on SDNET2018 train images,
not on the 80 used to make this choice.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from certus.data.loaders import ImageRecord
from certus.scores.registry import ScoreContext, ScoreResult, register

QUESTION = (
    "Does this photo show a structural defect on a building, such as a crack, "
    "plants growing on the structure, corrosion, or dampness? Answer yes or no."
)
ASSERTION = "Yes"
MODEL_TAG = "qwen2-vl-2b"


def prompt_hash() -> str:
    return hashlib.sha256(f"{QUESTION}\n--\n{ASSERTION}".encode()).hexdigest()[:12]


def to_defect_prob(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """p("Yes") back from S2 = -log p("Yes"). Not a calibrated p(defect),
    just the model's own answer probability -- fine as a conformal score
    input, which needs no calibration."""
    return np.exp(-values)


@register("s2")
def s2_generative_logprob(records: list[ImageRecord], ctx: ScoreContext) -> ScoreResult:
    from certus.models.generative import GenerativeVLM

    vlm = GenerativeVLM()
    paths = [ctx.image_roots[r.dataset] / r.relpath for r in records]
    values = -vlm.assertion_logprob(paths, QUESTION, ASSERTION)
    return ScoreResult(values=values, model_version=f"{MODEL_TAG}+assert:{prompt_hash()}")
