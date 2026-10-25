"""S3 -- verbalized confidence (roadmap Section 3.3, configs/score/s3.yaml):
ask the generative VLM for a 0-100 confidence that the image shows a
defect, and use 1 - c/100. Expected to be poorly calibrated; included
because it is what practitioners actually do without a statistics
background (baseline B11).

Answers that don't parse as a number in [0, 100] get c = 50 ("no idea"),
and the count of those is reported with the result -- a score that
silently coerces unparseable output would hide how often the model
ignores the instruction.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
from numpy.typing import NDArray

from certus.data.loaders import ImageRecord
from certus.scores.registry import ScoreContext, ScoreResult, register

QUESTION = (
    "How confident are you, from 0 to 100, that this photo shows a structural defect "
    "on a building, such as a crack, plants growing on the structure, corrosion, or "
    "dampness? Reply with only the number."
)
UNPARSEABLE = 50.0
MODEL_TAG = "qwen2-vl-2b"
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def prompt_hash() -> str:
    return hashlib.sha256(QUESTION.encode()).hexdigest()[:12]


def parse_confidence(answer: str) -> float | None:
    """First number in the answer, if it lies in [0, 100]."""
    m = _NUMBER.search(answer)
    if m is None:
        return None
    c = float(m.group())
    return c if 0.0 <= c <= 100.0 else None


def to_defect_prob(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """c/100 back from S3 = 1 - c/100."""
    return 1.0 - values


@register("s3")
def s3_verbalized_confidence(records: list[ImageRecord], ctx: ScoreContext) -> ScoreResult:
    from certus.models.generative import GenerativeVLM

    answers = GenerativeVLM().generate(
        [ctx.image_roots[r.dataset] / r.relpath for r in records], QUESTION
    )
    parsed = [parse_confidence(a) for a in answers]
    conf = np.array([UNPARSEABLE if c is None else c for c in parsed], dtype=np.float64)
    return ScoreResult(
        values=1.0 - conf / 100.0,
        model_version=f"{MODEL_TAG}+verbal:{prompt_hash()}",
        notes={"unparseable": sum(c is None for c in parsed)},
    )
