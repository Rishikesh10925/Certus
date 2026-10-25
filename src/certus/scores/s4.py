"""S4 -- detector objectness (roadmap Section 3.3, configs/score/s4.yaml):
1 - max confidence over predicted defect boxes. With no trained detector,
the boxes come from OWLv2 zero-shot, queried with one phrase per defect
class. Expected strong on localised defects (cracks, rust spots) and weak
on diffuse ones (damp patches).

The query strings are part of the score's definition: frozen here, hashed
into the model version, reported verbatim.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from certus.data.loaders import ImageRecord
from certus.scores.registry import ScoreContext, ScoreResult, register

QUERIES = (
    "a crack in concrete or brick",
    "a plant growing on a wall",
    "rust or corrosion",
    "a damp patch or water stain",
)
MODEL_TAG = "owlv2-b16"


def prompt_hash() -> str:
    return hashlib.sha256("\n".join(QUERIES).encode()).hexdigest()[:12]


def to_defect_prob(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Max box confidence back from S4 = 1 - max confidence."""
    return 1.0 - values


@register("s4")
def s4_detector_objectness(records: list[ImageRecord], ctx: ScoreContext) -> ScoreResult:
    from certus.models.detector import OpenVocabDetector

    conf = OpenVocabDetector().max_box_confidence(
        [ctx.image_roots[r.dataset] / r.relpath for r in records], list(QUERIES)
    )
    return ScoreResult(values=1.0 - conf, model_version=f"{MODEL_TAG}+zeroshot:{prompt_hash()}")
