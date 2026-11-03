"""Online S1 scoring for the service: one image in, its CLIP feature,
p(defect) and the most likely defect type out. Uses the exact prompts and
softmax of scores/s1.py, so a served score equals the calibrated one."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from certus.scores import s1

# One defect type per S1 defect prompt, in prompt order.
DEFECT_TYPES = ("crack", "vegetation", "corrosion", "dampness")


@dataclass(frozen=True)
class Scored:
    feature: NDArray[np.float32]  # L2-normalised CLIP image embedding
    p_defect: float
    defect_type: str  # the defect prompt with the most mass
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return 1.0 - self.p_defect  # S1; auto-clear iff score > lambda


class Scorer(Protocol):
    model_version: str

    def score(self, image_bytes: bytes) -> Scored:
        ...


def type_and_prob(
    feature: NDArray[np.float32], text_features: NDArray[np.float32]
) -> tuple[float, str]:
    p = float(s1.zero_shot_defect_prob(feature[None, :], text_features, len(s1.DEFECT_PROMPTS))[0])
    sims = text_features[: len(s1.DEFECT_PROMPTS)] @ feature
    return p, DEFECT_TYPES[int(np.argmax(sims))]


class ClipScorer:
    """Loads CLIP once; `score` is the per-request path."""

    def __init__(self, text_features: NDArray[np.float32], device: str | None = None) -> None:
        from certus.models.vlm import FrozenVLM

        self.vlm = FrozenVLM(device=device)
        self.text_features = text_features
        self.model_version = f"{s1.BACKBONE_TAG}+zeroshot:{s1.prompt_hash()}"
        self.score(_blank_jpeg())  # warm up kernels before the first request

    def score(self, image_bytes: bytes) -> Scored:
        t0 = time.perf_counter()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        t1 = time.perf_counter()
        feature = self.vlm.pil_features([image])[0]
        t2 = time.perf_counter()
        p, kind = type_and_prob(feature, self.text_features)
        t3 = time.perf_counter()
        return Scored(
            feature=feature,
            p_defect=p,
            defect_type=kind,
            timings_ms={
                "decode": 1000 * (t1 - t0),
                "embed": 1000 * (t2 - t1),
                "score": 1000 * (t3 - t2),
            },
        )


def _blank_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (128, 128, 128)).save(buf, format="JPEG")
    return buf.getvalue()
