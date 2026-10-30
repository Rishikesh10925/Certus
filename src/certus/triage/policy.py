"""Turns calibrated thresholds (lambda_hat, q_hat per site cluster) into a
three-outcome triage decision -- AUTO_CLEAR, FLAG, or ESCALATE -- rather
than a binary auto/human split, so a confident single-defect prediction
reaches a human with the defect type pre-populated instead of joining an
undifferentiated queue. See roadmap Section 3.4 for the reference
implementation.

AUTO_CLEAR is the conformal decision (s(x) > lambda). FLAG and ESCALATE
only route what a human already has to see, so they never change the
guarantee or the human-review fraction.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from certus.conformal.crc import auto_clear


class Triage(IntEnum):
    AUTO_CLEAR = 0
    FLAG = 1  # human review, defect type pre-populated
    ESCALATE = 2  # human review, no usable type


def decide(
    scores: ArrayLike, lam: float, confident_type: ArrayLike | None = None
) -> NDArray[np.int8]:
    """Triage code per image. `confident_type` marks images whose prediction
    set is a single defect class; without it nothing can be pre-populated,
    so every image not cleared escalates."""
    cleared = auto_clear(scores, lam)
    typed = (
        np.zeros(cleared.shape, dtype=bool)
        if confident_type is None
        else np.asarray(confident_type, dtype=bool)
    )
    if typed.shape != cleared.shape:
        raise ValueError(f"confident_type shape {typed.shape} != scores shape {cleared.shape}")
    out = np.full(cleared.shape, Triage.ESCALATE, dtype=np.int8)
    out[typed] = Triage.FLAG
    out[cleared] = Triage.AUTO_CLEAR
    return out


def human_review_fraction(decisions: ArrayLike) -> float:
    """HRF = (|FLAG| + |ESCALATE|) / N."""
    d = np.asarray(decisions)
    if d.size == 0:
        raise ValueError("no decisions")
    return float((d != Triage.AUTO_CLEAR).mean())
