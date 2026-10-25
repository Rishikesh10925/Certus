"""Nonconformity score registry. Every score (S1-S7, roadmap Section 3.3) is
registered here behind a single `@register('s3')` decorator interface so any
score can be swapped by config alone. S7 (the learned combination) must be
fitted strictly on a tuning split disjoint from calibration -- enforce that
separation here in code, not by discipline (Section 3.3, "three splits, not
two").

Orientation, shared by every score: the value is the nonconformity of the
hypothesis "this image contains a defect". High means the model sees little
evidence of a defect, so an image is a candidate for AUTO_CLEAR when its
score is high. S1 = 1 - p(defect) is the reference case.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from certus.data.loaders import ImageRecord

# Splits a fit_* function may ever see. "tune" is reserved for S7's
# held-out tuning split once it exists.
FIT_SPLITS = frozenset({"train", "tune"})


@dataclass(frozen=True)
class ScoreContext:
    image_roots: dict[str, Path]  # dataset name -> raw root
    cache_dir: Path


@dataclass(frozen=True)
class ScoreResult:
    values: NDArray[np.float64]  # one per input record, in input order
    model_version: str  # what produced them -- logged with every run


ScoreFn = Callable[[list[ImageRecord], ScoreContext], ScoreResult]

_REGISTRY: dict[str, ScoreFn] = {}


def register(score_id: str) -> Callable[[ScoreFn], ScoreFn]:
    def wrap(fn: ScoreFn) -> ScoreFn:
        if score_id in _REGISTRY:
            raise ValueError(f"score {score_id!r} registered twice")
        _REGISTRY[score_id] = fn
        return fn

    return wrap


def get_score(score_id: str) -> ScoreFn:
    _load_builtin_scores()
    try:
        return _REGISTRY[score_id]
    except KeyError:
        raise KeyError(f"unknown score {score_id!r}; have {available_scores()}") from None


def available_scores() -> list[str]:
    _load_builtin_scores()
    return sorted(_REGISTRY)


def defect_probability(score_id: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Maps a score back to a p(defect) in [0, 1], for methods that need a
    2-class probability vector (split CP, B1-B3)."""
    import importlib

    _load_builtin_scores()
    module = importlib.import_module(f"certus.scores.{score_id}")
    probs: NDArray[np.float64] = np.clip(module.to_defect_prob(values), 0.0, 1.0)
    return probs


def check_fit_splits(splits: Iterable[str]) -> None:
    """The guard every fit_* function calls first: fitting anything on
    calibration or test rows voids the conformal guarantee silently, so it
    fails loudly instead."""
    bad = sorted(set(splits) - FIT_SPLITS)
    if bad:
        raise ValueError(f"fit_* functions may only see {sorted(FIT_SPLITS)} rows, got {bad}")


def _load_builtin_scores() -> None:
    # Imported for the @register side effect; deferred so importing the
    # registry doesn't pull in torch/transformers.
    import certus.scores.s1  # noqa: F401
    import certus.scores.s2  # noqa: F401
