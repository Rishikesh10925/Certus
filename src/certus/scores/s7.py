"""S7 -- learned combination (roadmap Section 3.3, configs/score/s7.yaml):
a logistic regression over S1-S6, S7 = 1 - p(defect).

"Three splits, not two": the combiner is fit ONLY on the SDNET2018 tuning
set (`tune_records`, train split, 1000 balanced images), through the fit_*
split guard. Nothing it learns can have seen a calibration row, so S7 is
just another fixed function of the image as far as conformal calibration
is concerned. If any scored record is itself a tuning row, scoring
refuses -- an in-sample S7 would be optimistically good.

Base scores are read from the scores table, so S1-S6 must be cached for
both the tuning set and the records being scored. Each is standardised
with the tuning set's mean/SD before the fit.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from certus.data.loaders import ImageRecord
from certus.data.splits import tune_records
from certus.data.store import connect, model_versions, read_scores
from certus.scores.registry import ScoreContext, ScoreResult, check_fit_splits, register

BASE_SCORES = ("s1", "s2", "s3", "s4", "s5", "s6")


def _base_matrix(records: list[ImageRecord], ctx: ScoreContext) -> tuple[NDArray[np.float64], str]:
    conn = connect(ctx.db)
    cols, tags = [], []
    for sid in BASE_SCORES:
        versions = [v for v in model_versions(conn, sid) if "logreg" not in v]
        if len(versions) != 1:
            conn.close()
            raise ValueError(f"S7 needs exactly one cached version of {sid}, found {versions}")
        cols.append(read_scores(conn, records, sid, versions[0]))
        tags.append(f"{sid}={versions[0]}")
    conn.close()
    return np.stack(cols, axis=1), "|".join(tags)


def fit_combiner(
    x: NDArray[np.float64], y: NDArray[np.bool_], splits: list[str]
) -> tuple[LogisticRegression, NDArray[np.float64], NDArray[np.float64]]:
    """Standardise, then fit. Returns (model, mean, sd)."""
    check_fit_splits(splits)
    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-12
    model = LogisticRegression(max_iter=2000).fit((x - mu) / sd, y.astype(int))
    return model, mu, sd


def to_defect_prob(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 - values


@register("s7")
def s7_learned_combination(records: list[ImageRecord], ctx: ScoreContext) -> ScoreResult:
    tune = tune_records(ctx.manifest)
    tune_ids = {(r.dataset, r.relpath) for r in tune}
    if any((r.dataset, r.relpath) in tune_ids for r in records):
        raise ValueError("refusing to score S7 on its own tuning rows")
    x_tune, tags = _base_matrix(tune, ctx)
    model, mu, sd = fit_combiner(
        x_tune, np.array([r.defect_positive for r in tune]), [r.split for r in tune]
    )
    x, _ = _base_matrix(records, ctx)
    p_defect = model.predict_proba((x - mu) / sd)[:, 1]
    h = hashlib.sha256(tags.encode())
    h.update(np.ascontiguousarray(model.coef_).tobytes())
    return ScoreResult(
        values=1.0 - p_defect,
        model_version=f"logreg-s1..s6:{h.hexdigest()[:12]}",
        notes={
            f"coef_{sid}_x1000": int(round(1000 * c))
            for sid, c in zip(BASE_SCORES, model.coef_[0], strict=True)
        },
    )
