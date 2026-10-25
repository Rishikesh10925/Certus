"""Non-conformal baselines B1-B3 (blueprint Section 16), as prediction-set
methods so they sit on the same axes as split CP in E1/F2:

- B1 raw softmax: take classes in descending probability until their raw
  mass reaches 1 - alpha. What people do when they trust the softmax.
- B2 temperature scaling: B1 on probabilities rescaled by a temperature
  fit by NLL. The standard calibration fix -- still no finite-sample
  guarantee.
- B3 selective classification (softmax response): answer with the top
  class when max-prob >= tau, abstain (full set = send to a human)
  otherwise; tau is the most permissive threshold whose *empirical*
  miscoverage on the tuning rows is <= alpha. Plug-in, no +1 correction --
  exactly the gap conformal closes.

B2 and B3 fit on the same labelled rows a conformal method would
calibrate on (same data budget, fair comparison). That's legitimate for
them -- they promise nothing -- and those rows are passed to the fit_*
guard as "tune", their role here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from certus.scores.registry import check_fit_splits

FloatMatrix = NDArray[np.float64]
BoolMatrix = NDArray[np.bool_]


def naive_cumulative_sets(probs: FloatMatrix, alpha: float) -> BoolMatrix:
    """B1: smallest top-k set whose raw probability mass is >= 1 - alpha."""
    order = np.argsort(-probs, axis=1, kind="stable")
    cum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
    # include rank r iff the mass strictly before it is < 1 - alpha
    before = cum - np.take_along_axis(probs, order, axis=1)
    sorted_in = before < 1.0 - alpha
    sets = np.empty_like(sorted_in)
    np.put_along_axis(sets, order, sorted_in, axis=1)
    return sets


def temperature_scale(probs: FloatMatrix, temperature: float) -> FloatMatrix:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    out: FloatMatrix = e / e.sum(axis=1, keepdims=True)
    return out


def fit_temperature(probs: FloatMatrix, labels: NDArray[np.intp], splits: list[str]) -> float:
    """NLL-optimal temperature on tuning rows."""
    check_fit_splits(splits)
    rows = np.arange(len(labels))

    def nll(log_t: float) -> float:
        p = temperature_scale(probs, float(np.exp(log_t)))
        return float(-np.log(np.clip(p[rows, labels], 1e-12, 1.0)).mean())

    res = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(res.x))


def fit_selective_threshold(
    probs: FloatMatrix, labels: NDArray[np.intp], alpha: float, splits: list[str]
) -> float:
    """B3's tau: the lowest max-prob threshold (most answers, fewest
    abstentions) whose empirical miscoverage on the tuning rows is <= alpha.
    Miscoverage = answered-and-wrong / all rows; abstentions cover."""
    check_fit_splits(splits)
    conf = probs.max(axis=1)
    wrong = probs.argmax(axis=1) != labels
    order = np.argsort(-conf, kind="stable")
    # accepting the top-j most confident rows: errors = cumulative wrong
    errors = np.concatenate([[0], np.cumsum(wrong[order])]) / len(labels)
    ok = np.nonzero(errors <= alpha)[0]
    j = int(ok.max())  # j=0 (accept none) always qualifies
    return float(conf[order[j - 1]]) if j > 0 else float(np.inf)


def selective_sets(probs: FloatMatrix, tau: float) -> BoolMatrix:
    answered = probs.max(axis=1) >= tau
    sets = np.ones_like(probs, dtype=bool)
    top = probs.argmax(axis=1)
    sets[answered] = False
    sets[np.nonzero(answered)[0], top[answered]] = True
    return sets
