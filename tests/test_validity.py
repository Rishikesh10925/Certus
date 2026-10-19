"""The validity property test (roadmap Section 8.3) -- the single most
valuable file in the repository. On synthetic exchangeable data, empirical
coverage must land within Monte-Carlo error of nominal across a range of
alpha and n_cal, with an asymmetric tolerance: under-coverage is a
correctness violation, over-coverage is expected and bounded by
1/(n_cal + 1) -- but only for scores without ties. Deterministic APS gives
the lowest-ranked class a score of exactly 1.0 on every row, so its scores
tie and it over-covers by more than 1/(n_cal + 1) at small alpha; for it,
only the lower bound (the actual guarantee) is checked.

Each (method, alpha, n_cal) configuration runs 500 independent trials
(fresh calibration and test draws per trial, seeded deterministically so
CI never flakes). The
Monte-Carlo tolerance is 4 standard errors of the mean coverage across
trials, estimated from the trials themselves. The last test checks the test:
the naive np.quantile(1 - alpha) threshold must be caught under-covering.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from certus.conformal.split_cp import (
    aps_score_matrix,
    calibrate_split,
    lac_score_matrix,
    predict_sets,
    raps_score_matrix,
)
from certus.eval.coverage import marginal_coverage

N_TRIALS = 500
N_TEST = 500
N_CLASSES = 5
ALPHAS = [0.01, 0.05, 0.10, 0.20]  # matches configs/experiment/e1_validity.yaml
N_CALS = [50, 200, 1000]
TOLERANCE_SE = 4.0


def _softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    out: NDArray[np.float64] = e / e.sum(axis=-1, keepdims=True)
    return out


def _synthetic_classifier_data(
    rng: np.random.Generator, n_trials: int, n: int
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """i.i.d. (hence exchangeable) draws from a K-class problem, with a
    deliberately miscalibrated, noisy "model" -- conformal validity must not
    depend on the model being any good."""
    true_logits = rng.normal(scale=2.0, size=(n_trials, n, N_CLASSES))
    true_probs = _softmax(true_logits)
    u = rng.random(size=(n_trials, n, 1))
    labels = (u > true_probs.cumsum(axis=-1)).sum(axis=-1).astype(np.intp)
    labels = np.minimum(labels, N_CLASSES - 1)
    model_logits = 0.5 * true_logits + rng.normal(scale=1.0, size=true_logits.shape)
    return _softmax(model_logits), labels


ScoreFn = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]

# Each method maps (probs, u) -> (n, n_classes) score matrix; u is one
# uniform per row, ignored by the deterministic variants.
METHODS: dict[str, ScoreFn] = {
    "lac": lambda p, u: lac_score_matrix(p),
    "aps": lambda p, u: aps_score_matrix(p),
    "aps_randomized": lambda p, u: aps_score_matrix(p, u=u),
    "raps_randomized": lambda p, u: raps_score_matrix(p, lam=0.1, k_reg=1, u=u),
}
TIED_SCORES = {"aps"}  # no 1/(n+1) over-coverage bound for these


def _coverage_over_trials(
    alpha: float, n_cal: int, seed: int, score_fn: ScoreFn, calibrate: Callable[..., float]
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    probs, labels = _synthetic_classifier_data(rng, N_TRIALS, n_cal + N_TEST)
    u = rng.random(size=labels.shape)
    coverages = np.empty(N_TRIALS)
    for t in range(N_TRIALS):
        scores = score_fn(probs[t], u[t])
        q_hat = calibrate(scores[:n_cal], labels[t, :n_cal], alpha)
        sets = predict_sets(scores[n_cal:], q_hat)
        coverages[t] = marginal_coverage(sets, labels[t, n_cal:])
    return coverages


def _check_validity(
    coverages: NDArray[np.float64], alpha: float, n_cal: int, check_upper: bool = True
) -> None:
    mean = float(coverages.mean())
    se = float(coverages.std(ddof=1) / np.sqrt(coverages.size))
    slack = TOLERANCE_SE * se
    lower = 1.0 - alpha - slack
    upper = 1.0 - alpha + 1.0 / (n_cal + 1) + slack
    assert mean >= lower, f"UNDER-coverage {mean:.4f} < {lower:.4f} (alpha={alpha}, n={n_cal})"
    if check_upper:
        assert mean <= upper, f"over-coverage {mean:.4f} > {upper:.4f} (alpha={alpha}, n={n_cal})"


@pytest.mark.parametrize("n_cal", N_CALS)
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("method", list(METHODS))
def test_split_cp_coverage_is_valid(method: str, alpha: float, n_cal: int) -> None:
    seed = int(alpha * 1000) * 10_000 + n_cal
    coverages = _coverage_over_trials(alpha, n_cal, seed, METHODS[method], calibrate_split)
    _check_validity(coverages, alpha, n_cal, check_upper=method not in TIED_SCORES)


def _naive_calibrate(
    cal_scores: NDArray[np.float64], cal_labels: NDArray[np.intp], alpha: float
) -> float:
    true_scores = cal_scores[np.arange(cal_scores.shape[0]), cal_labels]
    return float(np.quantile(true_scores, 1.0 - alpha))


def test_validity_check_catches_the_naive_quantile() -> None:
    """At n_cal=20, alpha=0.1 the naive quantile's expected coverage is about
    18.1/21 = 0.862 -- the property test must flag it, or it's toothless."""
    coverages = _coverage_over_trials(
        0.1, 20, seed=1, score_fn=METHODS["lac"], calibrate=_naive_calibrate
    )
    with pytest.raises(AssertionError, match="UNDER-coverage"):
        _check_validity(coverages, alpha=0.1, n_cal=20)
