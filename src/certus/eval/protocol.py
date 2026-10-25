"""The 100-seed evaluation harness: random calibration/test partitions
within each protocol, bootstrap confidence intervals (clustered by image
where the unit of analysis is a region), and seed logging so every
coverage number in the paper is reported as mean +/- SD rather than a
single lucky draw. See roadmap Section 5.5.

Week 5 scope: the exchangeable protocol (E1). Each seed draws a fresh
random partition of one labelled pool into n_cal calibration rows and the
rest as test rows, runs one prediction-set method, and records coverage
and set size. Seeds are derived as (base_seed, seed_index), so any single
seed can be replayed in isolation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from certus.baselines import (
    fit_selective_threshold,
    fit_temperature,
    naive_cumulative_sets,
    selective_sets,
    temperature_scale,
)
from certus.conformal.split_cp import (
    aps_score_matrix,
    calibrate_split,
    lac_score_matrix,
    predict_sets,
    raps_score_matrix,
)
from certus.eval.coverage import marginal_coverage
from certus.eval.efficiency import mean_set_size

FloatMatrix = NDArray[np.float64]
Labels = NDArray[np.intp]
# (cal_probs, cal_labels, test_probs, alpha, rng) -> boolean test sets
SetMethod = Callable[
    [FloatMatrix, Labels, FloatMatrix, float, np.random.Generator], NDArray[np.bool_]
]

RAPS_LAM, RAPS_K_REG = 0.1, 1  # configs/conformal/split.yaml


def _conformal(
    score_fn: Callable[[FloatMatrix, NDArray[np.float64] | None], FloatMatrix],
) -> SetMethod:
    def method(
        cal_p: FloatMatrix,
        cal_y: Labels,
        test_p: FloatMatrix,
        alpha: float,
        rng: np.random.Generator,
    ) -> NDArray[np.bool_]:
        q_hat = calibrate_split(score_fn(cal_p, rng.random(len(cal_p))), cal_y, alpha)
        return predict_sets(score_fn(test_p, rng.random(len(test_p))), q_hat)

    return method


def _b1(
    cal_p: FloatMatrix, cal_y: Labels, test_p: FloatMatrix, alpha: float, rng: np.random.Generator
) -> NDArray[np.bool_]:
    return naive_cumulative_sets(test_p, alpha)


def _b2(
    cal_p: FloatMatrix, cal_y: Labels, test_p: FloatMatrix, alpha: float, rng: np.random.Generator
) -> NDArray[np.bool_]:
    t = fit_temperature(cal_p, cal_y, ["tune"] * len(cal_y))
    return naive_cumulative_sets(temperature_scale(test_p, t), alpha)


def _b3(
    cal_p: FloatMatrix, cal_y: Labels, test_p: FloatMatrix, alpha: float, rng: np.random.Generator
) -> NDArray[np.bool_]:
    tau = fit_selective_threshold(cal_p, cal_y, alpha, ["tune"] * len(cal_y))
    return selective_sets(test_p, tau)


METHODS: dict[str, SetMethod] = {
    "lac": _conformal(lambda p, u: lac_score_matrix(p)),
    "aps": _conformal(lambda p, u: aps_score_matrix(p, u=u)),
    "raps": _conformal(lambda p, u: raps_score_matrix(p, RAPS_LAM, RAPS_K_REG, u=u)),
    "b1_softmax": _b1,
    "b2_temperature": _b2,
    "b3_selective": _b3,
}
CONFORMAL = ("lac", "aps", "raps")


@dataclass(frozen=True)
class SeedResult:
    seed: int
    coverage: float
    mean_set_size: float
    singleton_frac: float


def run_protocol(
    probs: FloatMatrix,
    labels: Labels,
    method: str,
    alpha: float,
    n_cal: int,
    n_seeds: int = 100,
    base_seed: int = 0,
) -> list[SeedResult]:
    """n_seeds random cal/test partitions of the pool; one result per seed."""
    n = len(labels)
    if not 0 < n_cal < n:
        raise ValueError(f"n_cal={n_cal} must leave test rows in a pool of {n}")
    fn = METHODS[method]
    out = []
    for s in range(n_seeds):
        rng = np.random.default_rng([base_seed, s])
        perm = rng.permutation(n)
        cal, test = perm[:n_cal], perm[n_cal:]
        sets = fn(probs[cal], labels[cal], probs[test], alpha, rng)
        out.append(
            SeedResult(
                seed=s,
                coverage=marginal_coverage(sets, labels[test]),
                mean_set_size=mean_set_size(sets),
                singleton_frac=float((sets.sum(axis=1) == 1).mean()),
            )
        )
    return out


def summarize(results: list[SeedResult]) -> dict[str, float]:
    """mean, SD, and a 95% percentile bootstrap CI of the mean coverage
    over seeds, plus mean set size."""
    cov = np.array([r.coverage for r in results])
    rng = np.random.default_rng(0)
    boot = rng.choice(cov, size=(2000, cov.size), replace=True).mean(axis=1)
    return {
        "coverage_mean": float(cov.mean()),
        "coverage_sd": float(cov.std(ddof=1)) if cov.size > 1 else 0.0,
        "coverage_ci_lo": float(np.percentile(boot, 2.5)),
        "coverage_ci_hi": float(np.percentile(boot, 97.5)),
        "set_size_mean": float(np.mean([r.mean_set_size for r in results])),
        "singleton_frac_mean": float(np.mean([r.singleton_frac for r in results])),
        "n_seeds": float(cov.size),
    }


def seed_rows(results: list[SeedResult]) -> list[dict[str, float]]:
    return [asdict(r) for r in results]
