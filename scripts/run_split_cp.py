#!/usr/bin/env python
"""Runs the three split-CP baselines (LAC, APS, RAPS -- B5-B7) end to end on
real images with labels we can trust: SDNET2018's human cracked/uncracked
labels, calibrated on its `cal` split and evaluated on its `test` split.
The classifier is S1's zero-shot p(defect), as a 2-class probability
vector [p(no defect), p(defect)].

A plumbing-on-real-data check, not a headline result: CBRI's gold set is
still too small and too one-sided (Week 3/4) to calibrate on.

Usage:
    python scripts/run_split_cp.py --n-per-class 500 --alpha 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from certus.conformal.split_cp import (
    aps_score_matrix,
    calibrate_split,
    lac_score_matrix,
    predict_sets,
    raps_score_matrix,
)
from certus.data.loaders import ImageRecord
from certus.data.splits import read_manifest
from certus.eval.coverage import marginal_coverage
from certus.eval.efficiency import mean_set_size
from certus.scores.registry import ScoreContext, get_score


def _balanced(records: list[ImageRecord], n: int, rng: random.Random) -> list[ImageRecord]:
    pos = [r for r in records if r.defect_positive]
    neg = [r for r in records if not r.defect_positive]
    return rng.sample(pos, n) + rng.sample(neg, n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-class", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest/splits_v1.csv"))
    parser.add_argument("--secondary-root", type=Path, default=Path("data/raw/sdnet2018"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = read_manifest(args.manifest, "sdnet2018")
    cal = _balanced([r for r in records if r.split == "cal"], args.n_per_class, rng)
    test = _balanced([r for r in records if r.split == "test"], args.n_per_class, rng)

    ctx = ScoreContext({"sdnet2018": args.secondary_root}, args.cache_dir)
    s1 = get_score("s1")
    probs = {}
    for name, rows in (("cal", cal), ("test", test)):
        p_defect = 1.0 - s1(rows, ctx).values
        probs[name] = np.stack([1.0 - p_defect, p_defect], axis=1)
    labels = {
        name: np.array([int(r.defect_positive) for r in rows])
        for name, rows in (("cal", cal), ("test", test))
    }
    # One generator for both, so cal and test get independent draws.
    u_rng = np.random.default_rng(args.seed)
    u = {"cal": u_rng.random(len(cal)), "test": u_rng.random(len(test))}

    methods = {
        "lac": lambda p, uu: lac_score_matrix(p),
        "aps": lambda p, uu: aps_score_matrix(p, u=uu),
        "raps": lambda p, uu: raps_score_matrix(p, lam=0.1, k_reg=1, u=uu),
    }
    results = []
    for name, fn in methods.items():
        q_hat = calibrate_split(fn(probs["cal"], u["cal"]), labels["cal"], args.alpha)
        sets = predict_sets(fn(probs["test"], u["test"]), q_hat)
        results.append(
            {
                "method": name,
                "alpha": args.alpha,
                "n_cal": len(cal),
                "n_test": len(test),
                "q_hat": round(q_hat, 4),
                "coverage": round(marginal_coverage(sets, labels["test"]), 4),
                "mean_set_size": round(mean_set_size(sets), 4),
                "singleton_frac": round(float((sets.sum(axis=1) == 1).mean()), 4),
            }
        )
    for r in results:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
