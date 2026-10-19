#!/usr/bin/env python
"""Sanity check for a nonconformity score against labels we can trust.
CBRI's bootstrap labels can't do this (Week 3: most bootstrap negatives are
unannotated defects), and the gold set is almost all positives so far, so
this uses SDNET2018's human cracked/uncracked labels: a balanced sample
from one split, AUC of the score at telling them apart, overall and per
structure type (D=deck, P=pavement, W=wall). Scores are cached in the
scores table as a side effect.

Usage:
    python scripts/score_sanity.py --score s2 --split train --n-per-class 500
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from certus.data.splits import read_manifest
from certus.data.store import connect, write_scores
from certus.scores.registry import ScoreContext, get_score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest/splits_v1.csv"))
    parser.add_argument("--secondary-root", type=Path, default=Path("data/raw/sdnet2018"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--db", type=Path, default=Path("data/cache/certus.sqlite"))
    args = parser.parse_args()

    records = [r for r in read_manifest(args.manifest, "sdnet2018") if r.split == args.split]
    rng = random.Random(args.seed)
    sample = rng.sample([r for r in records if r.defect_positive], args.n_per_class)
    sample += rng.sample([r for r in records if not r.defect_positive], args.n_per_class)

    ctx = ScoreContext({"sdnet2018": args.secondary_root}, args.cache_dir)
    result = get_score(args.score)(sample, ctx)
    with connect(args.db) as conn:
        write_scores(conn, sample, args.score, result.model_version, result.values)
    conn.close()

    y = np.array([r.defect_positive for r in sample])
    # High score = little defect evidence, so negate for "defect" AUC.
    report: dict[str, object] = {
        "score": args.score,
        "model_version": result.model_version,
        "split": args.split,
        "n": len(sample),
        "auc": round(float(roc_auc_score(y, -result.values)), 4),
    }
    for t in "DPW":
        m = np.array([r.relpath.startswith(t) for r in sample])
        report[f"auc_{t}"] = round(float(roc_auc_score(y[m], -result.values[m])), 4)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
