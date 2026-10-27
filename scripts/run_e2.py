#!/usr/bin/env python
"""E2 -- score comparison (roadmap 5.2, contribution C1, Figure F3).
For each of S1-S7 on the SDNET2018 pool (seeded 3000 cal-split images,
human labels): AUC on the pool, and CRC at epsilon in {0.10, 0.05, 0.02,
0.01}, joint and conditional, n_cal = 2000, 100 seeds -- the
human-review fraction each score needs to hold the same guarantee.

Usage:
    python scripts/run_e2.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from certus.data.splits import read_manifest, sample_records
from certus.data.store import connect, model_versions, read_scores
from certus.eval.protocol import run_crc_protocol, summarize_crc
from certus.viz.figures import figure_f3

SCORES = ("s1", "s2", "s3", "s4", "s5", "s6", "s7")
EPSILONS = (0.10, 0.05, 0.02, 0.01)
N_CAL = 2000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest/splits_v1.csv"))
    parser.add_argument("--db", type=Path, default=Path("data/cache/certus.sqlite"))
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/e2/e2_scores.json"))
    parser.add_argument("--fig", type=Path, default=Path("paper/figures/F3_scores.png"))
    args = parser.parse_args()

    records = sample_records(read_manifest(args.manifest, "sdnet2018"), 3000, seed=0, split="cal")
    y = np.array([r.defect_positive for r in records])
    conn = connect(args.db)
    rows, auc, versions = [], {}, {}
    for score in SCORES:
        (versions[score],) = (v for v in model_versions(conn, score) if "logreg:" not in v)
        v = read_scores(conn, records, score, versions[score])
        auc[score] = float(roc_auc_score(y, -v))
        for conditional in (False, True):
            for eps in EPSILONS:
                res = run_crc_protocol(v, y, eps, N_CAL, conditional, args.n_seeds)
                rows.append(
                    {
                        "score": score,
                        "guarantee": "conditional" if conditional else "joint",
                        "epsilon": eps,
                        **summarize_crc(res, eps),
                    }
                )
    conn.close()

    report = {
        "experiment": "e2_scores",
        "manifest_hash": json.loads(args.manifest.with_suffix(".json").read_text())["overall_hash"],
        "pool": {"dataset": "sdnet2018", "split": "cal", "n": len(records), "n_pos": int(y.sum())},
        "n_cal": N_CAL,
        "n_seeds": args.n_seeds,
        "model_versions": versions,
        "auc_on_pool": auc,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    figure_f3(rows, [("joint", 0.05), ("conditional", 0.10)], args.fig, auc)

    print("AUC on pool:", {k: round(v, 3) for k, v in auc.items()})
    for g, e in (
        ("joint", 0.10),
        ("joint", 0.05),
        ("joint", 0.02),
        ("conditional", 0.10),
        ("conditional", 0.05),
    ):
        sel = sorted(
            (r for r in rows if r["guarantee"] == g and r["epsilon"] == e),
            key=lambda r: r["hrf_mean"],
        )
        print(
            f"{g:11s} eps={e:.2f}: " + "  ".join(f"{r['score']} {r['hrf_mean']:.3f}" for r in sel)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
