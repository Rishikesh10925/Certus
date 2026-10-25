#!/usr/bin/env python
"""E1 -- validity on exchangeable real data (roadmap 5.2, Week 5 gate).
For every (dataset, score, method, alpha, n_cal): 100 seeded random
cal/test partitions of the dataset's pool, coverage recorded per seed.
Writes the summary + provenance JSON, per-seed rows, and Figure F2.

Pools:
- sdnet2018: seeded 3000-image sample of its cal split, human labels.
- cbri: all 1518 images, bootstrap labels. Validity is a property of the
  procedure, so it must hold against *any* fixed labels -- even noisy
  ones. This checks correctness, not detection quality.

Gate: every conformal method's mean coverage within +/-2 points of 1-alpha.

Usage:
    python scripts/run_e1.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from certus.data.splits import read_manifest, sample_records
from certus.data.store import connect, model_versions, read_scores
from certus.eval.protocol import CONFORMAL, METHODS, run_protocol, summarize
from certus.scores.registry import defect_probability
from certus.viz.figures import figure_f2

POOLS = {"sdnet2018": {"split": "cal", "n": 3000}, "cbri": {"split": None, "n": None}}
SCORES = ("s1", "s2")
ALPHAS = (0.01, 0.05, 0.10, 0.20)  # configs/experiment/e1_validity.yaml
N_CALS = (50, 200, 1000)
TOLERANCE = 0.02


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest/splits_v1.csv"))
    parser.add_argument("--db", type=Path, default=Path("data/cache/certus.sqlite"))
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("results/e1"))
    parser.add_argument("--fig-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    manifest_hash = json.loads(args.manifest.with_suffix(".json").read_text())["overall_hash"]
    conn = connect(args.db)
    summary, seed_rows, provenance = [], [], {}
    for dataset, pool_def in POOLS.items():
        records = sample_records(
            read_manifest(args.manifest, dataset), pool_def["n"], seed=0, split=pool_def["split"]
        )
        labels = np.array([int(r.defect_positive) for r in records], dtype=np.intp)
        provenance[dataset] = {"pool": pool_def, "n": len(records), "n_pos": int(labels.sum())}
        for score in SCORES:
            (version,) = (v for v in model_versions(conn, score) if "logreg" not in v)
            provenance[dataset][f"{score}_model_version"] = version
            p = defect_probability(score, read_scores(conn, records, score, version))
            probs = np.stack([1.0 - p, p], axis=1)
            for method in METHODS:
                for alpha in ALPHAS:
                    for n_cal in N_CALS:
                        res = run_protocol(probs, labels, method, alpha, n_cal, args.n_seeds)
                        s = summarize(res)
                        row = {
                            "dataset": dataset,
                            "score": score,
                            "method": method,
                            "alpha": alpha,
                            "n_cal": n_cal,
                            **s,
                        }
                        if method in CONFORMAL:
                            row["gate_pass"] = abs(s["coverage_mean"] - (1 - alpha)) <= TOLERANCE
                        summary.append(row)
                        seed_rows += [
                            {
                                "dataset": dataset,
                                "score": score,
                                "method": method,
                                "alpha": alpha,
                                "n_cal": n_cal,
                                **vars(r),
                            }
                            for r in res
                        ]
    conn.close()

    gate = [r for r in summary if "gate_pass" in r]
    report = {
        "experiment": "e1_validity",
        "manifest_hash": manifest_hash,
        "n_seeds": args.n_seeds,
        "tolerance": TOLERANCE,
        "pools": provenance,
        "gate_pass": all(r["gate_pass"] for r in gate),
        "gate_cells": len(gate),
        "gate_failures": [r for r in gate if not r["gate_pass"]],
        "summary": summary,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "e1_validity.json").write_text(json.dumps(report, indent=1) + "\n")
    with gzip.open(args.out_dir / "e1_validity_seeds.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seed_rows[0]))
        w.writeheader()
        w.writerows(seed_rows)

    for score in SCORES:
        figure_f2(
            summary,
            score,
            list(POOLS),
            N_CALS,
            list(METHODS),
            args.fig_dir / f"F2_validity_{score}.png",
            TOLERANCE,
        )

    print(
        f"gate: {'PASS' if report['gate_pass'] else 'FAIL'} "
        f"({len(gate) - len(report['gate_failures'])}/{len(gate)} conformal cells within "
        f"±{TOLERANCE * 100:.0f} pts)"
    )
    for r in report["gate_failures"]:
        print(
            "  FAIL",
            {k: r[k] for k in ("dataset", "score", "method", "alpha", "n_cal")},
            round(r["coverage_mean"], 4),
        )
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
