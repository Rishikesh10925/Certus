#!/usr/bin/env python
"""Week 6 -- conformal risk control at epsilon in {0.10, 0.05, 0.02, 0.01}
for every cached score, 100 seeded cal/test partitions each, under both
guarantees:

- joint:       E[1{defect and auto-cleared}] <= epsilon (all images)
- conditional: P(auto-cleared | defect) <= epsilon (calibrate on defects)

Pool: the SDNET2018 E1 pool (seeded 3000-image sample of its cal split,
human labels). n_cal = 2000, so ~300 defects calibrate the conditional
guarantee -- enough for epsilon = 0.01 to be certifiable at all.

Usage:
    python scripts/run_crc.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from certus.data.splits import read_manifest, sample_records
from certus.data.store import connect, model_versions, read_scores
from certus.eval.protocol import run_crc_protocol, summarize_crc

EPSILONS = (0.10, 0.05, 0.02, 0.01)
N_CAL = 2000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest/splits_v1.csv"))
    parser.add_argument("--db", type=Path, default=Path("data/cache/certus.sqlite"))
    parser.add_argument("--scores", nargs="+", default=["s1", "s2", "s3", "s4"])
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/crc/crc_week6.json"))
    args = parser.parse_args()

    records = sample_records(read_manifest(args.manifest, "sdnet2018"), 3000, seed=0, split="cal")
    labels = np.array([r.defect_positive for r in records])
    conn = connect(args.db)
    rows, versions = [], {}
    for score in args.scores:
        found = [v for v in model_versions(conn, score) if "logreg" not in v]
        if len(found) != 1:
            print(f"skipping {score}: cached versions {found}")
            continue
        versions[score] = found[0]
        values = read_scores(conn, records, score, found[0])
        for conditional in (False, True):
            for eps in EPSILONS:
                res = run_crc_protocol(values, labels, eps, N_CAL, conditional, args.n_seeds)
                s = summarize_crc(res)
                rows.append(
                    {
                        "score": score,
                        "guarantee": "conditional" if conditional else "joint",
                        "epsilon": eps,
                        **s,
                        "valid": s["risk_mean"] <= eps + 3 * s["risk_se"],
                    }
                )
    conn.close()

    manifest_hash = json.loads(args.manifest.with_suffix(".json").read_text())["overall_hash"]
    report = {
        "experiment": "crc_week6",
        "manifest_hash": manifest_hash,
        "pool": {
            "dataset": "sdnet2018",
            "split": "cal",
            "n": len(records),
            "n_pos": int(labels.sum()),
        },
        "n_cal": N_CAL,
        "n_seeds": args.n_seeds,
        "model_versions": versions,
        "all_valid": all(r["valid"] for r in rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")

    print(f"{'score':5} {'guarantee':11} {'eps':>5} {'risk':>15} {'HRF':>15} {'finite':>6} valid")
    for r in rows:
        print(
            f"{r['score']:5} {r['guarantee']:11} {r['epsilon']:5.2f} "
            f"{r['risk_mean']:7.4f}±{r['risk_sd']:.4f} {r['hrf_mean']:7.3f}±{r['hrf_sd']:.3f} "
            f"{r['lam_finite_frac']:6.2f} {r['valid']}"
        )
    return 0 if report["all_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
