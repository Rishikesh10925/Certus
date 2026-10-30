#!/usr/bin/env python
"""E3 -- the efficiency curve (roadmap 5.2, contribution C2, Figure F4).
Human-review fraction across epsilon for the three best E2 scores on the
SDNET2018 pool, under CRC and LTT, joint and conditional, n_cal = 2000,
100 seeds, plus the cost break-even ratio at each point.

Conditional epsilon is floored by the ~305 calibration defects: below
1/306 = 0.0033 nothing can be certified, hence the sweep stops at 0.005.

Usage:
    python scripts/run_e3.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from certus.data.splits import read_manifest, sample_records
from certus.data.store import connect, model_versions, read_scores
from certus.eval.protocol import run_crc_protocol, summarize_crc
from certus.triage.economics import break_even_ratio, net_saving
from certus.viz.figures import figure_f4

SCORES = ("s7", "s1", "s5")
EPSILONS = (0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005)
COST_RATIOS = (10, 30, 100, 300, 1000)  # c_m / c_h
N_CAL = 2000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest/splits_v1.csv"))
    parser.add_argument("--db", type=Path, default=Path("data/cache/certus.sqlite"))
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/e3/e3_efficiency.json"))
    parser.add_argument("--fig", type=Path, default=Path("paper/figures/F4_efficiency.png"))
    args = parser.parse_args()

    records = sample_records(read_manifest(args.manifest, "sdnet2018"), 3000, seed=0, split="cal")
    y = np.array([r.defect_positive for r in records])
    prevalence = float(y.mean())
    conn = connect(args.db)
    rows, versions = [], {}
    for score in SCORES:
        (versions[score],) = (v for v in model_versions(conn, score) if "logreg:" not in v)
        v = read_scores(conn, records, score, versions[score])
        for method in ("crc", "ltt"):
            for conditional in (False, True):
                guarantee = "conditional" if conditional else "joint"
                for eps in EPSILONS:
                    res = run_crc_protocol(
                        v, y, eps, N_CAL, conditional, args.n_seeds, method=method
                    )
                    summary = summarize_crc(res, eps)
                    hrf = summary["hrf_mean"]
                    rows.append(
                        {
                            "score": score,
                            "method": method,
                            "guarantee": guarantee,
                            "epsilon": eps,
                            **summary,
                            "break_even_ratio": break_even_ratio(hrf, eps, prevalence, guarantee),
                            "net_saving_per_c_h": {
                                str(r): net_saving(hrf, eps, prevalence, guarantee, 1.0, float(r))
                                for r in COST_RATIOS
                            },
                        }
                    )
    conn.close()

    report = {
        "experiment": "e3_efficiency",
        "manifest_hash": json.loads(args.manifest.with_suffix(".json").read_text())["overall_hash"],
        "pool": {"dataset": "sdnet2018", "split": "cal", "n": len(records), "n_pos": int(y.sum())},
        "prevalence": prevalence,
        "n_cal": N_CAL,
        "n_seeds": args.n_seeds,
        "model_versions": versions,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    figure_f4(rows, SCORES, args.fig)

    print(f"prevalence {prevalence:.3f}")
    for r in rows:
        print(
            f"{r['score']} {r['method']} {r['guarantee']:11s} eps={r['epsilon']:<5g}"
            f" hrf={r['hrf_mean']:.3f}+-{r['hrf_sd']:.3f} risk={r['risk_mean']:.4f}"
            f" exceed={r['exceed_frac']:.2f} breakeven={r['break_even_ratio']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
