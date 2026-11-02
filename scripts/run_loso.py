#!/usr/bin/env python
"""E4, first pass -- leave-one-site-out with vanilla split CP and CRC.
Writes results/e4/loso_v1.json and the per-site gap table
results/e4/loso_v1.md.

Sites: SDNET2018's three structure types (deck, pavement, wall) from the
3000-image pool, plus CBRI's 175 gold-labelled images as a target-only
site (never calibrated on; its bootstrap labels are unusable). Scores S1
and S2, the only ones cached on CBRI. n_cal = 1000 source rows per seed,
100 seeds; the in-site reference splits the held-out site in half.

Usage:
    python scripts/run_loso.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from certus.data.splits import read_manifest, sample_records
from certus.data.store import connect, model_versions, read_scores
from certus.eval.loso import loso_folds, run_in_site, run_loso, summarize_loso
from certus.scores.registry import defect_probability

SCORES = ("s1", "s2")
SOURCE_SITES = ("deck", "pavement", "wall")
LEVELS = (0.10, 0.05)  # alpha for split CP and epsilon for CRC, paired
N_CAL = 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest/splits_v1.csv"))
    parser.add_argument("--gold", type=Path, default=Path("data/gold/gold_labels_v1.csv"))
    parser.add_argument("--db", type=Path, default=Path("data/cache/certus.sqlite"))
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/e4/loso_v1.json"))
    args = parser.parse_args()

    pool = sample_records(read_manifest(args.manifest, "sdnet2018"), 3000, seed=0, split="cal")
    cbri = {r.relpath: r for r in read_manifest(args.manifest, "cbri")}
    with args.gold.open(newline="", encoding="utf-8") as f:
        gold = [g for g in csv.DictReader(f) if g["defect_positive"]]
    records = pool + [cbri[g["relpath"]] for g in gold]
    sites = np.array([r.site_id.split("-")[0] for r in pool] + ["cbri"] * len(gold))
    y = np.array([r.defect_positive for r in pool] + [g["defect_positive"] == "yes" for g in gold])
    folds = loso_folds(sites, SOURCE_SITES)

    conn = connect(args.db)
    rows, versions = [], {}
    for score in SCORES:
        (versions[score],) = (v for v in model_versions(conn, score) if "logreg" not in v)
        v = read_scores(conn, records, score, versions[score])
        p = defect_probability(score, v)
        probs = np.stack([1.0 - p, p], axis=1)
        for level in LEVELS:
            for fold in folds:
                for setting, res in (
                    ("loso", run_loso(fold, probs, v, y, level, level, N_CAL, args.n_seeds)),
                    ("in_site", run_in_site(fold, probs, v, y, level, level, args.n_seeds)),
                ):
                    rows.append(
                        {
                            "score": score,
                            "level": level,
                            "site": fold.site,
                            "setting": setting,
                            "n_target": int(fold.target.size),
                            "target_prevalence": float(y[fold.target].mean()),
                            **summarize_loso(res, level, level),
                        }
                    )
    conn.close()

    report = {
        "experiment": "e4_loso_v1",
        "manifest_hash": json.loads(args.manifest.with_suffix(".json").read_text())["overall_hash"],
        "sites": {s: int((sites == s).sum()) for s in sorted(set(sites.tolist()))},
        "source_sites": SOURCE_SITES,
        "n_cal": N_CAL,
        "n_seeds": args.n_seeds,
        "model_versions": versions,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")

    lines = [
        "# LOSO v1 -- per-site gap (vanilla split CP and CRC conditional)",
        "",
        f"n_cal = {N_CAL} source rows, {args.n_seeds} seeds. Coverage gap = mean"
        " coverage - (1 - alpha). Miss ratio = mean P(cleared | defect) / epsilon;"
        " above 1 means the guarantee broke. `in-site` calibrates on half the"
        " held-out site (exchangeable reference).",
        "",
        "| score | level | held-out site | n | defects | coverage gap (LOSO / in-site)"
        " | miss ratio (LOSO / in-site) | seeds over eps (LOSO) | HRF (LOSO / in-site) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["setting"] != "loso":
            continue
        ref = next(
            q
            for q in rows
            if q["setting"] == "in_site" and all(q[k] == r[k] for k in ("score", "level", "site"))
        )
        lines.append(
            f"| {r['score']} | {r['level']:g} | {r['site']} | {r['n_target']}"
            f" | {r['target_prevalence']:.0%}"
            f" | {r['coverage_gap']:+.3f} / {ref['coverage_gap']:+.3f}"
            f" | {r['miss_rate_ratio']:.2f} / {ref['miss_rate_ratio']:.2f}"
            f" | {r['miss_exceed_frac']:.0%}"
            f" | {r['hrf_mean']:.2f} / {ref['hrf_mean']:.2f} |"
        )
    table = "\n".join(lines) + "\n"
    args.out.with_suffix(".md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
