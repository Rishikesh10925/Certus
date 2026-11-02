#!/usr/bin/env python
"""E4, Weeks 11-12 -- the four shift corrections under leave-one-site-out.
Writes results/e4/loso_v3.json and the side-by-side table
results/e4/loso_v3.md (v2, Week 11, had no localized CP).

Same folds, pool, scores, levels and n_cal as run_loso.py (v1), so the
vanilla rows reproduce v1 exactly. Adds Mondrian CP over k-means clusters
of the CLIP features (k in 2, 4, 8, fit on the source sites) and weighted
CP with cross-fitted domain-classifier weights; the weighted rows also log
the effective sample size and the domain classifier's AUC. Localized CP
weights by CLIP similarity with an adaptive bandwidth, k_frac in 0.05,
0.1, 0.25.

Usage:
    python scripts/run_loso_corrections.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from certus.data.splits import read_manifest, sample_records
from certus.data.store import connect, model_versions, read_scores
from certus.eval.loso import fold_weights, loso_folds, run_in_site, run_loso, summarize_loso
from certus.scores import s1
from certus.scores.registry import ScoreContext, defect_probability

SCORES = ("s1", "s2")
SOURCE_SITES = ("deck", "pavement", "wall")
LEVELS = (0.10, 0.05)  # alpha for split CP and epsilon for CRC, paired
N_CAL = 1000
CLUSTERS = (2, 4, 8)
K_FRACS = (0.05, 0.1, 0.25)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest/splits_v1.csv"))
    parser.add_argument("--gold", type=Path, default=Path("data/gold/gold_labels_v1.csv"))
    parser.add_argument("--db", type=Path, default=Path("data/cache/certus.sqlite"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/e4/loso_v3.json"))
    args = parser.parse_args()

    pool = sample_records(read_manifest(args.manifest, "sdnet2018"), 3000, seed=0, split="cal")
    cbri = {r.relpath: r for r in read_manifest(args.manifest, "cbri")}
    with args.gold.open(newline="", encoding="utf-8") as f:
        gold = [g for g in csv.DictReader(f) if g["defect_positive"]]
    records = pool + [cbri[g["relpath"]] for g in gold]
    sites = np.array([r.site_id.split("-")[0] for r in pool] + ["cbri"] * len(gold))
    y = np.array([r.defect_positive for r in pool] + [g["defect_positive"] == "yes" for g in gold])
    folds = loso_folds(sites, SOURCE_SITES)

    ctx = ScoreContext(
        image_roots={"cbri": Path("data/raw/cbri"), "sdnet2018": Path("data/raw/sdnet2018")},
        cache_dir=args.cache_dir,
        manifest=args.manifest,
        db=args.db,
    )
    feats = s1.load_features(records, ctx)
    weights = {f.site: fold_weights(f, feats, N_CAL, args.n_seeds) for f in folds}
    print("domain weights fitted", flush=True)

    conn = connect(args.db)
    rows, versions = [], {}
    for score in SCORES:
        (versions[score],) = (v for v in model_versions(conn, score) if "logreg" not in v)
        v = read_scores(conn, records, score, versions[score])
        p = defect_probability(score, v)
        probs = np.stack([1.0 - p, p], axis=1)
        for level in LEVELS:
            for fold in folds:
                common = (fold, probs, v, y, level, level, N_CAL, args.n_seeds)
                runs = {
                    "vanilla": run_loso(*common),
                    "in_site": run_in_site(fold, probs, v, y, level, level, args.n_seeds),
                    "weighted": run_loso(*common, method="weighted", weights=weights[fold.site]),
                }
                for k in CLUSTERS:
                    runs[f"mondrian_k{k}"] = run_loso(
                        *common, method="mondrian", features=feats, n_clusters=k
                    )
                for kf in K_FRACS:
                    runs[f"localized_{kf:g}"] = run_loso(
                        *common, method="localized", features=feats, k_frac=kf
                    )
                for method, res in runs.items():
                    rows.append(
                        {
                            "score": score,
                            "level": level,
                            "site": fold.site,
                            "method": method,
                            "n_target": int(fold.target.size),
                            "target_prevalence": float(y[fold.target].mean()),
                            **summarize_loso(res, level, level),
                        }
                    )
        print(f"{score} done", flush=True)
    conn.close()

    report = {
        "experiment": "e4_loso_v3_corrections",
        "manifest_hash": json.loads(args.manifest.with_suffix(".json").read_text())["overall_hash"],
        "sites": {s: int((sites == s).sum()) for s in sorted(set(sites.tolist()))},
        "source_sites": SOURCE_SITES,
        "n_cal": N_CAL,
        "n_seeds": args.n_seeds,
        "clusters": CLUSTERS,
        "k_fracs": K_FRACS,
        "model_versions": versions,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")

    methods = [
        "in_site",
        "vanilla",
        *(f"mondrian_k{k}" for k in CLUSTERS),
        "weighted",
        *(f"localized_{kf:g}" for kf in K_FRACS),
    ]
    lines = [
        "# LOSO v3 -- four shift corrections side by side",
        "",
        f"n_cal = {N_CAL} source rows, {args.n_seeds} seeds. Cells are"
        " miss ratio / HRF for conditional CRC: miss ratio = mean P(cleared |"
        " defect) / epsilon (above 1 = guarantee broken), HRF = share of the"
        " held-out site sent to a human. `in_site` is the exchangeable"
        " reference. Weighted also lists ESS / n and the domain-classifier AUC.",
        "",
        "| score | level | held-out site | " + " | ".join(methods) + " | ESS/n, AUC |",
        "|---|---|---|" + "---|" * (len(methods) + 1),
    ]
    cell = {(r["score"], r["level"], r["site"], r["method"]): r for r in rows}
    for score in SCORES:
        for level in LEVELS:
            for fold in folds:
                key = (score, level, fold.site)
                cells = [
                    f"{cell[(*key, m)]['miss_rate_ratio']:.2f} / {cell[(*key, m)]['hrf_mean']:.2f}"
                    for m in methods
                ]
                w = cell[(*key, "weighted")]
                lines.append(
                    f"| {score} | {level:g} | {fold.site} | "
                    + " | ".join(cells)
                    + f" | {w['ess_frac_mean']:.2f}, {w['domain_auc_mean']:.2f} |"
                )
    lines += [
        "",
        "Split CP marginal coverage gap (mean coverage - (1 - alpha)):",
        "",
        "| score | level | held-out site | " + " | ".join(methods) + " |",
        "|---|---|---|" + "---|" * len(methods),
    ]
    for score in SCORES:
        for level in LEVELS:
            for fold in folds:
                key = (score, level, fold.site)
                lines.append(
                    f"| {score} | {level:g} | {fold.site} | "
                    + " | ".join(f"{cell[(*key, m)]['coverage_gap']:+.3f}" for m in methods)
                    + " |"
                )
    table = "\n".join(lines) + "\n"
    args.out.with_suffix(".md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
