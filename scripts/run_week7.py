#!/usr/bin/env python
"""Week 7 -- Learn-then-Test and the label-noise study. Writes
results/week7/week7.json. Four parts:

A. CRC vs LTT (delta = 0.05) on the SDNET2018 pool, every cached score,
   joint and conditional, epsilon in {0.10, 0.05, 0.02, 0.01}: realised
   risk, share of seeds over epsilon, human-review fraction.
B. Agreement between the GPT-4V bootstrap labels and the 175 gold labels
   on CBRI: Cohen's kappa, binary and per class.
C. Guarantee degradation on SDNET2018: calibrate on labels with a
   fraction of defects hidden (CBRI's observed 42% and below), at random
   or worst case, and measure the realised miss rate against the clean
   labels.
D. The same on CBRI's real labels: calibrate on bootstrap vs gold (within
   the 175 gold images), evaluate against gold.

Usage:
    python scripts/run_week7.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from certus.data.splits import read_manifest, sample_records
from certus.data.store import connect, model_versions, read_scores
from certus.eval.label_noise import cohen_kappa, hide_defects
from certus.eval.protocol import run_crc_protocol, summarize_crc

EPSILONS = (0.10, 0.05, 0.02, 0.01)
DELTA = 0.05
N_CAL = 2000
GOLD_CLASSES = ("crack", "vegetation", "corrosion", "dampness")


def _version(conn, score: str) -> str:  # type: ignore[no-untyped-def]
    (v,) = (x for x in model_versions(conn, score) if "logreg" not in x)
    return v


def _row(res: list, eps: float, **keys: object) -> dict:  # type: ignore[type-arg]
    return {**keys, "epsilon": eps, **summarize_crc(res, eps)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest/splits_v1.csv"))
    parser.add_argument("--gold", type=Path, default=Path("data/gold/gold_labels_v1.csv"))
    parser.add_argument("--db", type=Path, default=Path("data/cache/certus.sqlite"))
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/week7/week7.json"))
    args = parser.parse_args()
    conn = connect(args.db)
    report: dict[str, object] = {
        "manifest_hash": json.loads(args.manifest.with_suffix(".json").read_text())["overall_hash"],
        "n_seeds": args.n_seeds,
        "delta": DELTA,
    }

    # A. CRC vs LTT
    sd = sample_records(read_manifest(args.manifest, "sdnet2018"), 3000, seed=0, split="cal")
    y_sd = np.array([r.defect_positive for r in sd])
    sd_scores = {s: read_scores(conn, sd, s, _version(conn, s)) for s in ("s1", "s2", "s3", "s4")}
    part_a = []
    for score, v in sd_scores.items():
        for conditional in (False, True):
            for method in ("crc", "ltt"):
                for eps in EPSILONS:
                    res = run_crc_protocol(
                        v, y_sd, eps, N_CAL, conditional, args.n_seeds, method=method, delta=DELTA
                    )
                    part_a.append(
                        _row(
                            res,
                            eps,
                            score=score,
                            method=method,
                            guarantee="conditional" if conditional else "joint",
                        )
                    )
    report["A_crc_vs_ltt"] = part_a

    # B. kappa, bootstrap vs gold, on CBRI
    cbri = {r.relpath: r for r in read_manifest(args.manifest, "cbri")}
    with args.gold.open(newline="", encoding="utf-8") as f:
        gold = [g for g in csv.DictReader(f) if g["defect_positive"]]
    recs = [cbri[g["relpath"]] for g in gold]
    y_gold = np.array([g["defect_positive"] == "yes" for g in gold])
    y_boot = np.array([r.defect_positive for r in recs])
    kappas = {"defect_positive": cohen_kappa(y_gold, y_boot)}
    for cls in GOLD_CLASSES:
        g = np.array([cls in x["classes"].split(";") for x in gold])
        # the bootstrap's "dampens" is a typo class for dampness (Week 1)
        b = np.array(
            [cls in r.classes or (cls == "dampness" and "dampens" in r.classes) for r in recs]
        )
        kappas[cls] = cohen_kappa(g, b)
    report["B_kappa"] = {
        "n": len(gold),
        "kappa": kappas,
        "gold_positive": int(y_gold.sum()),
        "bootstrap_positive": int(y_boot.sum()),
        "gold_defects_missed_by_bootstrap": int((y_gold & ~y_boot).sum()),
        "hidden_rate": float((y_gold & ~y_boot).sum() / y_gold.sum()),
    }
    hidden_rate = round(float((y_gold & ~y_boot).sum() / y_gold.sum()), 2)

    # C. degradation on SDNET2018 with controlled hidden defects
    part_c = []
    for score in ("s1", "s2"):
        v = sd_scores[score]
        for mode in ("random", "informative"):
            for rate in sorted({0.0, 0.10, 0.25, hidden_rate}):
                noisy = hide_defects(
                    y_sd, rate, np.random.default_rng(0), v if mode == "informative" else None
                )
                for conditional in (False, True):
                    for eps in EPSILONS[:3]:
                        res = run_crc_protocol(
                            v,
                            y_sd,
                            eps,
                            N_CAL,
                            conditional,
                            args.n_seeds,
                            calibration_labels=noisy,
                        )
                        part_c.append(
                            _row(
                                res,
                                eps,
                                score=score,
                                mode=mode,
                                hidden_rate=rate,
                                guarantee="conditional" if conditional else "joint",
                            )
                        )
    report["C_degradation_sdnet"] = part_c

    # D. CBRI: calibrate on bootstrap vs gold, evaluate against gold
    part_d = []
    for score in ("s1", "s2"):
        v = read_scores(conn, recs, score, _version(conn, score))
        for source, cal_y in (("gold", y_gold), ("bootstrap", y_boot)):
            for conditional in (False, True):
                for eps in (0.10, 0.05):
                    res = run_crc_protocol(
                        v, y_gold, eps, 100, conditional, args.n_seeds, calibration_labels=cal_y
                    )
                    part_d.append(
                        _row(
                            res,
                            eps,
                            score=score,
                            calibrated_on=source,
                            guarantee="conditional" if conditional else "joint",
                        )
                    )
    report["D_degradation_cbri"] = part_d
    conn.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["B_kappa"], indent=1))
    print(f"wrote {args.out}: A {len(part_a)} rows, C {len(part_c)} rows, D {len(part_d)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
