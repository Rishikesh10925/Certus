#!/usr/bin/env python
"""Intra-labeller consistency check for the gold sheet: a blind re-label
of a seeded random sample of gold images.

`init` writes a fresh sheet with the labels blanked and the rows shuffled,
so the re-label can't copy the first pass. Label it with the usual tool,
then `compare` reports agreement and Cohen's kappa against the first pass.

Usage:
    python scripts/relabel_check.py init
    python scripts/label_gold.py --labeller RT --sheet data/gold/relabel_check_v1.csv
    python scripts/relabel_check.py compare
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from certus.eval.label_noise import cohen_kappa

FIELDS = ["order", "relpath", "content_hash", "defect_positive", "classes", "labeller", "notes"]
CLASSES = ["crack", "vegetation", "corrosion", "dampness"]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def init(gold: Path, sheet: Path, n: int, seed: int) -> None:
    if sheet.exists():
        raise SystemExit(f"{sheet} exists; delete it to start over")
    rows = _read(gold)
    picked = np.random.default_rng(seed).choice(len(rows), size=n, replace=False)
    out = [
        {
            **rows[i],
            "order": str(k),
            "defect_positive": "",
            "classes": "",
            "labeller": "",
            "notes": "",
        }
        for k, i in enumerate(picked.tolist(), start=1)
    ]
    with sheet.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {len(out)} blank rows to {sheet}")


def compare(gold: Path, sheet: Path) -> None:
    first = {r["relpath"]: r for r in _read(gold)}
    second = [r for r in _read(sheet) if r["defect_positive"]]
    if not second:
        raise SystemExit(f"{sheet} has no labelled rows yet")
    a = np.array([first[r["relpath"]]["defect_positive"] == "yes" for r in second])
    b = np.array([r["defect_positive"] == "yes" for r in second])
    print(f"re-labelled {len(second)} images")
    print(f"defect yes/no: agree {int((a == b).sum())}/{len(b)}, kappa {cohen_kappa(a, b):.3f}")
    for c in CLASSES:
        ca = np.array([c in first[r["relpath"]]["classes"].split(";") for r in second])
        cb = np.array([c in r["classes"].split(";") for r in second])
        agree = int((ca == cb).sum())
        print(f"{c:10s}: agree {agree}/{len(cb)}, first {ca.sum()} / second {cb.sum()}")
    for r in second:
        old = first[r["relpath"]]["classes"]
        if old != r["classes"]:
            print(f"  changed: {r['relpath']}  '{old}' -> '{r['classes']}'")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "compare"])
    parser.add_argument("--gold", type=Path, default=Path("data/gold/gold_labels_v1.csv"))
    parser.add_argument("--sheet", type=Path, default=Path("data/gold/relabel_check_v1.csv"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.action == "init":
        init(args.gold, args.sheet, args.n, args.seed)
    else:
        compare(args.gold, args.sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
