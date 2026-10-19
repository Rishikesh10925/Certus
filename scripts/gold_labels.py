#!/usr/bin/env python
"""Gold-subset labelling sheet (blueprint "Gold subset": 300-500
hand-labelled images, 60-90 per week across Weeks 3-6). Every headline
conformal number is calibrated on these labels only; the GPT-4V-bootstrapped
Roboflow labels are training data, never calibration data.

Two subcommands:

    # Write the labelling sheet once: CBRI cal-split images (never train --
    # the detector was fit on those), in a fixed seeded order.
    python scripts/gold_labels.py init \
        --manifest data/manifest/splits_v1.csv --out data/gold/gold_labels_v1.csv

    # Count completed rows and check every hash still matches the manifest.
    python scripts/gold_labels.py status \
        --manifest data/manifest/splits_v1.csv --sheet data/gold/gold_labels_v1.csv

The sheet deliberately omits the bootstrapped classes: showing them to the
labeller would anchor the gold labels to GPT-4V and quietly inflate the
GPT-4V-vs-gold kappa the Week 7 label-noise study measures.

Labelling a row = opening data/raw/cbri/<relpath> and filling in
`defect_positive` (yes/no), `classes` (semicolon-separated, from
crack;vegetation;corrosion;dampness -- empty if no), and `labeller`.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

SEED = 0
SOURCE_SPLITS = ("cal", "test")  # cal first; test only once cal is exhausted
GOLD_CLASSES = frozenset({"crack", "vegetation", "corrosion", "dampness"})
FIELDS = ["order", "relpath", "content_hash", "defect_positive", "classes", "labeller", "notes"]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["dataset"] == "cbri"]


def build_sheet(manifest_rows: list[dict[str, str]], seed: int = SEED) -> list[dict[str, str]]:
    """Fixed labelling order: each source split shuffled with `seed`, cal
    before test. Label columns start empty."""
    rng = random.Random(seed)
    ordered: list[dict[str, str]] = []
    for split in SOURCE_SPLITS:
        rows = sorted((r for r in manifest_rows if r["split"] == split), key=lambda r: r["relpath"])
        rng.shuffle(rows)
        ordered.extend(rows)
    return [
        {
            "order": str(i + 1),
            "relpath": r["relpath"],
            "content_hash": r["content_hash"],
            "defect_positive": "",
            "classes": "",
            "labeller": "",
            "notes": "",
        }
        for i, r in enumerate(ordered)
    ]


def validate_sheet(
    sheet: list[dict[str, str]], manifest_rows: list[dict[str, str]]
) -> tuple[int, int, list[str]]:
    """Returns (n_labelled, n_labelled_positive, errors). A row counts as
    labelled once `defect_positive` is filled in."""
    hashes = {r["relpath"]: r["content_hash"] for r in manifest_rows}
    errors: list[str] = []
    n_labelled = n_pos = 0
    for row in sheet:
        where = f"row {row['order']} ({row['relpath']})"
        if hashes.get(row["relpath"]) != row["content_hash"]:
            errors.append(f"{where}: content_hash does not match the manifest")
        dp = row["defect_positive"].strip().lower()
        if not dp:
            continue
        classes = {c.strip() for c in row["classes"].split(";") if c.strip()}
        if dp not in ("yes", "no"):
            errors.append(f"{where}: defect_positive must be yes/no, got {dp!r}")
            continue
        if unknown := classes - GOLD_CLASSES:
            errors.append(f"{where}: unknown classes {sorted(unknown)}")
        if (dp == "yes") != bool(classes):
            errors.append(f"{where}: defect_positive={dp} but classes={sorted(classes)}")
        if not row["labeller"].strip():
            errors.append(f"{where}: labeller is empty")
        n_labelled += 1
        n_pos += dp == "yes"
    return n_labelled, n_pos, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_init = sub.add_parser("init")
    p_init.add_argument("--manifest", required=True, type=Path)
    p_init.add_argument("--out", required=True, type=Path)
    p_status = sub.add_parser("status")
    p_status.add_argument("--manifest", required=True, type=Path)
    p_status.add_argument("--sheet", required=True, type=Path)
    args = parser.parse_args()

    manifest_rows = _read_manifest(args.manifest)

    if args.cmd == "init":
        if args.out.exists():
            print(f"{args.out} already exists -- refusing to overwrite labels", file=sys.stderr)
            return 1
        sheet = build_sheet(manifest_rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(sheet)
        print(f"wrote {len(sheet)} rows to {args.out}")
        return 0

    with args.sheet.open(newline="", encoding="utf-8") as f:
        sheet = list(csv.DictReader(f))
    n_labelled, n_pos, errors = validate_sheet(sheet, manifest_rows)
    for e in errors:
        print(f"ERROR {e}", file=sys.stderr)
    print(f"gold labelled: {n_labelled}/{len(sheet)} ({n_pos} defect-positive)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
