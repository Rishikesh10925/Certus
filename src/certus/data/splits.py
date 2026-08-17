"""Generates and freezes the train/calibration/test split plus the
leave-one-site-out (LOSO) folds (roadmap Section 4.4). Splits are written
once to a versioned, content-hashed manifest on disk and every experiment
reads that manifest -- a split silently regenerating between runs is the
class of bug that produces months of uninterpretable results. The
calibration split must be physically inaccessible to any function whose
name begins with `fit_`.

Week 2 scope: CBRI keeps the train/valid/test split Roboflow already
assigned (re-splitting it would throw away a real, already-frozen
assignment for no reason). "valid" is renamed "cal" here since that's what
it's for in a conformal pipeline. The secondary dataset ships no split at
all, so it gets a fresh deterministic stratified one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

from certus.data.loaders import ImageRecord

_CBRI_SPLIT_RENAME = {"train": "train", "valid": "cal", "test": "test"}


def _content_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assign_secondary_split(
    records: list[ImageRecord], seed: int, train: float = 0.7, cal: float = 0.15
) -> list[ImageRecord]:
    """Deterministic stratified split (by defect_positive) for a dataset
    that ships no split of its own. `train + cal` must be < 1; the
    remainder is `test`."""
    rng = random.Random(seed)
    by_label: dict[bool, list[int]] = {True: [], False: []}
    for i, r in enumerate(records):
        by_label[r.defect_positive].append(i)

    split_of: dict[int, str] = {}
    for _, indices in by_label.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train)
        n_cal = int(n * cal)
        for i in shuffled[:n_train]:
            split_of[i] = "train"
        for i in shuffled[n_train : n_train + n_cal]:
            split_of[i] = "cal"
        for i in shuffled[n_train + n_cal :]:
            split_of[i] = "test"

    return [
        ImageRecord(
            dataset=r.dataset,
            relpath=r.relpath,
            site_id=r.site_id,
            camera_id=r.camera_id,
            split=split_of[i],
            defect_positive=r.defect_positive,
            classes=r.classes,
        )
        for i, r in enumerate(records)
    ]


def rename_cbri_splits(records: list[ImageRecord]) -> list[ImageRecord]:
    """Maps Roboflow's train/valid/test onto Certus's train/cal/test."""
    return [
        ImageRecord(
            dataset=r.dataset,
            relpath=r.relpath,
            site_id=r.site_id,
            camera_id=r.camera_id,
            split=_CBRI_SPLIT_RENAME.get(r.split, r.split),
            defect_positive=r.defect_positive,
            classes=r.classes,
        )
        for r in records
    ]


def freeze_manifest(
    records: list[ImageRecord], dataset_roots: dict[str, Path], manifest_path: Path
) -> str:
    """Writes one row per record (with its content hash) to a CSV manifest,
    plus a JSON sidecar with an overall manifest hash. Returns the overall
    hash -- the number that goes in the logbook and never changes for this
    manifest version, since freezing a split is the whole point.
    """
    rows = []
    for r in sorted(records, key=lambda r: (r.dataset, r.relpath)):
        file_hash = _content_hash(dataset_roots[r.dataset] / r.relpath)
        rows.append({**asdict(r), "classes": ",".join(r.classes), "content_hash": file_hash})

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    overall = hashlib.sha256()
    for row in rows:
        overall.update(row["content_hash"].encode("ascii"))
    overall_hash = overall.hexdigest()

    sidecar = manifest_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "manifest": manifest_path.name,
                "n_records": len(rows),
                "datasets": sorted({r["dataset"] for r in rows}),
                "overall_hash": overall_hash,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return overall_hash
