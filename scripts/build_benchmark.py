#!/usr/bin/env python
"""Packages the Certus calibration benchmark for Zenodo: everything
downstream of the images, so the conformal experiments can be re-run
without the datasets or a GPU. No image is included -- SDNET2018 and CBRI
are cited and must be downloaded from their sources.

Contents: frozen split manifest, hand-labelled gold labels (+ blind
relabel check), every cached nonconformity score (S1-S7) as CSV, the CLIP
ViT-B/32 features, and the results/ files behind every number in the
paper. Writes release/certus-benchmark-v<version>/ and a .zip of it, plus
.zenodo.json metadata for the upload form.

Usage:
    python scripts/build_benchmark.py --version 1.0.0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

CARD = """# Certus calibration benchmark v{version}

Companion data for "Where the Guarantee Breaks: Conformal Risk Control for
Building-Defect Triage under Site Shift" (Talpaliker, 2026) and the code at
https://github.com/Rishikesh10925/Certus .

It holds everything downstream of the images, so the conformal experiments
(split CP, conformal risk control, Learn-then-Test, leave-one-site-out with
Mondrian / weighted / localized correction) can be re-run on a CPU without
the datasets. **No images are included.**

## Contents

- `manifest/splits_v1.csv`: frozen 70/15/15 split of SDNET2018 and the
  CBRI split, one row per image (dataset, relpath, site_id, split,
  label); its hash is in `splits_v1.json`.
- `gold/gold_labels_v1.csv`: hand labels for all 175 CBRI images outside
  its training split, made blind to the bootstrap labels (defect yes/no
  and type). `gold/relabel_check_v1.csv`: blind re-label of 20 of them
  (20/20 agree).
- `scores/scores_v1.csv`: every cached nonconformity score (dataset,
  relpath, score_id s1-s7, model_version, value). Higher means less
  evidence of a defect.
- `features/*.npz`: L2-normalised CLIP ViT-B/32 image embeddings, keyed
  by relpath.
- `results/`: the JSON and Markdown result files behind every number in
  the paper.
- `SHA256SUMS`: checksums of every file.

{n_scores} score rows; score versions: {versions}.

## Source datasets (not redistributed)

- SDNET2018: Dorafshan, Thomas and Maguire, Data in Brief 21 (2018)
  1664-1668; dataset DOI 10.15142/T3TD19. Free for academic use; obtain the
  images from the source.
- CBRI building-defect dataset, Roboflow Universe export (version 1). The
  gold labels here are new; the images and bootstrap labels are not
  included.

The relpath column joins every file here to the images of those datasets.

## Licence

The files in this package (split manifest, gold labels, scores, features
and results) are released under CC BY 4.0. The source images remain under
their own terms.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--db", type=Path, default=Path("data/cache/certus.sqlite"))
    parser.add_argument("--features", type=Path, default=Path("data/cache/features"))
    parser.add_argument("--out", type=Path, default=Path("release"))
    args = parser.parse_args()

    root = args.out / f"certus-benchmark-v{args.version}"
    if root.exists():
        shutil.rmtree(root)
    for sub in ("manifest", "gold", "scores", "features"):
        (root / sub).mkdir(parents=True)

    for f in ("splits_v1.csv", "splits_v1.json"):
        shutil.copy2(Path("data/manifest") / f, root / "manifest" / f)
    for f in ("gold_labels_v1.csv", "relabel_check_v1.csv"):
        shutil.copy2(Path("data/gold") / f, root / "gold" / f)
    for f in ("sdnet2018_clip-vit-b32_by_relpath.npz", "cbri_clip-vit-b32_by_relpath.npz"):
        shutil.copy2(args.features / f, root / "features" / f)
    shutil.copytree("results", root / "results")

    # The Week 4 linear head on bootstrap labels (clip-vit-b32+logreg) was at
    # chance and is used nowhere; leaving it out avoids a second "s1".
    keep = "WHERE model_version NOT LIKE 'clip-vit-b32+logreg:%'"
    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        f"SELECT dataset, relpath, score_id, model_version, value FROM scores {keep}"
        " ORDER BY score_id, model_version, dataset, relpath"
    ).fetchall()
    versions = conn.execute(
        f"SELECT DISTINCT score_id, model_version FROM scores {keep} ORDER BY 1, 2"
    ).fetchall()
    conn.close()
    with (root / "scores/scores_v1.csv").open("w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["dataset", "relpath", "score_id", "model_version", "value"])
        w.writerows(rows)

    (root / "README.md").write_text(
        CARD.format(
            version=args.version,
            n_scores=f"{len(rows):,}",
            versions="; ".join(f"{s} `{v}`" for s, v in versions),
        ),
        encoding="utf-8",
    )
    files = sorted(p for p in root.rglob("*") if p.is_file())
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(root).as_posix()}\n" for p in files),
        encoding="utf-8",
    )
    zenodo = {
        "title": f"Certus calibration benchmark v{args.version}: scores, gold labels and"
        " splits for conformal risk control in building-defect triage",
        "upload_type": "dataset",
        "version": args.version,
        "creators": [
            {
                "name": "Talpaliker, Rishikesh",
                "affiliation": "School of Technology Management and Engineering,"
                " SVKM's NMIMS (Deemed to be University), Hyderabad",
            }
        ],
        "description": "Frozen splits, hand-labelled gold labels, cached nonconformity scores"
        " (S1-S7), CLIP ViT-B/32 features and result files for the Certus study of"
        " conformal risk control under site shift. No images are included; SDNET2018"
        " (doi:10.15142/T3TD19) and CBRI must be obtained from their sources.",
        "license": "cc-by-4.0",
        "keywords": [
            "conformal prediction",
            "conformal risk control",
            "distribution shift",
            "visual inspection",
            "crack detection",
            "SDNET2018",
        ],
        "related_identifiers": [
            {
                "identifier": "https://github.com/Rishikesh10925/Certus",
                "relation": "isSupplementTo",
                "resource_type": "software",
            }
        ],
    }
    (args.out / ".zenodo.json").write_text(json.dumps(zenodo, indent=1) + "\n")
    archive = shutil.make_archive(str(root), "zip", root_dir=args.out, base_dir=root.name)
    size = Path(archive).stat().st_size / 1e6
    print(f"{len(files) + 1} files, {len(rows):,} score rows -> {archive} ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
