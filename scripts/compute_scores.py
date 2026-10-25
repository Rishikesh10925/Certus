#!/usr/bin/env python
"""Computes a registered nonconformity score for every image of a dataset
in the frozen manifest and caches it in the SQLite scores table (roadmap
Section 3.6). Run once per (score, model version); every experiment then
reads from the table.

Usage:
    python scripts/compute_scores.py --score s1 --dataset cbri \
        --manifest data/manifest/splits_v1.csv \
        --cbri-root data/raw/cbri --cache-dir data/cache \
        --db data/cache/certus.sqlite

    # E1's SDNET2018 pool: a seeded 3000-image sample of the cal split
    python scripts/compute_scores.py --score s2 --dataset sdnet2018 \
        --manifest data/manifest/splits_v1.csv --split cal --sample 3000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from certus.data.loaders import load_cbri_images
from certus.data.splits import read_manifest, sample_records
from certus.data.store import connect, write_scores
from certus.scores.registry import ScoreContext, get_score


def _migrate_week2_cbri_cache(cbri_root: Path, cache_dir: Path) -> None:
    """Week 2 cached CBRI CLIP features as a bare array in
    load_cbri_images order. Re-key it by relpath once, so S1 doesn't
    re-run the backbone over 1518 images."""
    old = cache_dir / "features" / "cbri_clip_vitb32.npy"
    new = cache_dir / "features" / "cbri_clip-vit-b32_by_relpath.npz"
    if new.exists() or not old.exists():
        return
    relpaths = [r.relpath for r in load_cbri_images(cbri_root)]
    features = np.load(old)
    if features.shape[0] != len(relpaths):
        raise ValueError(f"{old} has {features.shape[0]} rows, expected {len(relpaths)}")
    np.savez(new, relpaths=np.array(relpaths), features=features)
    print(f"re-keyed {old.name} -> {new.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cbri-root", type=Path, default=Path("data/raw/cbri"))
    parser.add_argument("--secondary-root", type=Path, default=Path("data/raw/sdnet2018"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--db", type=Path, default=Path("data/cache/certus.sqlite"))
    parser.add_argument("--split", default=None, help="only this split")
    parser.add_argument("--sample", type=int, default=None, help="seeded subset of this size")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _migrate_week2_cbri_cache(args.cbri_root, args.cache_dir)
    records = sample_records(
        read_manifest(args.manifest, dataset=args.dataset), args.sample, args.seed, args.split
    )
    ctx = ScoreContext(
        image_roots={"cbri": args.cbri_root, "sdnet2018": args.secondary_root},
        cache_dir=args.cache_dir,
    )
    result = get_score(args.score)(records, ctx)

    with connect(args.db) as conn:
        write_scores(conn, records, args.score, result.model_version, result.values)
    conn.close()

    print(f"{args.score}@{result.model_version}: cached {len(records)} scores in {args.db}")
    if result.notes:
        print(f"  notes: {result.notes}")
    for split in ("train", "cal", "test"):
        mask = np.array([r.split == split for r in records])
        pos = np.array([r.defect_positive for r in records])
        v = result.values
        if mask.any():
            print(
                f"  {split:5s} n={mask.sum():4d}  mean score | defect={v[mask & pos].mean():.3f}"
                f"  | no defect={v[mask & ~pos].mean():.3f}  (manifest labels)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
