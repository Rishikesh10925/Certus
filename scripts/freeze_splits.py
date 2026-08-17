#!/usr/bin/env python
"""Freezes the Week 2 split manifest (roadmap Section 4.4) across CBRI and
the secondary dataset. Run once; the resulting manifest and its content
hash are what every downstream experiment reads -- re-running this script
after the fact regenerates the same splits (the split assignment is
deterministic in the seed) but will change file hashes if any source image
changed, which is the point: the hash is the tamper-evidence.

Usage:
    python scripts/freeze_splits.py \
        --cbri-root data/raw/cbri \
        --secondary-root data/raw/sdnet2018 \
        --manifest-out data/manifest/splits_v1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from certus.data.loaders import load_cbri_images, load_sdnet2018_images
from certus.data.splits import assign_secondary_split, freeze_manifest, rename_cbri_splits

SEED = 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cbri-root", required=True, type=Path)
    parser.add_argument("--secondary-root", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    args = parser.parse_args()

    cbri_records = rename_cbri_splits(load_cbri_images(args.cbri_root))
    secondary_records = assign_secondary_split(
        load_sdnet2018_images(args.secondary_root), seed=SEED
    )

    all_records = cbri_records + secondary_records
    overall_hash = freeze_manifest(
        all_records,
        dataset_roots={"cbri": args.cbri_root, "sdnet2018": args.secondary_root},
        manifest_path=args.manifest_out,
    )

    n_pos = sum(r.defect_positive for r in all_records)
    print(f"Froze {len(all_records)} records ({n_pos} defect-positive) to {args.manifest_out}")
    print(f"Manifest hash: {overall_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
