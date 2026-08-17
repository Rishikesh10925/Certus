#!/usr/bin/env python
"""Week 2 shift-verification diagnostic (roadmap Section 4.4, Week-2 hard
gate). Two separate checks, both against frozen-VLM features:

1. Within-CBRI proxy sites: CBRI ships no real site_id (Week 1 audit), so
   this clusters frozen-VLM embeddings into candidate proxy sites and then
   asks a logistic regression whether it can tell the clusters apart --
   AUC > 0.80 to trust them for LOSO, AUC < 0.65 is the pivot trigger
   (roadmap risk register).
2. CBRI vs. the secondary dataset: a domain classifier across the two
   actual acquisition sources, which is the genuine distribution shift the
   LOSO study depends on, independent of whether the within-CBRI proxy
   clusters turn out to be separable.

Both numbers get written to a JSON report and belong in the Week 2 logbook
entry verbatim, not paraphrased.

Usage:
    python scripts/shift_diagnostic.py \
        --cbri-root data/raw/cbri \
        --secondary-root data/raw/sdnet2018 \
        --cache-dir data/cache/features \
        --json-out docs/audit/week2_shift_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from certus.data.loaders import ImageRecord, load_cbri_images
from certus.data.sites import choose_k, cluster_sites, domain_classifier_auc
from certus.models.vlm import FrozenVLM

SEED = 0


def _cached_features(vlm: FrozenVLM, image_paths: list[Path], cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features = vlm.image_features(image_paths, batch_size=32)
    np.save(cache_path, features)
    return features


def run_cbri_proxy_site_diagnostic(
    cbri_root: Path, vlm: FrozenVLM, cache_dir: Path
) -> dict[str, object]:
    records = load_cbri_images(cbri_root)
    paths = [cbri_root / r.relpath for r in records]
    features = _cached_features(vlm, paths, cache_dir / "cbri_clip_vitb32.npy")

    k_scores = choose_k(features, range(3, 8), seed=SEED)
    best_k = max(k_scores, key=lambda k: k_scores[k])
    site_labels = cluster_sites(features, n_clusters=best_k, seed=SEED)
    auc = domain_classifier_auc(features, site_labels, seed=SEED)

    return {
        "n_images": len(records),
        "silhouette_by_k": k_scores,
        "chosen_k": best_k,
        "cluster_sizes": {int(k): int((site_labels == k).sum()) for k in np.unique(site_labels)},
        "domain_classifier_auc": auc,
    }


def run_cbri_vs_secondary_diagnostic(
    cbri_records: list[ImageRecord],
    cbri_root: Path,
    secondary_records: list[ImageRecord],
    secondary_root: Path,
    vlm: FrozenVLM,
    cache_dir: Path,
    sample_size: int = 1500,
) -> dict[str, object]:
    rng = random.Random(SEED)
    secondary_sample = (
        rng.sample(secondary_records, sample_size)
        if len(secondary_records) > sample_size
        else secondary_records
    )

    cbri_paths = [cbri_root / r.relpath for r in cbri_records]
    secondary_paths = [secondary_root / r.relpath for r in secondary_sample]

    cbri_features = _cached_features(vlm, cbri_paths, cache_dir / "cbri_clip_vitb32.npy")
    secondary_features = _cached_features(
        vlm, secondary_paths, cache_dir / "secondary_sample_clip_vitb32.npy"
    )

    features = np.concatenate([cbri_features, secondary_features], axis=0)
    labels = np.array([0] * len(cbri_features) + [1] * len(secondary_features))
    auc = domain_classifier_auc(features, labels, seed=SEED)

    return {
        "n_cbri": len(cbri_features),
        "n_secondary_sampled": len(secondary_features),
        "domain_classifier_auc": auc,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cbri-root", required=True, type=Path)
    parser.add_argument("--secondary-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/features"))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    vlm = FrozenVLM()
    report: dict[str, object] = {}

    print("Running within-CBRI proxy-site diagnostic...")
    report["cbri_proxy_sites"] = run_cbri_proxy_site_diagnostic(args.cbri_root, vlm, args.cache_dir)
    print(json.dumps(report["cbri_proxy_sites"], indent=2))

    if args.secondary_root is not None:
        from certus.data.loaders import load_sdnet2018_images

        print("Running CBRI-vs-secondary diagnostic...")
        cbri_records = load_cbri_images(args.cbri_root)
        secondary_records = load_sdnet2018_images(args.secondary_root)
        report["cbri_vs_secondary"] = run_cbri_vs_secondary_diagnostic(
            cbri_records,
            args.cbri_root,
            secondary_records,
            args.secondary_root,
            vlm,
            args.cache_dir,
        )
        print(json.dumps(report["cbri_vs_secondary"], indent=2))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
