#!/usr/bin/env python
"""Builds the service's calibration artifact from the whole SDNET2018
calibration split (8413 images, human labels): lambda_hat and q_hat per
structure type and pooled, per epsilon, plus the shift-check features.
Writes data/cache/service/calibration_v1.{json,npz} and the build time to
results/service/calibration_build.json (the blueprint reports calibration
overhead separately from request latency).

The service scores online with the same function as S1, so this also
checks that function against the S1 values stored for the research pool.

Usage:
    python scripts/build_service_calibration.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from certus.data.splits import read_manifest, sample_records
from certus.data.store import connect, model_versions, read_scores
from certus.scores import s1
from certus.scores.registry import ScoreContext
from certus.service import calibration as calib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest/splits_v1.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--db", type=Path, default=Path("data/cache/certus.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("data/cache/service/calibration_v1"))
    parser.add_argument(
        "--timing", type=Path, default=Path("results/service/calibration_build.json")
    )
    args = parser.parse_args()

    ctx = ScoreContext({"sdnet2018": Path("data/raw/sdnet2018")}, args.cache_dir)
    records = [r for r in read_manifest(args.manifest, "sdnet2018") if r.split == "cal"]
    t0 = time.perf_counter()
    feats = s1.load_features(records, ctx)
    t_feats = time.perf_counter() - t0
    text = s1.load_text_features(ctx)
    p = s1.zero_shot_defect_prob(feats, text, len(s1.DEFECT_PROMPTS))
    y = np.array([r.defect_positive for r in records])
    kind = np.array([r.site_id.split("-")[0] for r in records])

    # The served score must be the calibrated score.
    pool = sample_records(read_manifest(args.manifest, "sdnet2018"), 3000, seed=0, split="cal")
    conn = connect(args.db)
    (version,) = (v for v in model_versions(conn, "s1") if "logreg" not in v)
    stored = read_scores(conn, pool, "s1", version)
    conn.close()
    index = {r.relpath: i for i, r in enumerate(records)}
    online = 1.0 - p[[index[r.relpath] for r in pool]]
    max_diff = float(np.max(np.abs(online - stored)))
    if max_diff > 1e-5:
        raise SystemExit(f"online S1 differs from stored S1 by {max_diff}")
    model_version = f"{s1.BACKBONE_TAG}+zeroshot:{s1.prompt_hash()}"
    if model_version != version:
        raise SystemExit(f"S1 version {model_version} != stored {version}")

    t1 = time.perf_counter()
    clusters = {}
    for name in ("deck", "pavement", "wall", calib.POOLED):
        rows = np.ones(len(records), bool) if name == calib.POOLED else kind == name
        clusters[name] = calib.calibrate_cluster(name, feats[rows], p[rows], y[rows])
    t_cal = time.perf_counter() - t1
    manifest_hash = json.loads(args.manifest.with_suffix(".json").read_text())["overall_hash"]
    cal = calib.Calibration(f"v1-{manifest_hash[:8]}", model_version, clusters)
    calib.save(cal, args.out)

    timing = {
        "n_images": len(records),
        "features_seconds": t_feats,
        "calibration_seconds": t_cal,
        "features_were_cached": t_feats < 5.0,
        "online_vs_stored_s1_max_abs_diff": max_diff,
        "calibration_version": cal.version,
    }
    args.timing.parent.mkdir(parents=True, exist_ok=True)
    args.timing.write_text(json.dumps(timing, indent=1) + "\n")
    print(json.dumps(timing, indent=1))
    for c in clusters.values():
        lam = ", ".join(f"{e:g}: {v:.3f}" for e, v in c.lam.items())
        print(f"{c.name:9s} n={c.n_cal:5d} defects={c.n_defects:4d} shift>={c.shift_threshold:.3f}")
        print(f"          lambda {lam}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
