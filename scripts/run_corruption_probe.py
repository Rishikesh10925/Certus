#!/usr/bin/env python
"""Week 9 -- synthetic-corruption shift probe, the controlled second axis.
Calibrate on clean SDNET2018 pool rows, deploy on corrupted copies of the
test rows, and measure realised risk and HRF. Labels don't change, so any
movement is the shift alone.

Only the CLIP-feature scores (S1, S5, S6) run here: S7 would need the slow
VLM and detector passes on all 45,000 corrupted images. As a control, each
cell is also calibrated on corrupted rows, which restores exchangeability,
so the guarantee must come back.

Features are cached per (kind, severity) in data/cache/features/.

Usage:
    python scripts/run_corruption_probe.py
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sklearn.metrics import roc_auc_score

from certus.data.corruptions import KINDS, corrupt
from certus.data.loaders import ImageRecord
from certus.data.splits import read_manifest, sample_records
from certus.eval.protocol import run_crc_protocol, summarize_crc
from certus.scores import s1, s5, s6
from certus.scores.registry import ScoreContext

SEVERITIES = (1, 3, 5)
TARGETS = (("joint", 0.05), ("conditional", 0.10))
N_CAL = 2000
BATCH = 128


def corrupted_features(
    records: list[ImageRecord], kind: str, severity: int, root: Path, cache_dir: Path
) -> NDArray[np.float32]:
    path = cache_dir / "features" / f"corrupt_{kind}_{severity}_{s1.BACKBONE_TAG}.npz"
    relpaths = [r.relpath for r in records]
    if path.exists():
        with np.load(path) as z:
            if z["relpaths"].tolist() == relpaths:
                cached: NDArray[np.float32] = z["features"]
                return cached
    from certus.models.vlm import FrozenVLM

    model = FrozenVLM()

    def load(relpath: str) -> Image.Image:
        return corrupt(Image.open(root / relpath).convert("RGB"), kind, severity, relpath)

    chunks = []
    with ThreadPoolExecutor(8) as pool:
        for i in range(0, len(relpaths), BATCH):
            chunks.append(model.pil_features(list(pool.map(load, relpaths[i : i + BATCH]))))
    feats = np.concatenate(chunks, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, relpaths=np.array(relpaths), features=feats)
    return feats


def score_all(
    feats: NDArray[np.float32],
    text: NDArray[np.float32],
    relpaths: list[str],
    ref: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    n_defect = len(s1.DEFECT_PROMPTS)
    clean = s5.mc_clean_probs(feats, text, relpaths)
    return {
        "s1": 1.0 - s1.zero_shot_defect_prob(feats, text, n_defect),
        "s5": clean.mean(axis=1) - clean.std(axis=1, ddof=1),
        "s6": -s6.mahalanobis(feats, *ref),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest/splits_v1.csv"))
    parser.add_argument("--root", type=Path, default=Path("data/raw/sdnet2018"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("results/week9/corruption_probe.json"))
    args = parser.parse_args()

    records = sample_records(read_manifest(args.manifest, "sdnet2018"), 3000, seed=0, split="cal")
    y = np.array([r.defect_positive for r in records])
    relpaths = [r.relpath for r in records]
    ctx = ScoreContext(
        image_roots={"sdnet2018": args.root}, cache_dir=args.cache_dir, manifest=args.manifest
    )
    text = s1.load_text_features(ctx)
    ref_records = s6.reference_records(ctx)
    ref = s6.fit_reference(s1.load_features(ref_records, ctx), [r.split for r in ref_records])
    clean = score_all(s1.load_features(records, ctx), text, relpaths, ref)

    rows = []
    for score, v in clean.items():
        for guarantee, eps in TARGETS:
            res = run_crc_protocol(v, y, eps, N_CAL, guarantee == "conditional", args.n_seeds)
            rows.append(
                {
                    "kind": "clean",
                    "severity": 0,
                    "score": score,
                    "guarantee": guarantee,
                    "epsilon": eps,
                    "calibrated_on": "clean",
                    "auc": float(roc_auc_score(y, -v)),
                    **summarize_crc(res, eps),
                }
            )
    for kind in KINDS:
        for severity in SEVERITIES:
            feats = corrupted_features(records, kind, severity, args.root, args.cache_dir)
            shifted = score_all(feats, text, relpaths, ref)
            for score, v in shifted.items():
                auc = float(roc_auc_score(y, -v))
                for guarantee, eps in TARGETS:
                    conditional = guarantee == "conditional"
                    for cal, res in (
                        (
                            "clean",
                            run_crc_protocol(
                                clean[score],
                                y,
                                eps,
                                N_CAL,
                                conditional,
                                args.n_seeds,
                                test_scores=v,
                            ),
                        ),
                        (
                            "corrupted",
                            run_crc_protocol(v, y, eps, N_CAL, conditional, args.n_seeds),
                        ),
                    ):
                        rows.append(
                            {
                                "kind": kind,
                                "severity": severity,
                                "score": score,
                                "guarantee": guarantee,
                                "epsilon": eps,
                                "calibrated_on": cal,
                                "auc": auc,
                                **summarize_crc(res, eps),
                            }
                        )
            print(f"done {kind} {severity}", flush=True)

    report = {
        "experiment": "week9_corruption_probe",
        "manifest_hash": json.loads(args.manifest.with_suffix(".json").read_text())["overall_hash"],
        "pool": {"dataset": "sdnet2018", "split": "cal", "n": len(records), "n_pos": int(y.sum())},
        "n_cal": N_CAL,
        "n_seeds": args.n_seeds,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
