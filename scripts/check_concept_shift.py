#!/usr/bin/env python
"""Week 12 check: is the LOSO shift pure covariate shift in CLIP space?
Writes results/e4/concept_shift.json.

Weighted and localized CP correct for a change in p(x) only; they assume
P(defect | x) is the same at every site. S1 is a deterministic function of
the CLIP features x, so for S1 that assumption is all that weighting
needs. Test it directly: fit P(defect | x) (logistic on CLIP features) on
the source sites, predict on the held-out site, and compare the predicted
with the actual defect rate there -- overall and among the held-out
site's defects. A 5-fold in-site fit is the reference for what a correct
P(defect | x) gives on the same images.

Usage:
    python scripts/check_concept_shift.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from certus.data.splits import read_manifest, sample_records
from certus.scores import s1
from certus.scores.registry import ScoreContext

SITES = ("deck", "pavement", "wall")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest/splits_v1.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--out", type=Path, default=Path("results/e4/concept_shift.json"))
    args = parser.parse_args()

    pool = sample_records(read_manifest(args.manifest, "sdnet2018"), 3000, seed=0, split="cal")
    x = s1.load_features(pool, ScoreContext({}, args.cache_dir)).astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.array([r.defect_positive for r in pool])
    site = np.array([r.site_id.split("-")[0] for r in pool])

    rows = []
    for held in SITES:
        src, tgt = site != held, site == held
        clf = LogisticRegression(C=1.0, max_iter=5000).fit(x[src], y[src])
        p_loso = clf.predict_proba(x[tgt])[:, 1]
        p_in = cross_val_predict(
            LogisticRegression(C=1.0, max_iter=5000), x[tgt], y[tgt], cv=5, method="predict_proba"
        )[:, 1]
        yt = y[tgt]
        rows.append(
            {
                "site": held,
                "n": int(tgt.sum()),
                "actual_prevalence": float(yt.mean()),
                "predicted_prevalence_loso": float(p_loso.mean()),
                "predicted_prevalence_in_site": float(p_in.mean()),
                "mean_p_on_defects_loso": float(p_loso[yt].mean()),
                "mean_p_on_defects_in_site": float(p_in[yt].mean()),
                "auc_loso": float(roc_auc_score(yt, p_loso)),
                "auc_in_site": float(roc_auc_score(yt, p_in)),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"experiment": "concept_shift_check", "rows": rows}, indent=1) + "\n"
    )
    for r in rows:
        print(
            f"{r['site']:9s} prevalence actual {r['actual_prevalence']:.3f}"
            f" | predicted LOSO {r['predicted_prevalence_loso']:.3f}"
            f" in-site {r['predicted_prevalence_in_site']:.3f}"
            f" | mean p on defects LOSO {r['mean_p_on_defects_loso']:.3f}"
            f" in-site {r['mean_p_on_defects_in_site']:.3f}"
            f" | AUC LOSO {r['auc_loso']:.3f} in-site {r['auc_in_site']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
