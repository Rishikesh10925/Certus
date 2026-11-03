"""The service's calibration artifact: lambda_hat and q_hat per site cluster
and epsilon, plus what the shift check needs. Built offline by
scripts/build_service_calibration.py and loaded once at startup; the
service never recalibrates itself (feedback feeds a weekly batch job).

Clusters are the structure types the guarantee was validated on (deck,
pavement, wall) and `pooled`, the union, for sites of unknown type.
Leave-one-site-out (Week 10-12) showed the guarantee breaks when a site is
unlike the calibration set, so each cluster also keeps its calibration
features: an image whose nearest calibration neighbour is further away
than almost every calibration image's own nearest neighbour is outside
what the guarantee covers (`is_shifted`).

lambda_hat is conditional CRC on the cluster's defects (the miss-rate
guarantee the paper argues for); q_hat is split CP (LAC) at alpha =
epsilon, used only to decide FLAG vs ESCALATE.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from certus.conformal.crc import calibrate_crc
from certus.conformal.split_cp import calibrate_lac

POOLED = "pooled"
EPSILONS = (0.20, 0.10, 0.05, 0.02, 0.01)
SHIFT_QUANTILE = 0.01  # share of calibration images the shift check would flag


@dataclass(frozen=True)
class ClusterCalibration:
    name: str
    n_cal: int
    n_defects: int
    lam: dict[float, float]  # epsilon -> lambda_hat (inf = clear nothing)
    q: dict[float, float]  # alpha -> q_hat
    shift_threshold: float  # min nearest-neighbour cosine similarity
    features: NDArray[np.float32]  # L2-normalised calibration features

    def nearest_similarity(self, feature: NDArray[np.float32]) -> float:
        return float(np.max(self.features @ feature))

    def is_shifted(self, feature: NDArray[np.float32]) -> bool:
        return self.nearest_similarity(feature) < self.shift_threshold


@dataclass(frozen=True)
class Calibration:
    version: str
    model_version: str
    clusters: dict[str, ClusterCalibration]

    def cluster_for(self, site_id: str) -> str:
        """Site ids look like `wall-7001`; an unknown type goes to pooled."""
        prefix = site_id.split("-")[0].lower()
        return prefix if prefix in self.clusters else POOLED


def loo_nearest_similarity(features: NDArray[np.float32], block: int = 2048) -> NDArray[np.float64]:
    """Each row's cosine similarity to its nearest *other* row."""
    out = np.empty(features.shape[0])
    for i in range(0, features.shape[0], block):
        sims = features[i : i + block] @ features.T
        sims[np.arange(sims.shape[0]), np.arange(i, i + sims.shape[0])] = -np.inf
        out[i : i + block] = sims.max(axis=1)
    return out


def calibrate_cluster(
    name: str,
    features: NDArray[np.float32],
    p_defect: NDArray[np.float64],
    labels: NDArray[np.bool_],
    epsilons: tuple[float, ...] = EPSILONS,
    shift_quantile: float = SHIFT_QUANTILE,
) -> ClusterCalibration:
    scores = 1.0 - p_defect  # S1
    probs = np.stack([1.0 - p_defect, p_defect], axis=1)
    pos = labels.astype(bool)
    return ClusterCalibration(
        name=name,
        n_cal=int(labels.size),
        n_defects=int(pos.sum()),
        lam={e: calibrate_crc(scores[pos], labels[pos], e) for e in epsilons},
        q={e: calibrate_lac(probs, labels.astype(int), e) for e in epsilons},
        shift_threshold=float(np.quantile(loo_nearest_similarity(features), shift_quantile)),
        features=features.astype(np.float32),
    )


def save(cal: Calibration, path: Path) -> None:
    """Thresholds go in a JSON sidecar (readable, diffable); features in npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": cal.version,
        "model_version": cal.model_version,
        "clusters": {
            c.name: {
                "n_cal": c.n_cal,
                "n_defects": c.n_defects,
                "lam": {str(e): v for e, v in c.lam.items()},
                "q": {str(e): v for e, v in c.q.items()},
                "shift_threshold": c.shift_threshold,
            }
            for c in cal.clusters.values()
        },
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    np.savez_compressed(
        path.with_suffix(".npz"), **{c.name: c.features for c in cal.clusters.values()}
    )


def load(path: Path) -> Calibration:
    meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path.with_suffix(".npz")) as z:
        feats = {k: z[k] for k in z.files}
    clusters = {
        name: ClusterCalibration(
            name=name,
            n_cal=m["n_cal"],
            n_defects=m["n_defects"],
            lam={float(e): float(v) for e, v in m["lam"].items()},
            q={float(e): float(v) for e, v in m["q"].items()},
            shift_threshold=m["shift_threshold"],
            features=feats[name],
        )
        for name, m in meta["clusters"].items()
    }
    return Calibration(meta["version"], meta["model_version"], clusters)


def finite_or_none(x: float) -> float | None:
    """JSON has no infinity; lambda_hat = inf (clear nothing) is sent as null."""
    return x if math.isfinite(x) else None
