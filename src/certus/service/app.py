"""FastAPI triage service, per the contract in service/README.md:

- POST /v1/inspect             image + site_id -> triage decision + risk statement
- GET  /v1/calibration/{name}  lambda_hat and q_hat per epsilon for a site cluster
- POST /v1/feedback            human ground truth; stored, never recalibrates
- GET  /v1/audit/next          next randomly sampled auto-cleared image to label
- GET  /v1/metrics             Prometheus text format

Auto-clear is the conformal decision (S1 score > lambda_hat for the site's
cluster at the requested epsilon). The one addition is the shift check: an
image unlike the cluster's calibration set is escalated instead of
cleared, because leave-one-site-out showed the guarantee does not hold
there (paper, Section Results). A random share of auto-cleared images is
held for audit, which is how a deployment would notice a broken guarantee.

Run: uvicorn certus.service.app:build --factory  (env: CERTUS_CALIBRATION,
CERTUS_DB, CERTUS_DEVICE).
"""

from __future__ import annotations

import base64
import os
import sqlite3
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel

from certus.service import calibration as calib
from certus.service.calibration import Calibration, finite_or_none
from certus.service.scorer import Scorer
from certus.triage.policy import Triage

AUDIT_RATE = 0.05
LATENCY_WINDOW = 10_000
QUANTILES = (0.5, 0.95, 0.99)

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    site_id TEXT NOT NULL,
    cluster TEXT NOT NULL,
    epsilon REAL NOT NULL,
    score REAL NOT NULL,
    decision TEXT NOT NULL,
    shifted INTEGER NOT NULL,
    audit INTEGER NOT NULL,
    image BLOB
);
CREATE TABLE IF NOT EXISTS feedback (
    inspection_id INTEGER PRIMARY KEY REFERENCES inspections(id),
    created_at REAL NOT NULL,
    defect INTEGER NOT NULL,
    defect_type TEXT
);
"""


class Feedback(BaseModel):
    inspection_id: int
    defect: bool
    defect_type: str | None = None


def risk_statement(
    decision: Triage,
    cluster: str,
    epsilon: float,
    n_defects: int,
    shifted: bool,
    defect_type: str,
) -> str:
    """Plain English, one paragraph, for the console and the audit log."""
    scope = f"{cluster} sites" if cluster != calib.POOLED else "all calibrated site types"
    guarantee = (
        f"On {scope}, this threshold auto-clears at most {epsilon:.0%} of real defects on"
        f" average (conformal risk control on {n_defects} labelled defects)."
    )
    if shifted:
        return (
            "Sent to a human: this image is unlike every calibration image for "
            f"{scope}, so the guarantee does not cover it. {guarantee}"
        )
    if decision == Triage.AUTO_CLEAR:
        return (
            f"Auto-cleared. {guarantee} The guarantee assumes this image is drawn"
            " like the calibration images; it is an average over images, not a"
            " promise about this one."
        )
    if decision == Triage.FLAG:
        return f"Sent to a human, likely {defect_type} (pre-filled). {guarantee}"
    return f"Sent to a human; no defect type is certain enough to pre-fill. {guarantee}"


def _prometheus(
    counts: Counter[str], latencies: deque[float], cal: Calibration, escalate_on_shift: bool
) -> str:
    lines = [
        "# HELP certus_inspections_total Inspections by triage decision.",
        "# TYPE certus_inspections_total counter",
    ]
    for d in Triage:
        lines.append(f'certus_inspections_total{{decision="{d.name}"}} {counts[d.name]}')
    for name, help_text in (
        ("shift_flags", "Images outside their cluster's calibration set."),
        ("audit_sampled", "Auto-cleared images held for audit."),
        ("feedback", "Human labels received."),
        ("feedback_missed_defects", "Auto-cleared images a human labelled defective."),
    ):
        lines += [
            f"# HELP certus_{name}_total {help_text}",
            f"# TYPE certus_{name}_total counter",
            f"certus_{name}_total {counts[name]}",
        ]
    lines += [
        "# HELP certus_inspect_latency_seconds Server-side /v1/inspect latency.",
        "# TYPE certus_inspect_latency_seconds summary",
    ]
    lat = np.array(latencies) if latencies else np.zeros(1)
    for q in QUANTILES:
        lines.append(f'certus_inspect_latency_seconds{{quantile="{q}"}} {np.quantile(lat, q):.6f}')
    lines += [
        f"certus_inspect_latency_seconds_sum {float(np.sum(latencies)):.6f}",
        f"certus_inspect_latency_seconds_count {len(latencies)}",
        "# HELP certus_calibration_info Loaded calibration artifact.",
        "# TYPE certus_calibration_info gauge",
        f'certus_calibration_info{{version="{cal.version}",'
        f'escalate_on_shift="{escalate_on_shift}"}} 1',
    ]
    return "\n".join(lines) + "\n"


def create_app(
    cal: Calibration,
    scorer: Scorer,
    db_path: Path | str = ":memory:",
    audit_rate: float = AUDIT_RATE,
    seed: int = 0,
    escalate_on_shift: bool = True,
) -> FastAPI:
    app = FastAPI(title="Certus triage service", version="1.0")
    db = sqlite3.connect(str(db_path), check_same_thread=False)
    db.executescript(SCHEMA)
    lock = threading.Lock()
    rng = np.random.default_rng(seed)
    counts: Counter[str] = Counter()
    latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)

    @app.post("/v1/inspect")
    async def inspect(
        image: UploadFile = File(...),  # noqa: B008 -- FastAPI's declaration style
        site_id: str = Form(...),
        epsilon: float = Form(0.05),
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        cluster = cal.cluster_for(site_id)
        cc = cal.clusters[cluster]
        if epsilon not in cc.lam:
            raise HTTPException(422, f"epsilon must be one of {sorted(cc.lam)}")
        data = await image.read()
        try:
            scored = scorer.score(data)
        except Exception as err:  # PIL raises several types on bad input
            raise HTTPException(422, f"could not read image: {err}") from err
        lam, q = cc.lam[epsilon], cc.q[epsilon]
        shifted = cc.is_shifted(scored.feature)
        cleared = scored.score > lam and not (shifted and escalate_on_shift)
        # LAC set: class y is in it iff 1 - p_y <= q_hat.
        in_set = {"clean": scored.p_defect <= q, "defect": 1.0 - scored.p_defect <= q}
        if cleared:
            decision = Triage.AUTO_CLEAR
        elif in_set["defect"] and not in_set["clean"]:
            decision = Triage.FLAG
        else:
            decision = Triage.ESCALATE
        audit = decision == Triage.AUTO_CLEAR and rng.random() < audit_rate
        with lock:
            cur = db.execute(
                "INSERT INTO inspections (created_at, site_id, cluster, epsilon, score,"
                " decision, shifted, audit, image) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    site_id,
                    cluster,
                    epsilon,
                    scored.score,
                    decision.name,
                    int(shifted),
                    int(audit),
                    data if audit else None,
                ),
            )
            db.commit()
            counts[decision.name] += 1
            counts["shift_flags"] += int(shifted)
            counts["audit_sampled"] += int(audit)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        return {
            "inspection_id": cur.lastrowid,
            "decision": decision.name,
            "defect_type": scored.defect_type if decision == Triage.FLAG else None,
            "score": scored.score,
            "p_defect": scored.p_defect,
            "prediction_set": [k for k, v in in_set.items() if v],
            "site_id": site_id,
            "cluster": cluster,
            "epsilon": epsilon,
            "lambda_hat": finite_or_none(lam),
            "shift": {
                "flagged": shifted,
                "nearest_similarity": cc.nearest_similarity(scored.feature),
                "threshold": cc.shift_threshold,
            },
            "risk_statement": risk_statement(
                decision, cluster, epsilon, cc.n_defects, shifted, scored.defect_type
            ),
            "model_version": scorer.model_version,
            "calibration_version": cal.version,
            "timings_ms": {**scored.timings_ms, "total": 1000 * elapsed},
        }

    @app.get("/v1/calibration/{name}")
    def calibration(name: str) -> dict[str, Any]:
        if name not in cal.clusters:
            raise HTTPException(404, f"unknown cluster {name!r}; have {sorted(cal.clusters)}")
        cc = cal.clusters[name]
        return {
            "cluster": name,
            "calibration_version": cal.version,
            "model_version": cal.model_version,
            "n_cal": cc.n_cal,
            "n_defects": cc.n_defects,
            "guarantee": "conditional CRC: E[P(auto-cleared | defect)] <= epsilon",
            "thresholds": [
                {"epsilon": e, "lambda_hat": finite_or_none(cc.lam[e]), "q_hat": cc.q[e]}
                for e in sorted(cc.lam, reverse=True)
            ],
            "shift_threshold": cc.shift_threshold,
        }

    @app.post("/v1/feedback")
    def feedback(fb: Feedback) -> dict[str, Any]:
        with lock:
            row = db.execute(
                "SELECT decision FROM inspections WHERE id = ?", (fb.inspection_id,)
            ).fetchone()
            if row is None:
                raise HTTPException(404, f"no inspection {fb.inspection_id}")
            db.execute(
                "INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?)",
                (fb.inspection_id, time.time(), int(fb.defect), fb.defect_type),
            )
            db.commit()
            counts["feedback"] += 1
            missed = fb.defect and row[0] == Triage.AUTO_CLEAR.name
            counts["feedback_missed_defects"] += int(missed)
        return {
            "recorded": True,
            "missed_defect": missed,
            "recalibration": "not triggered; labels feed the weekly batch recalibration",
        }

    @app.get("/v1/audit/next", response_model=None)
    def audit_next() -> dict[str, Any] | Response:
        with lock:
            row = db.execute(
                "SELECT i.id, i.site_id, i.cluster, i.score, i.created_at, i.image"
                " FROM inspections i LEFT JOIN feedback f ON f.inspection_id = i.id"
                " WHERE i.audit = 1 AND f.inspection_id IS NULL ORDER BY i.id LIMIT 1"
            ).fetchone()
        if row is None:
            return Response(status_code=204)
        return {
            "inspection_id": row[0],
            "site_id": row[1],
            "cluster": row[2],
            "score": row[3],
            "created_at": row[4],
            "image_b64": base64.b64encode(row[5]).decode("ascii"),
        }

    @app.get("/v1/metrics")
    def metrics() -> Response:
        with lock:
            body = _prometheus(counts, latencies, cal, escalate_on_shift)
        return Response(body, media_type="text/plain; version=0.0.4")

    return app


def build() -> FastAPI:
    """Factory for uvicorn: loads the artifact and CLIP from the environment."""
    from certus.scores import s1
    from certus.scores.registry import ScoreContext
    from certus.service.scorer import ClipScorer

    cache = Path(os.environ.get("CERTUS_CACHE", "data/cache"))
    cal = calib.load(
        Path(os.environ.get("CERTUS_CALIBRATION", "data/cache/service/calibration_v1"))
    )
    text = s1.load_text_features(ScoreContext({}, cache))
    scorer = ClipScorer(text, device=os.environ.get("CERTUS_DEVICE") or None)
    if scorer.model_version != cal.model_version:
        raise RuntimeError(f"scorer {scorer.model_version} != calibration {cal.model_version}")
    return create_app(cal, scorer, os.environ.get("CERTUS_DB", "data/cache/service/service.sqlite"))
