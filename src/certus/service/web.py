"""Read-only endpoints for the web console (web/), plus serving its build.

The triage contract in app.py is unchanged; this module only adds what a
browser cannot do for itself:

- GET /v1/info                      service, calibration and model versions
- GET /v1/samples                   a seeded batch of held-out test images
                                    with ground truth, for the batch console
- GET /v1/samples/image             one of those images' bytes
- GET /v1/risk-statement            app.risk_statement, so the console never
                                    re-words the guarantee itself
- GET /v1/research                  which committed results files are served
- GET /v1/research/{name}           one results/ JSON (NaN sent as null)
- GET /v1/research/figures/{name}   one committed figure (F1-F9)

Batches are drawn exactly as the old Streamlit console drew them (test
split, site-id prefix, numpy default_rng(seed).choice without
replacement), so batch N shows the same images it always did.

If `static_dir` holds a built web console, it is served at / with a
single-page-app fallback.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from certus.data.loaders import ImageRecord
from certus.data.splits import read_manifest
from certus.service import calibration as calib
from certus.service.app import risk_statement
from certus.triage.policy import Triage

SAMPLE_DATASET = "sdnet2018"
SAMPLE_SPLIT = "test"
SITE_TYPES = ("wall", "deck", "pavement")
MAX_BATCH = 96

# Served results: name -> path under results/. Only these, never arbitrary paths.
RESEARCH_FILES = {
    "e1_validity": "e1/e1_validity.json",
    "e2_scores": "e2/e2_scores.json",
    "e3_efficiency": "e3/e3_efficiency.json",
    "loso_v3": "e4/loso_v3.json",
    "latency_cuda": "service/latency_cuda.json",
    "latency_cpu": "service/latency_cpu.json",
    "calibration_build": "service/calibration_build.json",
}


def _finite(obj: Any) -> Any:
    """JSON has no NaN or infinity; send them as null."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


def draw_batch(records: list[ImageRecord], site_type: str, seed: int, n: int) -> list[ImageRecord]:
    """Held-out test images of one structure type, the console's sampling."""
    pool = [r for r in records if r.split == SAMPLE_SPLIT and r.site_id.startswith(site_type)]
    if not pool:
        return []
    rng = np.random.default_rng(seed)
    return [pool[int(i)] for i in rng.choice(len(pool), min(n, len(pool)), replace=False)]


def mount_web(
    app: FastAPI,
    *,
    manifest: Path | None = None,
    data_root: Path | None = None,
    results_dir: Path | None = None,
    figures_dir: Path | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """Adds the console endpoints to an app built by app.create_app."""
    cal: calib.Calibration = app.state.calibration
    app.add_middleware(GZipMiddleware, minimum_size=2048)

    records: list[ImageRecord] = []
    if manifest is not None and manifest.is_file():
        records = read_manifest(manifest, SAMPLE_DATASET)
    by_relpath = {r.relpath: r for r in records if r.split == SAMPLE_SPLIT}
    samples_ok = bool(by_relpath) and data_root is not None and data_root.is_dir()

    @app.get("/v1/info")
    def info() -> dict[str, Any]:
        return {
            "service": app.title,
            "version": app.version,
            "calibration_version": cal.version,
            "model_version": app.state.model_version,
            "clusters": sorted(cal.clusters),
            "epsilons": sorted(next(iter(cal.clusters.values())).lam, reverse=True),
            "escalate_on_shift": app.state.escalate_on_shift,
            "audit_rate": app.state.audit_rate,
            "samples_available": samples_ok,
            "sample_dataset": SAMPLE_DATASET if samples_ok else None,
            "site_types": list(SITE_TYPES),
        }

    @app.get("/v1/samples")
    def samples(
        site_type: str = Query("wall"),
        seed: int = Query(0, ge=0),
        n: int = Query(48, ge=1, le=MAX_BATCH),
    ) -> dict[str, Any]:
        if not samples_ok:
            raise HTTPException(503, "sample images are not available on this server")
        if site_type not in SITE_TYPES:
            raise HTTPException(422, f"site_type must be one of {list(SITE_TYPES)}")
        batch = draw_batch(records, site_type, seed, n)
        return {
            "dataset": SAMPLE_DATASET,
            "split": SAMPLE_SPLIT,
            "site_type": site_type,
            "seed": seed,
            "items": [
                {
                    "relpath": r.relpath,
                    "site_id": r.site_id,
                    "defect": r.defect_positive,
                    "classes": list(r.classes),
                }
                for r in batch
            ],
        }

    @app.get("/v1/samples/image")
    def sample_image(relpath: str = Query(...)) -> FileResponse:
        # Only manifest test images: the lookup is the path-traversal guard.
        if not samples_ok or relpath not in by_relpath:
            raise HTTPException(404, "no such sample image")
        assert data_root is not None
        path = data_root / relpath
        if not path.is_file():
            raise HTTPException(404, "sample image missing on disk")
        return FileResponse(
            path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"}
        )

    @app.get("/v1/risk-statement")
    def statement(
        cluster: str,
        epsilon: float,
        decision: str = "AUTO_CLEAR",
        shifted: bool = False,
        defect_type: str = "",
    ) -> dict[str, str]:
        if cluster not in cal.clusters:
            raise HTTPException(404, f"unknown cluster {cluster!r}")
        if decision not in Triage.__members__:
            raise HTTPException(422, f"decision must be one of {list(Triage.__members__)}")
        n_defects = cal.clusters[cluster].n_defects
        text = risk_statement(Triage[decision], cluster, epsilon, n_defects, shifted, defect_type)
        return {"risk_statement": text}

    def _research_path(name: str) -> Path:
        if results_dir is None or name not in RESEARCH_FILES:
            raise HTTPException(404, f"unknown results file {name!r}")
        path = results_dir / RESEARCH_FILES[name]
        if not path.is_file():
            raise HTTPException(404, f"{RESEARCH_FILES[name]} is missing")
        return path

    @app.get("/v1/research")
    def research_index() -> dict[str, Any]:
        files = {
            name: (results_dir is not None and (results_dir / rel).is_file())
            for name, rel in RESEARCH_FILES.items()
        }
        figures = (
            sorted(p.stem for p in figures_dir.glob("F*.png"))
            if figures_dir is not None and figures_dir.is_dir()
            else []
        )
        return {"files": files, "figures": figures}

    @app.get("/v1/research/{name}")
    def research(name: str) -> Any:
        return _finite(json.loads(_research_path(name).read_text(encoding="utf-8")))

    @app.get("/v1/research/figures/{name}")
    def figure(name: str) -> FileResponse:
        if figures_dir is None or not name.startswith("F") or not name.replace("_", "").isalnum():
            raise HTTPException(404, "no such figure")
        path = figures_dir / f"{name}.png"
        if not path.is_file():
            raise HTTPException(404, "no such figure")
        return FileResponse(path, media_type="image/png")

    if static_dir is not None and (static_dir / "index.html").is_file():
        index = static_dir / "index.html"
        if (static_dir / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith("v1/"):
                raise HTTPException(404, "not found")
            candidate = (static_dir / path).resolve()
            if path and candidate.is_file() and static_dir.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(index)

    return app
