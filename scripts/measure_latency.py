#!/usr/bin/env python
"""Week 14: end-to-end latency of POST /v1/inspect, and the deployed
service's own miss rate. Starts the real service (uvicorn, CLIP on the
chosen device) as a subprocess, sends every image once over HTTP from a
single sequential client, and records client-side wall time per request
plus the server's stage timings.

Images: the whole SDNET2018 *test* split (never calibrated on; human
labels, so the realised miss rate per cluster can be checked against
epsilon) and the 175 gold-labelled CBRI images (whole facades, an unseen
site type: they exercise the shift check). Writes
results/service/latency_<device>.json.

Usage:
    python scripts/measure_latency.py --device cuda
    python scripts/measure_latency.py --device cpu --limit 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from certus.data.splits import read_manifest

QUANTILES = (0.5, 0.95, 0.99)
BUDGET_MS = 400.0  # blueprint objective O7: p95 under 400 ms


def wait_ready(url: str, proc: subprocess.Popen[bytes], timeout: float = 300.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"service exited with code {proc.returncode}")
        try:
            if httpx.get(f"{url}/v1/calibration/pooled", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise SystemExit("service did not come up")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest/splits_v1.csv"))
    parser.add_argument("--gold", type=Path, default=Path("data/gold/gold_labels_v1.csv"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0, help="random subset size (0 = all)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--out-dir", type=Path, default=Path("results/service"))
    args = parser.parse_args()

    sdnet = [r for r in read_manifest(args.manifest, "sdnet2018") if r.split == "test"]
    with args.gold.open(newline="", encoding="utf-8") as f:
        gold = {g["relpath"]: g["defect_positive"] == "yes" for g in csv.DictReader(f)}
    cbri = [r for r in read_manifest(args.manifest, "cbri") if r.relpath in gold]
    jobs = [
        (Path("data/raw/sdnet2018") / r.relpath, r.site_id, r.defect_positive, "sdnet2018")
        for r in sdnet
    ]
    jobs += [
        (Path("data/raw/cbri") / r.relpath, "cbri-facade", gold[r.relpath], "cbri") for r in cbri
    ]
    if args.limit:
        rng = np.random.default_rng(0)
        jobs = [jobs[i] for i in sorted(rng.choice(len(jobs), args.limit, replace=False))]

    db = args.out_dir / f"latency_{args.device}.sqlite"
    db.unlink(missing_ok=True)
    env = {**os.environ, "CERTUS_DEVICE": args.device, "CERTUS_DB": str(db)}
    url = f"http://127.0.0.1:{args.port}"
    t_start = time.perf_counter()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "certus.service.app:build",
            "--factory",
            "--port",
            str(args.port),
            "--log-level",
            "warning",
        ],
        env=env,
    )
    try:
        wait_ready(url, proc)
        startup_s = time.perf_counter() - t_start
        rows = []
        with httpx.Client(timeout=60.0) as client:
            for path, site, label, dataset in jobs:
                data = path.read_bytes()
                t0 = time.perf_counter()
                r = client.post(
                    f"{url}/v1/inspect",
                    files={"image": (path.name, data, "image/jpeg")},
                    data={"site_id": site, "epsilon": str(args.epsilon)},
                )
                wall_ms = 1000 * (time.perf_counter() - t0)
                r.raise_for_status()
                out = r.json()
                rows.append(
                    {
                        "dataset": dataset,
                        "cluster": out["cluster"],
                        "defect": label,
                        "decision": out["decision"],
                        "shifted": out["shift"]["flagged"],
                        "bytes": len(data),
                        "client_ms": wall_ms,
                        **{f"server_{k}_ms": v for k, v in out["timings_ms"].items()},
                    }
                )
        metrics = httpx.get(f"{url}/v1/metrics").text
    finally:
        proc.terminate()
        proc.wait(timeout=30)

    def q(values: list[float]) -> dict[str, float]:
        v = np.array(values)
        return {f"p{int(100 * x)}": float(np.quantile(v, x)) for x in QUANTILES} | {
            "mean": float(v.mean()),
            "max": float(v.max()),
        }

    lat = {
        "client_total": q([r["client_ms"] for r in rows]),
        **{
            f"server_{k}": q([r[f"server_{k}_ms"] for r in rows])
            for k in ("decode", "embed", "score", "total")
        },
        **{
            f"client_total_{d}": q([r["client_ms"] for r in rows if r["dataset"] == d])
            for d in ("sdnet2018", "cbri")
            if any(r["dataset"] == d for r in rows)
        },
    }
    outcome = {}
    for cluster in sorted({r["cluster"] for r in rows}):
        sel = [r for r in rows if r["cluster"] == cluster]
        defects = [r for r in sel if r["defect"]]
        cleared = [r["decision"] == "AUTO_CLEAR" for r in defects]
        outcome[cluster] = {
            "n": len(sel),
            "n_defects": len(defects),
            "miss_rate": float(np.mean(cleared)) if defects else None,
            "hrf": float(np.mean([r["decision"] != "AUTO_CLEAR" for r in sel])),
            "shift_flag_rate": float(np.mean([r["shifted"] for r in sel])),
            "decisions": {
                d: sum(r["decision"] == d for r in sel) for d in ("AUTO_CLEAR", "FLAG", "ESCALATE")
            },
        }
    report = {
        "device": args.device,
        "epsilon": args.epsilon,
        "n_requests": len(rows),
        "client": "single sequential httpx client, same machine",
        "startup_seconds": startup_s,
        "budget_ms": BUDGET_MS,
        "p95_within_budget": lat["client_total"]["p95"] < BUDGET_MS,
        "latency_ms": lat,
        "outcome_by_cluster": outcome,
        "metrics_snapshot": metrics,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"latency_{args.device}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    np.savez_compressed(
        args.out_dir / f"latency_{args.device}_requests.npz",
        client_ms=np.array([r["client_ms"] for r in rows]),
        server_decode_ms=np.array([r["server_decode_ms"] for r in rows]),
        server_embed_ms=np.array([r["server_embed_ms"] for r in rows]),
        server_total_ms=np.array([r["server_total_ms"] for r in rows]),
        dataset=np.array([r["dataset"] for r in rows]),
    )
    print(json.dumps({k: v for k, v in report.items() if k != "metrics_snapshot"}, indent=1))
    db.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
