# service/

FastAPI triage service (roadmap Section 3.7, Week 14). The code lives in
`src/certus/service/` so it is typed and tested with the rest of the
package; this directory holds the run instructions.

## Contract

- `POST /v1/inspect` -- multipart `image` + `site_id` (+ optional
  `epsilon`, one of 0.20/0.10/0.05/0.02/0.01, default 0.05). Returns the
  triage decision (AUTO_CLEAR / FLAG / ESCALATE), the S1 score, the
  split-CP prediction set, the shift check, a plain-English risk
  statement and per-stage timings.
- `GET /v1/calibration/{cluster}` -- lambda_hat and q_hat per epsilon for
  `deck`, `pavement`, `wall` or `pooled`.
- `POST /v1/feedback` -- `{"inspection_id", "defect", "defect_type"}`.
  Stored; does not trigger recalibration (that is a deliberate weekly
  batch job).
- `GET /v1/audit/next` -- next randomly sampled (5%) auto-cleared image
  awaiting a label, base64; 204 when none.
- `GET /v1/metrics` -- Prometheus text format.

The site id's prefix (`wall-7001` -> `wall`) picks the calibration
cluster; anything else uses `pooled`. An image unlike every calibration
image of its cluster (nearest-neighbour similarity below the 1st
percentile of the calibration set's own) is escalated, never cleared:
leave-one-site-out showed the guarantee does not hold on such images.

## Run

```
python scripts/build_service_calibration.py   # once; ~3 s with cached features
uvicorn certus.service.app:build --factory --port 8000
```

Environment: `CERTUS_CALIBRATION` (artifact path, default
`data/cache/service/calibration_v1`), `CERTUS_DB` (default
`data/cache/service/service.sqlite`), `CERTUS_DEVICE` (`cuda`/`cpu`).

Latency: `python scripts/measure_latency.py` (writes
`results/service/latency.json`).
