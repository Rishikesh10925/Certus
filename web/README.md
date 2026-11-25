# web/

The Certus console: a React + TypeScript single-page app (Vite, Tailwind
CSS, Radix UI primitives, Recharts, TanStack Query). It replaced the
Streamlit console. It talks only to the FastAPI service; every
number it shows comes from the service or from the committed `results/`
files the service exposes.

## Pages

| Page | What it does | Endpoints |
|---|---|---|
| Overview | What the guarantee says, thresholds at the default ε, service status, F1 | `/v1/info`, `/v1/calibration/*`, `/v1/metrics` |
| Batch triage | The old console: a seeded batch of held-out test images, inspected once each; the ε slider re-partitions it into auto-cleared vs. human review, with HRF, projected review hours, ground-truth misses, a per-ε curve and the risk statement | `/v1/samples`, `/v1/inspect`, `/v1/calibration/{type}`, `/v1/risk-statement` |
| Inspect image | Upload (or draw a held-out sample), choose site id and ε, get the full decision; label it | `/v1/inspect`, `/v1/feedback` |
| Audit queue | Label the random sample of auto-cleared images | `/v1/audit/next`, `/v1/feedback` |
| Calibration | λ̂ and q̂ per cluster and ε, shift threshold, guarantee | `/v1/calibration/*` |
| Monitoring | Live Prometheus counters and latency | `/v1/metrics` |
| Research results | E2, E3, LOSO and latency results, F1–F9 | `/v1/research/*` |
| Settings | Theme, default ε, batch size, review-cost inputs (browser only) | – |

The batch console inspects each image once at ε = 5% and then applies the
service's decision rule locally for other ε (`src/lib/triage.ts`, a
line-for-line copy of `POST /v1/inspect`). Score, p(defect) and the shift
check do not depend on ε, so this is exact; the page shows a live check
that the local rule agrees with the service on every image at ε = 5%.

## Run

Production (one process, one port):

```
npm ci && npm run build          # or: make web
uvicorn certus.service.app:build --factory --port 8000   # or: make serve
# open http://localhost:8000
```

Development (hot reload, `/v1` proxied to the service on :8000):

```
npm run dev                       # http://localhost:5173
```

`npm run typecheck`, `npm test` (Vitest) and `npm run build` run in CI.
