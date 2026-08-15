# service/

FastAPI + ONNX Runtime inspection service (roadmap Section 3.7, deliverable
due Week 14). Not implemented yet -- this is a placeholder so the directory
exists in git.

Before building, review the service contract in roadmap Section 3.7:

- `POST /v1/inspect` -- multipart image + site_id, returns a triage
  decision with an attached risk statement.
- `GET /v1/calibration/{cluster}` -- current lambda_hat and q_hat for a
  site cluster.
- `POST /v1/feedback` -- human-supplied ground truth; does not trigger
  recalibration (that is a deliberate weekly batch job).
- `GET /v1/audit/next` -- next randomly sampled image awaiting a label.
- `GET /v1/metrics` -- Prometheus format.
