// Parses the Prometheus text exposition served by GET /v1/metrics.

export interface Sample {
  name: string;
  labels: Record<string, string>;
  value: number;
}

const LINE = /^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)/;
const LABEL = /([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"/g;

export function parsePrometheus(text: string): Sample[] {
  const out: Sample[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const m = LINE.exec(line);
    if (!m) continue;
    const labels: Record<string, string> = {};
    for (const l of (m[2] ?? "").matchAll(LABEL)) labels[l[1]] = l[2];
    out.push({ name: m[1], labels, value: Number(m[3]) });
  }
  return out;
}

export interface ServiceMetrics {
  decisions: { AUTO_CLEAR: number; FLAG: number; ESCALATE: number };
  total: number;
  shiftFlags: number;
  auditSampled: number;
  feedback: number;
  feedbackMissed: number;
  latency: { p50: number; p95: number; p99: number; count: number; sum: number };
  calibrationVersion: string | null;
  escalateOnShift: boolean | null;
}

export function readMetrics(text: string): ServiceMetrics {
  const s = parsePrometheus(text);
  const get = (name: string, labels: Record<string, string> = {}) =>
    s.find(
      (x) => x.name === name && Object.entries(labels).every(([k, v]) => x.labels[k] === v),
    )?.value ?? 0;
  const decisions = {
    AUTO_CLEAR: get("certus_inspections_total", { decision: "AUTO_CLEAR" }),
    FLAG: get("certus_inspections_total", { decision: "FLAG" }),
    ESCALATE: get("certus_inspections_total", { decision: "ESCALATE" }),
  };
  const info = s.find((x) => x.name === "certus_calibration_info");
  const count = get("certus_inspect_latency_seconds_count");
  // The service reports zeros for the quantiles before the first request.
  const q = (v: string) => 1000 * get("certus_inspect_latency_seconds", { quantile: v });
  return {
    decisions,
    total: decisions.AUTO_CLEAR + decisions.FLAG + decisions.ESCALATE,
    shiftFlags: get("certus_shift_flags_total"),
    auditSampled: get("certus_audit_sampled_total"),
    feedback: get("certus_feedback_total"),
    feedbackMissed: get("certus_feedback_missed_defects_total"),
    latency: {
      p50: q("0.5"),
      p95: q("0.95"),
      p99: q("0.99"),
      count,
      sum: get("certus_inspect_latency_seconds_sum"),
    },
    calibrationVersion: info?.labels.version ?? null,
    escalateOnShift: info ? info.labels.escalate_on_shift === "True" : null,
  };
}
