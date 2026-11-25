// Re-partitions an already-inspected batch at any epsilon without calling the
// model again. This is the decision rule of POST /v1/inspect
// (src/certus/service/app.py), line for line:
//
//   cleared  = score > lambda_hat  and not (shifted and escalate_on_shift)
//   in_set   = {clean: p_defect <= q_hat, defect: 1 - p_defect <= q_hat}
//   decision = AUTO_CLEAR if cleared
//              else FLAG if in_set.defect and not in_set.clean
//              else ESCALATE
//
// score, p_defect and the shift check do not depend on epsilon, so one
// inspection per image is enough. lambda_hat = null is infinity (clear nothing).

import type { ClusterCalibration, Decision, Threshold } from "./types";

export interface Scored {
  score: number;
  p_defect: number;
  shifted: boolean;
}

export function lambdaValue(t: Pick<Threshold, "lambda_hat">): number {
  return t.lambda_hat ?? Number.POSITIVE_INFINITY;
}

export function predictionSet(pDefect: number, q: number): ("clean" | "defect")[] {
  const out: ("clean" | "defect")[] = [];
  if (pDefect <= q) out.push("clean");
  if (1 - pDefect <= q) out.push("defect");
  return out;
}

export function decide(
  s: Scored,
  t: Pick<Threshold, "lambda_hat" | "q_hat">,
  escalateOnShift = true,
): Decision {
  const cleared = s.score > lambdaValue(t) && !(s.shifted && escalateOnShift);
  if (cleared) return "AUTO_CLEAR";
  const set = predictionSet(s.p_defect, t.q_hat);
  if (set.includes("defect") && !set.includes("clean")) return "FLAG";
  return "ESCALATE";
}

export function thresholdFor(cal: ClusterCalibration, epsilon: number): Threshold {
  const t = cal.thresholds.find((x) => x.epsilon === epsilon);
  if (!t) throw new Error(`no threshold for epsilon ${epsilon} in cluster ${cal.cluster}`);
  return t;
}

export interface BatchSummary {
  n: number;
  cleared: number;
  human: number;
  /** Human-review fraction: (FLAG + ESCALATE) / N. */
  hrf: number;
  defects: number;
  /** Ground-truth defects that were auto-cleared (a miss). */
  missed: number;
  /** missed / defects in this batch; null if the batch has no defects. */
  batchMissRate: number | null;
  byDecision: Record<Decision, number>;
}

export function summarize<T extends Scored & { truth: boolean }>(
  items: T[],
  t: Pick<Threshold, "lambda_hat" | "q_hat">,
  escalateOnShift = true,
): BatchSummary {
  const byDecision: Record<Decision, number> = { AUTO_CLEAR: 0, FLAG: 0, ESCALATE: 0 };
  let missed = 0;
  let defects = 0;
  for (const it of items) {
    const d = decide(it, t, escalateOnShift);
    byDecision[d] += 1;
    if (it.truth) {
      defects += 1;
      if (d === "AUTO_CLEAR") missed += 1;
    }
  }
  const n = items.length;
  const cleared = byDecision.AUTO_CLEAR;
  return {
    n,
    cleared,
    human: n - cleared,
    hrf: n ? (n - cleared) / n : 0,
    defects,
    missed,
    batchMissRate: defects ? missed / defects : null,
    byDecision,
  };
}

/** Review hours per day = images/day x HRF x minutes per review / 60. */
export function reviewHours(perDay: number, hrf: number, minutesPerReview: number): number {
  return (perDay * hrf * minutesPerReview) / 60;
}
