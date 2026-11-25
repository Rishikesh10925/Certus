import { describe, expect, it } from "vitest";
import { decide, predictionSet, reviewHours, summarize } from "./triage";
import { parsePrometheus, readMetrics } from "./metrics";

const t = { lambda_hat: 0.8, q_hat: 0.6 };

describe("decide mirrors POST /v1/inspect", () => {
  it("auto-clears only strictly above lambda_hat", () => {
    expect(decide({ score: 0.81, p_defect: 0.19, shifted: false }, t)).toBe("AUTO_CLEAR");
    expect(decide({ score: 0.8, p_defect: 0.2, shifted: false }, t)).not.toBe("AUTO_CLEAR");
  });
  it("never clears a shifted image while the shift check escalates", () => {
    expect(decide({ score: 0.99, p_defect: 0.01, shifted: true }, t)).toBe("ESCALATE");
    expect(decide({ score: 0.99, p_defect: 0.01, shifted: true }, t, false)).toBe("AUTO_CLEAR");
  });
  it("null lambda_hat clears nothing", () => {
    expect(decide({ score: 1, p_defect: 0, shifted: false }, { lambda_hat: null, q_hat: 0.6 })).toBe(
      "ESCALATE",
    );
  });
  it("flags iff the LAC set is {defect}", () => {
    // p = 0.7: clean needs 0.7 <= 0.6 (no), defect needs 0.3 <= 0.6 (yes).
    expect(decide({ score: 0.3, p_defect: 0.7, shifted: false }, t)).toBe("FLAG");
    // p = 0.5: both in the set -> escalate.
    expect(decide({ score: 0.5, p_defect: 0.5, shifted: false }, t)).toBe("ESCALATE");
    expect(predictionSet(0.5, 0.6)).toEqual(["clean", "defect"]);
    expect(predictionSet(0.5, 0.4)).toEqual([]);
  });
});

describe("summarize", () => {
  it("counts HRF and ground-truth misses", () => {
    const items = [
      { score: 0.9, p_defect: 0.1, shifted: false, truth: true }, // cleared defect: a miss
      { score: 0.9, p_defect: 0.1, shifted: false, truth: false },
      { score: 0.3, p_defect: 0.7, shifted: false, truth: true },
      { score: 0.9, p_defect: 0.1, shifted: true, truth: false },
    ];
    const s = summarize(items, t);
    expect(s.cleared).toBe(2);
    expect(s.hrf).toBe(0.5);
    expect(s.missed).toBe(1);
    expect(s.batchMissRate).toBe(0.5);
    expect(s.byDecision).toEqual({ AUTO_CLEAR: 2, FLAG: 1, ESCALATE: 1 });
    expect(reviewHours(2000, 0.5, 1.5)).toBe(25);
  });
});

describe("metrics parser", () => {
  const text = `# HELP certus_inspections_total x
certus_inspections_total{decision="AUTO_CLEAR"} 3
certus_inspections_total{decision="FLAG"} 1
certus_inspections_total{decision="ESCALATE"} 6
certus_shift_flags_total 2
certus_feedback_missed_defects_total 1
certus_inspect_latency_seconds{quantile="0.95"} 0.098000
certus_inspect_latency_seconds_count 10
certus_calibration_info{version="v1-aeaa352b",escalate_on_shift="True"} 1
`;
  it("reads counters, quantiles and labels", () => {
    expect(parsePrometheus(text)).toHaveLength(8);
    const m = readMetrics(text);
    expect(m.total).toBe(10);
    expect(m.shiftFlags).toBe(2);
    expect(m.feedbackMissed).toBe(1);
    expect(m.latency.p95).toBeCloseTo(98);
    expect(m.calibrationVersion).toBe("v1-aeaa352b");
    expect(m.escalateOnShift).toBe(true);
  });
});
