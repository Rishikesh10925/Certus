export const pct = (x: number | null | undefined, digits = 0) =>
  x == null || !Number.isFinite(x) ? "—" : `${(100 * x).toFixed(digits)}%`;

export const fixed = (x: number | null | undefined, digits = 3) =>
  x == null || !Number.isFinite(x) ? "—" : x.toFixed(digits);

export const int = (x: number | null | undefined) =>
  x == null || !Number.isFinite(x) ? "—" : Math.round(x).toLocaleString("en-US");

export const ms = (x: number | null | undefined) =>
  x == null || !Number.isFinite(x) ? "—" : x >= 100 ? `${x.toFixed(0)} ms` : `${x.toFixed(1)} ms`;

/** Epsilon as the console shows it: 5%, 2%, 0.5%. */
export const eps = (e: number) => `${+(100 * e).toFixed(2)}%`;

export const lambda = (l: number | null) => (l == null ? "∞" : l.toFixed(3));

export const decisionLabel = {
  AUTO_CLEAR: "Auto-cleared",
  FLAG: "Flagged",
  ESCALATE: "Escalated",
} as const;

export const clusterLabel = (c: string) =>
  c === "pooled" ? "Pooled" : c.charAt(0).toUpperCase() + c.slice(1);

export const SCORE_NAMES: Record<string, string> = {
  s1: "S1 · CLIP zero-shot",
  s2: "S2 · Qwen2-VL log-prob",
  s3: "S3 · Verbalised confidence",
  s4: "S4 · OWLv2 objectness",
  s5: "S5 · MC-dropout bound",
  s6: "S6 · Feature typicality",
  s7: "S7 · Learned combination",
};

export const METHOD_NAMES: Record<string, string> = {
  in_site: "In-site (no shift)",
  vanilla: "Vanilla CRC",
  mondrian_k2: "Mondrian k=2",
  mondrian_k4: "Mondrian k=4",
  mondrian_k8: "Mondrian k=8",
  weighted: "Weighted",
  "localized_0.05": "Localized 5%",
  "localized_0.1": "Localized 10%",
  "localized_0.25": "Localized 25%",
};
