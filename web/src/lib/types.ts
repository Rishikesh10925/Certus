// Response shapes of the Certus service (src/certus/service/app.py, web.py).
// Field names match the JSON exactly.

export type Decision = "AUTO_CLEAR" | "FLAG" | "ESCALATE";
export const DECISIONS: Decision[] = ["AUTO_CLEAR", "FLAG", "ESCALATE"];

export interface ServiceInfo {
  service: string;
  version: string;
  calibration_version: string;
  model_version: string;
  clusters: string[];
  epsilons: number[];
  escalate_on_shift: boolean;
  audit_rate: number;
  samples_available: boolean;
  sample_dataset: string | null;
  site_types: string[];
}

export interface Threshold {
  epsilon: number;
  /** null means infinity: the threshold clears nothing. */
  lambda_hat: number | null;
  q_hat: number;
}

export interface ClusterCalibration {
  cluster: string;
  calibration_version: string;
  model_version: string;
  n_cal: number;
  n_defects: number;
  guarantee: string;
  thresholds: Threshold[];
  shift_threshold: number;
}

export interface InspectResult {
  inspection_id: number;
  decision: Decision;
  defect_type: string | null;
  score: number;
  p_defect: number;
  prediction_set: ("clean" | "defect")[];
  site_id: string;
  cluster: string;
  epsilon: number;
  lambda_hat: number | null;
  shift: { flagged: boolean; nearest_similarity: number; threshold: number };
  risk_statement: string;
  model_version: string;
  calibration_version: string;
  timings_ms: Record<string, number>;
}

export interface FeedbackResult {
  recorded: boolean;
  missed_defect: boolean;
  recalibration: string;
}

export interface AuditItem {
  inspection_id: number;
  site_id: string;
  cluster: string;
  score: number;
  created_at: number;
  image_b64: string;
}

export interface SampleItem {
  relpath: string;
  site_id: string;
  defect: boolean;
  classes: string[];
}

export interface SampleBatch {
  dataset: string;
  split: string;
  site_type: string;
  seed: number;
  items: SampleItem[];
}

export interface ResearchIndex {
  files: Record<string, boolean>;
  figures: string[];
}
