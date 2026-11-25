import type {
  AuditItem,
  ClusterCalibration,
  FeedbackResult,
  InspectResult,
  ResearchIndex,
  SampleBatch,
  ServiceInfo,
} from "./types";

/** Same origin: Vite proxies /v1 in development, FastAPI serves the build in production. */
const BASE = "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
  get unreachable() {
    return this.status === 0;
  }
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(BASE + path, init);
  } catch {
    throw new ApiError("The Certus service is not reachable.", 0);
  }
  // The Vite proxy answers 502/504 when the service is down.
  if (res.status === 502 || res.status === 504) {
    throw new ApiError("The Certus service is not reachable.", 0);
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail))
        detail = body.detail.map((d: { msg?: string }) => d.msg).join("; ");
    } catch {
      /* not JSON */
    }
    throw new ApiError(detail, res.status);
  }
  return res;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  return (await request(path, init)).json() as Promise<T>;
}

export const api = {
  info: () => json<ServiceInfo>("/v1/info"),

  calibration: (cluster: string) =>
    json<ClusterCalibration>(`/v1/calibration/${encodeURIComponent(cluster)}`),

  inspect: (image: Blob, filename: string, siteId: string, epsilon: number) => {
    const form = new FormData();
    form.append("image", image, filename);
    form.append("site_id", siteId);
    form.append("epsilon", String(epsilon));
    return json<InspectResult>("/v1/inspect", { method: "POST", body: form });
  },

  feedback: (inspectionId: number, defect: boolean, defectType: string | null) =>
    json<FeedbackResult>("/v1/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inspection_id: inspectionId, defect, defect_type: defectType }),
    }),

  /** null when nothing is waiting (the service answers 204). */
  auditNext: async (): Promise<AuditItem | null> => {
    const res = await request("/v1/audit/next");
    return res.status === 204 ? null : ((await res.json()) as AuditItem);
  },

  metrics: async () => (await request("/v1/metrics")).text(),

  samples: (siteType: string, seed: number, n: number) =>
    json<SampleBatch>(`/v1/samples?site_type=${siteType}&seed=${seed}&n=${n}`),

  sampleImageUrl: (relpath: string) => `${BASE}/v1/samples/image?relpath=${encodeURIComponent(relpath)}`,

  sampleImage: async (relpath: string) =>
    (await request(`/v1/samples/image?relpath=${encodeURIComponent(relpath)}`)).blob(),

  riskStatement: (p: {
    cluster: string;
    epsilon: number;
    decision?: string;
    shifted?: boolean;
    defect_type?: string;
  }) => {
    const q = new URLSearchParams({
      cluster: p.cluster,
      epsilon: String(p.epsilon),
      decision: p.decision ?? "AUTO_CLEAR",
      shifted: String(p.shifted ?? false),
      defect_type: p.defect_type ?? "",
    });
    return json<{ risk_statement: string }>(`/v1/risk-statement?${q}`);
  },

  researchIndex: () => json<ResearchIndex>("/v1/research"),
  research: <T>(name: string) => json<T>(`/v1/research/${name}`),
  figureUrl: (name: string) => `${BASE}/v1/research/figures/${name}`,
};
