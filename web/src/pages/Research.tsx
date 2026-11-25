import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, Area } from "recharts";
import { Maximize2 } from "lucide-react";
import { api } from "@/lib/api";
import { cn } from "@/lib/cn";
import { clusterLabel, eps, fixed, int, METHOD_NAMES, ms, pct, SCORE_NAMES } from "@/lib/format";
import { useResearch, useResearchIndex } from "@/lib/queries";
import { Badge, Card, CardBody, CardHeader, PageHeader, Segmented, Skeleton, Stat } from "@/components/ui/primitives";
import { Dialog, Tabs, TabsContent, TabsList } from "@/components/ui/radix";
import { ErrorState } from "@/components/States";
import { ChartTooltip, useChartColors } from "@/components/charts";

/* ---- result file shapes (only the fields used here) ---- */

interface E3Row { score: string; method: string; guarantee: string; epsilon: number; risk_mean: number; hrf_mean: number; hrf_ci_lo: number; hrf_ci_hi: number; exceed_frac: number; break_even_ratio: number | null }
interface E3 { n_seeds: number; n_cal: number; prevalence: number; pool: { n: number; n_pos: number }; rows: E3Row[] }
interface E2Row { score: string; guarantee: string; epsilon: number; risk_mean: number; risk_sd: number; hrf_mean: number; hrf_sd: number; exceed_frac: number; lam_finite_frac: number }
interface E2 { n_seeds: number; n_cal: number; auc_on_pool: Record<string, number>; model_versions: Record<string, string>; rows: E2Row[] }
interface LosoRow { score: string; level: number; site: string; method: string; n_target: number; miss_rate_mean: number; miss_rate_ratio: number; miss_exceed_frac: number; hrf_mean: number; coverage_gap: number }
interface Loso { n_seeds: number; n_cal: number; sites: Record<string, number>; rows: LosoRow[] }
interface Lat { device: string; n_requests: number; budget_ms: number; p95_within_budget: boolean; latency_ms: Record<string, { p50: number; p95: number; p99: number; mean: number }>; outcome_by_cluster?: Record<string, { n: number; n_defects: number; miss_rate: number; hrf: number; shift_flag_rate: number; decisions: Record<string, number> }>; epsilon: number }

const TABS = [
  { value: "efficiency", label: "Efficiency (E3)" },
  { value: "scores", label: "Scores (E2)" },
  { value: "shift", label: "Site shift (E4)" },
  { value: "service", label: "Service" },
  { value: "figures", label: "Figures" },
];

function Loading() {
  return <Skeleton className="h-80" />;
}

export default function ResearchPage() {
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") ?? "efficiency";
  const e2 = useResearch<E2>("e2_scores");
  const e3 = useResearch<E3>("e3_efficiency");
  const loso = useResearch<Loso>("loso_v3");
  const cuda = useResearch<Lat>("latency_cuda");

  const best = useMemo(() => {
    if (!e2.data) return null;
    const rows = e2.data.rows.filter((r) => r.guarantee === "conditional" && r.epsilon === 0.05);
    const min = rows.reduce((a, b) => (b.hrf_mean < a.hrf_mean ? b : a), rows[0]);
    const auc = Object.entries(e2.data.auc_on_pool).reduce((a, b) => (b[1] > a[1] ? b : a));
    return { min, auc };
  }, [e2.data]);
  const wall = loso.data?.rows.find((r) => r.score === "s1" && r.site === "wall" && r.level === 0.1 && r.method === "vanilla");

  return (
    <>
      <PageHeader
        eyebrow="Understand"
        title="Research results"
        description="Every number here is read from the committed results/ files: means over 100 seeds on human-labelled SDNET2018 data unless stated otherwise. Nothing is recomputed in the browser."
      />

      <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat
          label="Review needed at ε = 5%"
          value={best ? pct(best.min.hrf_mean) : "—"}
          hint={best ? `Best of 7 scores (${best.min.score.toUpperCase()}), conditional CRC` : " "}
        />
        <Stat label="Best score AUC" value={best ? fixed(best.auc[1], 2) : "—"} hint={best ? SCORE_NAMES[best.auc[0]] : " "} />
        <Stat label="Miss rate on a new site" value={wall ? `${wall.miss_rate_ratio.toFixed(1)}× ε` : "—"} tone="defect" hint="S1, held-out walls, ε = 10%, uncorrected" />
        <Stat label="Service p95 latency" value={cuda.data ? ms(cuda.data.latency_ms.client_total.p95) : "—"} hint={`Laptop GPU, ${cuda.data ? int(cuda.data.n_requests) : "…"} requests`} />
      </div>

      <Tabs value={tab} onValueChange={(v) => setParams({ tab: v }, { replace: true })}>
        <TabsList tabs={TABS} className="mb-6" />
        <TabsContent value="efficiency">{e3.isError ? <ErrorState error={e3.error} /> : e3.data ? <Efficiency data={e3.data} /> : <Loading />}</TabsContent>
        <TabsContent value="scores">{e2.isError ? <ErrorState error={e2.error} /> : e2.data ? <Scores data={e2.data} /> : <Loading />}</TabsContent>
        <TabsContent value="shift">{loso.isError ? <ErrorState error={loso.error} /> : loso.data ? <Shift data={loso.data} /> : <Loading />}</TabsContent>
        <TabsContent value="service"><Service /></TabsContent>
        <TabsContent value="figures"><Figures /></TabsContent>
      </Tabs>
    </>
  );
}

/* ---------------- E3 ---------------- */

function Efficiency({ data }: { data: E3 }) {
  const colors = useChartColors();
  const [guarantee, setGuarantee] = useState("conditional");
  const [method, setMethod] = useState("crc");
  const scores = [...new Set(data.rows.map((r) => r.score))].sort();
  const epsAll = [...new Set(data.rows.map((r) => r.epsilon))].sort((a, b) => a - b);
  const rows = data.rows.filter((r) => r.guarantee === guarantee && r.method === method);
  const chart = epsAll.map((e) => {
    const o: Record<string, unknown> = { eps: eps(e) };
    for (const s of scores) {
      const r = rows.find((x) => x.score === s && x.epsilon === e);
      if (r) {
        o[s] = r.hrf_mean;
        o[`${s}_ci`] = [r.hrf_ci_lo, r.hrf_ci_hi];
      }
    }
    return o;
  });
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Human-review fraction against the guaranteed miss rate"
          description={`Pool of ${int(data.pool.n)} calibration-split images (${pct(data.prevalence, 1)} defective), n_cal = ${int(data.n_cal)}, ${data.n_seeds} seeds. Bands are bootstrap 95% intervals.`}
          action={
            <div className="flex flex-wrap gap-2">
              <Segmented ariaLabel="Guarantee" value={guarantee} onChange={setGuarantee} options={[{ value: "conditional", label: "Conditional" }, { value: "joint", label: "Joint" }]} />
              <Segmented ariaLabel="Method" value={method} onChange={setMethod} options={[{ value: "crc", label: "CRC" }, { value: "ltt", label: "LTT" }]} />
            </div>
          }
        />
        <CardBody>
          <div className="h-80">
            <ResponsiveContainer>
              <ComposedChart data={chart} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
                <CartesianGrid stroke={colors.grid} vertical={false} />
                <XAxis dataKey="eps" tick={{ fill: colors.axis, fontSize: 12 }} tickLine={false} axisLine={{ stroke: colors.grid }} />
                <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v)} tick={{ fill: colors.axis, fontSize: 12 }} tickLine={false} axisLine={false} />
                <Tooltip content={<ChartTooltip format={(v) => (Array.isArray(v) ? "" : pct(v as number, 1))} labelFormat={(l) => `ε = ${l}`} />} />
                <Legend
                  wrapperStyle={{ fontSize: 12 }}
                  payload={scores.map((s, i) => ({ value: SCORE_NAMES[s] ?? s, type: "plainline", id: s, color: colors.series[i], payload: { strokeDasharray: "" } }))}
                />
                {scores.map((s, i) => (
                  <Area key={`${s}-ci`} dataKey={`${s}_ci`} stroke="none" fill={colors.series[i]} fillOpacity={0.18} legendType="none" name={`${s} CI`} isAnimationActive={false} activeDot={false} tooltipType="none" />
                ))}
                {scores.map((s, i) => (
                  <Line key={s} dataKey={s} name={SCORE_NAMES[s] ?? s} stroke={colors.series[i]} strokeWidth={2.25} dot={{ r: 3 }} type="monotone" />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Detail" description="Break-even: automation beats full review only while a missed defect costs less than this many reviews." />
        <CardBody className="overflow-x-auto pt-3">
          <table className="w-full min-w-[640px] text-[13px]">
            <thead>
              <tr className="border-b border-border text-left">
                {["Score", "ε", "Realised miss rate", "HRF", "95% CI", "Seeds over ε", "Break-even ratio"].map((h, i) => (
                  <th key={h} className={cn("label py-2 pr-4 font-medium", i > 1 && "text-right")}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows
                .filter((r) => r.epsilon <= 0.2)
                .sort((a, b) => a.score.localeCompare(b.score) || b.epsilon - a.epsilon)
                .map((r) => (
                  <tr key={`${r.score}-${r.epsilon}`} className="border-b border-border/60 last:border-0">
                    <td className="py-2 pr-4">{SCORE_NAMES[r.score] ?? r.score}</td>
                    <td className="num py-2 pr-4">{eps(r.epsilon)}</td>
                    <td className="num py-2 pr-4 text-right">{pct(r.risk_mean, 2)}</td>
                    <td className="num py-2 pr-4 text-right font-medium">{pct(r.hrf_mean, 1)}</td>
                    <td className="num py-2 pr-4 text-right text-subtle">{pct(r.hrf_ci_lo, 1)}–{pct(r.hrf_ci_hi, 1)}</td>
                    <td className="num py-2 pr-4 text-right">{pct(r.exceed_frac)}</td>
                    <td className="num py-2 text-right">{r.break_even_ratio == null ? "—" : `${r.break_even_ratio.toFixed(1)}×`}</td>
                  </tr>
                ))}
            </tbody>
          </table>
          <p className="mt-3 text-[12px] text-subtle">
            CRC bounds the miss rate in expectation; "seeds over ε" is the share of single calibration draws whose realised miss rate exceeded ε.
          </p>
        </CardBody>
      </Card>
    </div>
  );
}

/* ---------------- E2 ---------------- */

function Scores({ data }: { data: E2 }) {
  const colors = useChartColors();
  const [guarantee, setGuarantee] = useState("conditional");
  const epsList = [...new Set(data.rows.map((r) => r.epsilon))].sort((a, b) => b - a);
  const [e, setE] = useState(0.05);
  const rows = data.rows.filter((r) => r.guarantee === guarantee && r.epsilon === e).sort((a, b) => a.score.localeCompare(b.score));
  const chart = rows.map((r) => ({ name: r.score.toUpperCase(), hrf: r.hrf_mean, risk: r.risk_mean }));
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Seven off-the-shelf scores under the same guarantee"
          description={`Identical splits, n_cal = ${int(data.n_cal)}, ${data.n_seeds} seeds. The guarantee holds for every score; the score only sets the review cost.`}
          action={
            <div className="flex flex-wrap gap-2">
              <Segmented ariaLabel="Guarantee" value={guarantee} onChange={setGuarantee} options={[{ value: "conditional", label: "Conditional" }, { value: "joint", label: "Joint" }]} />
              <Segmented ariaLabel="Epsilon" value={e} onChange={setE} options={epsList.map((x) => ({ value: x, label: eps(x) }))} />
            </div>
          }
        />
        <CardBody>
          <div className="h-72">
            <ResponsiveContainer>
              <BarChart data={chart} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
                <CartesianGrid stroke={colors.grid} vertical={false} />
                <XAxis dataKey="name" tick={{ fill: colors.axis, fontSize: 12 }} tickLine={false} axisLine={{ stroke: colors.grid }} />
                <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v)} tick={{ fill: colors.axis, fontSize: 12 }} tickLine={false} axisLine={false} />
                <Tooltip cursor={{ fill: colors.grid, opacity: 0.4 }} content={<ChartTooltip format={(v) => pct(v as number, 1)} />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="hrf" name="Human-review fraction" fill={colors.series[0]} radius={[5, 5, 0, 0]} maxBarSize={44} />
                <Bar dataKey="risk" name="Realised miss rate" fill={colors.defect} radius={[5, 5, 0, 0]} maxBarSize={44} />
                <ReferenceLine y={e} stroke={colors.defect} strokeDasharray="4 4" label={{ value: `ε = ${eps(e)}`, fill: colors.defect, fontSize: 11, position: "insideTopRight" }} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardBody>
      </Card>
      <Card>
        <CardBody className="overflow-x-auto">
          <table className="w-full min-w-[680px] text-[13px]">
            <thead>
              <tr className="border-b border-border text-left">
                {["Score", "AUC", "Realised miss rate (± sd)", "HRF (± sd)", "Seeds over ε", "Finite λ̂"].map((h, i) => (
                  <th key={h} className={cn("label py-2 pr-4 font-medium", i > 0 && "text-right")}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.score} className="border-b border-border/60 last:border-0">
                  <td className="py-2 pr-4">
                    <div>{SCORE_NAMES[r.score] ?? r.score}</div>
                    <div className="font-mono text-[11px] text-subtle">{data.model_versions[r.score]}</div>
                  </td>
                  <td className="num py-2 pr-4 text-right">{fixed(data.auc_on_pool[r.score], 3)}</td>
                  <td className="num py-2 pr-4 text-right">{pct(r.risk_mean, 2)} <span className="text-subtle">± {pct(r.risk_sd, 2)}</span></td>
                  <td className="num py-2 pr-4 text-right font-medium">{pct(r.hrf_mean, 1)} <span className="font-normal text-subtle">± {pct(r.hrf_sd, 1)}</span></td>
                  <td className="num py-2 pr-4 text-right">{pct(r.exceed_frac)}</td>
                  <td className="num py-2 text-right">{pct(r.lam_finite_frac)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-[12px] text-subtle">
            "Finite λ̂" is the share of seeds in which the threshold clears anything at all; S3 answers only 0 or 50, so it clears nothing at small ε.
          </p>
        </CardBody>
      </Card>
    </div>
  );
}

/* ---------------- E4 LOSO ---------------- */

function Shift({ data }: { data: Loso }) {
  const colors = useChartColors();
  const [score, setScore] = useState("s1");
  const [level, setLevel] = useState(0.1);
  const [family, setFamily] = useState("main");
  const sites = ["deck", "pavement", "wall", "cbri"].filter((s) => s in data.sites);
  const methods =
    family === "main"
      ? ["in_site", "vanilla", "mondrian_k8", "weighted", "localized_0.05"]
      : ["vanilla", "mondrian_k2", "mondrian_k4", "mondrian_k8", "localized_0.05", "localized_0.1", "localized_0.25"];
  const rows = data.rows.filter((r) => r.score === score && r.level === level);
  const chart = sites.map((s) => {
    const o: Record<string, unknown> = { site: s === "cbri" ? "CBRI (gold)" : clusterLabel(s) };
    for (const m of methods) o[m] = rows.find((r) => r.site === s && r.method === m)?.miss_rate_ratio ?? null;
    return o;
  });
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Leave-one-site-out: miss rate on the held-out site, as a multiple of ε"
          description={`Calibrate on the other SDNET2018 structure types (n_cal = ${int(data.n_cal)}), deploy on the held-out one; CBRI is a target only. ${data.n_seeds} seeds. Above 1.0× the guarantee is broken.`}
          action={
            <div className="flex flex-wrap gap-2">
              <Segmented ariaLabel="Score" value={score} onChange={setScore} options={[{ value: "s1", label: "S1" }, { value: "s2", label: "S2" }]} />
              <Segmented ariaLabel="Level" value={level} onChange={setLevel} options={[{ value: 0.1, label: "ε 10%" }, { value: 0.05, label: "ε 5%" }]} />
              <Segmented ariaLabel="Methods" value={family} onChange={setFamily} options={[{ value: "main", label: "Corrections" }, { value: "sweep", label: "k sweep" }]} />
            </div>
          }
        />
        <CardBody>
          <div className="h-80">
            <ResponsiveContainer>
              <BarChart data={chart} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
                <CartesianGrid stroke={colors.grid} vertical={false} />
                <XAxis dataKey="site" tick={{ fill: colors.axis, fontSize: 12 }} tickLine={false} axisLine={{ stroke: colors.grid }} />
                <YAxis tickFormatter={(v) => `${v}×`} tick={{ fill: colors.axis, fontSize: 12 }} tickLine={false} axisLine={false} />
                <Tooltip cursor={{ fill: colors.grid, opacity: 0.4 }} content={<ChartTooltip format={(v) => `${(v as number).toFixed(2)}× ε`} />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine y={1} stroke={colors.defect} strokeDasharray="4 4" label={{ value: "ε", fill: colors.defect, fontSize: 11, position: "right" }} />
                {methods.map((m, i) => (
                  <Bar key={m} dataKey={m} name={METHOD_NAMES[m] ?? m} fill={colors.series[i]} radius={[4, 4, 0, 0]} maxBarSize={22} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Recovered validity costs review" description="Every repair raises the human-review fraction. CBRI fails safe: almost every image is sent to a human." />
        <CardBody className="overflow-x-auto pt-3">
          <table className="w-full min-w-[720px] text-[13px]">
            <thead>
              <tr className="border-b border-border text-left">
                <th className="label py-2 pr-4 font-medium">Method</th>
                {sites.map((s) => (
                  <th key={s} className="label py-2 pr-4 text-right font-medium" colSpan={1}>{s === "cbri" ? "CBRI" : clusterLabel(s)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {methods.map((m) => (
                <tr key={m} className="border-b border-border/60 last:border-0">
                  <td className="py-2 pr-4">{METHOD_NAMES[m] ?? m}</td>
                  {sites.map((s) => {
                    const r = rows.find((x) => x.site === s && x.method === m);
                    return (
                      <td key={s} className="num py-2 pr-4 text-right">
                        {r ? (
                          <>
                            <span className={cn("font-medium", r.miss_rate_ratio > 1 ? "text-defect" : "text-clear")}>{r.miss_rate_ratio.toFixed(2)}×</span>
                            <span className="ml-2 text-subtle">HRF {pct(r.hrf_mean)}</span>
                          </>
                        ) : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </CardBody>
      </Card>
    </div>
  );
}

/* ---------------- Service ---------------- */

function Service() {
  const colors = useChartColors();
  const cuda = useResearch<Lat>("latency_cuda");
  const cpu = useResearch<Lat>("latency_cpu");
  if (cuda.isError) return <ErrorState error={cuda.error} />;
  if (!cuda.data || !cpu.data) return <Loading />;
  const chart = (["p50", "p95", "p99"] as const).map((q) => ({
    q,
    gpu: cuda.data.latency_ms.client_total[q],
    cpu: cpu.data.latency_ms.client_total[q],
  }));
  const out = cuda.data.outcome_by_cluster ?? {};
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
      <Card>
        <CardHeader title="End-to-end latency" description={`Client-measured, single sequential client on the same machine. Budget ${cuda.data.budget_ms} ms at p95.`} />
        <CardBody>
          <div className="h-72">
            <ResponsiveContainer>
              <BarChart data={chart} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                <CartesianGrid stroke={colors.grid} vertical={false} />
                <XAxis dataKey="q" tick={{ fill: colors.axis, fontSize: 12 }} tickLine={false} axisLine={{ stroke: colors.grid }} />
                <YAxis tick={{ fill: colors.axis, fontSize: 12 }} tickLine={false} axisLine={false} unit=" ms" width={70} />
                <Tooltip cursor={{ fill: colors.grid, opacity: 0.4 }} content={<ChartTooltip format={(v) => ms(v as number)} />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine y={cuda.data.budget_ms} stroke={colors.defect} strokeDasharray="4 4" label={{ value: "budget", fill: colors.defect, fontSize: 11, position: "insideTopRight" }} />
                <Bar dataKey="gpu" name={`GPU (${int(cuda.data.n_requests)} req)`} fill={colors.series[0]} radius={[5, 5, 0, 0]} maxBarSize={48} />
                <Bar dataKey="cpu" name={`CPU only (${int(cpu.data.n_requests)} req)`} fill={colors.series[2]} radius={[5, 5, 0, 0]} maxBarSize={48} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 flex gap-2 text-[12px]">
            <Badge tone={cuda.data.p95_within_budget ? "clear" : "escalate"}>GPU {cuda.data.p95_within_budget ? "within" : "over"} budget</Badge>
            <Badge tone={cpu.data.p95_within_budget ? "clear" : "escalate"}>CPU {cpu.data.p95_within_budget ? "within" : "over"} budget</Badge>
          </div>
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Held-out test images through the service" description={`Decisions at ε = ${eps(cuda.data.epsilon)} on the SDNET2018 test split; "pooled" here is the 175 CBRI gold images (an unseen site).`} />
        <CardBody className="overflow-x-auto pt-3">
          <table className="w-full min-w-[560px] text-[13px]">
            <thead>
              <tr className="border-b border-border text-left">
                {["Cluster", "Images", "Defects", "Miss rate", "HRF", "Shift flags"].map((h, i) => (
                  <th key={h} className={cn("label py-2 pr-4 font-medium", i > 0 && "text-right")}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(out).map(([c, r]) => (
                <tr key={c} className="border-b border-border/60 last:border-0">
                  <td className="py-2 pr-4">{c === "pooled" ? "CBRI → pooled" : clusterLabel(c)}</td>
                  <td className="num py-2 pr-4 text-right">{int(r.n)}</td>
                  <td className="num py-2 pr-4 text-right">{int(r.n_defects)}</td>
                  <td className={cn("num py-2 pr-4 text-right font-medium", r.miss_rate > cuda.data.epsilon ? "text-defect" : "")}>{pct(r.miss_rate, 1)}</td>
                  <td className="num py-2 pr-4 text-right">{pct(r.hrf, 1)}</td>
                  <td className="num py-2 text-right">{pct(r.shift_flag_rate, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-[12px] text-subtle">One test set is one draw; the bound is on the expected miss rate, so a single cluster can land slightly above ε.</p>
        </CardBody>
      </Card>
    </div>
  );
}

/* ---------------- Figures ---------------- */

const CAPTIONS: Record<string, string> = {
  F1_system: "System overview",
  F2_validity_s1: "Split-CP validity, S1",
  F2_validity_s2: "Split-CP validity, S2",
  F3_scores: "Score comparison under CRC",
  F4_efficiency: "Efficiency curve",
  F5_loso: "Leave-one-site-out miss rate",
  F6_recovery: "Recovery by shift correction",
  F7_class_coverage: "Class-conditional coverage",
  F8_economics: "Break-even economics",
  F9_latency: "Service latency",
};

function Figures() {
  const idx = useResearchIndex();
  const [open, setOpen] = useState<string | null>(null);
  if (idx.isError) return <ErrorState error={idx.error} />;
  if (!idx.data) return <Loading />;
  return (
    <>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {idx.data.figures.map((f) => (
          <button key={f} onClick={() => setOpen(f)} className="group text-left">
            <Card className="overflow-hidden transition-all duration-200 group-hover:-translate-y-0.5 group-hover:shadow-pop">
              <div className="relative aspect-[4/3] bg-white p-3">
                <img src={api.figureUrl(f)} alt={CAPTIONS[f] ?? f} loading="lazy" className="h-full w-full object-contain" />
                <Maximize2 className="absolute right-3 top-3 h-4 w-4 text-neutral-400 opacity-0 transition-opacity group-hover:opacity-100" />
              </div>
              <div className="flex items-center justify-between border-t border-border px-4 py-3">
                <span className="text-[13px] font-medium">{CAPTIONS[f] ?? f}</span>
                <span className="font-mono text-[11px] text-subtle">{f.split("_")[0]}</span>
              </div>
            </Card>
          </button>
        ))}
      </div>
      {open && (
        <Dialog open onOpenChange={(o) => !o && setOpen(null)} title={CAPTIONS[open] ?? open} description={`figures/${open}.png · regenerated by make figures`} className="max-w-5xl">
          <div className="rounded-lg bg-white p-3">
            <img src={api.figureUrl(open)} alt={CAPTIONS[open] ?? open} className="w-full" />
          </div>
        </Dialog>
      )}
    </>
  );
}
