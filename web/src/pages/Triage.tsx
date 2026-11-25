import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertOctagon,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Clock,
  Eye,
  EyeOff,
  ImageOff,
  RefreshCw,
  ShieldAlert,
  Users,
} from "lucide-react";
import { useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Link } from "react-router-dom";
import { api } from "@/lib/api";
import { cn } from "@/lib/cn";
import { clusterLabel, eps, fixed, lambda, pct } from "@/lib/format";
import { useCalibration, useInfo, useRiskStatement } from "@/lib/queries";
import { useSettings } from "@/lib/settings";
import { decide, predictionSet, reviewHours, summarize, thresholdFor } from "@/lib/triage";
import type { Decision, InspectResult, SampleItem } from "@/lib/types";
import { Badge, Button, Card, CardBody, CardHeader, Field, Input, PageHeader, Segmented, Skeleton, Stat } from "@/components/ui/primitives";
import { Dialog, StepSlider, Tip } from "@/components/ui/radix";
import { DECISION_EXPLAIN, DecisionBadge, FeedbackForm, KV, RiskStatement, ScoreScale, ShiftMeter } from "@/components/domain";
import { EmptyState, ErrorState } from "@/components/States";
import { ChartTooltip, useChartColors } from "@/components/charts";

/** ε every image in a batch is inspected at. Score, p(defect) and the shift
 * check do not depend on ε, so the batch is then re-partitioned locally. */
const INSPECT_EPS = 0.05;

interface BatchItem extends SampleItem {
  result: InspectResult;
  url: string;
}

async function runPool<T, R>(items: T[], limit: number, fn: (t: T, i: number) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i], i);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return out;
}

function useBatch(siteType: string, seed: number, n: number, concurrency: number, enabled: boolean) {
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const query = useQuery({
    queryKey: ["batch", siteType, seed, n],
    enabled,
    staleTime: Infinity,
    gcTime: 60 * 60 * 1000,
    retry: false, // never re-post a half-inspected batch silently
    queryFn: async () => {
      const batch = await api.samples(siteType, seed, n);
      setProgress({ done: 0, total: batch.items.length });
      let done = 0;
      return runPool(batch.items, concurrency, async (it): Promise<BatchItem> => {
        const blob = await api.sampleImage(it.relpath);
        const name = it.relpath.split("/").pop() ?? "image.jpg";
        const result = await api.inspect(blob, name, it.site_id, INSPECT_EPS);
        setProgress({ done: ++done, total: batch.items.length });
        return { ...it, result, url: api.sampleImageUrl(it.relpath) };
      });
    },
  });
  return { ...query, progress };
}

function Thumb({
  item,
  decision,
  showTruth,
  onClick,
}: {
  item: BatchItem;
  decision: Decision;
  showTruth: boolean;
  onClick: () => void;
}) {
  const shifted = item.result.shift.flagged;
  return (
    <button
      onClick={onClick}
      className={cn(
        "group relative aspect-square overflow-hidden rounded-lg bg-muted text-left ring-offset-2 ring-offset-surface transition-all duration-200 hover:-translate-y-0.5 hover:shadow-pop focus-visible:ring-2 focus-visible:ring-accent",
        showTruth && item.defect ? "ring-[3px] ring-defect" : "ring-1 ring-border",
      )}
      aria-label={`${item.relpath}, score ${item.result.score.toFixed(2)}`}
    >
      <img src={item.url} alt="" loading="lazy" className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105" />
      <div className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-1 bg-gradient-to-t from-black/75 to-transparent px-1.5 pb-1 pt-4">
        <span className="num text-[10.5px] font-medium text-white">{item.result.score.toFixed(2)}</span>
        <span className="flex items-center gap-0.5">
          {shifted && <ShieldAlert className="h-3 w-3 text-violet-300" aria-label="shift" />}
          {decision === "FLAG" && <span className="h-1.5 w-1.5 rounded-full bg-amber-400" aria-label="flagged" />}
        </span>
      </div>
    </button>
  );
}

function Grid({
  title,
  icon,
  items,
  decisions,
  showTruth,
  onOpen,
  tone,
  empty,
}: {
  title: string;
  icon: React.ReactNode;
  items: BatchItem[];
  decisions: Map<string, Decision>;
  showTruth: boolean;
  onOpen: (it: BatchItem) => void;
  tone: "clear" | "escalate";
  empty: string;
}) {
  const defects = items.filter((i) => i.defect).length;
  return (
    <Card className="flex flex-col">
      <CardHeader
        title={
          <span className="flex items-center gap-2">
            <span className={cn("grid h-6 w-6 place-items-center rounded-md", tone === "clear" ? "bg-clear/10 text-clear" : "bg-escalate/10 text-escalate")}>{icon}</span>
            {title}
            <span className="num text-subtle">{items.length}</span>
          </span>
        }
        action={showTruth && <Badge tone={defects && tone === "clear" ? "defect" : "neutral"}>{defects} defective</Badge>}
      />
      <CardBody>
        {items.length === 0 ? (
          <div className="grid h-32 place-items-center rounded-lg border border-dashed border-border text-[13px] text-subtle">{empty}</div>
        ) : (
          <div className="grid grid-cols-4 gap-2 sm:grid-cols-6 xl:grid-cols-6 2xl:grid-cols-8">
            {items.map((it) => (
              <Thumb key={it.relpath} item={it} decision={decisions.get(it.relpath)!} showTruth={showTruth} onClick={() => onOpen(it)} />
            ))}
          </div>
        )}
      </CardBody>
    </Card>
  );
}

export default function TriagePage() {
  const { settings, update } = useSettings();
  const info = useInfo();
  const qc = useQueryClient();
  const [siteType, setSiteType] = useState("wall");
  const [seed, setSeed] = useState(0);
  const [epsilon, setEpsilon] = useState(settings.epsilon);
  const [showTruth, setShowTruth] = useState(settings.showTruth);
  const [open, setOpen] = useState<BatchItem | null>(null);
  const cal = useCalibration(siteType);
  const ready = info.data?.samples_available === true;
  const batch = useBatch(siteType, seed, settings.batchSize, settings.concurrency, ready);

  const epsGrid = useMemo(() => (cal.data ? cal.data.thresholds.map((t) => t.epsilon).sort((a, b) => a - b) : []), [cal.data]);
  const eSel = epsGrid.includes(epsilon) ? epsilon : epsGrid.includes(0.05) ? 0.05 : epsGrid[0];
  const t = cal.data && eSel != null ? thresholdFor(cal.data, eSel) : null;
  const escalate = info.data?.escalate_on_shift ?? true;

  const scored = useMemo(
    () => (batch.data ?? []).map((b) => ({ ...b, score: b.result.score, p_defect: b.result.p_defect, shifted: b.result.shift.flagged, truth: b.defect })),
    [batch.data],
  );
  const decisions = useMemo(() => new Map(t ? scored.map((s) => [s.relpath, decide(s, t, escalate)] as const) : []), [scored, t, escalate]);
  const summary = t && scored.length ? summarize(scored, t, escalate) : null;
  const cleared = scored.filter((s) => decisions.get(s.relpath) === "AUTO_CLEAR");
  const human = scored.filter((s) => decisions.get(s.relpath) !== "AUTO_CLEAR");

  // The service decided each image at INSPECT_EPS; the local rule must agree there.
  const agreement = useMemo(() => {
    if (!cal.data || !scored.length || !cal.data.thresholds.some((x) => x.epsilon === INSPECT_EPS)) return null;
    const t5 = thresholdFor(cal.data, INSPECT_EPS);
    const mismatches = scored.filter((s) => decide(s, t5, escalate) !== s.result.decision).length;
    return { n: scored.length, mismatches };
  }, [cal.data, scored, escalate]);

  const curve = useMemo(
    () =>
      cal.data && scored.length
        ? [...cal.data.thresholds]
            .sort((a, b) => a.epsilon - b.epsilon)
            .map((th) => {
              const s = summarize(scored, th, escalate);
              return { eps: eps(th.epsilon), epsilon: th.epsilon, hrf: s.hrf, miss: s.batchMissRate };
            })
        : [],
    [cal.data, scored, escalate],
  );

  const risk = useRiskStatement({ cluster: siteType, epsilon: eSel ?? 0.05 });
  const colors = useChartColors();

  if (info.isError) return <ErrorState error={info.error} onRetry={() => info.refetch()} />;
  if (info.data && !ready) {
    return (
      <>
        <PageHeader title="Batch triage" />
        <Card>
          <EmptyState icon={<ImageOff className="h-5 w-5" />} title="No sample images on this server" action={<Link to="/inspect"><Button>Inspect your own image</Button></Link>}>
            The batch console draws held-out SDNET2018 test images from <code className="font-mono">data/raw/sdnet2018</code>,
            which is not present here. Set <code className="font-mono">CERTUS_SAMPLES</code> to the dataset folder and restart the
            service, or inspect single images instead.
          </EmptyState>
        </Card>
      </>
    );
  }

  return (
    <>
      <PageHeader
        eyebrow="Operate"
        title="Batch triage"
        description={`A batch of ${settings.batchSize} held-out SDNET2018 test images, each inspected once by the service. Moving ε re-partitions the batch against the λ̂ the service publishes for that ε, so it responds instantly.`}
        actions={
          <>
            <Button variant="outline" size="sm" onClick={() => setShowTruth((v) => !v)} aria-pressed={showTruth}>
              {showTruth ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />} Ground truth
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={batch.isFetching}
              onClick={() => qc.resetQueries({ queryKey: ["batch", siteType, seed, settings.batchSize] })}
            >
              <RefreshCw className="h-3.5 w-3.5" /> Re-inspect
            </Button>
          </>
        }
      />

      {/* Controls */}
      <Card className="mb-6">
        <CardBody className="grid gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
          <div>
            <div className="mb-3 flex items-baseline justify-between">
              <span className="text-[13px] font-medium">Guaranteed maximum missed-defect rate (ε)</span>
              <span className="num text-[22px] font-semibold text-accent">{eSel != null ? eps(eSel) : "—"}</span>
            </div>
            {epsGrid.length ? (
              <StepSlider values={epsGrid} value={eSel!} onChange={(v) => setEpsilon(v)} format={eps} ariaLabel="Epsilon" />
            ) : (
              <Skeleton className="h-10" />
            )}
            <div className="mt-3 text-[12px] text-subtle">
              λ̂ for {clusterLabel(siteType)} at ε = {eSel != null ? eps(eSel) : "—"}: <span className="num text-fg">{t ? lambda(t.lambda_hat) : "—"}</span>
              {t?.lambda_hat == null && t && " (clears nothing)"} · q̂ <span className="num text-fg">{t ? fixed(t.q_hat) : "—"}</span>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="col-span-2">
              <div className="mb-1.5 text-[13px] font-medium">Structure type</div>
              <Segmented
                ariaLabel="Structure type"
                value={siteType}
                onChange={(v) => setSiteType(v)}
                options={(info.data?.site_types ?? ["wall", "deck", "pavement"]).map((s) => ({ value: s, label: clusterLabel(s) }))}
              />
            </div>
            <Field label="Batch" htmlFor="seed">
              <div className="flex gap-1">
                <Button variant="outline" size="icon" aria-label="Previous batch" disabled={seed === 0} onClick={() => setSeed((s) => Math.max(0, s - 1))}>
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <Input id="seed" type="number" min={0} value={seed} onChange={(e) => setSeed(Math.max(0, Math.floor(Number(e.target.value) || 0)))} className="num text-center" />
                <Button variant="outline" size="icon" aria-label="Next batch" onClick={() => setSeed((s) => s + 1)}>
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Images/day" htmlFor="perday">
                <Input id="perday" type="number" min={1} step={100} value={settings.imagesPerDay} onChange={(e) => update({ imagesPerDay: Math.max(1, Number(e.target.value) || 1) })} className="num" />
              </Field>
              <Field label="Min/review" htmlFor="mpr">
                <Input id="mpr" type="number" min={0.1} step={0.1} value={settings.minutesPerReview} onChange={(e) => update({ minutesPerReview: Math.max(0.1, Number(e.target.value) || 0.1) })} className="num" />
              </Field>
            </div>
          </div>
        </CardBody>
      </Card>

      {batch.isError ? (
        <ErrorState error={batch.error} onRetry={() => batch.refetch()} />
      ) : batch.isPending || !summary ? (
        <Card>
          <CardBody className="py-10">
            <div className="mx-auto max-w-md text-center">
              <div className="text-[15px] font-semibold">Inspecting the batch…</div>
              <p className="mt-1 text-[13px] text-subtle">
                Each image goes through <code className="font-mono">POST /v1/inspect</code> once ({clusterLabel(siteType)}, batch {seed}).
              </p>
              <div className="mt-5 h-2 overflow-hidden rounded-full bg-muted">
                <div className="h-full rounded-full bg-accent transition-all duration-300" style={{ width: `${batch.progress.total ? (100 * batch.progress.done) / batch.progress.total : 4}%` }} />
              </div>
              <div className="num mt-2 text-[12px] text-subtle">
                {batch.progress.done} / {batch.progress.total || settings.batchSize}
              </div>
            </div>
          </CardBody>
        </Card>
      ) : (
        <>
          {/* Metrics */}
          <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat label="Sent to a human (HRF)" value={pct(summary.hrf)} icon={<Users className="h-4 w-4" />} hint={`${summary.byDecision.FLAG} flagged · ${summary.byDecision.ESCALATE} escalated`} />
            <Stat label="Auto-cleared" value={`${summary.cleared} of ${summary.n}`} tone="clear" icon={<CheckCircle2 className="h-4 w-4" />} hint={`${scored.filter((s) => s.shifted).length} held back by the shift check`} />
            <Stat
              label="Projected review hours / day"
              value={reviewHours(settings.imagesPerDay, summary.hrf, settings.minutesPerReview).toFixed(1)}
              icon={<Clock className="h-4 w-4" />}
              hint={`${settings.imagesPerDay.toLocaleString()} images × ${pct(summary.hrf)} × ${settings.minutesPerReview} min`}
            />
            <Stat
              label="Defects auto-cleared (this batch)"
              value={showTruth ? `${summary.missed} of ${summary.defects}` : "hidden"}
              tone={showTruth && summary.missed ? "defect" : undefined}
              icon={<AlertOctagon className="h-4 w-4" />}
              hint="Ground truth, shown for the demo. The guarantee is an average over many batches, so one batch can exceed ε."
            />
          </div>

          {/* Partition */}
          <div className="mb-6 grid gap-4 xl:grid-cols-2">
            <Grid title="Auto-cleared" tone="clear" icon={<CheckCircle2 className="h-3.5 w-3.5" />} items={cleared} decisions={decisions} showTruth={showTruth} onOpen={setOpen} empty="Nothing clears at this ε." />
            <Grid title="To a human" tone="escalate" icon={<Users className="h-3.5 w-3.5" />} items={human} decisions={decisions} showTruth={showTruth} onOpen={setOpen} empty="Every image clears at this ε." />
          </div>
          <div className="mb-6 flex flex-wrap items-center gap-x-5 gap-y-2 text-[12px] text-subtle">
            {showTruth && <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm ring-[3px] ring-defect" /> Defect in the ground truth</span>}
            <span className="flex items-center gap-1.5"><ShieldAlert className="h-3.5 w-3.5 text-shift" /> Failed the shift check (never cleared)</span>
            <span className="flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-flag" /> Flagged with a defect type</span>
            <span className="num">Number = score s = 1 − p(defect)</span>
          </div>

          {/* Curve + statement */}
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
            <Card>
              <CardHeader title="This batch across every ε" description="Human-review fraction and the batch's own miss rate (defects auto-cleared ÷ defects). One batch is a single draw; the guarantee bounds the average." />
              <CardBody>
                <div className="h-64">
                  <ResponsiveContainer>
                    <LineChart data={curve} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
                      <CartesianGrid stroke={colors.grid} vertical={false} />
                      <XAxis dataKey="eps" tick={{ fill: colors.axis, fontSize: 12 }} tickLine={false} axisLine={{ stroke: colors.grid }} />
                      <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v)} tick={{ fill: colors.axis, fontSize: 12 }} tickLine={false} axisLine={false} />
                      <Tooltip content={<ChartTooltip format={(v) => pct(v as number, 1)} />} />
                      <Legend iconType="plainline" wrapperStyle={{ fontSize: 12 }} />
                      {<Line name="ε (the bound)" type="linear" dataKey="epsilon" stroke={colors.axis} strokeDasharray="4 4" dot={false} strokeWidth={1.5} />}
                      <Line name="Human-review fraction" type="monotone" dataKey="hrf" stroke={colors.series[0]} strokeWidth={2.25} dot={{ r: 3 }} />
                      {showTruth && <Line name="Batch miss rate" type="monotone" dataKey="miss" stroke={colors.defect} strokeWidth={2.25} dot={{ r: 3 }} />}
                      {eSel != null && <ReferenceLine x={eps(eSel)} stroke={colors.accent} strokeOpacity={0.5} />}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardBody>
            </Card>
            <Card>
              <CardHeader title="Risk statement" />
              <CardBody className="space-y-4">
                <RiskStatement
                  text={
                    risk.data ? (
                      <>
                        {risk.data.risk_statement.replace(/^Auto-cleared\. /, "")}{" "}
                        {cal.data && `Calibration ${cal.data.calibration_version}, ${cal.data.n_cal.toLocaleString()} images.`}
                      </>
                    ) : (
                      <Skeleton className="h-16" />
                    )
                  }
                />
                {agreement && (
                  <Tip content="The service decided every image at ε = 5%. The console's local re-partition uses the same rule; this checks they agree.">
                    <div className={cn("flex items-center gap-2 text-[12px]", agreement.mismatches ? "text-escalate" : "text-subtle")}>
                      {agreement.mismatches ? <AlertOctagon className="h-3.5 w-3.5" /> : <CheckCircle2 className="h-3.5 w-3.5 text-clear" />}
                      {agreement.mismatches
                        ? `${agreement.mismatches} of ${agreement.n} local decisions disagree with the service at ε = 5%`
                        : `Local re-partition matches the service on all ${agreement.n} images at ε = 5%`}
                    </div>
                  </Tip>
                )}
              </CardBody>
            </Card>
          </div>
        </>
      )}

      {open && t && eSel != null && (
        <Dialog open onOpenChange={(o) => !o && setOpen(null)} title={open.relpath} description={`${open.site_id} · cluster ${open.result.cluster} · inspection #${open.result.inspection_id}`}>
          <ImageDetail item={open} t={t} epsilon={eSel} escalate={escalate} showTruth={showTruth} />
        </Dialog>
      )}
    </>
  );
}

function ImageDetail({
  item,
  t,
  epsilon,
  escalate,
  showTruth,
}: {
  item: BatchItem;
  t: { lambda_hat: number | null; q_hat: number };
  epsilon: number;
  escalate: boolean;
  showTruth: boolean;
}) {
  const s = { score: item.result.score, p_defect: item.result.p_defect, shifted: item.result.shift.flagged };
  const d = decide(s, t, escalate);
  const set = predictionSet(s.p_defect, t.q_hat);
  return (
    <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
      <div>
        <img src={item.url} alt={item.relpath} className="w-full rounded-xl ring-1 ring-border" />
        {showTruth && (
          <div className="mt-3 flex items-center gap-2 text-[13px]">
            <span className="text-subtle">Ground truth:</span>
            <Badge tone={item.defect ? "defect" : "clear"}>{item.defect ? `Defect${item.classes.length ? ` · ${item.classes.join(", ")}` : ""}` : "No defect"}</Badge>
          </div>
        )}
      </div>
      <div className="space-y-5">
        <div>
          <div className="flex items-center gap-2">
            <DecisionBadge decision={d} className="text-[12px]" />
            <span className="text-[12px] text-subtle">at ε = {eps(epsilon)}</span>
          </div>
          <p className="mt-2 text-[13px] text-subtle">{DECISION_EXPLAIN[d]}</p>
        </div>
        <ScoreScale score={s.score} lam={t.lambda_hat} />
        <dl className="divide-y divide-border">
          <KV k="p(defect)" v={fixed(s.p_defect)} />
          <KV k="Split-CP set (q̂)" v={`{${set.join(", ")}} · q̂ ${fixed(t.q_hat)}`} />
          <KV k="Most likely type" v={item.result.defect_type ?? "—"} />
          <KV k="Service decision at ε = 5%" v={item.result.decision} />
        </dl>
        <ShiftMeter similarity={item.result.shift.nearest_similarity} threshold={item.result.shift.threshold} flagged={item.result.shift.flagged} />
        <div>
          <div className="label mb-2">Record a human label</div>
          <FeedbackForm inspectionId={item.result.inspection_id} suggestedType={item.classes[0]} />
        </div>
      </div>
    </div>
  );
}
