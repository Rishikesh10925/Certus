import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowDown, ArrowDownUp, ArrowUp, ChevronLeft, ChevronRight, CheckCircle2, ImageOff, Lock, Minus, Plus, RefreshCw, Target, AlertOctagon } from "lucide-react";
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Link } from "react-router-dom";
import { api } from "@/lib/api";
import { cn } from "@/lib/cn";
import { clusterLabel, eps, fixed, int, lambda, pct } from "@/lib/format";
import { useCalibration, useInfo, useRiskStatement } from "@/lib/queries";
import { useSettings } from "@/lib/settings";
import { decide, lambdaValue, predictionSet, reviewHours, summarize, thresholdFor, type BatchSummary } from "@/lib/triage";
import type { ClusterCalibration, Decision, InspectResult, SampleItem, Threshold } from "@/lib/types";
import { Button, Card, CardBody, CardHeader, Input, PageHeader, Segmented, Skeleton } from "@/components/ui/primitives";
import { Sheet, Switch, Tip } from "@/components/ui/radix";
import {
  DECISION_ICON,
  DECISION_META,
  DECISION_SOFT,
  DECISION_SOLID,
  DecisionLegend,
  DecisionSwatch,
  FeedbackForm,
  GuaranteeFormula,
  KV,
  ScoreScale,
  ShiftMark,
  ShiftMeter,
} from "@/components/domain";
import { EmptyState, ErrorState } from "@/components/States";
import { axisProps, ChartTooltip, useChartColors } from "@/components/charts";

/** ε every image in a batch is inspected at. Score, p(defect) and the shift
 * check do not depend on ε, so the batch is then re-partitioned locally. */
const INSPECT_EPS = 0.05;

interface BatchItem extends SampleItem {
  result: InspectResult;
  url: string;
}

type ScoredItem = BatchItem & { score: number; p_defect: number; shifted: boolean; truth: boolean };

interface Placed {
  item: ScoredItem;
  decision: Decision;
  /** Would clear on score, but the shift check held it back. */
  held: boolean;
  rank: number;
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

const reducedMotion = () => typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ======================================================================== */

export default function TriagePage() {
  const { settings, update } = useSettings();
  const info = useInfo();
  const qc = useQueryClient();
  const [siteType, setSiteType] = useState("wall");
  const [seed, setSeed] = useState(0);
  const [epsilon, setEpsilon] = useState(settings.epsilon);
  const [showTruth, setShowTruth] = useState(settings.showTruth);
  const [openKey, setOpenKey] = useState<string | null>(null);
  const cal = useCalibration(siteType);
  const ready = info.data?.samples_available === true;
  const batch = useBatch(siteType, seed, settings.batchSize, settings.concurrency, ready);

  const epsGrid = useMemo(() => (cal.data ? cal.data.thresholds.map((t) => t.epsilon).sort((a, b) => a - b) : []), [cal.data]);
  const eSel = epsGrid.includes(epsilon) ? epsilon : epsGrid.includes(0.05) ? 0.05 : epsGrid[0];
  const t = cal.data && eSel != null ? thresholdFor(cal.data, eSel) : null;
  const escalate = info.data?.escalate_on_shift ?? true;

  const scored = useMemo<ScoredItem[]>(
    () => (batch.data ?? []).map((b) => ({ ...b, score: b.result.score, p_defect: b.result.p_defect, shifted: b.result.shift.flagged, truth: b.defect })),
    [batch.data],
  );
  const summary = t && scored.length ? summarize(scored, t, escalate) : null;

  /* Order of the wall: cleared (cleanest first), then shift holds, then the rest by score. */
  const placed = useMemo<Placed[]>(() => {
    if (!t) return [];
    const lam = lambdaValue(t);
    const rows = scored.map((item) => ({ item, decision: decide(item, t, escalate), held: item.score > lam && item.shifted && escalate }));
    const byScore = (a: { item: ScoredItem }, b: { item: ScoredItem }) => b.item.score - a.item.score;
    const cleared = rows.filter((r) => r.decision === "AUTO_CLEAR").sort(byScore);
    const held = rows.filter((r) => r.held).sort(byScore);
    const rest = rows.filter((r) => r.decision !== "AUTO_CLEAR" && !r.held).sort(byScore);
    return [...cleared, ...held, ...rest].map((r, rank) => ({ ...r, rank: rank + 1 }));
  }, [scored, t, escalate]);
  const nCleared = placed.filter((p) => p.decision === "AUTO_CLEAR").length;
  const nHeld = placed.filter((p) => p.held).length;

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

  /* ---- FLIP: snapshot tile positions before ε changes, play after layout ---- */
  const wallRef = useRef<HTMLUListElement>(null);
  const flipFrom = useRef<Map<string, DOMRect> | null>(null);
  const snapshot = useCallback(() => {
    if (!wallRef.current || reducedMotion()) return;
    const m = new Map<string, DOMRect>();
    wallRef.current.querySelectorAll<HTMLElement>("[data-flip]").forEach((el) => m.set(el.dataset.flip!, el.getBoundingClientRect()));
    flipFrom.current = m;
  }, []);
  const changeEpsilon = useCallback(
    (v: number) => {
      if (v === eSel) return;
      snapshot();
      setEpsilon(v);
    },
    [eSel, snapshot],
  );
  useLayoutEffect(() => {
    const from = flipFrom.current;
    flipFrom.current = null;
    if (!from || !wallRef.current) return;
    wallRef.current.querySelectorAll<HTMLElement>("[data-flip]").forEach((el) => {
      const f = from.get(el.dataset.flip!);
      if (!f) {
        el.animate([{ opacity: 0, transform: "scale(.97)" }, { opacity: 1, transform: "none" }], { duration: 320, easing: "ease-out" });
        return;
      }
      const l = el.getBoundingClientRect();
      const dx = f.left - l.left;
      const dy = f.top - l.top;
      if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
        el.animate([{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "none" }], {
          duration: 560,
          easing: "cubic-bezier(.2,.8,.2,1)",
          delay: Math.min(220, Math.abs(dy) / 6),
        });
      }
    });
  }, [eSel]);

  /* ---- control bar height, for the sticky KPI column ---- */
  const barRef = useRef<HTMLElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const bar = barRef.current;
    if (!bar) return;
    const ro = new ResizeObserver(() => rootRef.current?.style.setProperty("--cb-h", `${bar.offsetHeight}px`));
    ro.observe(bar);
    return () => ro.disconnect();
  }, []);

  /* ---- drawer ---- */
  const openIdx = openKey ? placed.findIndex((p) => p.item.relpath === openKey) : -1;
  const opened = openIdx >= 0 ? placed[openIdx] : null;
  const step = (dir: number) => {
    const n = openIdx + dir;
    if (n >= 0 && n < placed.length) setOpenKey(placed[n].item.relpath);
  };
  // Return focus to the tile of the image last shown (Prev/Next may have moved it).
  const lastShown = useRef<string | null>(null);
  useEffect(() => {
    if (openKey) lastShown.current = openKey;
  }, [openKey]);

  if (info.isError) return <ErrorState className="mx-auto max-w-[1440px]" error={info.error} onRetry={() => info.refetch()} />;
  if (info.data && !ready) {
    return (
      <div className="mx-auto max-w-[1440px]">
        <PageHeader title="Batch triage" />
        <Card>
          <EmptyState icon={<ImageOff className="h-5 w-5" />} title="No sample images on this server" action={<Link to="/inspect"><Button>Inspect your own image</Button></Link>}>
            The batch console draws held-out SDNET2018 test images from <code className="font-mono">data/raw/sdnet2018</code>,
            which is not present here. Set <code className="font-mono">CERTUS_SAMPLES</code> to the dataset folder and restart the
            service, or inspect single images instead.
          </EmptyState>
        </Card>
      </div>
    );
  }

  const loading = batch.isPending || (!summary && !batch.isError);
  const dataset = (info.data?.sample_dataset ?? "sample").toUpperCase();

  return (
    <div ref={rootRef}>
      {/* Title row */}
      <div className="mb-4 flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0 max-w-3xl">
          <p className="mb-1 text-[12px] font-semibold uppercase tracking-[0.12em] text-accent">{clusterLabel(siteType)} cluster · live batch</p>
          <h1 className="text-[26px] font-bold leading-[1.05] tracking-[-0.035em] md:text-[32px]">Batch triage</h1>
          <p className="mt-1.5 text-[14px] text-subtle">
            {settings.batchSize} held-out {dataset} {siteType} images, batch {seed}, each inspected once by{" "}
            <code className="font-mono text-[12.5px]">POST /v1/inspect</code>. Moving ε re-partitions them against the λ̂ the service
            publishes for that ε{cal.data ? <> (calibration <span className="num">{cal.data.calibration_version}</span>)</> : null}.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
          <DecisionLegend />
          <Button
            variant="outline"
            size="sm"
            disabled={batch.isFetching}
            onClick={() => qc.resetQueries({ queryKey: ["batch", siteType, seed, settings.batchSize] })}
          >
            <RefreshCw className={cn("h-3.5 w-3.5", batch.isFetching && "animate-spin")} /> Re-inspect
          </Button>
        </div>
      </div>

      {/* Sticky control bar, full bleed */}
      <section
        ref={barRef}
        aria-label="Triage controls"
        className="relative z-30 -mx-4 border-y md:sticky md:top-0 border-border bg-surface shadow-card md:-mx-6 xl:-mx-8"
      >
        <div className="flex flex-wrap items-end gap-x-8 gap-y-3 px-4 py-3 md:px-6 xl:px-8">
          <EpsilonControl grid={epsGrid} cal={cal.data} value={eSel} onChange={changeEpsilon} t={t} />
          <div className="hidden flex-1 xl:block" />
          <ControlGroup label="Structure type" id="ctl-struct">
            <Segmented
              ariaLabel="Structure type"
              value={siteType}
              onChange={(v) => setSiteType(v)}
              options={(info.data?.site_types ?? ["wall", "deck", "pavement"]).map((s) => ({ value: s, label: clusterLabel(s) }))}
            />
          </ControlGroup>
          <ControlGroup label="Batch" id="ctl-seed">
            <div className="flex items-center gap-1">
              <Button variant="outline" size="icon" className="h-9 w-9" aria-label="Previous batch" disabled={seed === 0} onClick={() => setSeed((s) => Math.max(0, s - 1))}>
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Input
                aria-label="Batch number"
                type="number"
                min={0}
                value={seed}
                onChange={(e) => setSeed(Math.max(0, Math.floor(Number(e.target.value) || 0)))}
                className="num w-16 text-center font-semibold"
              />
              <Button variant="outline" size="icon" className="h-9 w-9" aria-label="Next batch" onClick={() => setSeed((s) => s + 1)}>
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </ControlGroup>
          <ControlGroup label="Evaluation" id="ctl-gt">
            <div className="flex h-9 items-center gap-2.5">
              <Switch id="gt" checked={showTruth} onCheckedChange={setShowTruth} />
              <label htmlFor="gt" className="cursor-pointer text-[14px] font-semibold">Show ground truth</label>
            </div>
          </ControlGroup>
        </div>
      </section>

      {batch.isError ? (
        <ErrorState className="mt-6" error={batch.error} onRetry={() => batch.refetch()} />
      ) : loading || !summary || !t ? (
        <LoadingWall done={batch.progress.done} total={batch.progress.total || settings.batchSize} site={clusterLabel(siteType)} seed={seed} n={settings.batchSize} />
      ) : (
        <>
          {/* Compact HRF strip: tablets and phones see the headline number before the wall */}
          <HrfStrip summary={summary} epsilon={eSel!} className="mt-5 lg:hidden" />

          <div className="mt-5 grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_300px] xl:grid-cols-[minmax(0,1fr)_340px]">
            <section aria-labelledby="wall-title" className="min-w-0">
              <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                <h2 id="wall-title" className="text-[15px] font-bold tracking-[-0.01em]">
                  Image wall <span className="font-medium text-subtle">· {summary.n} images</span>
                </h2>
                <span className="flex items-center gap-1.5 text-[13px] text-faint">
                  <ArrowDownUp className="h-3.5 w-3.5" aria-hidden />
                  Sorted by score, cleanest first · the line is λ̂
                </span>
              </div>
              {nCleared > 0 && (
                <p className="mb-2.5 flex items-center gap-2 text-[12px] font-bold uppercase tracking-[0.1em] text-clear after:h-px after:flex-1 after:bg-clear/35">
                  <ArrowUp className="h-3.5 w-3.5" aria-hidden /> Auto-cleared · score above <span className="normal-case">λ̂</span>
                </p>
              )}
              <ul ref={wallRef} className="grid grid-cols-[repeat(auto-fill,minmax(148px,1fr))] gap-2 sm:gap-2.5">
                {nCleared === 0 && <NothingClears key="empty" t={t} epsilon={eSel!} top={Math.max(...scored.map((s) => s.score))} />}
                {placed.slice(0, nCleared).map((p) => (
                  <Tile key={p.item.relpath} p={p} showTruth={showTruth} onOpen={() => setOpenKey(p.item.relpath)} />
                ))}
                <Divider key="divider" lam={t.lambda_hat} up={nCleared} down={summary.human} />
                {placed.slice(nCleared).map((p) => (
                  <Tile key={p.item.relpath} p={p} showTruth={showTruth} onOpen={() => setOpenKey(p.item.relpath)} />
                ))}
                {summary.human === 0 && (
                  <li className="col-span-full rounded-xl border-2 border-dashed border-border-strong bg-surface p-5 text-[14px] text-subtle">
                    Every image in this batch clears at ε = {eps(eSel!)}. None goes to a reviewer.
                  </li>
                )}
              </ul>
              <WallKey showTruth={showTruth} />
            </section>

            <aside
              aria-label="Batch metrics"
              className="grid gap-3 sm:grid-cols-2 lg:sticky lg:top-[calc(var(--cb-h,96px)+16px)] lg:max-h-[calc(100vh-var(--cb-h,96px)-32px)] lg:grid-cols-1 lg:overflow-y-auto lg:p-0.5"
            >
              <HrfCard summary={summary} epsilon={eSel!} className="hidden lg:block" />
              <div className="grid grid-cols-2 gap-3 sm:col-span-2 lg:col-span-1">
                <MiniStat label="Auto-cleared" value={summary.cleared} note={summary.cleared === 0 ? "nothing clears at this ε" : <>score above λ̂, no shift</>} tone="clear" />
                <MiniStat
                  label="Shift holds"
                  value={nHeld}
                  note={<ShiftNote held={nHeld} items={scored} cal={cal.data} />}
                  tone={nHeld ? "shift" : undefined}
                />
              </div>
              <ReviewLoad
                hrf={summary.hrf}
                perDay={settings.imagesPerDay}
                minutes={settings.minutesPerReview}
                onPerDay={(v) => update({ imagesPerDay: v })}
                onMinutes={(v) => update({ minutesPerReview: v })}
              />
              <Card className="border-accent/35 bg-accent-soft p-4 md:p-5">
                <p className="kpi-label text-accent">
                  <Target className="h-3.5 w-3.5" aria-hidden /> The guarantee
                </p>
                <div className="mt-2 text-[14.5px] leading-relaxed" aria-live="polite">
                  {risk.data ? (
                    risk.data.risk_statement.replace(/^Auto-cleared\. /, "")
                  ) : risk.isError ? (
                    <span className="text-danger">The risk statement could not be loaded.</span>
                  ) : (
                    <Skeleton className="h-14" />
                  )}
                </div>
                {cal.data && (
                  <div className="mt-3 border-t border-accent/25 pt-3">
                    <div className="text-[12px] text-subtle">{cal.data.guarantee.split(":")[0]}</div>
                    <GuaranteeFormula className="mt-0.5" />
                  </div>
                )}
                {agreement && (
                  <Tip content="The service decided every image at ε = 5%. The console's local re-partition uses the same rule; this checks they agree.">
                    <div tabIndex={0} className={cn("mt-3 flex items-start gap-2 rounded-md text-[12.5px]", agreement.mismatches ? "text-danger" : "text-subtle")}>
                      {agreement.mismatches ? <AlertOctagon className="mt-0.5 h-3.5 w-3.5 shrink-0" /> : <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-clear" />}
                      {agreement.mismatches
                        ? `${agreement.mismatches} of ${agreement.n} local decisions disagree with the service at ε = 5%`
                        : `Local re-partition matches the service on all ${agreement.n} images at ε = 5%`}
                    </div>
                  </Tip>
                )}
              </Card>
              {showTruth && <TruthCard summary={summary} epsilon={eSel!} />}
              {cal.data && (
                <p className="px-1 text-[12px] leading-relaxed text-faint sm:col-span-2 lg:col-span-1">
                  Model <code className="break-all font-mono text-[11.5px] text-subtle">{cal.data.model_version}</code>
                  <br />
                  Calibration <code className="font-mono text-[11.5px] text-subtle">{cal.data.calibration_version}</code> · n ={" "}
                  <span className="num">{int(cal.data.n_cal)}</span> ({int(cal.data.n_defects)} defects) · shift threshold{" "}
                  <span className="num">{fixed(cal.data.shift_threshold)}</span>
                </p>
              )}
            </aside>
          </div>

          <BatchCurve curve={curve} epsilon={eSel!} showTruth={showTruth} />
        </>
      )}

      {opened && t && eSel != null && (
        <Sheet
          open
          onOpenChange={(o) => {
            if (!o) {
              const key = lastShown.current;
              setOpenKey(null);
              // Radix restores focus to the tile that opened the drawer; move it to the one last shown.
              requestAnimationFrame(() => wallRef.current?.querySelector<HTMLButtonElement>(`[data-key="${CSS.escape(key ?? "")}"]`)?.focus());
            }
          }}
          title={opened.item.site_id}
          description={`inspection #${opened.item.result.inspection_id} · rank ${opened.rank} of ${placed.length}`}
          onKeyDown={(e) => {
            const tag = (e.target as HTMLElement).tagName;
            if (tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement).closest("[role=listbox]")) return;
            if (e.key === "ArrowLeft") step(-1);
            if (e.key === "ArrowRight") step(1);
          }}
          footer={
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" disabled={openIdx <= 0} onClick={() => step(-1)}>
                <ChevronLeft className="h-4 w-4" /> Previous
              </Button>
              <Button variant="outline" size="sm" disabled={openIdx >= placed.length - 1} onClick={() => step(1)}>
                Next <ChevronRight className="h-4 w-4" />
              </Button>
              <span className="ml-auto hidden items-center gap-1 text-[12px] text-faint md:flex">
                <Kbd>←</Kbd> <Kbd>→</Kbd> step · <Kbd>Esc</Kbd> close
              </span>
            </div>
          }
        >
          <ImageDetail key={opened.item.relpath} p={opened} t={t} epsilon={eSel} showTruth={showTruth} />
        </Sheet>
      )}
    </div>
  );
}

/* ======================================================================== */
/* Controls                                                                  */

function ControlGroup({ label, id, children, className }: { label: ReactNode; id: string; children: ReactNode; className?: string }) {
  return (
    <div role="group" aria-labelledby={id} className={cn("flex min-w-0 flex-col gap-1.5", className)}>
      <span id={id} className="text-[12px] font-semibold tracking-[0.01em] text-subtle">
        {label}
      </span>
      {children}
    </div>
  );
}

function EpsilonControl({
  grid,
  cal,
  value,
  onChange,
  t,
}: {
  grid: number[];
  cal: ClusterCalibration | undefined;
  value: number | undefined;
  onChange: (v: number) => void;
  t: Threshold | null;
}) {
  const labelId = useId();
  const groupRef = useRef<HTMLDivElement>(null);
  const i = value != null ? grid.indexOf(value) : -1;
  const go = (n: number, focus = false) => {
    const v = grid[Math.max(0, Math.min(grid.length - 1, n))];
    if (v == null) return;
    onChange(v);
    if (focus) requestAnimationFrame(() => groupRef.current?.querySelector<HTMLButtonElement>(`[data-eps="${v}"]`)?.focus());
  };
  const onKey = (e: React.KeyboardEvent) => {
    let n: number | null = null;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") n = i + 1;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") n = i - 1;
    if (e.key === "Home") n = 0;
    if (e.key === "End") n = grid.length - 1;
    if (n == null) return;
    e.preventDefault();
    go(n, true);
  };
  return (
    <div className="flex w-full min-w-0 flex-col gap-1.5 sm:w-auto">
      <span id={labelId} className="text-[12px] font-semibold tracking-[0.01em] text-subtle">
        Guaranteed max missed-defect rate (ε)
      </span>
      <div className="flex flex-wrap items-stretch gap-x-3 gap-y-2">
        {grid.length && cal ? (
          <div className="flex min-w-0 basis-full items-stretch gap-[3px] rounded-[10px] border border-border-strong bg-muted p-[3px] sm:basis-auto">
            <button
              type="button"
              className="hidden w-[34px] place-items-center rounded-md text-subtle hover:bg-sunken hover:text-fg disabled:cursor-not-allowed disabled:opacity-35 sm:grid"
              aria-label="Stricter: lower ε"
              disabled={i <= 0}
              onClick={() => go(i - 1)}
            >
              <Minus className="h-4 w-4" />
            </button>
            <div ref={groupRef} role="radiogroup" aria-labelledby={labelId} className="flex min-w-0 flex-1 gap-[3px]" onKeyDown={onKey}>
              {grid.map((e) => {
                const row = cal.thresholds.find((x) => x.epsilon === e)!;
                const on = e === value;
                return (
                  <button
                    key={e}
                    type="button"
                    role="radio"
                    aria-checked={on}
                    tabIndex={on ? 0 : -1}
                    data-eps={e}
                    aria-label={`ε ${eps(e)}, λ̂ ${row.lambda_hat == null ? "infinity" : fixed(row.lambda_hat)}`}
                    onClick={() => onChange(e)}
                    className={cn(
                      "flex min-w-0 flex-1 flex-col items-center rounded-md px-1 pb-1 pt-1.5 transition-[background-color,color,box-shadow] duration-200 focus-visible:ring-offset-0 sm:w-[70px] sm:flex-none",
                      on ? "bg-accent text-accent-ink shadow-[0_4px_14px_-6px_rgb(var(--accent))]" : "text-subtle hover:bg-sunken",
                    )}
                  >
                    <span className={cn("num text-[17px] font-bold leading-[1.1] sm:text-[19px]", on ? "text-accent-ink" : "text-fg")}>{eps(e)}</span>
                    <span className={cn("num text-[11px]", on ? "text-accent-ink" : "text-faint")}>
                      {row.lambda_hat == null ? "∞" : row.lambda_hat.toFixed(3).replace(/^0/, "")}
                    </span>
                  </button>
                );
              })}
            </div>
            <button
              type="button"
              className="hidden w-[34px] place-items-center rounded-md text-subtle hover:bg-sunken hover:text-fg disabled:cursor-not-allowed disabled:opacity-35 sm:grid"
              aria-label="More lenient: higher ε"
              disabled={i >= grid.length - 1}
              onClick={() => go(i + 1)}
            >
              <Plus className="h-4 w-4" />
            </button>
          </div>
        ) : (
          <Skeleton className="h-[52px] w-full sm:w-[440px]" />
        )}
        <div className="flex gap-4 rounded-r-md border-l-[3px] border-accent bg-accent-soft px-3.5 py-1.5" aria-live="polite">
          <div className="flex flex-col">
            <span className="text-[11.5px] text-subtle">
              <span className="sym text-[14px]">λ̂</span> auto-clear above
            </span>
            <span className="num text-[17px] font-semibold leading-tight">{t ? lambda(t.lambda_hat) : "—"}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[11.5px] text-subtle">
              <span className="sym text-[14px]">q̂</span> set cutoff
            </span>
            <span className="num text-[17px] font-semibold leading-tight">{t ? fixed(t.q_hat) : "—"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded border border-b-2 border-border-strong bg-muted px-1.5 font-mono text-[11px] text-subtle">{children}</kbd>;
}

/* ======================================================================== */
/* Wall                                                                      */

const RAIL: Record<Decision, string> = { AUTO_CLEAR: "bg-clear", FLAG: "rail-flag", ESCALATE: "rail-escalate" };

function Tile({ p, showTruth, onOpen }: { p: Placed; showTruth: boolean; onOpen: () => void }) {
  const { item, decision, held, rank } = p;
  const Icon = DECISION_ICON[decision];
  const missed = showTruth && item.defect && decision === "AUTO_CLEAR";
  const meta = DECISION_META[decision];
  const label = [
    `Rank ${rank}`,
    item.site_id,
    `score ${item.score.toFixed(3)}`,
    meta.name + (meta.qualifier ? `, ${meta.qualifier}` : ""),
    held ? "held by the shift check" : null,
    showTruth ? (item.defect ? (missed ? "ground truth: defect, auto-cleared (missed)" : "ground truth: defect") : "ground truth: clean") : null,
  ]
    .filter(Boolean)
    .join(", ");
  return (
    <li data-flip={item.relpath} className="min-w-0">
      <button
        type="button"
        data-key={item.relpath}
        onClick={onOpen}
        aria-label={`${label}. Open details.`}
        className={cn(
          "group relative block w-full overflow-hidden rounded-[10px] bg-surface text-left shadow-card transition-[transform,box-shadow] duration-200 [transition-timing-function:cubic-bezier(.2,.8,.2,1)] hover:-translate-y-[3px] hover:shadow-pop active:translate-y-[-1px] active:scale-[.99]",
          missed ? "ring-[3px] ring-flag" : "ring-1 ring-border",
        )}
      >
        <span className={cn("absolute inset-y-0 left-0 z-[2] w-1", RAIL[decision])} aria-hidden />
        <span className="relative block aspect-square overflow-hidden bg-sunken">
          <img
            src={item.url}
            alt=""
            loading="lazy"
            width={256}
            height={256}
            className={cn(
              "h-full w-full object-cover transition-transform duration-500 [transition-timing-function:cubic-bezier(.2,.8,.2,1)] group-hover:scale-[1.04]",
              decision !== "AUTO_CLEAR" && "saturate-[.92]",
            )}
          />
          <span className={cn("absolute left-2 top-2 inline-flex items-center gap-1 rounded-md py-[3px] pl-1.5 pr-2 text-[11.5px] font-bold tracking-[0.02em] shadow-sm", DECISION_SOLID[decision])}>
            <Icon className="h-[13px] w-[13px]" />
            {meta.short}
          </span>
          <span className="num absolute right-2 top-2 rounded-[5px] bg-[rgb(4_8_20/0.6)] px-1.5 py-0.5 text-[11px] font-semibold text-white">
            #{String(rank).padStart(2, "0")}
          </span>
          {held && (
            <span className="absolute left-2 top-[34px] inline-flex items-center gap-1 rounded-[5px] bg-shift px-1.5 py-0.5 text-[10.5px] font-bold text-shift-ink">
              <ShiftMark className="h-3 w-3" /> Shift hold
            </span>
          )}
          {showTruth && item.defect && (
            <span
              className={cn(
                "absolute bottom-2 left-2 inline-flex items-center gap-1 rounded-[5px] py-0.5 pl-1 pr-1.5 text-[10.5px] font-bold shadow-[0_1px_3px_rgb(0_0_0/0.3)]",
                missed ? "bg-flag text-flag-ink" : "bg-white text-[#0b1324]",
              )}
            >
              <Target className={cn("h-3 w-3", !missed && "text-[#be123c]")} aria-hidden />
              {missed ? "Missed" : "Defect"}
            </span>
          )}
        </span>
        <span className="flex items-baseline justify-between gap-2 border-t border-border py-1.5 pl-3 pr-2.5">
          <span className="num text-[15px] font-bold leading-none">
            <span className="sr-only">score </span>
            {item.score.toFixed(3)}
          </span>
          <span className="num truncate text-[11px] text-faint">{item.site_id}</span>
        </span>
      </button>
    </li>
  );
}

function Divider({ lam, up, down }: { lam: number | null; up: number; down: number }) {
  return (
    <li
      data-flip="divider"
      className="relative col-span-full my-0.5 py-3.5"
      aria-label={`Threshold λ̂ ${lambda(lam)}: ${up} auto-cleared above, ${down} to human review below`}
    >
      <div
        className="absolute inset-x-1 top-1/2 h-1 -translate-y-1/2 rounded-sm bg-accent shadow-[0_0_0_4px_rgb(var(--accent)/0.16)] before:absolute before:-left-1 before:top-1/2 before:h-3 before:w-3 before:-translate-y-1/2 before:rounded-full before:bg-accent after:absolute after:-right-1 after:top-1/2 after:h-3 after:w-3 after:-translate-y-1/2 after:rounded-full after:bg-accent"
        aria-hidden
      />
      <div className="relative flex justify-center" aria-hidden>
        <div className="flex max-w-full items-stretch overflow-hidden rounded-[10px] border-2 border-accent bg-surface shadow-pop">
          <span className="flex items-center gap-1.5 whitespace-nowrap px-2.5 py-1.5 text-[12px] font-semibold text-clear sm:px-3.5 sm:text-[13px]">
            <ArrowUp className="h-3.5 w-3.5" />
            <b className="num text-[15px]">{up}</b>
            <span className="hidden sm:inline">auto-cleared</span>
            <span className="sm:hidden">clear</span>
          </span>
          <span className="num flex items-center gap-1 whitespace-nowrap bg-accent px-2.5 py-1.5 text-[13px] font-bold text-accent-ink sm:px-3.5 sm:text-[14px]">
            <span className="sym text-[17px]">λ̂</span> = {lambda(lam)}
          </span>
          <span className="flex items-center gap-1.5 whitespace-nowrap px-2.5 py-1.5 text-[12px] font-semibold sm:px-3.5 sm:text-[13px]">
            <ArrowDown className="h-3.5 w-3.5" />
            <b className="num text-[15px]">{down}</b>
            <span className="hidden sm:inline">to a human</span>
            <span className="sm:hidden">review</span>
          </span>
        </div>
      </div>
    </li>
  );
}

function NothingClears({ t, epsilon, top }: { t: Threshold; epsilon: number; top: number }) {
  return (
    <li data-flip="empty" className="col-span-full grid grid-cols-[auto_1fr] items-start gap-4 rounded-xl border-2 border-dashed border-border-strong bg-surface p-5 md:p-6">
      <span className="grid h-11 w-11 place-items-center rounded-xl bg-accent-soft text-accent">
        <Lock className="h-6 w-6" aria-hidden />
      </span>
      <div>
        <h3 className="text-[18px] font-bold tracking-[-0.02em]">Nothing auto-clears at ε = {eps(epsilon)}</h3>
        <p className="mt-1 max-w-[62ch] text-[14px] leading-relaxed text-subtle">
          {t.lambda_hat == null ? (
            <>The calibration set cannot certify any threshold this strict, so λ̂ = ∞ and every image goes to a reviewer. </>
          ) : (
            <>
              The threshold λ̂ = <span className="num text-fg">{lambda(t.lambda_hat)}</span> sits above the cleanest image in this batch (score{" "}
              <span className="num text-fg">{top.toFixed(3)}</span>), so every image goes to a reviewer.{" "}
            </>
          )}
          That is the guarantee doing its job: a tighter ε buys certainty with human time. Loosen ε to let the cleanest images through.
        </p>
      </div>
    </li>
  );
}

function WallKey({ showTruth }: { showTruth: boolean }) {
  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-[12.5px] text-subtle">
      <span className="flex items-center gap-1.5">
        <span className="h-3.5 w-1 rounded-sm bg-clear" aria-hidden />
        <span className="rail-flag h-3.5 w-1 rounded-sm" aria-hidden />
        <span className="rail-escalate h-3.5 w-1 rounded-sm" aria-hidden />
        Edge stripe: solid clear, dashed flag, dotted escalate
      </span>
      <span className="flex items-center gap-1.5">
        <span className="inline-flex items-center gap-1 rounded-[5px] bg-shift px-1.5 py-0.5 text-[10.5px] font-bold text-shift-ink">
          <ShiftMark className="h-3 w-3" /> Shift hold
        </span>
        would clear, but failed the shift check
      </span>
      {showTruth && (
        <span className="flex items-center gap-1.5">
          <span className="inline-flex items-center gap-1 rounded-[5px] bg-flag px-1.5 py-0.5 text-[10.5px] font-bold text-flag-ink">
            <Target className="h-3 w-3" aria-hidden /> Missed
          </span>
          a real defect that was auto-cleared
        </span>
      )}
      <span className="num">score s = 1 − p(defect)</span>
    </div>
  );
}

function LoadingWall({ done, total, site, seed, n }: { done: number; total: number; site: string; seed: number; n: number }) {
  return (
    <div className="mt-5 grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_300px] xl:grid-cols-[minmax(0,1fr)_340px]" aria-busy="true">
      <div className="min-w-0">
        <Card className="mb-4 p-5">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <div className="text-[15px] font-bold">Inspecting the batch…</div>
            <div className="num text-[13px] text-subtle" aria-live="polite">
              {done} / {total}
            </div>
          </div>
          <p className="mt-1 text-[13px] text-subtle">
            Each image goes through <code className="font-mono">POST /v1/inspect</code> once ({site}, batch {seed}). This takes a few seconds.
          </p>
          <div
            className="mt-4 h-2 overflow-hidden rounded-full bg-sunken"
            role="progressbar"
            aria-label="Images inspected"
            aria-valuemin={0}
            aria-valuemax={total}
            aria-valuenow={done}
          >
            <div className="h-full rounded-full bg-accent transition-[width] duration-300" style={{ width: `${total ? Math.max(3, (100 * done) / total) : 3}%` }} />
          </div>
        </Card>
        <div className="grid grid-cols-[repeat(auto-fill,minmax(148px,1fr))] gap-2 sm:gap-2.5">
          {Array.from({ length: Math.min(n, 18) }, (_, i) => (
            <Skeleton key={i} className="aspect-[1/1.18] rounded-[10px]" />
          ))}
        </div>
      </div>
      <div className="hidden gap-3 lg:grid">
        <Skeleton className="h-64 rounded-xl" />
        <Skeleton className="h-28 rounded-xl" />
        <Skeleton className="h-44 rounded-xl" />
      </div>
    </div>
  );
}

/* ======================================================================== */
/* KPIs                                                                      */

function StackBar({ summary, className }: { summary: BatchSummary; className?: string }) {
  const { AUTO_CLEAR: c, FLAG: f, ESCALATE: e } = summary.byDecision;
  return (
    <div className={cn("flex h-3.5 gap-0.5 overflow-hidden rounded bg-sunken", className)} aria-hidden>
      <i className="block min-w-0 basis-0 bg-clear transition-[flex-grow] duration-500" style={{ flexGrow: c }} />
      <i className="hatch-flag block min-w-0 basis-0 transition-[flex-grow] duration-500" style={{ flexGrow: f }} />
      <i className="block min-w-0 basis-0 bg-escalate transition-[flex-grow] duration-500" style={{ flexGrow: e }} />
    </div>
  );
}

function HrfCard({ summary, epsilon, className }: { summary: BatchSummary; epsilon: number; className?: string }) {
  const rows: { d: Decision; n: number }[] = [
    { d: "AUTO_CLEAR", n: summary.byDecision.AUTO_CLEAR },
    { d: "FLAG", n: summary.byDecision.FLAG },
    { d: "ESCALATE", n: summary.byDecision.ESCALATE },
  ];
  return (
    <Card className={cn("p-5", className)}>
      <p className="kpi-label">Human-review fraction (HRF)</p>
      <div className="mb-0.5 mt-1.5 flex items-baseline gap-1" aria-live="polite">
        <span className="num text-[64px] font-bold leading-[.95] tracking-[-0.05em]">{(100 * summary.hrf).toFixed(0)}</span>
        <span className="num text-[26px] font-semibold text-subtle">%</span>
      </div>
      <p className="text-[13.5px] text-subtle">
        <b className="num text-fg">{summary.human}</b> of <b className="num text-fg">{summary.n}</b> images go to a person at ε = {eps(epsilon)}
      </p>
      <StackBar summary={summary} className="my-4" />
      <ul className="grid gap-1.5">
        {rows.map(({ d, n }) => (
          <li key={d} className="grid grid-cols-[22px_1fr_auto_auto] items-center gap-2 text-[13.5px]">
            <DecisionSwatch decision={d} />
            <span>
              {DECISION_META[d].name} {DECISION_META[d].qualifier && <small className="text-faint">{DECISION_META[d].qualifier}</small>}
            </span>
            <span className="num min-w-[2ch] text-right text-[15px] font-bold">{n}</span>
            <span className="num min-w-[5ch] text-right text-[12.5px] text-faint">{pct(n / summary.n, 1)}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

/** About 80px: the headline number and the split, shown above the wall below lg. */
function HrfStrip({ summary, epsilon, className }: { summary: BatchSummary; epsilon: number; className?: string }) {
  return (
    <Card className={cn("flex items-center gap-4 px-4 py-3", className)}>
      <div className="shrink-0">
        <p className="kpi-label text-[10.5px]">HRF</p>
        <div className="flex items-baseline gap-0.5">
          <span className="num text-[36px] font-bold leading-none tracking-[-0.04em]">{(100 * summary.hrf).toFixed(0)}</span>
          <span className="num text-[16px] font-semibold text-subtle">%</span>
        </div>
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-[12.5px] text-subtle">
          <b className="num text-fg">{summary.human}</b> of <b className="num text-fg">{summary.n}</b> to a person at ε = {eps(epsilon)}
        </p>
        <StackBar summary={summary} className="mt-1.5 h-3" />
        <div className="num mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[12px]">
          {(["AUTO_CLEAR", "FLAG", "ESCALATE"] as const).map((d) => (
            <span key={d} className="flex items-center gap-1">
              <DecisionSwatch decision={d} className="h-3.5 w-3.5 rounded-[4px] [&>svg]:h-2.5 [&>svg]:w-2.5" />
              <b>{summary.byDecision[d]}</b> <span className="font-sans text-faint">{DECISION_META[d].short.toLowerCase()}</span>
            </span>
          ))}
        </div>
      </div>
    </Card>
  );
}

function MiniStat({ label, value, note, tone }: { label: string; value: ReactNode; note: ReactNode; tone?: "clear" | "shift" }) {
  return (
    <Card className="p-4">
      <p className="kpi-label">{label}</p>
      <div className={cn("num mb-1 mt-2 text-[34px] font-bold leading-none tracking-[-0.03em]", tone === "clear" && "text-clear", tone === "shift" && "text-shift")}>
        {value}
      </div>
      <p className="text-[12.5px] leading-snug text-subtle">{note}</p>
    </Card>
  );
}

function ShiftNote({ held, items, cal }: { held: number; items: ScoredItem[]; cal: ClusterCalibration | undefined }) {
  if (held > 0) return <>would clear, but look unlike the calibration data</>;
  const flagged = items.filter((i) => i.shifted).length;
  if (flagged > 0) return <>{flagged} shifted, none above λ̂</>;
  const minSim = Math.min(...items.map((i) => i.result.shift.nearest_similarity));
  return (
    <>
      all in-distribution (min sim <span className="num">{fixed(minSim)}</span>
      {cal && <> ≥ <span className="num">{fixed(cal.shift_threshold)}</span></>})
    </>
  );
}

/** A number input that keeps what the user typed and only commits valid values. */
function NumberField({
  id,
  label,
  value,
  onCommit,
  min,
  step,
  integer,
  invalidText,
  onValidity,
}: {
  id: string;
  label: string;
  value: number;
  onCommit: (v: number) => void;
  onValidity: (valid: boolean) => void;
  min: number;
  step: number;
  integer?: boolean;
  invalidText: string;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText((cur) => (Number(cur) === value ? cur : String(value))), [value]);
  const n = Number(text);
  const valid = text.trim() !== "" && Number.isFinite(n) && n >= min && (!integer || Number.isInteger(n));
  useEffect(() => onValidity(valid), [valid, onValidity]);
  return (
    <div className="min-w-0">
      <label htmlFor={id} className="mb-1 block text-[12px] font-semibold text-subtle">
        {label}
      </label>
      <Input
        id={id}
        type="number"
        inputMode={integer ? "numeric" : "decimal"}
        min={min}
        step={step}
        value={text}
        aria-invalid={!valid}
        aria-describedby={!valid ? `${id}-err` : undefined}
        onChange={(e) => {
          setText(e.target.value);
          const v = Number(e.target.value);
          if (e.target.value.trim() !== "" && Number.isFinite(v) && v >= min && (!integer || Number.isInteger(v))) onCommit(v);
        }}
        className="num text-[15px] font-semibold"
      />
      {!valid && (
        <p id={`${id}-err`} className="mt-1 text-[11.5px] text-danger" role="alert">
          {invalidText}
        </p>
      )}
    </div>
  );
}

function ReviewLoad({
  hrf,
  perDay,
  minutes,
  onPerDay,
  onMinutes,
}: {
  hrf: number;
  perDay: number;
  minutes: number;
  onPerDay: (v: number) => void;
  onMinutes: (v: number) => void;
}) {
  const h = reviewHours(perDay, hrf, minutes);
  const [bad, setBad] = useState({ perDay: false, minutes: false });
  const perDayValid = useCallback((v: boolean) => setBad((b) => (b.perDay === !v ? b : { ...b, perDay: !v })), []);
  const minutesValid = useCallback((v: boolean) => setBad((b) => (b.minutes === !v ? b : { ...b, minutes: !v })), []);
  const invalid = bad.perDay || bad.minutes;
  return (
    <Card className="p-4 md:p-5">
      <p className="kpi-label">Projected review load</p>
      <div className="my-3 grid grid-cols-2 gap-3">
        <NumberField id="perday" label="Images / day" value={perDay} onCommit={onPerDay} min={1} step={100} integer invalidText="A whole number, 1 or more." onValidity={perDayValid} />
        <NumberField id="mpr" label="Min / review" value={minutes} onCommit={onMinutes} min={0.1} step={0.1} invalidText="0.1 minutes or more." onValidity={minutesValid} />
      </div>
      <div className="flex items-baseline gap-1.5" aria-live="polite">
        <span className="num text-[40px] font-bold leading-none tracking-[-0.04em]">{invalid ? "—" : h >= 100 ? h.toFixed(0) : h.toFixed(1)}</span>
        <span className="text-[14px] font-semibold text-subtle">reviewer-hours / day</span>
      </div>
      <p className="num mt-1.5 text-[12px] leading-snug text-faint">
        {invalid ? "Enter valid numbers to project the load." : <>{int(perDay)} × {pct(hrf, 1)} × {minutes} min ÷ 60 ≈ {(h / 8).toFixed(1)} reviewer-shifts (8 h)</>}
      </p>
    </Card>
  );
}

function TruthCard({ summary, epsilon }: { summary: BatchSummary; epsilon: number }) {
  const k = summary.missed;
  const D = summary.defects;
  const rate = summary.batchMissRate;
  return (
    <Card className="p-4 sm:col-span-2 lg:col-span-1 md:p-5">
      <p className="kpi-label">Defects auto-cleared</p>
      <div className={cn("num mb-0.5 mt-2 text-[34px] font-bold tracking-[-0.03em]", k > 0 && "text-flag")}>
        {k} <span className="text-[22px] text-faint">of {D}</span>
      </div>
      <p className="text-[13px] text-subtle">
        {D === 0 ? (
          "This batch has no real defects in its ground truth."
        ) : (
          <>
            true defects in this batch were auto-cleared (<span className="num">{pct(rate, 1)}</span> vs ε = {eps(epsilon)}).{" "}
            {k === 0 ? "Every real defect reached a reviewer." : rate! > epsilon ? "This batch landed above ε." : "Within ε for this batch."}
          </>
        )}
      </p>
      <p className="mt-2.5 border-t border-dashed border-border-strong pt-2 text-[12.5px] text-subtle">
        The guarantee is an <b className="text-fg">average over many batches</b>. A single batch of {summary.n} can land above ε by chance without the method being wrong.
      </p>
    </Card>
  );
}

function BatchCurve({ curve, epsilon, showTruth }: { curve: { eps: string; epsilon: number; hrf: number; miss: number | null }[]; epsilon: number; showTruth: boolean }) {
  const colors = useChartColors();
  return (
    <Card className="mt-6">
      <CardHeader
        title="This batch across every ε"
        description="Human-review fraction and the batch's own miss rate (defects auto-cleared ÷ defects). One batch is a single draw; the guarantee bounds the average."
      />
      <CardBody>
        <div className="h-64">
          <ResponsiveContainer>
            <LineChart data={curve} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
              <CartesianGrid stroke={colors.grid} vertical={false} />
              <XAxis dataKey="eps" {...axisProps(colors)} axisLine={{ stroke: colors.grid }} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v)} {...axisProps(colors)} axisLine={false} />
              <Tooltip content={<ChartTooltip format={(v) => pct(v as number, 1)} labelFormat={(l) => `ε = ${l}`} />} />
              <Legend iconType="plainline" wrapperStyle={{ fontSize: 12 }} />
              <Line name="ε (the bound)" type="linear" dataKey="epsilon" stroke={colors.axis} strokeDasharray="4 4" dot={false} strokeWidth={1.5} />
              <Line name="Human-review fraction" type="monotone" dataKey="hrf" stroke={colors.accent} strokeWidth={2.5} dot={{ r: 3.5, fill: colors.accent }} />
              {showTruth && <Line name="Batch miss rate" type="monotone" dataKey="miss" stroke={colors.flag} strokeWidth={2.25} dot={{ r: 3, fill: colors.flag }} />}
              <ReferenceLine x={eps(epsilon)} stroke={colors.accent} strokeOpacity={0.45} strokeWidth={6} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardBody>
    </Card>
  );
}

/* ======================================================================== */
/* Drawer                                                                    */

function ImageDetail({ p, t, epsilon, showTruth }: { p: Placed; t: Threshold; epsilon: number; showTruth: boolean }) {
  const { item, decision: d, held } = p;
  const set = predictionSet(item.p_defect, t.q_hat);
  const inClean = set.includes("clean");
  const inDefect = set.includes("defect");
  const lam = t.lambda_hat;
  const Icon = DECISION_ICON[d];
  const meta = DECISION_META[d];
  const s = item.score.toFixed(3);
  const why =
    d === "AUTO_CLEAR"
      ? `Score ${s} > λ̂ ${lambda(lam)} and the image looks like the calibration data.`
      : held
        ? `Score ${s} clears λ̂ ${lambda(lam)}, but the shift check held it back for a human.`
        : d === "FLAG"
          ? `Score ${s} ≤ λ̂ ${lambda(lam)}; the prediction set is {defect} only, so the reviewer gets the defect type pre-filled.`
          : `Score ${s} ≤ λ̂ ${lambda(lam)}; the prediction set is ${
              inClean && inDefect ? "{clean, defect}: the model cannot decide." : inClean ? "{clean}, but not confident enough to clear." : "empty: neither label is certain."
            }`;
  const missed = item.defect && d === "AUTO_CLEAR";
  return (
    <div className="space-y-5 p-5">
      <div className="relative aspect-square overflow-hidden rounded-xl bg-sunken">
        <img src={item.url} alt={`${clusterLabel(item.result.cluster)} inspection photo, site ${item.site_id}`} className="h-full w-full object-cover" />
      </div>

      <div className={cn("grid grid-cols-[auto_1fr] items-center gap-3 rounded-[10px] p-4", DECISION_SOFT[d])}>
        <span className={cn("grid h-11 w-11 place-items-center rounded-xl", DECISION_SOLID[d])}>
          <Icon className="h-6 w-6" />
        </span>
        <div>
          <h3 className="text-[20px] font-bold tracking-[-0.02em]">
            {meta.name}
            {meta.qualifier && <span className="font-medium text-subtle"> · {meta.qualifier}</span>}
          </h3>
          <p className="mt-0.5 text-[13.5px] text-subtle">{why}</p>
        </div>
      </div>

      <section>
        <h4 className="mb-2.5 flex flex-wrap justify-between gap-x-3 text-[12px] font-bold uppercase tracking-[0.08em] text-subtle">
          Score vs threshold
          <span className="num normal-case tracking-normal text-fg">at ε = {eps(epsilon)}</span>
        </h4>
        <ScoreScale score={item.score} lam={lam} />
      </section>

      <dl className="grid grid-cols-2 gap-3">
        {[
          ["p(defect)", fixed(item.p_defect)],
          ["Score = 1 − p", fixed(item.score)],
          ["Site", item.site_id],
        ].map(([k, v]) => (
          <div key={k} className="min-w-0 rounded-[10px] border border-border bg-muted px-3 py-2.5">
            <dt className="text-[12px] text-subtle">{k}</dt>
            <dd className="num mt-0.5 truncate text-[18px] font-bold">{v}</dd>
          </div>
        ))}
        <div className="min-w-0 rounded-[10px] border border-border bg-muted px-3 py-2.5">
          <dt className="text-[12px] text-subtle">Ground truth</dt>
          {showTruth ? (
            <dd className={cn("mt-0.5 text-[15px] font-bold leading-snug", item.defect ? "text-defect" : "text-clear")}>
              {item.defect ? `Defect${item.classes.length ? ` · ${item.classes.join(", ")}` : ""}` : "No defect"}
              {missed && " (missed)"}
            </dd>
          ) : (
            <dd className="mt-0.5 text-[13px] text-faint">Hidden · turn on “Show ground truth”</dd>
          )}
        </div>
      </dl>

      <section>
        <h4 className="mb-2.5 flex justify-between text-[12px] font-bold uppercase tracking-[0.08em] text-subtle">
          Prediction set
          <span className="num normal-case tracking-normal text-fg">
            <span className="sym">q̂</span> {fixed(t.q_hat)}
          </span>
        </h4>
        <div className="flex items-center gap-2 font-mono text-[16px]" aria-label={`Prediction set: {${set.join(", ")}}`}>
          <span className="text-[22px] text-faint" aria-hidden>{"{"}</span>
          {(["clean", "defect"] as const).map((c) => {
            const on = set.includes(c);
            return (
              <span
                key={c}
                className={cn(
                  "rounded-md border px-2.5 py-1 font-semibold",
                  on ? "border-subtle bg-muted text-fg" : "border-dashed border-border-strong text-faint line-through",
                )}
              >
                {c}
              </span>
            );
          })}
          <span className="text-[22px] text-faint" aria-hidden>{"}"}</span>
        </div>
        <p className="mt-2 text-[12.5px] text-subtle">
          clean is in the set when p(defect) ≤ q̂; defect when 1 − p(defect) ≤ q̂.
        </p>
      </section>

      <section>
        <h4 className="mb-2.5 text-[12px] font-bold uppercase tracking-[0.08em] text-subtle">Shift check</h4>
        <ShiftMeter similarity={item.result.shift.nearest_similarity} threshold={item.result.shift.threshold} flagged={item.result.shift.flagged} />
        <p className="mt-2 text-[12.5px] text-subtle">
          {item.result.shift.flagged
            ? "Below the threshold: this photo looks unlike the calibration set, so it cannot auto-clear."
            : "The nearest calibration image is similar enough; the guarantee applies."}
        </p>
      </section>

      <dl className="divide-y divide-border">
        <KV k="Most likely type" v={item.result.defect_type ?? "—"} />
        <KV k="Cluster" v={clusterLabel(item.result.cluster)} mono={false} />
        <KV k="Service decision at ε = 5%" v={item.result.decision} />
        <KV k="File" v={<span title={item.relpath}>{item.relpath}</span>} />
      </dl>

      <section>
        <h4 className="mb-2.5 text-[12px] font-bold uppercase tracking-[0.08em] text-subtle">Record a human label</h4>
        <FeedbackForm inspectionId={item.result.inspection_id} suggestedType={item.classes[0]} />
      </section>
    </div>
  );
}
