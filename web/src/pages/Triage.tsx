import { useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertOctagon, CheckCircle2, ChevronLeft, ChevronRight, ImageOff, RefreshCw, ShieldMinus } from "lucide-react";
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Link } from "react-router-dom";
import { api } from "@/lib/api";
import { cn } from "@/lib/cn";
import { clusterLabel, eps, fixed, int, pct } from "@/lib/format";
import { useCalibration, useInfo, useRiskStatement } from "@/lib/queries";
import { useSettings } from "@/lib/settings";
import { decide, lambdaValue, predictionSet, reviewHours, summarize, thresholdFor, type BatchSummary } from "@/lib/triage";
import type { ClusterCalibration, Decision, InspectResult, SampleItem, Threshold } from "@/lib/types";
import { Button, Card, CardBody, CardHeader, PageHeader, Segmented, Skeleton } from "@/components/ui/primitives";
import { Sheet, Switch, Tip } from "@/components/ui/radix";
import {
  DECISION_BORDER,
  DECISION_ICON,
  DECISION_META,
  DECISION_SOFT,
  DECISION_TEXT,
  DefectMark,
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

/** Escalate tiles are split into score bands measured down from λ̂, so the wall echoes the rail. */
interface Band {
  key: string;
  title: ReactNode;
  items: Placed[];
}
function escalateBands(rows: Placed[], lam: number): Band[] {
  const ref = Number.isFinite(lam) ? lam : 1;
  const refTxt = Number.isFinite(lam) ? "λ̂" : "1.0";
  const held = rows.filter((r) => r.held);
  const rest = rows.filter((r) => !r.held);
  const near = rest.filter((r) => ref - r.item.score <= 0.1);
  const mid = rest.filter((r) => ref - r.item.score > 0.1 && ref - r.item.score <= 0.3);
  const far = rest.filter((r) => ref - r.item.score > 0.3);
  const f = (x: number) => Math.max(0, x).toFixed(2);
  return [
    { key: "held", title: <>Held back by the shift check <span className="text-faint">· score above λ̂</span></>, items: held },
    { key: "near", title: <>Near {refTxt} <span className="num text-faint">{f(ref - 0.1)}–{f(ref)}</span></>, items: near },
    { key: "mid", title: <>Mid <span className="num text-faint">{f(ref - 0.3)}–{f(ref - 0.1)}</span></>, items: mid },
    { key: "far", title: <>Far below {refTxt} <span className="num text-faint">&lt; {f(ref - 0.3)}</span></>, items: far },
  ].filter((b) => b.items.length);
}

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

  const groups = useMemo(() => {
    if (!t) return null;
    const lam = lambdaValue(t);
    const rows: Placed[] = scored.map((item) => ({ item, decision: decide(item, t, escalate), held: item.score > lam && item.shifted && escalate }));
    const clear = rows.filter((r) => r.decision === "AUTO_CLEAR").sort((a, b) => b.item.score - a.item.score);
    // Flag: most defect-like first. Escalate: closest to clearing first.
    const flag = rows.filter((r) => r.decision === "FLAG").sort((a, b) => a.item.score - b.item.score);
    const esc = rows.filter((r) => r.decision === "ESCALATE").sort((a, b) => b.item.score - a.item.score);
    const bands = escalateBands(esc, lam);
    const ordered = [...clear, ...flag, ...bands.flatMap((b) => b.items)];
    const byKey = new Map(rows.map((r) => [r.item.relpath, r] as const));
    return { clear, flag, esc, bands, ordered, byKey, lam };
  }, [scored, t, escalate]);
  const nHeld = groups ? groups.esc.filter((p) => p.held).length : 0;
  const nShifted = scored.filter((s) => s.shifted).length;

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
  const groupsRef = useRef<HTMLDivElement>(null);
  const flipFrom = useRef<{ rects: Map<string, DOMRect>; prev: Map<string, Decision> } | null>(null);
  const changeEpsilon = useCallback(
    (v: number) => {
      if (v === eSel) return;
      if (groupsRef.current && groups && !reducedMotion()) {
        const rects = new Map<string, DOMRect>();
        groupsRef.current.querySelectorAll<HTMLElement>("[data-flip]").forEach((el) => rects.set(el.dataset.flip!, el.getBoundingClientRect()));
        flipFrom.current = { rects, prev: new Map([...groups.byKey].map(([k, r]) => [k, r.decision])) };
      }
      setEpsilon(v);
    },
    [eSel, groups],
  );
  useLayoutEffect(() => {
    const from = flipFrom.current;
    flipFrom.current = null;
    if (!from || !groupsRef.current || !groups) return;
    groupsRef.current.querySelectorAll<HTMLElement>("[data-flip]").forEach((el) => {
      const key = el.dataset.flip!;
      const f = from.rects.get(key);
      if (!f) return;
      const l = el.getBoundingClientRect();
      const dx = f.left - l.left;
      const dy = f.top - l.top;
      if (Math.abs(dx) < 1 && Math.abs(dy) < 1) return;
      const moved = from.prev.get(key) !== groups.byKey.get(key)?.decision;
      el.animate(
        [
          { transform: `translate(${dx}px, ${dy}px)`, opacity: moved ? 0.35 : 1 },
          { transform: "none", opacity: 1 },
        ],
        { duration: moved ? 620 : 480, easing: "cubic-bezier(.2,.7,.1,1)" },
      );
    });
  }, [eSel, groups]);

  /* ---- drawer ---- */
  const ordered = groups?.ordered ?? [];
  const openIdx = openKey ? ordered.findIndex((p) => p.item.relpath === openKey) : -1;
  const opened = openIdx >= 0 ? ordered[openIdx] : null;
  const step = (dir: number) => {
    if (!ordered.length || openIdx < 0) return;
    setOpenKey(ordered[(openIdx + dir + ordered.length) % ordered.length].item.relpath);
  };
  const returnFocus = useRef<HTMLElement | null>(null);
  const openFrom = (key: string, from?: HTMLElement | null) => {
    returnFocus.current = from ?? (document.activeElement as HTMLElement | null);
    setOpenKey(key);
  };

  if (info.isError) return <ErrorState className="mx-auto max-w-[1440px]" error={info.error} onRetry={() => info.refetch()} />;
  if (info.data && !ready) {
    return (
      <div className="mx-auto max-w-[1440px]">
        <PageHeader eyebrow="triage" title="Batch triage" />
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
  const dataset = info.data?.sample_dataset ?? "sample";

  return (
    <div className="mx-auto grid max-w-[1600px] gap-4">
      {/* Header */}
      <header className="flex flex-wrap items-end justify-between gap-3 animate-rise">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 font-mono text-[11.5px] text-faint">
            <span>triage</span>
            <span aria-hidden>/</span>
            <b className="font-medium text-subtle">batch</b>
            <span aria-hidden>/</span>
            <span>{siteType}</span>
          </div>
          <h1 className="mt-1 text-[22px] font-semibold leading-[1.1] tracking-[-0.025em] md:text-[26px]">Batch triage</h1>
          <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-[12.5px] text-subtle">
            <span>
              {settings.batchSize} {siteType} images · batch {seed}
              {cal.data && <> · calibrated on {int(cal.data.n_cal)} ({int(cal.data.n_defects)} defects)</>}
            </span>
            <span className="hidden font-mono text-faint sm:inline">
              {dataset} · each inspected once by POST /v1/inspect{cal.data ? ` · ${cal.data.model_version.split(":")[0]}` : ""}
            </span>
          </div>
        </div>
        <Button
          variant="outline"
          size="sm"
          disabled={batch.isFetching}
          onClick={() => qc.resetQueries({ queryKey: ["batch", siteType, seed, settings.batchSize] })}
        >
          <RefreshCw className={cn("h-3.5 w-3.5", batch.isFetching && "animate-spin")} strokeWidth={1.8} /> Re-inspect
        </Button>
      </header>

      {/* Control strip */}
      <section aria-label="Triage controls" className="flex flex-wrap rounded-xl border border-border bg-surface animate-rise [animation-delay:50ms]">
        <Ctl label="Structure type" className="basis-[46%] sm:basis-auto">
          <Segmented
            ariaLabel="Structure type"
            value={siteType}
            onChange={(v) => setSiteType(v)}
            options={(info.data?.site_types ?? ["wall", "deck", "pavement"]).map((s) => ({ value: s, label: clusterLabel(s) }))}
          />
        </Ctl>
        <EpsilonControl grid={epsGrid} value={eSel} onChange={changeEpsilon} />
        <Ctl label="Thresholds in force" className="basis-1/2 sm:basis-auto">
          <div className="flex h-8 items-baseline gap-5 pt-1 font-mono tabular-nums" aria-live="polite">
            <span className="flex items-baseline gap-1.5">
              <span className="font-sans text-[13.5px] text-subtle">λ̂</span>
              <span className="text-[16px] font-medium text-accent">{t ? (t.lambda_hat == null ? "∞" : t.lambda_hat.toFixed(4)) : "—"}</span>
            </span>
            <span className="flex items-baseline gap-1.5">
              <span className="font-sans text-[13.5px] text-subtle">q̂</span>
              <span className="text-[16px] font-medium">{t ? t.q_hat.toFixed(4) : "—"}</span>
            </span>
          </div>
        </Ctl>
        <Ctl label="Batch" className="basis-1/2 sm:basis-auto">
          <div className="flex items-center gap-1">
            <Button variant="outline" size="icon" aria-label="Previous batch" disabled={seed === 0} onClick={() => setSeed((s) => Math.max(0, s - 1))}>
              <ChevronLeft className="h-4 w-4" strokeWidth={1.8} />
            </Button>
            <input
              aria-label="Batch number"
              type="number"
              min={0}
              value={seed}
              onChange={(e) => setSeed(Math.max(0, Math.floor(Number(e.target.value) || 0)))}
              className="num h-8 w-14 rounded-[5px] border border-border-strong bg-muted text-center text-[13px] focus:border-accent focus:outline-none focus-visible:outline-none focus:ring-1 focus:ring-accent"
            />
            <Button variant="outline" size="icon" aria-label="Next batch" onClick={() => setSeed((s) => s + 1)}>
              <ChevronRight className="h-4 w-4" strokeWidth={1.8} />
            </Button>
          </div>
        </Ctl>
        <Ctl label="Evaluation" className="basis-1/2 sm:basis-auto">
          <div className="flex h-8 items-center gap-2.5">
            <Switch id="gt" checked={showTruth} onCheckedChange={setShowTruth} />
            <label htmlFor="gt" className="cursor-pointer whitespace-nowrap text-[12.5px]">
              <span className="hidden sm:inline">Show ground truth</span>
              <span className="sm:hidden">Ground truth</span>
            </label>
          </div>
        </Ctl>
      </section>

      {batch.isError ? (
        <ErrorState error={batch.error} onRetry={() => batch.refetch()} />
      ) : loading || !summary || !t || !groups ? (
        <LoadingState done={batch.progress.done} total={batch.progress.total || settings.batchSize} site={clusterLabel(siteType)} seed={seed} />
      ) : (
        <>
          <MetricsStrip
            summary={summary}
            epsilon={eSel!}
            held={nHeld}
            shifted={nShifted}
            cal={cal.data}
            showTruth={showTruth}
            perDay={settings.imagesPerDay}
            minutes={settings.minutesPerReview}
            onPerDay={(v) => update({ imagesPerDay: v })}
            onMinutes={(v) => update({ minutesPerReview: v })}
          />

          <RiskBar
            text={risk.data?.risk_statement.replace(/^Auto-cleared\. /, "")}
            error={risk.isError}
            summary={summary}
            showTruth={showTruth}
            cal={cal.data}
            agreement={agreement}
          />

          <ScoreRail
            items={groups.ordered}
            t={t}
            showTruth={showTruth}
            summary={summary}
            onOpen={(key, el) => openFrom(key, el)}
          />

          <div ref={groupsRef} className="grid gap-4">
            <Panel
              title={
                <>
                  <DecisionGlyph d="AUTO_CLEAR" />
                  <PanelTitle>Auto-cleared</PanelTitle>
                  <span className="font-mono text-[12.5px] text-subtle">
                    {summary.cleared} of {summary.n}
                  </span>
                  <span className="text-[11.5px] text-faint">score &gt; λ̂ and in distribution · no human needed</span>
                </>
              }
            >
              <TileGrid>
                {groups.clear.length === 0 && <NothingClears t={t} epsilon={eSel!} top={Math.max(...scored.map((s) => s.score))} nDefects={cal.data?.n_defects} />}
                {groups.clear.map((p) => (
                  <Tile key={p.item.relpath} p={p} lam={t.lambda_hat} showTruth={showTruth} onOpen={(el) => openFrom(p.item.relpath, el)} />
                ))}
              </TileGrid>
            </Panel>

            <Panel
              title={
                <>
                  <svg viewBox="0 0 16 16" className="h-4 w-4 fill-none stroke-current text-subtle" strokeWidth={1.6} aria-hidden>
                    <circle cx="8" cy="5" r="2.6" />
                    <path d="M2.8 14c.6-2.8 2.7-4.4 5.2-4.4s4.6 1.6 5.2 4.4" />
                  </svg>
                  <PanelTitle>To a human</PanelTitle>
                  <span className="font-mono text-[12.5px] text-subtle">
                    {summary.human} of {summary.n} · {pct(summary.hrf, 1)}
                  </span>
                </>
              }
            >
              <SubGroup d="FLAG" count={groups.flag.length} desc="prediction set is {defect}: likely defect · most defect-like first">
                <TileGrid>
                  {groups.flag.length === 0 && <EmptyNote>No image has a {"{defect}"}-only prediction set at this ε.</EmptyNote>}
                  {groups.flag.map((p) => (
                    <Tile key={p.item.relpath} p={p} lam={t.lambda_hat} showTruth={showTruth} onOpen={(el) => openFrom(p.item.relpath, el)} />
                  ))}
                </TileGrid>
              </SubGroup>
              <SubGroup d="ESCALATE" count={groups.esc.length} desc="uncertain, below λ̂ or shifted · closest to clearing first">
                {groups.esc.length === 0 ? (
                  <TileGrid>
                    <EmptyNote>No uncertain images at this ε.</EmptyNote>
                  </TileGrid>
                ) : (
                  <div className="grid gap-4">
                    {groups.bands.map((b) => (
                      <div key={b.key}>
                        <div className="mb-2 flex items-center gap-2 text-[11.5px] text-subtle after:h-px after:flex-1 after:bg-border">
                          <span className="whitespace-nowrap">{b.title}</span>
                          <span className="font-mono text-faint">{b.items.length}</span>
                        </div>
                        <TileGrid>
                          {b.items.map((p) => (
                            <Tile key={p.item.relpath} p={p} lam={t.lambda_hat} showTruth={showTruth} onOpen={(el) => openFrom(p.item.relpath, el)} />
                          ))}
                        </TileGrid>
                      </div>
                    ))}
                  </div>
                )}
              </SubGroup>
            </Panel>
          </div>

          <BatchCurve curve={curve} epsilon={eSel!} showTruth={showTruth} />
        </>
      )}

      {opened && t && eSel != null && (
        <Sheet
          open
          onOpenChange={(o) => {
            if (!o) {
              const el = returnFocus.current;
              setOpenKey(null);
              requestAnimationFrame(() => el?.focus());
            }
          }}
          title={
            <span className="flex items-baseline gap-2">
              {opened.item.site_id}
              <span className="font-mono text-[11.5px] font-normal text-faint">#{opened.item.result.inspection_id}</span>
            </span>
          }
          description={`${openIdx + 1} of ${ordered.length} · ${DECISION_META[opened.decision].name.toLowerCase()} · ε ${eps(eSel)}`}
          onKeyDown={(e) => {
            const el = e.target as HTMLElement;
            if (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.closest("[role=listbox]")) return;
            if (e.key === "ArrowLeft") {
              e.preventDefault();
              step(-1);
            }
            if (e.key === "ArrowRight") {
              e.preventDefault();
              step(1);
            }
          }}
          headerActions={
            <div className="flex gap-1.5">
              <Button variant="outline" size="icon" aria-label="Previous image (←)" onClick={() => step(-1)}>
                <ChevronLeft className="h-4 w-4" strokeWidth={1.8} />
              </Button>
              <Button variant="outline" size="icon" aria-label="Next image (→)" onClick={() => step(1)}>
                <ChevronRight className="h-4 w-4" strokeWidth={1.8} />
              </Button>
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
/* Building blocks                                                           */

function Ctl({ label, children, className, labelId }: { label: ReactNode; children: ReactNode; className?: string; labelId?: string }) {
  const auto = useId();
  const id = labelId ?? auto;
  return (
    <div
      role="group"
      aria-labelledby={id}
      className={cn(
        "grid min-w-0 flex-1 content-start gap-1.5 border-b border-border px-4 py-2.5 sm:flex-none sm:border-b-0 sm:border-r last:border-r-0",
        className,
      )}
    >
      <span id={id} className="flex items-center gap-1.5 whitespace-nowrap text-[11.5px] text-subtle">
        {label}
      </span>
      {children}
    </div>
  );
}

function EpsilonControl({ grid, value, onChange }: { grid: number[]; value: number | undefined; onChange: (v: number) => void }) {
  const labelId = useId();
  const groupRef = useRef<HTMLDivElement>(null);
  const [thumb, setThumb] = useState<{ left: number; width: number } | null>(null);
  const i = value != null ? grid.indexOf(value) : -1;

  // The sliding highlight follows the selected segment.
  useLayoutEffect(() => {
    const g = groupRef.current;
    const measure = () => {
      const on = g?.querySelector<HTMLElement>('[aria-checked="true"]');
      if (on) setThumb({ left: on.offsetLeft, width: on.offsetWidth });
    };
    measure();
    if (!g) return;
    const ro = new ResizeObserver(measure);
    ro.observe(g);
    document.fonts?.ready.then(measure).catch(() => {});
    return () => ro.disconnect();
  }, [value, grid.length]);

  const onKey = (e: React.KeyboardEvent) => {
    let n: number | null = null;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") n = i + 1;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") n = i - 1;
    if (e.key === "Home") n = 0;
    if (e.key === "End") n = grid.length - 1;
    if (n == null) return;
    e.preventDefault();
    const v = grid[Math.max(0, Math.min(grid.length - 1, n))];
    onChange(v);
    requestAnimationFrame(() => groupRef.current?.querySelector<HTMLButtonElement>(`[data-eps="${v}"]`)?.focus());
  };

  return (
    <Ctl label="Guaranteed max missed-defect rate (ε)" labelId={labelId} className="basis-full sm:flex-[1_1_360px]">
      {grid.length ? (
        <div
          ref={groupRef}
          role="radiogroup"
          aria-labelledby={labelId}
          onKeyDown={onKey}
          className="relative flex gap-0.5 rounded-[5px] border border-border bg-sunken p-0.5 sm:inline-flex sm:w-fit"
        >
          {thumb && (
            <span
              className="absolute inset-y-0.5 rounded-[4px] bg-surface shadow-[0_0_0_1px_rgb(var(--border-strong)),0_1px_2px_rgb(0_0_0/0.25)] transition-[left,width] duration-300 [transition-timing-function:cubic-bezier(.2,.7,.1,1)]"
              style={{ left: thumb.left, width: thumb.width }}
              aria-hidden
            />
          )}
          {grid.map((e) => {
            const on = e === value;
            return (
              <button
                key={e}
                type="button"
                role="radio"
                aria-checked={on}
                tabIndex={on ? 0 : -1}
                data-eps={e}
                aria-label={`ε ${eps(e)}`}
                onClick={() => onChange(e)}
                className={cn(
                  "relative z-[1] h-7 min-w-0 flex-1 rounded-[4px] px-2 font-mono text-[12.5px] transition-colors sm:min-w-[52px] sm:flex-none",
                  on ? "font-semibold text-accent" : "text-subtle hover:text-fg",
                )}
              >
                {eps(e)}
              </button>
            );
          })}
        </div>
      ) : (
        <Skeleton className="h-8 w-full sm:w-[290px]" />
      )}
    </Ctl>
  );
}

function Panel({ title, children, className, bodyClassName }: { title: ReactNode; children: ReactNode; className?: string; bodyClassName?: string }) {
  return (
    <section className={cn("rounded-xl border border-border bg-surface", className)}>
      <div className="flex min-h-[42px] flex-wrap items-center gap-x-2.5 gap-y-1 border-b border-border px-4 py-2.5">{title}</div>
      <div className={cn("p-3 sm:p-4", bodyClassName)}>{children}</div>
    </section>
  );
}

function PanelTitle({ children }: { children: ReactNode }) {
  return <h2 className="text-[11.5px] font-semibold uppercase tracking-[0.09em] text-subtle">{children}</h2>;
}

function DecisionGlyph({ d, className }: { d: Decision; className?: string }) {
  const Icon = DECISION_ICON[d];
  return <Icon className={cn(DECISION_TEXT[d], className)} />;
}

function SubGroup({ d, count, desc, children }: { d: Decision; count: number; desc: string; children: ReactNode }) {
  return (
    <div className="[&+&]:mt-5 [&+&]:border-t [&+&]:border-border [&+&]:pt-4">
      <h3 className="mb-3 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[12.5px] font-semibold">
        <DecisionGlyph d={d} />
        {DECISION_META[d].name}
        <span className="font-mono font-normal text-faint">{count}</span>
        <span className="text-[11.5px] font-normal text-faint">{desc}</span>
      </h3>
      {children}
    </div>
  );
}

function TileGrid({ children }: { children: ReactNode }) {
  return <ul className="grid min-h-[40px] grid-cols-2 gap-2.5 sm:grid-cols-[repeat(auto-fill,minmax(196px,1fr))] sm:gap-3">{children}</ul>;
}

function EmptyNote({ children }: { children: ReactNode }) {
  return <li className="col-span-full rounded-lg border border-dashed border-border-strong bg-muted p-3 text-[12.5px] text-subtle">{children}</li>;
}

/* ======================================================================== */
/* Metrics, risk                                                             */

function MetricsStrip({
  summary,
  epsilon,
  held,
  shifted,
  cal,
  showTruth,
  perDay,
  minutes,
  onPerDay,
  onMinutes,
}: {
  summary: BatchSummary;
  epsilon: number;
  held: number;
  shifted: number;
  cal: ClusterCalibration | undefined;
  showTruth: boolean;
  perDay: number;
  minutes: number;
  onPerDay: (v: number) => void;
  onMinutes: (v: number) => void;
}) {
  const n = summary.n;
  const [bad, setBad] = useState({ perDay: false, minutes: false });
  const perDayValid = useCallback((v: boolean) => setBad((b) => (b.perDay === !v ? b : { ...b, perDay: !v })), []);
  const minutesValid = useCallback((v: boolean) => setBad((b) => (b.minutes === !v ? b : { ...b, minutes: !v })), []);
  const h = reviewHours(perDay, summary.hrf, minutes);
  const cell = "grid min-w-0 content-start gap-1 border-b border-r border-border px-4 py-3";
  return (
    <section
      aria-label="Key metrics"
      className={cn(
        "grid overflow-hidden rounded-xl border border-border bg-surface animate-rise [animation-delay:100ms]",
        "grid-cols-2 md:grid-cols-3",
        showTruth ? "xl:grid-cols-[repeat(4,minmax(0,1fr))_minmax(0,1.3fr)]" : "xl:grid-cols-[repeat(3,minmax(0,1fr))_minmax(0,1.3fr)]",
        "[&>*]:-mb-px [&>*]:-mr-px",
      )}
    >
      <div className={cell}>
        <div className="text-[11.5px] text-subtle">Human-review fraction</div>
        <div className="num text-[21px] font-medium leading-[1.1] tracking-[-0.02em] md:text-[24px]" aria-live="polite">
          {pct(summary.hrf, 1)}
        </div>
        <div className="mt-1 flex h-[3px] overflow-hidden rounded-sm bg-sunken" aria-hidden>
          <i className="block bg-flag transition-[width] duration-500" style={{ width: `${(100 * summary.byDecision.FLAG) / n}%` }} />
          <i className="block bg-escalate transition-[width] duration-500" style={{ width: `${(100 * summary.byDecision.ESCALATE) / n}%` }} />
          <i className="block bg-clear transition-[width] duration-500" style={{ width: `${(100 * summary.byDecision.AUTO_CLEAR) / n}%` }} />
        </div>
        <div className="font-mono text-[11px] text-faint">
          {summary.byDecision.FLAG} flag · {summary.byDecision.ESCALATE} esc
        </div>
      </div>
      <div className={cell}>
        <div className="flex items-center gap-1.5 text-[11.5px] text-subtle">
          <DecisionGlyph d="AUTO_CLEAR" /> Auto-cleared
        </div>
        <div className="num flex items-baseline gap-1.5 text-[21px] font-medium leading-[1.1] md:text-[24px]">
          {summary.cleared}
          <small className="text-[12.5px] font-normal text-faint">/ {n}</small>
        </div>
        <div className="text-[11.5px] text-faint">{summary.cleared ? `${pct(summary.cleared / n, 1)} of batch skips review` : "nothing clears at this ε"}</div>
      </div>
      <div className={cell}>
        <div className="flex items-center gap-1.5 text-[11.5px] text-subtle">
          <ShiftMark className="text-shift" /> <span className="sm:hidden">Shift holds</span>
          <span className="hidden sm:inline">Held back by shift check</span>
        </div>
        <div className="num flex items-baseline gap-1.5 text-[21px] font-medium leading-[1.1] md:text-[24px]">
          {held}
          <small className="text-[12.5px] font-normal text-faint">images</small>
        </div>
        <div className="text-[11.5px] text-faint">
          {shifted} of {n} outside similarity {cal ? fixed(cal.shift_threshold) : "—"}
        </div>
      </div>
      {showTruth && (
        <div className={cell}>
          <div className="flex items-center gap-1.5 text-[11.5px] text-subtle">
            <DefectMark className="text-defect" /> Defects auto-cleared
          </div>
          <div className={cn("num flex items-baseline gap-1.5 text-[21px] font-medium leading-[1.1] md:text-[24px]", summary.missed > 0 && "text-defect")}>
            {summary.missed}
            <small className="text-[12.5px] font-normal text-faint">of {summary.defects}</small>
          </div>
          <div className="text-[11.5px] text-faint">
            {pct(summary.batchMissRate ?? 0, 1)} this batch vs ε {eps(epsilon)}
          </div>
        </div>
      )}
      <div className={cn(cell, "col-span-2 md:col-span-1", showTruth ? "md:col-span-2 xl:col-span-1" : "")}>
        <div className="text-[11.5px] text-subtle">Projected review load</div>
        <div className="num flex items-baseline gap-1.5 text-[21px] font-medium leading-[1.1] md:text-[24px]" aria-live="polite">
          {bad.perDay || bad.minutes ? "—" : h >= 100 ? h.toFixed(0) : h.toFixed(1)}
          <small className="text-[12.5px] font-normal text-faint">h / day</small>
        </div>
        <div className="mt-0.5 flex flex-wrap gap-1.5">
          <InlineNumber label="Images per day" unit="img/day" value={perDay} min={1} step={100} integer onCommit={onPerDay} onValidity={perDayValid} />
          <InlineNumber label="Minutes per review" unit="min/review" value={minutes} min={0.1} step={0.1} onCommit={onMinutes} onValidity={minutesValid} />
        </div>
        {(bad.perDay || bad.minutes) && (
          <p className="text-[11px] text-danger" role="alert">
            {bad.perDay ? "Images per day: a whole number, 1 or more. " : ""}
            {bad.minutes ? "Minutes per review: 0.1 or more." : ""}
          </p>
        )}
      </div>
    </section>
  );
}

/** A compact number field with a unit, committing only valid values. */
function InlineNumber({
  label,
  unit,
  value,
  min,
  step,
  integer,
  onCommit,
  onValidity,
}: {
  label: string;
  unit: string;
  value: number;
  min: number;
  step: number;
  integer?: boolean;
  onCommit: (v: number) => void;
  onValidity: (valid: boolean) => void;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText((cur) => (Number(cur) === value ? cur : String(value))), [value]);
  const n = Number(text);
  const valid = text.trim() !== "" && Number.isFinite(n) && n >= min && (!integer || Number.isInteger(n));
  useEffect(() => onValidity(valid), [valid, onValidity]);
  return (
    <label
      className={cn(
        "inline-flex h-6 items-center rounded-[3px] border bg-muted text-[11.5px] text-faint focus-within:border-accent focus-within:ring-1 focus-within:ring-accent",
        valid ? "border-border-strong" : "border-danger",
      )}
    >
      <span className="sr-only">{label}</span>
      <input
        type="number"
        inputMode={integer ? "numeric" : "decimal"}
        min={min}
        step={step}
        value={text}
        aria-invalid={!valid}
        onChange={(e) => {
          setText(e.target.value);
          const v = Number(e.target.value);
          if (e.target.value.trim() !== "" && Number.isFinite(v) && v >= min && (!integer || Number.isInteger(v))) onCommit(v);
        }}
        className={cn("h-full w-[58px] bg-transparent px-1.5 text-right font-mono tabular-nums focus:outline-none focus-visible:outline-none", valid ? "text-fg" : "text-danger")}
      />
      <span className="whitespace-nowrap pl-0.5 pr-1.5">{unit}</span>
    </label>
  );
}

function RiskBar({
  text,
  error,
  summary,
  showTruth,
  cal,
  agreement,
}: {
  text: string | undefined;
  error: boolean;
  summary: BatchSummary;
  showTruth: boolean;
  cal: ClusterCalibration | undefined;
  agreement: { n: number; mismatches: number } | null;
}) {
  return (
    <div className="flex flex-wrap items-start gap-x-3 gap-y-2 rounded-r-lg border-l-2 border-signal bg-accent-soft px-4 py-2.5 text-[13.5px] leading-relaxed animate-rise [animation-delay:150ms]" aria-live="polite">
      <svg viewBox="0 0 16 16" className="mt-1 h-4 w-4 shrink-0 fill-none stroke-current text-accent" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        <path d="M8 1.5l5.5 2v4.2c0 3.2-2.4 5.8-5.5 6.8-3.1-1-5.5-3.6-5.5-6.8V3.5z" />
        <path d="M5.6 8l1.7 1.7 3.2-3.4" />
      </svg>
      <div className="min-w-0 flex-1 basis-72">
        {text ?? (error ? <span className="text-danger">The risk statement could not be loaded.</span> : <Skeleton className="h-5 w-3/4" />)}
        {showTruth && summary.defects > 0 && (
          <span className="text-subtle">
            {" "}
            This batch: {summary.missed} of {summary.defects} true defects auto-cleared ({pct(summary.batchMissRate, 1)}). The guarantee is an
            average over many batches, so a single batch of {summary.n} can land above ε.
          </span>
        )}
      </div>
      <div className="flex shrink-0 flex-col items-start gap-1 text-[11.5px] text-subtle md:items-end">
        {cal && (
          <span className="flex items-baseline gap-2">
            <span className="text-faint">{cal.guarantee.split(":")[0]}</span>
            <GuaranteeFormula className="text-[13px] text-fg" />
          </span>
        )}
        {agreement && (
          <Tip content="The service decided every image at ε = 5%. The console re-partitions locally with the same rule; this checks they agree.">
            <span tabIndex={0} className={cn("flex items-center gap-1.5 rounded-sm", agreement.mismatches ? "text-danger" : "")}>
              {agreement.mismatches ? <AlertOctagon className="h-3.5 w-3.5" /> : <CheckCircle2 className="h-3.5 w-3.5 text-clear" />}
              {agreement.mismatches
                ? `${agreement.mismatches} of ${agreement.n} local decisions disagree with the service at ε = 5%`
                : `matches the service on all ${agreement.n} at ε = 5%`}
            </span>
          </Tip>
        )}
      </div>
    </div>
  );
}

/* ======================================================================== */
/* Score rail                                                                */

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setW(el.clientWidth);
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

const RING: Record<Decision, string> = {
  AUTO_CLEAR: "rgb(var(--clear))",
  FLAG: "rgb(var(--flag))",
  ESCALATE: "rgb(var(--escalate))",
};

function ScoreRail({
  items,
  t,
  showTruth,
  summary,
  onOpen,
}: {
  items: Placed[];
  t: Threshold;
  showTruth: boolean;
  summary: BatchSummary;
  onOpen: (key: string, from: HTMLElement | null) => void;
}) {
  const [plotRef, W] = useWidth<HTMLDivElement>();
  const listId = useId();
  const [active, setActive] = useState(-1);
  const [peek, setPeek] = useState<{ p: Placed; x: number; y: number } | null>(null);
  const prev = useRef<Map<string, Decision>>(new Map());
  const [blips, setBlips] = useState<Set<string>>(new Set());

  const lam = lambdaValue(t);
  const lamX = Number.isFinite(lam) ? lam : 1;
  const b = Math.min(t.q_hat, 1 - t.q_hat);

  // Beeswarm stacked up from the axis: positions depend only on the scores.
  const layout = useMemo(() => {
    if (!W) return null;
    const ts = W < 500 ? 22 : W < 900 ? 28 : 34;
    const gap = W < 500 ? 3 : 4;
    const sorted = [...items].sort((a, b) => a.item.score - b.item.score);
    const placed: { cx: number; r: number }[] = [];
    let maxRow = 0;
    const pos = new Map<string, { cx: number; r: number }>();
    for (const p of sorted) {
      const cx = Math.min(Math.max(p.item.score * W, ts / 2), W - ts / 2);
      let r = 0;
      while (placed.some((q) => q.r === r && Math.abs(q.cx - cx) < ts + gap)) r++;
      placed.push({ cx, r });
      pos.set(p.item.relpath, { cx, r });
      maxRow = Math.max(maxRow, r);
    }
    return { ts, gap, pos, order: sorted.map((p) => p.item.relpath), height: Math.max(104, 16 + (maxRow + 1) * (ts + gap) + 8) };
  }, [items, W]);

  // A short pulse on the thumbs whose decision changed with ε.
  useEffect(() => {
    const changed = new Set<string>();
    for (const p of items) {
      const was = prev.current.get(p.item.relpath);
      if (was && was !== p.decision) changed.add(p.item.relpath);
    }
    prev.current = new Map(items.map((p) => [p.item.relpath, p.decision]));
    if (!changed.size || reducedMotion()) return;
    setBlips(changed);
    const id = setTimeout(() => setBlips(new Set()), 750);
    return () => clearTimeout(id);
  }, [items]);

  const byKey = useMemo(() => new Map(items.map((p) => [p.item.relpath, p])), [items]);
  const showPeek = (p: Placed, el: HTMLElement) => {
    const r = el.getBoundingClientRect();
    const x = Math.max(8, Math.min(window.innerWidth - 176, r.left + r.width / 2 - 84));
    const y = r.top - 204 < 8 ? r.bottom + 12 : r.top - 204;
    setPeek({ p, x, y });
  };

  const onKey = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (!layout) return;
    const ord = layout.order;
    if (["ArrowRight", "ArrowLeft", "Home", "End"].includes(e.key)) {
      e.preventDefault();
      let n = active < 0 ? 0 : active;
      if (active >= 0) {
        if (e.key === "ArrowRight") n = Math.min(ord.length - 1, active + 1);
        if (e.key === "ArrowLeft") n = Math.max(0, active - 1);
      }
      if (e.key === "Home") n = 0;
      if (e.key === "End") n = ord.length - 1;
      setActive(n);
      const el = document.getElementById(`${listId}-${n}`);
      const p = byKey.get(ord[n]);
      if (el && p) showPeek(p, el);
    } else if ((e.key === "Enter" || e.key === " ") && active >= 0) {
      e.preventDefault();
      setPeek(null);
      onOpen(ord[active], e.currentTarget);
    }
  };

  const ticks = Array.from({ length: 21 }, (_, k) => k / 20);
  const flagCap = `≤ ${b.toFixed(2)}`;

  return (
    <section aria-labelledby={`${listId}-t`} className="rounded-xl border border-border bg-surface animate-rise [animation-delay:200ms]">
      <div className="flex min-h-[42px] flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-border px-4 py-2.5">
        <h2 id={`${listId}-t`} className="text-[11.5px] font-semibold uppercase tracking-[0.09em] text-subtle">
          Score rail
        </h2>
        <ul className="flex flex-wrap gap-x-3.5 gap-y-1 text-[11.5px] text-subtle" aria-hidden>
          <li className="flex items-center gap-1.5"><DecisionGlyph d="FLAG" />Flag</li>
          <li className="flex items-center gap-1.5"><DecisionGlyph d="ESCALATE" />Escalate</li>
          <li className="flex items-center gap-1.5"><DecisionGlyph d="AUTO_CLEAR" />Auto-clear</li>
          {showTruth && <li className="flex items-center gap-1.5"><DefectMark className="text-defect" />True defect</li>}
        </ul>
        <span className="w-full font-mono text-[11.5px] text-faint sm:ml-auto sm:w-auto">
          {summary.byDecision.FLAG} flag · {summary.byDecision.ESCALATE} esc · {summary.cleared} clear
        </span>
      </div>
      <div className="px-3 pb-3 pt-4 sm:px-4">
        <div className="relative mt-[30px]">
          <div
            ref={plotRef}
            id={listId}
            role="listbox"
            tabIndex={0}
            aria-label="Images plotted by score. Use the arrow keys to move and Enter to open."
            aria-activedescendant={active >= 0 ? `${listId}-${active}` : undefined}
            onKeyDown={onKey}
            onBlur={() => {
              setActive(-1);
              setPeek(null);
            }}
            className="relative transition-[height] duration-300 focus-visible:outline-offset-4"
            style={{ height: layout?.height ?? 104 }}
          >
            {/* zones */}
            <div
              className="absolute inset-y-0 left-0 border-r border-dashed border-flag/55 bg-gradient-to-b from-transparent to-flag/15 transition-[width] duration-500 [transition-timing-function:cubic-bezier(.2,.7,.1,1)]"
              style={{ width: `${b * 100}%` }}
              aria-hidden
            />
            {Number.isFinite(lam) && (
              <div
                className="hatch-clear-zone absolute inset-y-0 right-0 transition-[left] duration-500 [transition-timing-function:cubic-bezier(.2,.7,.1,1)]"
                style={{ left: `${lamX * 100}%` }}
                aria-hidden
              />
            )}
            {/* the λ̂ cut */}
            <div
              className="pointer-events-none absolute -bottom-1.5 -top-[30px] z-[4] w-0 transition-[left] duration-500 [transition-timing-function:cubic-bezier(.2,.7,.1,1)]"
              style={{ left: `${lamX * 100}%` }}
              aria-hidden
            >
              <span
                className={cn(
                  "absolute bottom-0 left-[-1px] top-[22px] w-0.5",
                  Number.isFinite(lam) ? "bg-signal shadow-[0_0_12px_rgb(var(--signal)/0.55)]" : "bg-[repeating-linear-gradient(180deg,rgb(var(--signal))_0_4px,transparent_4px_8px)]",
                )}
              />
              <span
                className={cn(
                  "absolute left-0 top-0 whitespace-nowrap rounded-[3px] bg-signal px-[7px] py-0.5 font-mono text-[11.5px] font-semibold text-accent-ink",
                  lamX > 0.86 ? "-translate-x-full rounded-tr-none" : "-translate-x-1/2",
                )}
              >
                {Number.isFinite(lam) ? `λ̂ ${lam.toFixed(4)}` : "λ̂ = ∞"}
              </span>
            </div>
            {/* thumbs */}
            {layout &&
              layout.order.map((key, idx) => {
                const p = byKey.get(key)!;
                const pos = layout.pos.get(key)!;
                const Icon = DECISION_ICON[p.decision];
                const isActive = idx === active;
                return (
                  <div
                    key={key}
                    id={`${listId}-${idx}`}
                    role="option"
                    aria-selected={isActive}
                    aria-label={`${p.item.site_id}, score ${p.item.score.toFixed(3)}, ${DECISION_META[p.decision].name}${showTruth && p.item.defect ? ", true defect" : ""}`}
                    onClick={() => {
                      setPeek(null);
                      onOpen(key, plotRef.current);
                    }}
                    onMouseEnter={(e) => showPeek(p, e.currentTarget)}
                    onMouseLeave={() => setPeek(null)}
                    className={cn(
                      "absolute z-[2] cursor-pointer overflow-visible rounded-[3px] bg-sunken transition-[left,bottom,box-shadow,transform] duration-[450ms] [transition-timing-function:cubic-bezier(.2,.7,.1,1)] hover:z-[6] hover:scale-[1.6]",
                      isActive && "z-[6] scale-[1.6] outline outline-2 outline-offset-2 outline-accent",
                      blips.has(key) && "animate-[blip_.7s_cubic-bezier(.2,.7,.1,1)]",
                    )}
                    style={{
                      width: layout.ts,
                      height: layout.ts,
                      left: pos.cx - layout.ts / 2,
                      bottom: 8 + pos.r * (layout.ts + layout.gap),
                      boxShadow: `0 0 0 1.5px ${RING[p.decision]}, 0 0 0 3px rgb(var(--surface))`,
                    }}
                  >
                    <img src={p.item.url} alt="" loading="lazy" className="pointer-events-none h-full w-full rounded-[3px] object-cover" />
                    {/* decision glyph: colour is never the only signal */}
                    <span
                      className={cn("absolute -bottom-1 -right-1 grid place-items-center rounded-full bg-surface", DECISION_TEXT[p.decision])}
                      style={{ width: layout.ts < 26 ? 11 : 13, height: layout.ts < 26 ? 11 : 13 }}
                      aria-hidden
                    >
                      <Icon className={layout.ts < 26 ? "h-[9px] w-[9px]" : "h-[11px] w-[11px]"} />
                    </span>
                    {showTruth && p.item.defect && (
                      <span
                        className="absolute -left-1 -top-1 h-[9px] w-[9px] rounded-full border-[1.5px] border-surface bg-defect"
                        aria-hidden
                      />
                    )}
                  </div>
                );
              })}
          </div>
          {/* axis */}
          <div className="relative h-[34px] border-t border-border-strong" aria-hidden>
            {ticks.map((v, k) => (
              <span
                key={k}
                className={cn(
                  "absolute top-0 -translate-x-1/2 pt-2 font-mono text-[10.5px] text-faint before:absolute before:left-1/2 before:top-0 before:w-px before:bg-border-strong",
                  k % 2 ? "text-transparent before:h-[3px]" : "before:h-[5px]",
                  k % 4 === 2 && "max-sm:text-transparent",
                  k === 0 && "translate-x-0 before:left-0",
                  k === 20 && "-translate-x-full before:left-auto before:right-0",
                )}
                style={{ left: `${v * 100}%` }}
              >
                {k % 2 ? "" : v.toFixed(1)}
              </span>
            ))}
          </div>
          <div className="relative mt-0.5 h-[18px] text-[11.5px] font-medium" aria-hidden>
            <span className="absolute left-0 flex items-center gap-1.5 whitespace-nowrap text-flag">
              <DecisionGlyph d="FLAG" />Flag <span className="font-mono font-normal text-faint">{flagCap}</span>
            </span>
            <span
              className="absolute hidden -translate-x-1/2 items-center gap-1.5 whitespace-nowrap text-escalate transition-[left] duration-500 sm:flex"
              style={{ left: `${((b + lamX) / 2) * 100}%` }}
            >
              <DecisionGlyph d="ESCALATE" />Escalate
            </span>
            {Number.isFinite(lam) && (
              <span className="absolute right-0 flex items-center gap-1.5 whitespace-nowrap text-clear">
                <DecisionGlyph d="AUTO_CLEAR" />Clear
              </span>
            )}
          </div>
          <div className="mt-1 flex justify-between text-[10.5px] uppercase tracking-[0.08em] text-faint">
            <span>← likely defect</span>
            <span className="hidden sm:inline">
              <b className="font-medium text-subtle">score</b> = 1 − p(defect)
            </span>
            <span>looks clean →</span>
          </div>
        </div>
      </div>
      {peek && (
        <div
          className="pointer-events-none fixed z-50 w-[168px] rounded-lg bg-surface p-1.5 shadow-pop animate-fade-in"
          style={{ left: peek.x, top: peek.y }}
          aria-hidden
        >
          <img src={peek.p.item.url} alt="" className="aspect-square w-full rounded object-cover" />
          <div className="flex items-center justify-between gap-2 px-0.5 pt-1.5 text-[11.5px]">
            <span className="truncate font-mono">
              {peek.p.item.score.toFixed(3)} · {peek.p.item.site_id}
            </span>
            <span className={cn("flex shrink-0 items-center gap-1", DECISION_TEXT[peek.p.decision])}>
              <DecisionGlyph d={peek.p.decision} />
            </span>
          </div>
        </div>
      )}
    </section>
  );
}

/* ======================================================================== */
/* Tiles                                                                     */

const BAR: Record<Decision, string> = { AUTO_CLEAR: "bg-clear", FLAG: "bg-flag", ESCALATE: "bg-escalate" };
const HOVER: Record<Decision, string> = {
  AUTO_CLEAR: "hover:border-clear hover:shadow-[0_0_0_1px_rgb(var(--clear)),0_10px_24px_-14px_rgb(0_0_0/0.6)]",
  FLAG: "hover:border-flag hover:shadow-[0_0_0_1px_rgb(var(--flag)),0_10px_24px_-14px_rgb(0_0_0/0.6)]",
  ESCALATE: "hover:border-escalate hover:shadow-[0_0_0_1px_rgb(var(--escalate)),0_10px_24px_-14px_rgb(0_0_0/0.6)]",
};

function Tile({ p, lam, showTruth, onOpen }: { p: Placed; lam: number | null; showTruth: boolean; onOpen: (el: HTMLElement) => void }) {
  const { item, decision, held } = p;
  const missed = showTruth && item.defect && decision === "AUTO_CLEAR";
  const lamX = lam == null ? 1 : lam;
  const meta = DECISION_META[decision];
  const label = [
    item.site_id,
    `score ${item.score.toFixed(3)}`,
    meta.name,
    held ? "held by the shift check" : null,
    showTruth && item.defect ? (missed ? "true defect, missed" : "true defect") : null,
  ]
    .filter(Boolean)
    .join(", ");
  return (
    <li data-flip={item.relpath} className="min-w-0">
      <button
        type="button"
        onClick={(e) => onOpen(e.currentTarget)}
        aria-label={`${label}. Open details`}
        className={cn(
          "group relative block w-full overflow-hidden rounded-lg border bg-muted text-left transition-[border-color,box-shadow] duration-200",
          missed ? "border-defect shadow-[0_0_0_1px_rgb(var(--defect))]" : cn("border-border", HOVER[decision]),
        )}
      >
        <span className="relative block aspect-square overflow-hidden bg-sunken">
          <img
            src={item.url}
            alt=""
            loading="lazy"
            width={256}
            height={256}
            className="h-full w-full object-cover transition-transform duration-500 [transition-timing-function:cubic-bezier(.2,.7,.1,1)] group-hover:scale-[1.04]"
          />
          {/* score gauge on a dark backing chip so it reads on pale concrete */}
          <span className="absolute inset-x-2 top-2 block rounded-[4px] bg-[rgb(11_13_16/0.6)] px-1 py-[5px]" aria-hidden>
            <span className="relative block h-[3px] rounded-sm bg-white/25">
              <i className="absolute inset-y-0 left-0 rounded-sm bg-white/90" style={{ width: `${item.score * 100}%` }} />
              {lam != null && (
                <b
                  className="absolute -bottom-[3px] -top-[3px] w-[3px] -translate-x-1/2 rounded-[1px] bg-[#C8F31D] shadow-[0_0_0_1px_rgb(var(--tick-outline))] transition-[left] duration-500"
                  style={{ left: `${lamX * 100}%` }}
                />
              )}
            </span>
          </span>
          {held && (
            <span className="absolute left-2 top-[30px] inline-flex items-center gap-1 rounded-[3px] bg-shift px-1.5 py-px text-[10.5px] font-semibold text-shift-ink">
              <ShiftMark className="h-3 w-3" /> shift hold
            </span>
          )}
          {showTruth && item.defect && (
            <span
              className={cn(
                "absolute right-0 top-[30px] inline-flex max-w-[85%] items-center gap-1 truncate rounded-l-[3px] bg-defect py-0.5 pl-1.5 pr-[7px] text-[10.5px] font-semibold tracking-[0.04em] text-white dark:text-[#1a0b0e]",
              )}
            >
              <DefectMark className="h-2.5 w-2.5" />
              <span className="truncate">
                {missed ? "Missed · " : ""}Defect{item.classes.length ? ` · ${item.classes.join(", ")}` : ""}
              </span>
            </span>
          )}
          <span className={cn("absolute inset-x-0 bottom-0 h-[3px]", BAR[decision])} aria-hidden />
        </span>
        <span className="flex items-center gap-1.5 border-t border-border px-2.5 py-[7px]">
          <DecisionGlyph d={decision} />
          <span className="num text-[13.5px] font-medium">{item.score.toFixed(3)}</span>
          <span className="ml-auto min-w-0 truncate font-mono text-[11px] text-faint">{item.site_id}</span>
        </span>
      </button>
    </li>
  );
}

function NothingClears({ t, epsilon, top, nDefects }: { t: Threshold; epsilon: number; top: number; nDefects?: number }) {
  return (
    <li className="col-span-full grid grid-cols-[auto_1fr] items-start gap-x-3.5 gap-y-1 rounded-lg border border-dashed border-border-strong bg-muted p-4">
      <ShieldMinus className="row-span-2 h-7 w-7 text-accent" strokeWidth={1.5} aria-hidden />
      <b className="font-semibold">Nothing auto-clears at ε = {eps(epsilon)}</b>
      <p className="max-w-[70ch] text-[12.5px] text-subtle">
        {t.lambda_hat == null ? (
          <>
            With only {nDefects != null ? int(nDefects) : "the"} calibration defects, no finite threshold can certify ε = {eps(epsilon)}, so λ̂ = ∞. Every image goes to a
            human. Loosen ε to let clean-looking images clear.
          </>
        ) : (
          <>
            No image in this batch scores above λ̂ = <span className="num text-fg">{t.lambda_hat.toFixed(4)}</span> (the cleanest scores{" "}
            <span className="num text-fg">{top.toFixed(3)}</span>). Every image goes to a human, which is what a {eps(epsilon)} guarantee costs here.
          </>
        )}
      </p>
    </li>
  );
}

function LoadingState({ done, total, site, seed }: { done: number; total: number; site: string; seed: number }) {
  return (
    <div className="grid gap-4" aria-busy="true">
      <div className="rounded-xl border border-border bg-surface px-4 py-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div className="text-[13.5px] font-medium">Inspecting the batch…</div>
          <div className="num text-[12.5px] text-subtle" aria-live="polite">
            {done} / {total}
          </div>
        </div>
        <p className="mt-0.5 text-[12.5px] text-faint">
          Each image goes through <code className="font-mono">POST /v1/inspect</code> once ({site}, batch {seed}). This takes a few seconds.
        </p>
        <div className="mt-3 h-[3px] overflow-hidden rounded-sm bg-sunken" role="progressbar" aria-label="Images inspected" aria-valuemin={0} aria-valuemax={total} aria-valuenow={done}>
          <div className="h-full bg-signal transition-[width] duration-300" style={{ width: `${total ? Math.max(3, (100 * done) / total) : 3}%` }} />
        </div>
      </div>
      <Skeleton className="h-[88px] rounded-xl" />
      <Skeleton className="h-[230px] rounded-xl" />
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-[repeat(auto-fill,minmax(196px,1fr))]">
        {Array.from({ length: 10 }, (_, i) => (
          <Skeleton key={i} className="aspect-[1/1.17] rounded-lg" />
        ))}
      </div>
    </div>
  );
}

function BatchCurve({ curve, epsilon, showTruth }: { curve: { eps: string; epsilon: number; hrf: number; miss: number | null }[]; epsilon: number; showTruth: boolean }) {
  const colors = useChartColors();
  return (
    <Card>
      <CardHeader
        title="This batch across every ε"
        description="Human-review fraction and the batch's own miss rate (defects auto-cleared ÷ defects). One batch is a single draw; the guarantee bounds the average."
      />
      <CardBody>
        <div className="h-60">
          <ResponsiveContainer>
            <LineChart data={curve} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
              <CartesianGrid stroke={colors.grid} vertical={false} />
              <XAxis dataKey="eps" {...axisProps(colors)} axisLine={{ stroke: colors.grid }} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v)} {...axisProps(colors)} axisLine={false} />
              <Tooltip content={<ChartTooltip format={(v) => pct(v as number, 1)} labelFormat={(l) => `ε = ${l}`} />} />
              <Legend iconType="plainline" wrapperStyle={{ fontSize: 12 }} />
              <Line name="ε (the bound)" type="linear" dataKey="epsilon" stroke={colors.axis} strokeDasharray="4 4" dot={false} strokeWidth={1.5} />
              <Line name="Human-review fraction" type="monotone" dataKey="hrf" stroke={colors.signal} strokeWidth={2} dot={{ r: 3, fill: colors.signal, strokeWidth: 0 }} />
              {showTruth && <Line name="Batch miss rate" type="monotone" dataKey="miss" stroke={colors.defect} strokeWidth={2} dot={{ r: 3, fill: colors.defect, strokeWidth: 0 }} />}
              <ReferenceLine x={eps(epsilon)} stroke={colors.signal} strokeOpacity={0.35} strokeWidth={6} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardBody>
    </Card>
  );
}

/* ======================================================================== */
/* Drawer                                                                    */

function Blk({ title, meta, children }: { title: ReactNode; meta?: ReactNode; children: ReactNode }) {
  return (
    <section>
      <h3 className="mb-2 flex justify-between gap-3 text-[11.5px] font-semibold uppercase tracking-[0.09em] text-subtle">
        {title}
        {meta && <span className="font-mono font-normal normal-case tracking-normal text-faint">{meta}</span>}
      </h3>
      {children}
    </section>
  );
}

function ImageDetail({ p, t, epsilon, showTruth }: { p: Placed; t: Threshold; epsilon: number; showTruth: boolean }) {
  const { item, decision: d, held } = p;
  const set = predictionSet(item.p_defect, t.q_hat);
  const inClean = set.includes("clean");
  const inDefect = set.includes("defect");
  const lam = t.lambda_hat;
  const lamTxt = lam == null ? "∞" : lam.toFixed(4);
  const missed = item.defect && d === "AUTO_CLEAR";
  const reason =
    d === "AUTO_CLEAR"
      ? `score ${item.score.toFixed(3)} > λ̂ ${lamTxt} and in distribution`
      : d === "FLAG"
        ? `prediction set at ε ${eps(epsilon)} is {defect} only`
        : held
          ? "would clear, but held back by the shift check"
          : inClean && inDefect
            ? "prediction set is {clean, defect}: ambiguous"
            : inClean
              ? `below λ̂ ${lamTxt}; prediction set is {clean} but not confident enough to clear`
              : `below λ̂ ${lamTxt}; the prediction set is empty`;
  const title = d === "AUTO_CLEAR" ? "Auto-cleared" : d === "FLAG" ? "Flag: likely defect" : "Escalate: uncertain";
  return (
    <div className="grid content-start gap-4 p-4">
      <div className="relative overflow-hidden rounded-lg border border-border bg-sunken">
        <img src={item.url} alt={`${clusterLabel(item.result.cluster)} surface photo from ${item.site_id}, inspection ${item.result.inspection_id}`} className="aspect-square w-full object-cover" />
        {showTruth && item.defect && (
          <span className="absolute right-0 top-3 inline-flex items-center gap-1 rounded-l-[3px] bg-defect py-0.5 pl-1.5 pr-2 text-[11px] font-semibold text-white dark:text-[#1a0b0e]">
            <DefectMark className="h-2.5 w-2.5" />
            Ground truth: defect{item.classes.length ? ` · ${item.classes.join(", ")}` : ""}
          </span>
        )}
      </div>

      <div className={cn("flex items-center gap-2.5 rounded-lg border px-3 py-2.5", DECISION_SOFT[d], DECISION_BORDER[d])}>
        <DecisionGlyph d={d} className="h-[18px] w-[18px]" />
        <div className="min-w-0">
          <b className={cn("text-[13px] font-semibold uppercase tracking-[0.04em]", DECISION_TEXT[d])}>{title}</b>
          <div className="text-[12.5px] text-subtle">{reason}</div>
        </div>
      </div>

      <dl className="grid grid-cols-3 rounded-lg border border-border">
        {[
          ["score", item.score.toFixed(3)],
          ["p(defect)", item.p_defect.toFixed(3)],
          ["site", item.site_id],
        ].map(([k, v]) => (
          <div key={k} className="min-w-0 border-r border-border px-3 py-2.5 last:border-r-0">
            <dt className="text-[11.5px] text-faint">{k}</dt>
            <dd className="num mt-0.5 truncate text-[13.5px] sm:text-[15px]">{v}</dd>
          </div>
        ))}
      </dl>

      <Blk title="Score vs threshold" meta={`ε ${eps(epsilon)}`}>
        <ScoreScale score={item.score} lam={lam} q={t.q_hat} decision={d} />
      </Blk>

      <Blk title="Prediction set" meta={<>q̂ {t.q_hat.toFixed(4)}</>}>
        <div className="flex items-center gap-2 font-mono text-[13.5px]" aria-label={`Prediction set: {${set.join(", ")}}`}>
          <span className="text-[18px] text-faint" aria-hidden>{"{"}</span>
          {(["clean", "defect"] as const).map((c) => {
            const on = set.includes(c);
            return (
              <span
                key={c}
                className={cn(
                  "rounded-[3px] border px-2.5 py-[3px]",
                  on ? "border-subtle bg-sunken text-fg" : "border-border-strong text-faint line-through decoration-faint/60",
                )}
              >
                {c}
              </span>
            );
          })}
          <span className="text-[18px] text-faint" aria-hidden>{"}"}</span>
        </div>
        <p className="mt-1.5 text-[11.5px] text-faint">
          A label is in the set when its nonconformity is at most q̂: clean if p(defect) ≤ q̂, defect if 1 − p(defect) ≤ q̂.
        </p>
      </Blk>

      <Blk title="Distribution shift" meta={item.result.shift.flagged ? "shifted" : "in distribution"}>
        <ShiftMeter similarity={item.result.shift.nearest_similarity} threshold={item.result.shift.threshold} flagged={item.result.shift.flagged} />
      </Blk>

      {showTruth && (
        <div className={cn("flex items-center gap-2 rounded-lg border px-3 py-2.5 text-[12.5px]", item.defect ? "border-defect/50 bg-defect-soft" : "border-border")}>
          {item.defect ? <DefectMark className="text-defect" /> : <DecisionGlyph d="AUTO_CLEAR" />}
          <span>
            Ground truth: <b className="font-semibold">{item.defect ? `defect${item.classes.length ? ` (${item.classes.join(", ")})` : ""}` : "no defect"}</b>
            {missed && ". This one was auto-cleared: a miss counted against ε."}
          </span>
        </div>
      )}

      <dl className="divide-y divide-border border-y border-border">
        <KV k="Most likely type" v={item.result.defect_type ?? "—"} />
        <KV k="Cluster" v={clusterLabel(item.result.cluster)} mono={false} />
        <KV k="Service decision at ε = 5%" v={item.result.decision} />
        <KV k="File" v={<span title={item.relpath}>{item.relpath}</span>} />
      </dl>

      <Blk title="Record a human label">
        <FeedbackForm inspectionId={item.result.inspection_id} suggestedType={item.classes[0]} />
      </Blk>
    </div>
  );
}
