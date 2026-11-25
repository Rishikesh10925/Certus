import { useMutation } from "@tanstack/react-query";
import { Dices, History, ImagePlus, ScanSearch, Timer, Trash2, UploadCloud, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { cn } from "@/lib/cn";
import { clusterLabel, eps, fixed, lambda, ms } from "@/lib/format";
import { useInfo } from "@/lib/queries";
import { useSettings } from "@/lib/settings";
import type { InspectResult, SampleItem } from "@/lib/types";
import { Badge, Button, Card, CardBody, CardHeader, Field, Input, PageHeader } from "@/components/ui/primitives";
import { Select, Tip } from "@/components/ui/radix";
import { DECISION_EXPLAIN, DECISION_ICON, DECISION_TONE, DecisionBadge, FeedbackForm, KV, RiskStatement, ScoreScale, ShiftMeter } from "@/components/domain";
import { EmptyState, ErrorState } from "@/components/States";
import { decisionLabel } from "@/lib/format";

const MAX_BYTES = 20 * 1024 * 1024;
const ACCEPT = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff", "image/gif"];

interface Picked {
  file: Blob;
  name: string;
  url: string;
  truth?: SampleItem; // set when the image is a held-out sample with a known label
}

interface HistoryEntry {
  result: InspectResult;
  thumb: string;
  name: string;
  truth?: SampleItem;
  at: number;
}

/** Which calibration cluster the service will use for a site id (calibration.cluster_for). */
function clusterFor(siteId: string, clusters: string[]) {
  const prefix = siteId.split("-")[0].toLowerCase();
  return clusters.includes(prefix) ? prefix : "pooled";
}

export default function InspectPage() {
  const { settings } = useSettings();
  const info = useInfo();
  const [picked, setPicked] = useState<Picked | null>(null);
  const [siteId, setSiteId] = useState("wall-7001");
  const [epsilon, setEpsilon] = useState(settings.epsilon);
  const [fileError, setFileError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [active, setActive] = useState<HistoryEntry | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const epsilons = info.data?.epsilons ?? [];
  useEffect(() => {
    if (epsilons.length && !epsilons.includes(epsilon)) setEpsilon(epsilons.includes(0.05) ? 0.05 : epsilons[0]);
  }, [epsilons, epsilon]);

  const siteError = !siteId.trim() ? "A site id is required." : siteId.length > 120 ? "Too long." : null;
  const cluster = info.data ? clusterFor(siteId.trim(), info.data.clusters) : null;

  const accept = useCallback((file: File | Blob, name: string, truth?: SampleItem) => {
    setFileError(null);
    if (file.type && !ACCEPT.includes(file.type)) {
      setFileError(`Unsupported file type ${file.type}. Use JPEG, PNG, WebP, BMP, TIFF or GIF.`);
      return;
    }
    if (file.size > MAX_BYTES) {
      setFileError(`The file is ${(file.size / 1e6).toFixed(1)} MB; the limit is 20 MB.`);
      return;
    }
    if (file.size === 0) {
      setFileError("The file is empty.");
      return;
    }
    setPicked((old) => {
      if (old && !history.some((h) => h.thumb === old.url)) URL.revokeObjectURL(old.url);
      return { file, name, url: URL.createObjectURL(file), truth };
    });
  }, [history]);

  const sample = useMutation({
    mutationFn: async () => {
      const type = ["wall", "deck", "pavement"][Math.floor(Math.random() * 3)];
      const batch = await api.samples(type, Math.floor(Math.random() * 100_000), 1);
      const it = batch.items[0];
      return { it, blob: await api.sampleImage(it.relpath) };
    },
    onSuccess: ({ it, blob }) => {
      accept(blob, it.relpath.split("/").pop() ?? "sample.jpg", it);
      setSiteId(it.site_id);
    },
  });

  const inspect = useMutation({
    mutationFn: () => api.inspect(picked!.file, picked!.name, siteId.trim(), epsilon),
    onSuccess: (result) => {
      const entry = { result, thumb: picked!.url, name: picked!.name, truth: picked!.truth, at: Date.now() };
      setHistory((h) => [entry, ...h].slice(0, 30));
      setActive(entry);
    },
  });

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) accept(f, f.name);
  };

  return (
    <>
      <PageHeader
        eyebrow="Operate"
        title="Inspect an image"
        description="Send one image through the triage service. You get the decision, the score against the calibrated threshold, the split-CP prediction set, the shift check and the plain-English risk statement."
      />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
        {/* Left: input */}
        <div className="space-y-4">
          <Card>
            <CardBody className="space-y-4">
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
                role="button"
                tabIndex={0}
                aria-label="Choose or drop an image"
                className={cn(
                  "group relative grid aspect-[4/3] cursor-pointer place-items-center overflow-hidden rounded-xl border-2 border-dashed transition-colors",
                  dragging ? "border-accent bg-accent/5" : "border-border hover:border-accent/50 hover:bg-muted/50",
                )}
              >
                {picked ? (
                  <>
                    <img src={picked.url} alt="Selected" className="h-full w-full object-contain" />
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setPicked(null);
                      }}
                      className="absolute right-2 top-2 rounded-md bg-black/60 p-1 text-white opacity-0 transition-opacity group-hover:opacity-100"
                      aria-label="Remove image"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </>
                ) : (
                  <div className="px-6 text-center">
                    <UploadCloud className="mx-auto h-8 w-8 text-subtle transition-transform group-hover:-translate-y-0.5" />
                    <div className="mt-3 text-[14px] font-medium">Drop an image or click to browse</div>
                    <div className="mt-1 text-[12px] text-subtle">JPEG, PNG, WebP, BMP, TIFF · up to 20 MB</div>
                  </div>
                )}
                <input
                  ref={inputRef}
                  type="file"
                  accept={ACCEPT.join(",")}
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) accept(f, f.name);
                    e.target.value = "";
                  }}
                />
              </div>
              {picked && (
                <div className="flex items-center justify-between gap-2 text-[12px] text-subtle">
                  <span className="truncate">{picked.name}</span>
                  {picked.truth && <Badge tone="accent">held-out sample</Badge>}
                </div>
              )}
              {fileError && <ErrorState compact error={new Error(fileError)} />}
              {info.data?.samples_available && (
                <Button variant="outline" size="sm" className="w-full" loading={sample.isPending} onClick={() => sample.mutate()}>
                  <Dices className="h-3.5 w-3.5" /> Use a random held-out test image
                </Button>
              )}
              {sample.isError && <ErrorState compact error={sample.error} />}

              <Field
                label="Site id"
                htmlFor="site"
                error={siteError}
                hint={cluster && <>Calibrated against the <span className="font-medium text-fg">{clusterLabel(cluster)}</span> cluster{cluster === "pooled" && " (unknown site type)"}.</>}
              >
                <Input id="site" value={siteId} onChange={(e) => setSiteId(e.target.value)} aria-invalid={!!siteError} placeholder="wall-7001" />
              </Field>
              <div className="flex flex-wrap gap-1.5">
                {["wall-7001", "deck-7001", "pavement-7001", "facade-1"].map((s) => (
                  <button key={s} onClick={() => setSiteId(s)} className={cn("rounded-md px-2 py-0.5 font-mono text-[11px] ring-1 ring-inset transition-colors", siteId === s ? "bg-accent/10 text-accent ring-accent/30" : "text-subtle ring-border hover:text-fg")}>
                    {s}
                  </button>
                ))}
              </div>
              <Field label="Guaranteed maximum missed-defect rate (ε)" htmlFor="eps">
                <Select id="eps" value={String(epsilon)} onValueChange={(v) => setEpsilon(Number(v))} options={epsilons.map((e) => ({ value: String(e), label: eps(e) }))} />
              </Field>
              <Button size="lg" className="w-full" disabled={!picked || !!siteError || !info.data} loading={inspect.isPending} onClick={() => inspect.mutate()}>
                <ScanSearch className="h-4 w-4" /> Inspect
              </Button>
              {inspect.isError && <ErrorState compact error={inspect.error} />}
            </CardBody>
          </Card>

          {history.length > 0 && (
            <Card>
              <CardHeader
                title={<span className="flex items-center gap-2"><History className="h-4 w-4 text-subtle" /> This session</span>}
                action={
                  <Button variant="ghost" size="sm" onClick={() => { setHistory([]); setActive(null); }}>
                    <Trash2 className="h-3.5 w-3.5" /> Clear
                  </Button>
                }
              />
              <CardBody className="space-y-1 pt-3">
                {history.map((h) => (
                  <button key={h.result.inspection_id} onClick={() => setActive(h)} className={cn("flex w-full items-center gap-3 rounded-lg p-1.5 text-left transition-colors", active?.result.inspection_id === h.result.inspection_id ? "bg-muted" : "hover:bg-muted/60")}>
                    <img src={h.thumb} alt="" className="h-10 w-10 rounded-md object-cover ring-1 ring-border" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[13px] font-medium">{h.name}</div>
                      <div className="num text-[11px] text-subtle">#{h.result.inspection_id} · {h.result.site_id} · ε {eps(h.result.epsilon)}</div>
                    </div>
                    <DecisionBadge decision={h.result.decision} />
                  </button>
                ))}
              </CardBody>
            </Card>
          )}
        </div>

        {/* Right: result */}
        <div>
          {inspect.isPending ? (
            <Card className="grid min-h-[420px] place-items-center">
              <div className="text-center">
                <ScanSearch className="mx-auto h-8 w-8 animate-pulse text-accent" />
                <div className="mt-3 text-[14px] font-medium">Embedding and scoring…</div>
                <div className="mt-1 text-[12px] text-subtle">CLIP ViT-B/32 zero-shot (S1)</div>
              </div>
            </Card>
          ) : active ? (
            <ResultPanel key={active.result.inspection_id} entry={active} />
          ) : (
            <Card className="min-h-[420px]">
              <EmptyState icon={<ImagePlus className="h-5 w-5" />} title="No inspection yet">
                Choose an image, set the site id and ε, then press Inspect. Each inspection is logged by the service and can receive a human label.
              </EmptyState>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}

function ResultPanel({ entry }: { entry: HistoryEntry }) {
  const r = entry.result;
  const Icon = DECISION_ICON[r.decision];
  const tone = DECISION_TONE[r.decision];
  const toneBg = { clear: "from-clear/15 text-clear", flag: "from-flag/15 text-flag", escalate: "from-escalate/15 text-escalate" }[tone];
  const timings = Object.entries(r.timings_ms).filter(([k]) => k !== "total");
  const total = r.timings_ms.total ?? timings.reduce((a, [, v]) => a + v, 0);
  return (
    <div className="space-y-4 animate-fade-in">
      <Card className="overflow-hidden">
        <div className={cn("bg-gradient-to-br to-transparent p-5", toneBg)}>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="grid h-11 w-11 place-items-center rounded-xl bg-surface shadow-sm ring-1 ring-border">
                <Icon className="h-5 w-5" />
              </div>
              <div>
                <div className="text-[22px] font-semibold tracking-tight">{decisionLabel[r.decision]}</div>
                <div className="text-[13px] text-subtle">
                  {r.decision === "FLAG" && r.defect_type ? `Likely ${r.defect_type} (pre-filled for the reviewer)` : DECISION_EXPLAIN[r.decision].split(":")[0]}
                </div>
              </div>
            </div>
            <div className="num text-right text-[12px] text-subtle">
              <div>inspection #{r.inspection_id}</div>
              <div>{r.site_id} → {clusterLabel(r.cluster)} · ε {eps(r.epsilon)}</div>
            </div>
          </div>
        </div>
        <CardBody className="space-y-5">
          <RiskStatement text={r.risk_statement} />
          <div>
            <div className="mb-3 flex items-center justify-between">
              <span className="label">Score vs threshold</span>
              <Tip content="Auto-clear iff s > λ̂ (conformal risk control). λ̂ = ∞ means this ε clears nothing.">
                <span className="cursor-help text-[12px] text-subtle underline decoration-dotted">how is this decided?</span>
              </Tip>
            </div>
            <ScoreScale score={r.score} lam={r.lambda_hat} />
          </div>
        </CardBody>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="Model output" />
          <CardBody>
            <dl className="divide-y divide-border">
              <KV k="Score s = 1 − p(defect)" v={fixed(r.score)} />
              <KV k="p(defect)" v={fixed(r.p_defect)} />
              <KV k="λ̂ at this ε" v={lambda(r.lambda_hat)} />
              <KV k="Split-CP prediction set" v={<span className="flex justify-end gap-1">{r.prediction_set.length ? r.prediction_set.map((c) => <Badge key={c} tone={c === "defect" ? "defect" : "clear"}>{c}</Badge>) : <Badge>empty</Badge>}</span>} />
              <KV k="Defect type" v={r.defect_type ?? "—"} />
              {entry.truth && <KV k="Ground truth (held-out)" v={<Badge tone={entry.truth.defect ? "defect" : "clear"}>{entry.truth.defect ? `defect${entry.truth.classes.length ? ` · ${entry.truth.classes.join(", ")}` : ""}` : "no defect"}</Badge>} />}
            </dl>
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Shift check" description="Nearest-neighbour similarity to the cluster's calibration images." />
          <CardBody>
            <ShiftMeter similarity={r.shift.nearest_similarity} threshold={r.shift.threshold} flagged={r.shift.flagged} />
            <p className="mt-4 text-[12px] text-subtle">
              Below the threshold (the 1st percentile of the calibration set's own similarities) the image is escalated, never cleared:
              leave-one-site-out showed the guarantee does not hold on such images.
            </p>
          </CardBody>
        </Card>
        <Card>
          <CardHeader title={<span className="flex items-center gap-2"><Timer className="h-4 w-4 text-subtle" /> Server timings</span>} description={`${ms(total)} total`} />
          <CardBody className="space-y-2.5">
            {timings.map(([k, v]) => (
              <div key={k}>
                <div className="flex justify-between text-[12px]"><span className="capitalize text-subtle">{k}</span><span className="num">{ms(v)}</span></div>
                <div className="mt-1 h-1.5 rounded-full bg-muted"><div className="h-full rounded-full bg-accent/70" style={{ width: `${Math.max(1, (100 * v) / (total || 1))}%` }} /></div>
              </div>
            ))}
            <dl className="mt-3 divide-y divide-border border-t border-border pt-1">
              <KV k="Model" v={r.model_version} />
              <KV k="Calibration" v={r.calibration_version} />
            </dl>
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Human label" description="POST /v1/feedback. Stored for the weekly recalibration; it never changes thresholds live." />
          <CardBody>
            <FeedbackForm inspectionId={r.inspection_id} suggestedType={r.defect_type ?? entry.truth?.classes[0]} />
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
