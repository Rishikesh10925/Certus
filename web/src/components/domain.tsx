import { useMutation } from "@tanstack/react-query";
import { AlertOctagon, CheckCircle2, Flag, ShieldAlert, ShieldCheck, Siren } from "lucide-react";
import { useState, type ReactNode } from "react";
import { api } from "@/lib/api";
import { cn } from "@/lib/cn";
import { decisionLabel, fixed, lambda } from "@/lib/format";
import type { Decision, FeedbackResult } from "@/lib/types";
import { Badge, Button } from "./ui/primitives";
import { Select } from "./ui/radix";
import { ErrorState } from "./States";

export const DECISION_TONE = { AUTO_CLEAR: "clear", FLAG: "flag", ESCALATE: "escalate" } as const;
export const DECISION_ICON = { AUTO_CLEAR: ShieldCheck, FLAG: Flag, ESCALATE: Siren } as const;

export function DecisionBadge({ decision, className }: { decision: Decision; className?: string }) {
  const Icon = DECISION_ICON[decision];
  return (
    <Badge tone={DECISION_TONE[decision]} className={className}>
      <Icon className="h-3 w-3" />
      {decisionLabel[decision]}
    </Badge>
  );
}

export const DECISION_EXPLAIN: Record<Decision, string> = {
  AUTO_CLEAR: "Score above λ̂ and inside the calibration distribution: skips human review.",
  FLAG: "Sent to a human with the defect type pre-filled (the split-CP set is {defect}).",
  ESCALATE: "Sent to a human with no pre-filled type (set is empty, {clean} or both, or the image failed the shift check).",
};

/**
 * Score s = 1 − p(defect) on [0, 1]. The auto-clear region is s > λ̂.
 * λ̂ = null (infinity) means nothing clears at this ε.
 */
export function ScoreScale({ score, lam, compact }: { score: number; lam: number | null; compact?: boolean }) {
  const lamPos = lam == null ? 100 : Math.min(100, Math.max(0, lam * 100));
  const sPos = Math.min(100, Math.max(0, score * 100));
  const above = lam != null && score > lam;
  return (
    <div>
      <div className={cn("relative rounded-full bg-muted ring-1 ring-inset ring-border", compact ? "h-2" : "h-3")}>
        <div
          className="absolute inset-y-0 right-0 rounded-r-full bg-clear/25"
          style={{ left: `${lamPos}%` }}
          aria-hidden
        />
        <div className="absolute inset-y-[-4px] w-0.5 rounded bg-fg/70" style={{ left: `${lamPos}%` }} aria-hidden />
        <div
          className={cn(
            "absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-surface shadow transition-all duration-300",
            above ? "bg-clear" : "bg-escalate",
          )}
          style={{ left: `${sPos}%` }}
          aria-hidden
        />
      </div>
      {!compact && (
        <div className="mt-2 flex justify-between text-[11px] text-subtle">
          <span>0 · defect-like</span>
          <span className="num">
            s = {fixed(score)} {above ? ">" : "≤"} λ̂ = {lambda(lam)}
          </span>
          <span>clean-like · 1</span>
        </div>
      )}
    </div>
  );
}

/** Nearest-neighbour cosine similarity to the cluster's calibration set vs the 1st-percentile threshold. */
export function ShiftMeter({ similarity, threshold, flagged }: { similarity: number; threshold: number; flagged: boolean }) {
  // Show the informative range around the threshold.
  const lo = Math.min(similarity, threshold) - 0.05;
  const hi = 1;
  const pos = (v: number) => `${Math.min(100, Math.max(0, ((v - lo) / (hi - lo)) * 100))}%`;
  return (
    <div>
      <div className="flex items-center justify-between text-[13px]">
        <span className="flex items-center gap-1.5 font-medium">
          {flagged ? <ShieldAlert className="h-4 w-4 text-shift" /> : <CheckCircle2 className="h-4 w-4 text-clear" />}
          {flagged ? "Outside the calibration distribution" : "Within the calibration distribution"}
        </span>
      </div>
      <div className="relative mt-3 h-2 rounded-full bg-muted ring-1 ring-inset ring-border">
        <div className="absolute inset-y-0 left-0 rounded-l-full bg-shift/20" style={{ width: pos(threshold) }} />
        <div className="absolute inset-y-[-4px] w-0.5 bg-shift" style={{ left: pos(threshold) }} />
        <div
          className={cn("absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-surface shadow", flagged ? "bg-shift" : "bg-clear")}
          style={{ left: pos(similarity) }}
        />
      </div>
      <div className="num mt-2 flex justify-between text-[11px] text-subtle">
        <span>similarity {fixed(similarity)}</span>
        <span>threshold {fixed(threshold)}</span>
      </div>
    </div>
  );
}

export function KV({ k, v, mono = true }: { k: ReactNode; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5 text-[13px]">
      <dt className="text-subtle">{k}</dt>
      <dd className={cn("min-w-0 truncate text-right", mono && "num")}>{v}</dd>
    </div>
  );
}

export function RiskStatement({ text, className }: { text: ReactNode; className?: string }) {
  return (
    <blockquote className={cn("rounded-lg border-l-2 border-accent bg-accent/5 px-4 py-3 text-[13.5px] leading-relaxed", className)}>
      {text}
    </blockquote>
  );
}

export const DEFECT_TYPES = ["crack", "vegetation", "corrosion", "dampness", "other"];

/** POST /v1/feedback: a human's ground truth. Stored; never recalibrates. */
export function FeedbackForm({
  inspectionId,
  suggestedType,
  onDone,
  submitLabel = "Submit label",
}: {
  inspectionId: number;
  suggestedType?: string | null;
  onDone?: (r: FeedbackResult) => void;
  submitLabel?: string;
}) {
  const [defect, setDefect] = useState<boolean | null>(null);
  const [type, setType] = useState(suggestedType ?? "crack");
  const m = useMutation({
    mutationFn: () => api.feedback(inspectionId, defect!, defect ? type : null),
    onSuccess: (r) => onDone?.(r),
  });

  if (m.isSuccess) {
    return (
      <div className="space-y-2 animate-fade-in">
        <div className="flex items-center gap-2 text-[13px] font-medium text-clear">
          <CheckCircle2 className="h-4 w-4" /> Label recorded for inspection #{inspectionId}
        </div>
        {m.data.missed_defect && (
          <div className="flex items-start gap-2 rounded-lg border border-defect/30 bg-defect/5 p-3 text-[13px] text-defect">
            <AlertOctagon className="mt-0.5 h-4 w-4 shrink-0" />
            This image was auto-cleared but is defective: a missed defect. It is counted in the
            service's missed-defect metric.
          </div>
        )}
        <p className="text-[12px] text-subtle">Recalibration: {m.data.recalibration}.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2">
        {[
          { v: true, label: "Defect present" },
          { v: false, label: "No defect" },
        ].map((o) => (
          <button
            key={String(o.v)}
            onClick={() => setDefect(o.v)}
            aria-pressed={defect === o.v}
            className={cn(
              "h-10 rounded-lg border text-[13px] font-medium transition-all",
              defect === o.v
                ? o.v
                  ? "border-defect bg-defect/10 text-defect"
                  : "border-clear bg-clear/10 text-clear"
                : "border-border bg-surface text-subtle hover:text-fg",
            )}
          >
            {o.label}
          </button>
        ))}
      </div>
      {defect && (
        <Select
          ariaLabel="Defect type"
          value={type}
          onValueChange={setType}
          options={DEFECT_TYPES.map((t) => ({ value: t, label: t.charAt(0).toUpperCase() + t.slice(1) }))}
        />
      )}
      {m.isError && <ErrorState compact error={m.error} />}
      <Button className="w-full" disabled={defect === null} loading={m.isPending} onClick={() => m.mutate()}>
        {submitLabel}
      </Button>
    </div>
  );
}
