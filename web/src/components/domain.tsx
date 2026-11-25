import { useMutation } from "@tanstack/react-query";
import { AlertOctagon, CheckCircle2, ShieldAlert } from "lucide-react";
import { useState, type ReactNode } from "react";
import { api } from "@/lib/api";
import { cn } from "@/lib/cn";
import { decisionLabel, fixed, lambda } from "@/lib/format";
import type { Decision, FeedbackResult } from "@/lib/types";
import { Badge, Button } from "./ui/primitives";
import { Select } from "./ui/radix";
import { ErrorState } from "./States";

export const DECISION_TONE = { AUTO_CLEAR: "clear", FLAG: "flag", ESCALATE: "escalate" } as const;

type IconProps = { className?: string };

/* Decision marks: a check, a warning triangle and a question diamond, so the
   three outcomes differ in shape as well as colour. */
export function ClearMark({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" className={cn("h-4 w-4 shrink-0", className)} aria-hidden>
      <path d="M5 12.5l4.2 4.2L19 7" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
export function FlagMark({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" className={cn("h-4 w-4 shrink-0", className)} aria-hidden>
      <path d="M12 3.5L22 20.5H2z" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinejoin="round" />
      <path d="M12 10v4.6" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" />
      <circle cx="12" cy="17.4" r="1.4" fill="currentColor" />
    </svg>
  );
}
export function EscalateMark({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" className={cn("h-4 w-4 shrink-0", className)} aria-hidden>
      <path d="M12 2.5l9.5 9.5-9.5 9.5L2.5 12z" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinejoin="round" />
      <path d="M9.6 9.6a2.5 2.5 0 1 1 3.4 2.3c-.7.3-1 .8-1 1.5v.4" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
      <circle cx="12" cy="16.6" r="1.2" fill="currentColor" />
    </svg>
  );
}
export function ShiftMark({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" className={cn("h-4 w-4 shrink-0", className)} aria-hidden>
      <path d="M4 8h12l-3-3M20 16H8l3 3" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export const DECISION_ICON = { AUTO_CLEAR: ClearMark, FLAG: FlagMark, ESCALATE: EscalateMark } as const;

/** Short names and plain-English qualifiers used on tiles, legends and the drawer. */
export const DECISION_META: Record<Decision, { short: string; name: string; qualifier: string | null }> = {
  AUTO_CLEAR: { short: "Clear", name: "Auto-clear", qualifier: null },
  FLAG: { short: "Flag", name: "Flag", qualifier: "likely defect" },
  ESCALATE: { short: "Escalate", name: "Escalate", qualifier: "uncertain" },
};

/** Solid fills for chips and swatches (text uses the matching *-ink token). */
export const DECISION_SOLID: Record<Decision, string> = {
  AUTO_CLEAR: "bg-clear text-clear-ink",
  FLAG: "bg-flag text-flag-ink",
  ESCALATE: "bg-escalate text-escalate-ink",
};
export const DECISION_SOFT: Record<Decision, string> = {
  AUTO_CLEAR: "bg-clear-soft",
  FLAG: "bg-flag-soft",
  ESCALATE: "bg-escalate-soft",
};

export function DecisionBadge({
  decision,
  className,
  solid,
}: {
  decision: Decision;
  className?: string;
  solid?: boolean;
}) {
  const Icon = DECISION_ICON[decision];
  if (solid) {
    return (
      <span className={cn("inline-flex items-center gap-1 whitespace-nowrap rounded-md px-1.5 py-0.5 text-[11.5px] font-bold", DECISION_SOLID[decision], className)}>
        <Icon className="h-3 w-3" />
        {decisionLabel[decision]}
      </span>
    );
  }
  return (
    <Badge tone={DECISION_TONE[decision]} className={className}>
      <Icon className="h-3 w-3" />
      {decisionLabel[decision]}
    </Badge>
  );
}

/** A filled square swatch with the decision mark, for legends and breakdowns. */
export function DecisionSwatch({ decision, className }: { decision: Decision; className?: string }) {
  const Icon = DECISION_ICON[decision];
  return (
    <span className={cn("grid h-[22px] w-[22px] shrink-0 place-items-center rounded-md", DECISION_SOLID[decision], className)} aria-hidden>
      <Icon className="h-[13px] w-[13px]" />
    </span>
  );
}

export function DecisionLegend({ className }: { className?: string }) {
  return (
    <ul className={cn("flex flex-wrap gap-x-4 gap-y-2 text-[13px] text-subtle", className)} aria-label="Decision legend">
      {(["AUTO_CLEAR", "FLAG", "ESCALATE"] as const).map((d) => (
        <li key={d} className="flex items-center gap-1.5">
          <DecisionSwatch decision={d} className="h-[18px] w-[18px]" />
          {DECISION_META[d].name}
          {DECISION_META[d].qualifier && <span className="text-faint">· {DECISION_META[d].qualifier}</span>}
        </li>
      ))}
    </ul>
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
      <div
        className={cn("relative mx-1.5", compact ? "h-5" : "h-[30px]")}
        role="img"
        aria-label={`Score ${fixed(score)} on a 0 to 1 scale; auto-clear threshold λ̂ ${lambda(lam)}${above ? ", above the threshold" : ", not above the threshold"}`}
      >
        <div className={cn("absolute inset-x-0 overflow-hidden rounded-full bg-sunken", compact ? "top-1.5 h-2" : "top-2.5 h-2.5")}>
          <div className="hatch-clear-zone absolute inset-y-0 right-0" style={{ left: `${lamPos}%` }} />
        </div>
        {lam != null && (
          <div className="absolute inset-y-0.5 w-[3px] -translate-x-1/2 rounded-sm bg-accent" style={{ left: `${lamPos}%` }} aria-hidden />
        )}
        <div
          className={cn(
            "absolute -translate-x-1/2 rounded-full border-[3px] border-surface bg-fg shadow-[0_0_0_1px_rgb(var(--fg))] transition-[left] duration-300",
            compact ? "top-[3px] h-3.5 w-3.5" : "top-[7px] h-4 w-4",
          )}
          style={{ left: `${sPos}%` }}
          aria-hidden
        />
      </div>
      {!compact && (
        <>
          <div className="mx-1.5 mt-0.5 flex justify-between gap-2 font-mono text-[11.5px] text-faint">
            <span>0 · looks defective</span>
            <span>clean · 1</span>
          </div>
          <div className="num mx-1.5 mt-1 text-[12px] text-subtle">
            s = <span className="font-semibold text-fg">{fixed(score)}</span> {above ? ">" : "≤"} λ̂ = <span className="font-semibold text-fg">{lambda(lam)}</span>
            {lam == null ? " (nothing clears)" : <span className="text-faint"> · hatched = auto-clear zone</span>}
          </div>
        </>
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
      <div className="flex items-center gap-1.5 text-[13px] font-semibold">
        {flagged ? <ShieldAlert className="h-4 w-4 text-shift" /> : <CheckCircle2 className="h-4 w-4 text-clear" />}
        {flagged ? "Outside the calibration distribution" : "Within the calibration distribution"}
      </div>
      <div
        className="relative mt-3 h-2.5 rounded-full bg-sunken"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={Number(similarity.toFixed(3))}
        aria-label={`Nearest calibration similarity ${fixed(similarity)}, threshold ${fixed(threshold)}`}
      >
        <div className={cn("absolute inset-y-0 left-0 rounded-full", flagged ? "bg-shift" : "bg-accent")} style={{ width: pos(similarity) }} />
        <div className="absolute -inset-y-1.5 w-0.5 bg-fg" style={{ left: pos(threshold) }} />
      </div>
      <div className="num mt-2 flex flex-wrap justify-between gap-x-3 text-[11.5px] text-faint">
        <span>similarity <span className="font-semibold text-fg">{fixed(similarity)}</span></span>
        <span>threshold {fixed(threshold)}</span>
      </div>
    </div>
  );
}

export function KV({ k, v, mono = true }: { k: ReactNode; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2 text-[13px]">
      <dt className="shrink-0 text-subtle">{k}</dt>
      <dd className={cn("min-w-0 truncate text-right font-medium", mono && "num")}>{v}</dd>
    </div>
  );
}

export function RiskStatement({ text, className }: { text: ReactNode; className?: string }) {
  return (
    <blockquote className={cn("rounded-r-md border-l-[3px] border-accent bg-accent-soft px-4 py-3 text-[14px] leading-relaxed", className)}>
      {text}
    </blockquote>
  );
}

/** The guarantee, typeset: 𝔼[ P(auto-clear | defect) ] ≤ ε. */
export function GuaranteeFormula({ className }: { className?: string }) {
  return (
    <span
      className={cn("inline-flex flex-wrap items-baseline gap-x-1 text-[15px]", className)}
      role="math"
      aria-label="The expected probability of auto-clearing a defect is at most epsilon"
    >
      <span className="sym not-italic" aria-hidden>𝔼[</span>
      <span className="sym" aria-hidden>P</span>
      <span className="sym not-italic" aria-hidden>(</span>
      <span aria-hidden>auto-clear</span>
      <span className="sym not-italic" aria-hidden>|</span>
      <span aria-hidden>defect</span>
      <span className="sym not-italic" aria-hidden>)]</span>
      <span className="sym not-italic px-1" aria-hidden>≤</span>
      <span className="sym" aria-hidden>ε</span>
    </span>
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
          <div className="flex items-start gap-2 rounded-md border border-defect/30 bg-flag-soft p-3 text-[13px] text-defect">
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
              "h-10 rounded-md border text-[13.5px] font-semibold transition-all",
              defect === o.v
                ? o.v
                  ? "border-defect bg-flag-soft text-defect ring-1 ring-inset ring-defect"
                  : "border-clear bg-clear-soft text-clear ring-1 ring-inset ring-clear"
                : "border-border-strong bg-muted text-subtle hover:bg-sunken hover:text-fg",
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
