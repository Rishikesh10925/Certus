import { useMutation } from "@tanstack/react-query";
import { AlertOctagon, CheckCircle2, ShieldAlert, ShieldCheck } from "lucide-react";
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

/* Decision glyphs: a filled circle-check, triangle and diamond, so the three
   outcomes differ in shape as well as colour. The inner mark is cut out in
   the surface colour; on a solid chip pass `knock` to cut it in the chip's ink. */
export function ClearMark({ className }: IconProps) {
  return (
    <svg viewBox="0 0 12 12" className={cn("h-3 w-3 shrink-0", className)} aria-hidden>
      <circle cx="6" cy="6" r="5.25" fill="currentColor" />
      <path d="M3.6 6.2l1.6 1.6 3.2-3.4" fill="none" stroke="rgb(var(--surface))" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
export function FlagMark({ className }: IconProps) {
  return (
    <svg viewBox="0 0 12 12" className={cn("h-3 w-3 shrink-0", className)} aria-hidden>
      <path d="M6 .9l5.3 9.6H.7z" fill="currentColor" strokeLinejoin="round" />
      <path d="M6 4.4v2.6M6 8.7v.1" stroke="rgb(var(--surface))" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}
export function EscalateMark({ className }: IconProps) {
  return (
    <svg viewBox="0 0 12 12" className={cn("h-3 w-3 shrink-0", className)} aria-hidden>
      <path d="M6 .6L11.4 6 6 11.4.6 6z" fill="currentColor" />
      <circle cx="6" cy="6" r="1.5" fill="rgb(var(--surface))" />
    </svg>
  );
}
/** Ground truth: a real defect (a target). */
export function DefectMark({ className }: IconProps) {
  return (
    <svg viewBox="0 0 12 12" className={cn("h-3 w-3 shrink-0", className)} aria-hidden>
      <circle cx="6" cy="6" r="3.2" fill="currentColor" />
      <circle cx="6" cy="6" r="5.3" fill="none" stroke="currentColor" strokeWidth="1" />
    </svg>
  );
}
export function ShiftMark({ className }: IconProps) {
  return (
    <svg viewBox="0 0 16 16" className={cn("h-3.5 w-3.5 shrink-0", className)} aria-hidden>
      <path d="M2.5 5.5h9l-2.2-2.2M13.5 10.5h-9l2.2 2.2" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
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

/** Decision colour as text (glyphs take it through currentColor). */
export const DECISION_TEXT: Record<Decision, string> = {
  AUTO_CLEAR: "text-clear",
  FLAG: "text-flag",
  ESCALATE: "text-escalate",
};
/** Solid fills (text uses the matching *-ink token). */
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
export const DECISION_BORDER: Record<Decision, string> = {
  AUTO_CLEAR: "border-clear/40",
  FLAG: "border-flag/40",
  ESCALATE: "border-escalate/40",
};

export function DecisionBadge({ decision, className }: { decision: Decision; className?: string; solid?: boolean }) {
  const Icon = DECISION_ICON[decision];
  return (
    <Badge tone={DECISION_TONE[decision]} className={className}>
      <Icon className="h-3 w-3" />
      {decisionLabel[decision]}
    </Badge>
  );
}

/** The decision glyph in its colour, for legends and breakdowns. */
export function DecisionSwatch({ decision, className }: { decision: Decision; className?: string }) {
  const Icon = DECISION_ICON[decision];
  return <Icon className={cn(DECISION_TEXT[decision], className)} />;
}

export function DecisionLegend({ className }: { className?: string }) {
  return (
    <ul className={cn("flex flex-wrap gap-x-3.5 gap-y-1.5 text-[11.5px] text-subtle", className)} aria-label="Decision legend">
      {(["FLAG", "ESCALATE", "AUTO_CLEAR"] as const).map((d) => (
        <li key={d} className="flex items-center gap-1.5">
          <DecisionSwatch decision={d} />
          {DECISION_META[d].name}
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
 * Score s = 1 − p(defect) on [0, 1]: the flag zone (s ≤ min(q̂, 1 − q̂)), the
 * escalate band and the auto-clear zone above λ̂. λ̂ = null (infinity) means
 * nothing clears at this ε. q̂ is optional; without it only λ̂ is drawn.
 */
export function ScoreScale({
  score,
  lam,
  q,
  compact,
  decision,
}: {
  score: number;
  lam: number | null;
  q?: number;
  compact?: boolean;
  decision?: Decision;
}) {
  const lamX = lam == null ? 1 : Math.min(1, Math.max(0, lam));
  const b = q == null ? null : Math.min(q, 1 - q);
  const sPos = Math.min(100, Math.max(0, score * 100));
  const above = lam != null && score > lam;
  const ring = decision ? { AUTO_CLEAR: "rgb(var(--clear))", FLAG: "rgb(var(--flag))", ESCALATE: "rgb(var(--escalate))" }[decision] : "rgb(var(--fg))";
  return (
    <div>
      <div
        className={cn("relative overflow-hidden rounded-[5px] bg-sunken", compact ? "h-4" : "h-[34px]")}
        role="img"
        aria-label={`Score ${fixed(score)} on a 0 to 1 scale; auto-clear threshold λ̂ ${lambda(lam)}${above ? ", above the threshold" : ", not above the threshold"}`}
      >
        {b != null && <span className="absolute inset-y-0 left-0 bg-flag/15" style={{ width: `${b * 100}%` }} />}
        {b != null && <span className="absolute inset-y-0 bg-escalate/15" style={{ left: `${b * 100}%`, width: `${Math.max(0, lamX - b) * 100}%` }} />}
        <span className="hatch-clear-zone absolute inset-y-0 right-0" style={{ left: `${lamX * 100}%` }} />
        {lam != null && <span className="absolute inset-y-0 -ml-px w-0.5 bg-signal" style={{ left: `${lamX * 100}%` }} />}
        <span
          className={cn("absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-fg transition-[left] duration-300", compact ? "h-2.5 w-2.5" : "h-3.5 w-3.5")}
          style={{ left: `${sPos}%`, boxShadow: `0 0 0 3px rgb(var(--surface)), 0 0 0 4.5px ${ring}` }}
        />
      </div>
      {!compact && (
        <div className="relative mt-1 h-4 font-mono text-[10.5px] text-faint">
          <span className="absolute left-0">0</span>
          <span
            className="absolute font-semibold text-accent"
            style={{ left: `${lamX * 100}%`, transform: lamX > 0.85 ? "translateX(-100%)" : lamX < 0.15 ? "none" : "translateX(-50%)" }}
          >
            λ̂ {lambda(lam)}
          </span>
          {lamX <= 0.85 && <span className="absolute right-0">1</span>}
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
      <div className="flex items-center gap-1.5 text-[12.5px]">
        {flagged ? <ShieldAlert className="h-3.5 w-3.5 text-shift" /> : <CheckCircle2 className="h-3.5 w-3.5 text-clear" />}
        {flagged ? "Outside the calibration distribution" : "Within the calibration distribution"}
      </div>
      <div
        className="relative mt-2.5 h-2 rounded bg-sunken"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={Number(similarity.toFixed(3))}
        aria-label={`Nearest calibration similarity ${fixed(similarity)}, threshold ${fixed(threshold)}`}
      >
        <div className={cn("absolute inset-y-0 left-0 rounded", flagged ? "bg-shift" : "bg-clear")} style={{ width: pos(similarity) }} />
        <div className="absolute -inset-y-[5px] -ml-px w-0.5 bg-fg" style={{ left: pos(threshold) }} />
      </div>
      <div className="mt-1.5 flex flex-wrap justify-between gap-x-3 font-mono text-[11.5px] text-faint">
        <span>nearest similarity {fixed(similarity)}</span>
        <span>threshold {fixed(threshold)}</span>
      </div>
    </div>
  );
}

export function KV({ k, v, mono = true }: { k: ReactNode; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2 text-[13px]">
      <dt className="shrink-0 text-subtle">{k}</dt>
      <dd className={cn("min-w-0 truncate text-right", mono && "num")}>{v}</dd>
    </div>
  );
}

/** The risk-statement bar: signal rule on the left, shield glyph. */
export function RiskStatement({ text, className }: { text: ReactNode; className?: string }) {
  return (
    <div className={cn("flex items-start gap-3 rounded-r-lg border-l-2 border-signal bg-accent-soft px-4 py-2.5 text-[13.5px] leading-relaxed", className)}>
      <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-accent" aria-hidden />
      <div className="min-w-0">{text}</div>
    </div>
  );
}

/** The guarantee, typeset: 𝔼[ P(auto-clear | defect) ] ≤ ε. */
export function GuaranteeFormula({ className }: { className?: string }) {
  return (
    <span
      className={cn("inline-flex flex-wrap items-baseline gap-x-1 text-[14px]", className)}
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
          <div className="flex items-start gap-2 rounded-md border border-defect/30 bg-defect-soft p-3 text-[13px] text-defect">
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
              "h-9 rounded-[5px] border text-[13px] font-medium transition-colors",
              defect === o.v
                ? o.v
                  ? "border-defect bg-defect-soft text-defect"
                  : "border-clear bg-clear-soft text-clear"
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
