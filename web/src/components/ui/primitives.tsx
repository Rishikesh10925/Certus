import { forwardRef, type ButtonHTMLAttributes, type HTMLAttributes, type ReactNode } from "react";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/cn";

/* ---------- Button ---------- */

type Variant = "primary" | "secondary" | "ghost" | "danger" | "outline";
type Size = "sm" | "md" | "lg" | "icon";

const variants: Record<Variant, string> = {
  primary: "bg-accent text-accent-ink hover:bg-accent/90 active:translate-y-px",
  secondary: "border border-border bg-sunken text-fg hover:border-border-strong",
  outline: "border border-border-strong bg-surface text-fg hover:bg-sunken active:translate-y-px",
  ghost: "text-subtle hover:bg-sunken hover:text-fg",
  danger: "bg-danger text-white hover:bg-danger/90 dark:text-bg",
};
const sizes: Record<Size, string> = {
  sm: "h-8 px-3 text-[12.5px] gap-1.5",
  md: "h-9 px-3.5 text-[13.5px] gap-2",
  lg: "h-10 px-4 text-[14px] gap-2",
  icon: "h-8 w-8",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", loading, disabled, children, ...props }, ref) => (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={cn(
        "inline-flex shrink-0 select-none items-center justify-center whitespace-nowrap rounded-[5px] font-medium transition-[background-color,border-color,color,transform] duration-150 disabled:pointer-events-none disabled:opacity-45",
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    >
      {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
      {children}
    </button>
  ),
);
Button.displayName = "Button";

/* ---------- Card ---------- */

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("rounded-xl border border-border bg-surface", className)}
      {...props}
    />
  );
}

export function CardHeader({
  title,
  description,
  action,
  className,
}: {
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-wrap items-start justify-between gap-x-4 gap-y-2 border-b border-border px-4 py-3", className)}>
      <div className="min-w-0 flex-1 basis-64">
        <h3 className="text-[14px] font-semibold tracking-[-0.01em]">{title}</h3>
        {description && <p className="mt-0.5 text-[12.5px] leading-snug text-faint">{description}</p>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

export function CardBody({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-4", className)} {...props} />;
}

/* ---------- Badge ---------- */

type Tone = "neutral" | "clear" | "flag" | "escalate" | "shift" | "accent" | "defect" | "danger";
const tones: Record<Tone, string> = {
  neutral: "bg-muted text-subtle ring-border",
  clear: "bg-clear-soft text-clear ring-clear/30",
  flag: "bg-flag-soft text-flag ring-flag/30",
  escalate: "bg-escalate-soft text-escalate ring-escalate/30",
  shift: "bg-shift/10 text-shift ring-shift/30",
  accent: "bg-accent-soft text-accent ring-accent/30",
  defect: "bg-defect-soft text-defect ring-defect/30",
  danger: "bg-danger/10 text-danger ring-danger/30",
};

export function Badge({
  tone = "neutral",
  className,
  ...props
}: HTMLAttributes<HTMLSpanElement> & { tone?: Tone }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-[4px] px-1.5 py-0.5 text-[11.5px] font-medium ring-1 ring-inset",
        tones[tone],
        className,
      )}
      {...props}
    />
  );
}

/* ---------- Stat ---------- */

/** Uppercase labels would turn ε into Ε and λ̂ into Λ̂; keep Greek symbols in their own case. */
export function keepGreekCase(label: ReactNode): ReactNode {
  if (typeof label !== "string" || !/[Ͱ-Ͽ]/.test(label)) return label;
  return label.split(/([Ͱ-Ͽ]̂?)/).map((part, i) =>
    i % 2 ? <span key={i} className="normal-case">{part}</span> : part,
  );
}

export function Stat({
  label,
  value,
  hint,
  tone,
  className,
  icon,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "clear" | "flag" | "escalate" | "shift" | "accent" | "defect" | "danger";
  className?: string;
  icon?: ReactNode;
}) {
  const color = tone ? { clear: "text-clear", flag: "text-flag", escalate: "text-escalate", shift: "text-shift", accent: "text-accent", defect: "text-defect", danger: "text-danger" }[tone] : "";
  return (
    <Card className={cn("px-4 py-3", className)}>
      <div className="flex items-start justify-between gap-2">
        <div className="text-[12px] text-subtle">{label}</div>
        {icon && <div className="text-faint">{icon}</div>}
      </div>
      <div className={cn("num mt-1.5 text-[24px] font-medium leading-[1.1] tracking-[-0.02em]", color)}>{value}</div>
      {hint && <div className="mt-1 text-[11.5px] leading-snug text-faint">{hint}</div>}
    </Card>
  );
}

/* ---------- Skeleton / Spinner ---------- */

export function Skeleton({ className }: { className?: string }) {
  return (
    <div className={cn("relative overflow-hidden rounded-md bg-sunken", className)}>
      <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-white/30 to-transparent dark:via-white/5" />
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cn("h-4 w-4 animate-spin text-subtle", className)} aria-label="Loading" />;
}

/* ---------- Inputs ---------- */

export const Input = forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "h-9 w-full rounded-[5px] border border-border-strong bg-muted px-3 text-[13.5px] text-fg placeholder:text-faint transition-colors hover:border-faint focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent focus-visible:outline-none aria-[invalid=true]:border-danger aria-[invalid=true]:text-danger",
        className,
      )}
      {...props}
    />
  ),
);
Input.displayName = "Input";

export function Field({
  label,
  hint,
  error,
  children,
  htmlFor,
}: {
  label: ReactNode;
  hint?: ReactNode;
  error?: ReactNode;
  children: ReactNode;
  htmlFor?: string;
}) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={htmlFor} className="block text-[12px] font-medium text-subtle">
        {label}
      </label>
      {children}
      {error ? (
        <p className="text-[12px] text-danger" role="alert">{error}</p>
      ) : hint ? (
        <p className="text-[12px] text-subtle">{hint}</p>
      ) : null}
    </div>
  );
}

/* ---------- Segmented control ---------- */

/** A single-choice radio group with roving focus (arrow keys, Home, End). */
export function Segmented<T extends string | number>({
  value,
  onChange,
  options,
  className,
  ariaLabel,
  size = "md",
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: ReactNode; disabled?: boolean; title?: string }[];
  className?: string;
  ariaLabel?: string;
  size?: "sm" | "md";
}) {
  const enabled = options.filter((o) => !o.disabled);
  const onKey = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const i = enabled.findIndex((o) => o.value === value);
    let n: number | null = null;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") n = Math.min(enabled.length - 1, i + 1);
    if (e.key === "ArrowLeft" || e.key === "ArrowUp") n = Math.max(0, i - 1);
    if (e.key === "Home") n = 0;
    if (e.key === "End") n = enabled.length - 1;
    if (n == null || n === i) return;
    e.preventDefault();
    onChange(enabled[n].value);
    const btn = e.currentTarget.querySelector<HTMLButtonElement>(`[data-seg-index="${options.indexOf(enabled[n])}"]`);
    btn?.focus();
  };
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      onKeyDown={onKey}
      className={cn("inline-flex max-w-full gap-0.5 rounded-[5px] border border-border bg-sunken p-0.5", className)}
    >
      {options.map((o, idx) => {
        const on = o.value === value;
        return (
          <button
            key={String(o.value)}
            type="button"
            role="radio"
            data-seg-index={idx}
            aria-checked={on}
            disabled={o.disabled}
            title={o.title}
            tabIndex={on ? 0 : -1}
            onClick={() => onChange(o.value)}
            className={cn(
              "whitespace-nowrap rounded-[4px] transition-[background-color,color,box-shadow] disabled:cursor-not-allowed disabled:opacity-50",
              size === "sm" ? "h-6 px-2 text-[12px]" : "h-7 px-2.5 text-[12.5px]",
              on ? "bg-surface font-medium text-fg shadow-[0_0_0_1px_rgb(var(--border-strong)),0_1px_2px_rgb(0_0_0/0.25)]" : "text-subtle hover:text-fg",
            )}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/* ---------- Page header ---------- */

export function PageHeader({
  title,
  description,
  actions,
  eyebrow,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  eyebrow?: ReactNode;
}) {
  return (
    <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
      <div className="min-w-0 max-w-3xl">
        {eyebrow && (
          <p className="mb-1 flex gap-1.5 font-mono text-[11.5px] lowercase text-faint">
            <span>{eyebrow}</span>
            {typeof title === "string" && (
              <>
                <span aria-hidden>/</span>
                <b className="font-medium text-subtle">{title}</b>
              </>
            )}
          </p>
        )}
        <h1 className="text-[22px] font-semibold leading-[1.1] tracking-[-0.025em] md:text-[26px]">{title}</h1>
        {description && <p className="mt-1.5 text-[13px] leading-relaxed text-subtle">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}
