import { forwardRef, type ButtonHTMLAttributes, type HTMLAttributes, type ReactNode } from "react";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/cn";

/* ---------- Button ---------- */

type Variant = "primary" | "secondary" | "ghost" | "danger" | "outline";
type Size = "sm" | "md" | "lg" | "icon";

const variants: Record<Variant, string> = {
  primary: "bg-accent text-accent-fg hover:bg-accent/90 shadow-sm",
  secondary: "bg-muted text-fg hover:bg-border/70",
  outline: "border border-border bg-surface text-fg hover:bg-muted",
  ghost: "text-subtle hover:bg-muted hover:text-fg",
  danger: "bg-escalate text-white hover:bg-escalate/90 shadow-sm",
};
const sizes: Record<Size, string> = {
  sm: "h-8 px-3 text-[13px] gap-1.5",
  md: "h-9 px-4 gap-2",
  lg: "h-11 px-5 text-[15px] gap-2",
  icon: "h-9 w-9",
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
        "inline-flex shrink-0 select-none items-center justify-center rounded-lg font-medium transition-colors duration-150 disabled:pointer-events-none disabled:opacity-50",
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
      className={cn("rounded-xl border border-border bg-surface shadow-card", className)}
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
    <div className={cn("flex items-start justify-between gap-4 px-5 pt-5", className)}>
      <div className="min-w-0">
        <h3 className="text-[15px] font-semibold tracking-tight">{title}</h3>
        {description && <p className="mt-0.5 text-[13px] text-subtle">{description}</p>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

export function CardBody({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-5", className)} {...props} />;
}

/* ---------- Badge ---------- */

type Tone = "neutral" | "clear" | "flag" | "escalate" | "shift" | "accent" | "defect";
const tones: Record<Tone, string> = {
  neutral: "bg-muted text-subtle ring-border",
  clear: "bg-clear/10 text-clear ring-clear/25",
  flag: "bg-flag/10 text-flag ring-flag/25",
  escalate: "bg-escalate/10 text-escalate ring-escalate/25",
  shift: "bg-shift/10 text-shift ring-shift/25",
  accent: "bg-accent/10 text-accent ring-accent/25",
  defect: "bg-defect/10 text-defect ring-defect/25",
};

export function Badge({
  tone = "neutral",
  className,
  ...props
}: HTMLAttributes<HTMLSpanElement> & { tone?: Tone }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-md px-1.5 py-0.5 text-[11px] font-medium ring-1 ring-inset",
        tones[tone],
        className,
      )}
      {...props}
    />
  );
}

/* ---------- Stat ---------- */

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
  tone?: "clear" | "flag" | "escalate" | "shift" | "accent" | "defect";
  className?: string;
  icon?: ReactNode;
}) {
  const color = tone ? { clear: "text-clear", flag: "text-flag", escalate: "text-escalate", shift: "text-shift", accent: "text-accent", defect: "text-defect" }[tone] : "";
  return (
    <Card className={cn("p-4", className)}>
      <div className="flex items-center justify-between gap-2">
        <div className="label">{label}</div>
        {icon && <div className="text-subtle">{icon}</div>}
      </div>
      <div className={cn("num mt-2 text-[26px] font-semibold leading-none", color)}>{value}</div>
      {hint && <div className="mt-2 text-[12px] leading-snug text-subtle">{hint}</div>}
    </Card>
  );
}

/* ---------- Skeleton / Spinner ---------- */

export function Skeleton({ className }: { className?: string }) {
  return (
    <div className={cn("relative overflow-hidden rounded-md bg-muted", className)}>
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
        "h-9 w-full rounded-lg border border-border bg-surface px-3 text-[14px] text-fg placeholder:text-subtle/70 transition-colors focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/25 aria-[invalid=true]:border-escalate",
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
      <label htmlFor={htmlFor} className="block text-[13px] font-medium">
        {label}
      </label>
      {children}
      {error ? (
        <p className="text-[12px] text-escalate">{error}</p>
      ) : hint ? (
        <p className="text-[12px] text-subtle">{hint}</p>
      ) : null}
    </div>
  );
}

/* ---------- Segmented control ---------- */

export function Segmented<T extends string | number>({
  value,
  onChange,
  options,
  className,
  ariaLabel,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: ReactNode }[];
  className?: string;
  ariaLabel?: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={cn("inline-flex rounded-lg border border-border bg-muted p-0.5", className)}
    >
      {options.map((o) => (
        <button
          key={String(o.value)}
          role="radio"
          aria-checked={o.value === value}
          onClick={() => onChange(o.value)}
          className={cn(
            "h-7 rounded-md px-3 text-[13px] font-medium transition-all",
            o.value === value
              ? "bg-surface text-fg shadow-sm ring-1 ring-border"
              : "text-subtle hover:text-fg",
          )}
        >
          {o.label}
        </button>
      ))}
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
    <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
      <div className="min-w-0 max-w-3xl">
        {eyebrow && <div className="label mb-2 text-accent">{eyebrow}</div>}
        <h1 className="text-[26px] font-semibold tracking-tight md:text-[28px]">{title}</h1>
        {description && <p className="mt-1.5 text-[14px] text-subtle">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}
