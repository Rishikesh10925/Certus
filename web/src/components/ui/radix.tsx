import * as SliderPrimitive from "@radix-ui/react-slider";
import * as SelectPrimitive from "@radix-ui/react-select";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import * as SwitchPrimitive from "@radix-ui/react-switch";
import { Check, ChevronDown, X } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

/* ---------- Slider over a discrete grid ---------- */

export function StepSlider<T>({
  values,
  value,
  onChange,
  format,
  ariaLabel,
}: {
  values: T[];
  value: T;
  onChange: (v: T) => void;
  format: (v: T) => string;
  ariaLabel: string;
}) {
  const index = Math.max(0, values.indexOf(value));
  return (
    <div>
      <SliderPrimitive.Root
        className="relative flex h-6 w-full touch-none select-none items-center"
        min={0}
        max={values.length - 1}
        step={1}
        value={[index]}
        onValueChange={([i]) => onChange(values[i])}
        aria-label={ariaLabel}
      >
        <SliderPrimitive.Track className="relative h-1.5 grow overflow-hidden rounded-full bg-sunken ring-1 ring-inset ring-border">
          <SliderPrimitive.Range className="absolute h-full bg-accent" />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb
          aria-valuetext={format(values[index])}
          className="block h-5 w-5 rounded-full border-2 border-accent bg-surface shadow-md transition-transform hover:scale-110 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-accent/30"
        />
      </SliderPrimitive.Root>
      <div className="mt-1.5 flex justify-between">
        {values.map((v, i) => (
          <button
            key={i}
            onClick={() => onChange(v)}
            className={cn(
              "num text-[11px] transition-colors",
              i === index ? "font-semibold text-accent" : "text-subtle hover:text-fg",
            )}
          >
            {format(v)}
          </button>
        ))}
      </div>
    </div>
  );
}

/* ---------- Select ---------- */

export function Select({
  value,
  onValueChange,
  options,
  className,
  ariaLabel,
  id,
}: {
  value: string;
  onValueChange: (v: string) => void;
  options: { value: string; label: ReactNode }[];
  className?: string;
  ariaLabel?: string;
  id?: string;
}) {
  return (
    <SelectPrimitive.Root value={value} onValueChange={onValueChange}>
      <SelectPrimitive.Trigger
        id={id}
        aria-label={ariaLabel}
        className={cn(
          "inline-flex h-9 w-full items-center justify-between gap-2 rounded-md border border-border-strong bg-muted px-3 text-[14px] transition-colors hover:border-faint focus:outline-none focus:ring-2 focus:ring-accent/30",
          className,
        )}
      >
        <SelectPrimitive.Value />
        <SelectPrimitive.Icon>
          <ChevronDown className="h-4 w-4 text-subtle" />
        </SelectPrimitive.Icon>
      </SelectPrimitive.Trigger>
      <SelectPrimitive.Portal>
        <SelectPrimitive.Content
          position="popper"
          sideOffset={4}
          className="z-50 min-w-[var(--radix-select-trigger-width)] overflow-hidden rounded-[10px] border border-border-strong bg-surface p-1 shadow-pop animate-fade-in"
        >
          <SelectPrimitive.Viewport>
            {options.map((o) => (
              <SelectPrimitive.Item
                key={o.value}
                value={o.value}
                className="relative flex h-8 cursor-default select-none items-center rounded-md pl-7 pr-3 text-[14px] outline-none data-[highlighted]:bg-accent-soft data-[highlighted]:text-fg"
              >
                <SelectPrimitive.ItemIndicator className="absolute left-2">
                  <Check className="h-3.5 w-3.5 text-accent" />
                </SelectPrimitive.ItemIndicator>
                <SelectPrimitive.ItemText>{o.label}</SelectPrimitive.ItemText>
              </SelectPrimitive.Item>
            ))}
          </SelectPrimitive.Viewport>
        </SelectPrimitive.Content>
      </SelectPrimitive.Portal>
    </SelectPrimitive.Root>
  );
}

/* ---------- Tabs ---------- */

export const Tabs = TabsPrimitive.Root;
export const TabsContent = (props: TabsPrimitive.TabsContentProps) => (
  <TabsPrimitive.Content {...props} className={cn("animate-fade-in focus:outline-none", props.className)} />
);

export function TabsList({ tabs, className }: { tabs: { value: string; label: ReactNode }[]; className?: string }) {
  return (
    <TabsPrimitive.List
      className={cn("no-scrollbar flex gap-1 overflow-x-auto border-b border-border", className)}
    >
      {tabs.map((t) => (
        <TabsPrimitive.Trigger
          key={t.value}
          value={t.value}
          className="relative -mb-px whitespace-nowrap border-b-[3px] border-transparent px-3 pb-2.5 pt-1 text-[13.5px] font-medium text-subtle transition-colors hover:text-fg focus-visible:ring-offset-0 data-[state=active]:border-accent data-[state=active]:font-bold data-[state=active]:text-fg"
        >
          {t.label}
        </TabsPrimitive.Trigger>
      ))}
    </TabsPrimitive.List>
  );
}

/* ---------- Dialog ---------- */

export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  className,
}: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-[rgb(4_8_20/0.55)] animate-fade-in" />
        <DialogPrimitive.Content
          className={cn(
            "fixed left-1/2 top-1/2 z-50 max-h-[90vh] w-[calc(100vw-32px)] max-w-3xl -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-xl border border-border bg-surface p-5 shadow-pop animate-fade-in focus:outline-none md:p-6",
            className,
          )}
        >
          <div className="mb-4 flex items-start justify-between gap-4">
            <div>
              <DialogPrimitive.Title className="text-[17px] font-bold tracking-tight">
                {title}
              </DialogPrimitive.Title>
              {description ? (
                <DialogPrimitive.Description className="mt-0.5 text-[13px] text-subtle">
                  {description}
                </DialogPrimitive.Description>
              ) : (
                <DialogPrimitive.Description className="sr-only">Details</DialogPrimitive.Description>
              )}
            </div>
            <DialogPrimitive.Close className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-border text-subtle hover:bg-muted hover:text-fg" aria-label="Close (Esc)">
              <X className="h-4 w-4" />
            </DialogPrimitive.Close>
          </div>
          {children}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

/* ---------- Tooltip ---------- */

export const TooltipProvider = TooltipPrimitive.Provider;

export function Tip({ content, children }: { content: ReactNode; children: ReactNode }) {
  return (
    <TooltipPrimitive.Root delayDuration={150}>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content
          sideOffset={6}
          className="z-50 max-w-xs rounded-md bg-fg px-2.5 py-1.5 text-[12.5px] leading-snug text-surface shadow-pop animate-fade-in"
        >
          {content}
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  );
}

/* ---------- Switch ---------- */

export function Switch({
  checked,
  onCheckedChange,
  id,
  ariaLabel,
}: {
  checked: boolean;
  onCheckedChange: (v: boolean) => void;
  id?: string;
  ariaLabel?: string;
}) {
  return (
    <SwitchPrimitive.Root
      id={id}
      aria-label={ariaLabel}
      checked={checked}
      onCheckedChange={onCheckedChange}
      className="relative h-6 w-[42px] shrink-0 rounded-full border border-border-strong bg-sunken transition-colors data-[state=checked]:border-accent data-[state=checked]:bg-accent"
    >
      <SwitchPrimitive.Thumb className="block h-[18px] w-[18px] translate-x-[2px] rounded-full bg-surface shadow-[0_1px_3px_rgb(0_0_0/0.35)] transition-transform data-[state=checked]:translate-x-[20px]" />
    </SwitchPrimitive.Root>
  );
}

/* ---------- Sheet (side / bottom drawer) ---------- */

/**
 * A modal drawer. Radix handles focus trapping, Esc and returning focus.
 * side="right" becomes a bottom sheet below the md breakpoint.
 */
export function Sheet({
  open,
  onOpenChange,
  title,
  description,
  children,
  footer,
  side = "right",
  className,
  onKeyDown,
}: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  side?: "right" | "left";
  className?: string;
  onKeyDown?: (e: React.KeyboardEvent<HTMLDivElement>) => void;
}) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-[rgb(4_8_20/0.55)] animate-fade-in" />
        <DialogPrimitive.Content
          onKeyDown={onKeyDown}
          className={cn(
            "fixed z-50 flex flex-col bg-surface shadow-pop focus:outline-none",
            side === "right"
              ? "inset-x-0 bottom-0 h-[92dvh] rounded-t-[18px] border-t border-border animate-slide-in-up md:inset-y-0 md:left-auto md:right-0 md:h-auto md:w-[min(540px,100vw)] md:rounded-none md:border-l md:border-t-0 md:animate-slide-in-right"
              : "inset-y-0 left-0 w-[min(300px,86vw)] border-r border-border animate-slide-in-left",
            className,
          )}
        >
          <div className="flex items-center gap-3 border-b border-border py-3 pl-5 pr-4">
            <div className="min-w-0 flex-1">
              <DialogPrimitive.Title className="truncate text-[16px] font-bold tracking-tight">{title}</DialogPrimitive.Title>
              {description ? (
                <DialogPrimitive.Description className="num truncate text-[12px] text-faint">{description}</DialogPrimitive.Description>
              ) : (
                <DialogPrimitive.Description className="sr-only">Details</DialogPrimitive.Description>
              )}
            </div>
            <DialogPrimitive.Close
              className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-border text-fg hover:border-border-strong hover:bg-muted"
              aria-label="Close (Esc)"
            >
              <X className="h-[18px] w-[18px]" />
            </DialogPrimitive.Close>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
          {footer && <div className="border-t border-border px-5 py-3">{footer}</div>}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
