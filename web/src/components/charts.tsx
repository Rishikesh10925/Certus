import { useMemo } from "react";
import { useSettings } from "@/lib/settings";

/** Chart colours from the theme tokens, recomputed when the theme flips. */
export function useChartColors() {
  const { dark } = useSettings();
  return useMemo(() => {
    // Children render before the provider's effect flips the class, so make
    // sure the tokens read below belong to the theme being rendered.
    document.documentElement.classList.toggle("dark", dark);
    const css = getComputedStyle(document.documentElement);
    const v = (name: string, a = 1) => `rgb(${css.getPropertyValue(`--${name}`).trim()} / ${a})`;
    return {
      grid: v("border"),
      axis: v("faint"),
      surface: v("surface"),
      fg: v("fg"),
      accent: v("accent"),
      defect: v("defect"),
      clear: v("clear"),
      flag: v("flag"),
      escalate: v("escalate"),
      shift: v("shift"),
      // Categorical series, led by the indigo accent and chosen to stay
      // distinct in both themes (and from the decision colours' meaning).
      series: dark
        ? ["#8b93ff", "#22d3ee", "#fbbf24", "#f472b6", "#a3e635", "#fb923c", "#c4b5fd", "#5eead4", "#94a3b8"]
        : ["#4338ca", "#0e7490", "#b45309", "#be185d", "#4d7c0f", "#c2410c", "#7c3aed", "#0f766e", "#475569"],
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dark]);
}

interface TooltipProps {
  active?: boolean;
  label?: string | number;
  payload?: { name?: string; value?: number | string | null; color?: string; dataKey?: string | number }[];
  format?: (v: number | string, key?: string | number) => string;
  labelFormat?: (l: string | number) => string;
}

export function ChartTooltip({ active, payload, label, format, labelFormat }: TooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-border-strong bg-surface px-3 py-2 text-[12px] shadow-pop">
      {label != null && <div className="num mb-1 font-semibold">{labelFormat ? labelFormat(label) : label}</div>}
      {payload.map((p, i) =>
        p.value == null ? null : (
          <div key={i} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-subtle">
              <span className="h-2 w-2 rounded-[2px]" style={{ background: p.color }} />
              {p.name}
            </span>
            <span className="num font-medium">{format ? format(p.value, p.dataKey) : String(p.value)}</span>
          </div>
        ),
      )}
    </div>
  );
}

/** Shared Recharts axis / grid props so every chart reads the same. */
export function axisProps(colors: { axis: string; grid: string }) {
  return {
    tick: { fill: colors.axis, fontSize: 12, fontFamily: '"JetBrains Mono Variable", ui-monospace, monospace' },
    tickLine: false,
  } as const;
}
