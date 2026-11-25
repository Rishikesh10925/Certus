import { useMemo } from "react";
import { useSettings } from "@/lib/settings";

/** Chart colours from the theme tokens, recomputed when the theme flips. */
export function useChartColors() {
  const { dark } = useSettings();
  return useMemo(() => {
    const css = getComputedStyle(document.documentElement);
    const v = (name: string, a = 1) => `rgb(${css.getPropertyValue(`--${name}`).trim()} / ${a})`;
    return {
      grid: v("border"),
      axis: v("subtle"),
      fg: v("fg"),
      accent: v("accent"),
      defect: v("defect"),
      clear: v("clear"),
      flag: v("flag"),
      escalate: v("escalate"),
      shift: v("shift"),
      // Categorical series, chosen to stay distinct in both themes.
      series: dark
        ? ["#2dd4bf", "#818cf8", "#fbbf24", "#f472b6", "#60a5fa", "#a3e635", "#fb923c", "#c084fc", "#94a3b8"]
        : ["#0f766e", "#4f46e5", "#b45309", "#be185d", "#2563eb", "#4d7c0f", "#c2410c", "#7e22ce", "#475569"],
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
    <div className="rounded-lg border border-border bg-surface px-3 py-2 text-[12px] shadow-pop">
      {label != null && <div className="mb-1 font-medium">{labelFormat ? labelFormat(label) : label}</div>}
      {payload.map((p, i) =>
        p.value == null ? null : (
          <div key={i} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-subtle">
              <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
              {p.name}
            </span>
            <span className="num font-medium">{format ? format(p.value, p.dataKey) : String(p.value)}</span>
          </div>
        ),
      )}
    </div>
  );
}
