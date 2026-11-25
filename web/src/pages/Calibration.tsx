import { useQueries } from "@tanstack/react-query";
import { Gauge, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "@/lib/api";
import { clusterLabel, eps, fixed, int, lambda, pct } from "@/lib/format";
import { useInfo } from "@/lib/queries";
import type { ClusterCalibration } from "@/lib/types";
import { Badge, Card, CardBody, CardHeader, PageHeader, Segmented, Skeleton, Stat } from "@/components/ui/primitives";
import { ErrorState } from "@/components/States";
import { ChartTooltip, useChartColors } from "@/components/charts";

export default function CalibrationPage() {
  const info = useInfo();
  const clusters = info.data?.clusters ?? [];
  const results = useQueries({
    queries: clusters.map((c) => ({ queryKey: ["calibration", c], queryFn: () => api.calibration(c), staleTime: Infinity })),
  });
  const [sel, setSel] = useState("wall");
  const colors = useChartColors();

  if (info.isError) return <ErrorState error={info.error} onRetry={() => info.refetch()} />;
  const all = results.map((r) => r.data).filter(Boolean) as ClusterCalibration[];
  const err = results.find((r) => r.isError)?.error;
  const cur = all.find((c) => c.cluster === sel) ?? all[0];
  const order = ["deck", "pavement", "wall", "pooled"].filter((c) => clusters.includes(c));

  const epsAsc = cur ? [...cur.thresholds].sort((a, b) => a.epsilon - b.epsilon).map((t) => t.epsilon) : [];
  const chart = epsAsc.map((e) => {
    const row: Record<string, number | string | null> = { eps: eps(e) };
    for (const c of all) row[c.cluster] = c.thresholds.find((t) => t.epsilon === e)?.lambda_hat ?? null;
    return row;
  });

  return (
    <>
      <PageHeader
        eyebrow="Understand"
        title="Calibration"
        description="The thresholds the service loaded at startup. λ̂ is conformal risk control on each cluster's labelled defects; q̂ is split conformal (LAC) at α = ε and only decides FLAG versus ESCALATE. The service never recalibrates itself."
        actions={cur && <Badge tone="accent" className="text-[12px]">{cur.calibration_version}</Badge>}
      />
      {err ? <ErrorState error={err} /> : null}

      <Card className="mb-6">
        <CardBody className="flex flex-col gap-3 md:flex-row md:items-center">
          <ShieldCheck className="h-5 w-5 shrink-0 text-accent" />
          <div className="min-w-0 flex-1">
            <div className="text-[14px] font-medium">Guarantee: <span className="font-mono text-[13px]">{cur?.guarantee ?? "…"}</span></div>
            <p className="mt-0.5 text-[12.5px] text-subtle">
              Holds on average over calibration draws, assuming new images are exchangeable with the cluster's calibration images.
              It is not a promise about any single image, and it broke under site shift (see Research → Site shift).
            </p>
          </div>
        </CardBody>
      </Card>

      <div className="mb-4">
        <Segmented ariaLabel="Cluster" value={cur?.cluster ?? sel} onChange={setSel} options={order.map((c) => ({ value: c, label: clusterLabel(c) }))} />
      </div>

      {!cur ? (
        <div className="grid gap-3 md:grid-cols-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-24" />)}</div>
      ) : (
        <>
          <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Calibration images" value={int(cur.n_cal)} />
            <Stat label="Labelled defects" value={int(cur.n_defects)} hint={`${pct(cur.n_defects / cur.n_cal, 1)} prevalence`} />
            <Stat label="Shift threshold" value={fixed(cur.shift_threshold)} tone="shift" hint="Min. nearest-neighbour cosine similarity" />
            <Stat label="Model" value={<span className="text-[14px]">{cur.model_version.split("+")[0]}</span>} hint={cur.model_version} />
          </div>

          <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
            <Card>
              <CardHeader title={`${clusterLabel(cur.cluster)} thresholds`} description="Auto-clear iff s > λ̂. Smaller ε → higher λ̂ → fewer images cleared." />
              <CardBody className="pt-3">
                <div className="overflow-x-auto">
                  <table className="w-full text-[13px]">
                    <thead>
                      <tr className="border-b border-border text-left">
                        <th className="label py-2 pr-4 font-medium">ε (max miss rate)</th>
                        <th className="label py-2 pr-4 text-right font-medium">λ̂ (CRC)</th>
                        <th className="label py-2 text-right font-medium">q̂ (LAC)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cur.thresholds.map((t) => (
                        <tr key={t.epsilon} className="border-b border-border/60 last:border-0">
                          <td className="num py-2.5 pr-4 font-medium">{eps(t.epsilon)}</td>
                          <td className="num py-2.5 pr-4 text-right">{lambda(t.lambda_hat)}{t.lambda_hat == null && <span className="ml-1 text-subtle">(clears nothing)</span>}</td>
                          <td className="num py-2.5 text-right">{fixed(t.q_hat)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardBody>
            </Card>
            <Card>
              <CardHeader title={<span className="flex items-center gap-2"><Gauge className="h-4 w-4 text-subtle" /> λ̂ by cluster</span>} description="How strict the auto-clear threshold gets as the guaranteed miss rate tightens." />
              <CardBody>
                <div className="h-72">
                  <ResponsiveContainer>
                    <LineChart data={chart} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
                      <CartesianGrid stroke={colors.grid} vertical={false} />
                      <XAxis dataKey="eps" tick={{ fill: colors.axis, fontSize: 12 }} tickLine={false} axisLine={{ stroke: colors.grid }} />
                      <YAxis domain={["auto", "auto"]} tickFormatter={(v) => (v as number).toFixed(2)} tick={{ fill: colors.axis, fontSize: 12 }} tickLine={false} axisLine={false} />
                      <Tooltip content={<ChartTooltip format={(v) => fixed(v as number)} labelFormat={(l) => `ε = ${l}`} />} />
                      <Legend iconType="plainline" wrapperStyle={{ fontSize: 12 }} />
                      {order.map((c, i) => (
                        <Line key={c} name={clusterLabel(c)} dataKey={c} type="monotone" stroke={colors.series[i]} strokeWidth={c === cur.cluster ? 3 : 1.5} strokeOpacity={c === cur.cluster ? 1 : 0.55} dot={{ r: c === cur.cluster ? 3.5 : 2 }} connectNulls={false} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardBody>
            </Card>
          </div>
        </>
      )}
    </>
  );
}
