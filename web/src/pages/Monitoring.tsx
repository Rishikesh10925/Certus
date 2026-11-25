import { Activity, AlertOctagon, ClipboardCheck, MessageSquare, ShieldAlert, Users } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { decisionLabel, int, ms, pct } from "@/lib/format";
import { useInfo, useMetrics } from "@/lib/queries";
import type { ServiceMetrics } from "@/lib/metrics";
import { DECISIONS } from "@/lib/types";
import { Badge, Card, CardBody, CardHeader, PageHeader, Skeleton, Stat } from "@/components/ui/primitives";
import { ErrorState, EmptyState } from "@/components/States";
import { ChartTooltip, useChartColors } from "@/components/charts";

const POLL = 5000;

/** Rolling window of p95 readings taken while this page is open. */
function useHistory(m: ServiceMetrics | undefined) {
  const [rows, setRows] = useState<{ t: string; p50: number; p95: number; total: number }[]>([]);
  const last = useRef<number>(-1);
  useEffect(() => {
    if (!m) return;
    const key = m.latency.count;
    if (key === last.current && rows.length) return;
    last.current = key;
    setRows((r) => [...r, { t: new Date().toLocaleTimeString(), p50: m.latency.p50, p95: m.latency.p95, total: m.total }].slice(-60));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [m]);
  return rows;
}

export default function MonitoringPage() {
  const metrics = useMetrics(POLL);
  const info = useInfo();
  const colors = useChartColors();
  const history = useHistory(metrics.data);
  const m = metrics.data;

  if (metrics.isError) return <ErrorState error={metrics.error} onRetry={() => metrics.refetch()} />;

  const tone = { AUTO_CLEAR: colors.clear, FLAG: colors.flag, ESCALATE: colors.escalate };
  const bars = m ? DECISIONS.map((d) => ({ name: decisionLabel[d], value: m.decisions[d], d })) : [];
  const hrf = m && m.total ? (m.decisions.FLAG + m.decisions.ESCALATE) / m.total : null;

  return (
    <>
      <PageHeader
        eyebrow="Understand"
        title="Monitoring"
        description="Live counters from GET /v1/metrics (Prometheus format), polled every 5 s. Counters are kept in memory by the service process and reset when it restarts."
        actions={
          <Badge tone="accent" className="text-[12px]">
            <span className="mr-1 h-1.5 w-1.5 animate-pulse rounded-full bg-accent" /> live
          </Badge>
        }
      />
      {!m ? (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">{Array.from({ length: 8 }, (_, i) => <Skeleton key={i} className="h-24" />)}</div>
      ) : (
        <>
          <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <Stat label="Inspections" value={int(m.total)} icon={<Activity className="h-4 w-4" />} />
            <Stat label="Human-review share" value={pct(hrf)} icon={<Users className="h-4 w-4" />} hint="Across all ε requested" />
            <Stat label="Shift flags" value={int(m.shiftFlags)} tone="shift" icon={<ShieldAlert className="h-4 w-4" />} hint={m.total ? `${pct(m.shiftFlags / m.total, 1)} of inspections` : undefined} />
            <Stat label="Held for audit" value={int(m.auditSampled)} icon={<ClipboardCheck className="h-4 w-4" />} />
            <Stat label="Labels received" value={int(m.feedback)} icon={<MessageSquare className="h-4 w-4" />} />
            <Stat label="Missed defects" value={int(m.feedbackMissed)} tone={m.feedbackMissed ? "defect" : undefined} icon={<AlertOctagon className="h-4 w-4" />} hint="Labelled defective after auto-clear" />
          </div>

          <div className="grid gap-4 xl:grid-cols-2">
            <Card>
              <CardHeader title="Decisions" description="AUTO_CLEAR skips review; FLAG and ESCALATE go to a human." />
              <CardBody>
                {m.total === 0 ? (
                  <EmptyState title="No inspections yet" className="py-10">Run a batch or inspect an image to see decisions here.</EmptyState>
                ) : (
                  <div className="h-64">
                    <ResponsiveContainer>
                      <BarChart data={bars} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
                        <CartesianGrid stroke={colors.grid} vertical={false} />
                        <XAxis dataKey="name" tick={{ fill: colors.axis, fontSize: 12 }} tickLine={false} axisLine={{ stroke: colors.grid }} />
                        <YAxis allowDecimals={false} tick={{ fill: colors.axis, fontSize: 12 }} tickLine={false} axisLine={false} />
                        <Tooltip cursor={{ fill: colors.grid, opacity: 0.4 }} content={<ChartTooltip format={(v) => int(v as number)} />} />
                        <Bar dataKey="value" name="Inspections" radius={[6, 6, 0, 0]} maxBarSize={72}>
                          {bars.map((b) => <Cell key={b.d} fill={tone[b.d]} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </CardBody>
            </Card>
            <Card>
              <CardHeader
                title="Server latency"
                description={`Window of the last ${int(Math.min(m.latency.count, 10000))} /v1/inspect requests on ${info.data ? "this service" : "the service"}.`}
              />
              <CardBody>
                <div className="mb-4 grid grid-cols-3 gap-3">
                  {(["p50", "p95", "p99"] as const).map((q) => (
                    <div key={q} className="rounded-lg bg-muted/60 p-3">
                      <div className="label">{q}</div>
                      <div className="num mt-1 text-[20px] font-semibold">{m.latency.count ? ms(m.latency[q]) : "—"}</div>
                    </div>
                  ))}
                </div>
                <div className="h-40">
                  {history.length < 2 ? (
                    <div className="grid h-full place-items-center rounded-lg border border-dashed border-border text-[12px] text-subtle">
                      The trend fills in while this page stays open.
                    </div>
                  ) : (
                    <ResponsiveContainer>
                      <AreaChart data={history} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
                        <defs>
                          <linearGradient id="lat" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor={colors.accent} stopOpacity={0.35} />
                            <stop offset="100%" stopColor={colors.accent} stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid stroke={colors.grid} vertical={false} />
                        <XAxis dataKey="t" tick={{ fill: colors.axis, fontSize: 11 }} tickLine={false} axisLine={false} minTickGap={40} />
                        <YAxis tickFormatter={(v) => `${Math.round(v)}`} tick={{ fill: colors.axis, fontSize: 11 }} tickLine={false} axisLine={false} unit=" ms" width={64} />
                        <Tooltip content={<ChartTooltip format={(v) => ms(v as number)} />} />
                        <Area type="monotone" name="p95" dataKey="p95" stroke={colors.accent} fill="url(#lat)" strokeWidth={2} />
                        <Area type="monotone" name="p50" dataKey="p50" stroke={colors.series[1]} fill="none" strokeWidth={1.5} />
                      </AreaChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </CardBody>
            </Card>
          </div>
          <p className="mt-4 text-[12px] text-subtle">
            Calibration <span className="num">{m.calibrationVersion ?? "—"}</span> · shift check {m.escalateOnShift ? "escalates" : "is advisory only"}.
          </p>
        </>
      )}
    </>
  );
}
