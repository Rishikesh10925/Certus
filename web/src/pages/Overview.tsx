import { useQueries } from "@tanstack/react-query";
import { ArrowRight, BookOpen, ClipboardCheck, Gauge, ScanSearch, SlidersHorizontal } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "@/lib/api";
import { clusterLabel, eps, int, lambda, pct } from "@/lib/format";
import { useInfo, useMetrics } from "@/lib/queries";
import { useSettings } from "@/lib/settings";
import type { ClusterCalibration } from "@/lib/types";
import { Badge, Card, CardBody, CardHeader, Skeleton } from "@/components/ui/primitives";
import { ErrorState } from "@/components/States";
import { KV } from "@/components/domain";

const ACTIONS = [
  { to: "/triage", icon: SlidersHorizontal, title: "Batch triage", text: "Move ε and watch a batch of held-out images re-partition into auto-cleared and human review." },
  { to: "/inspect", icon: ScanSearch, title: "Inspect an image", text: "Upload one image and get the decision, the score against λ̂, the shift check and the risk statement." },
  { to: "/audit", icon: ClipboardCheck, title: "Audit queue", text: "Label the random sample of auto-cleared images: how a deployment notices a broken guarantee." },
  { to: "/research", icon: BookOpen, title: "Research results", text: "Efficiency curves, score comparison, site shift and latency from the committed results." },
];

export default function OverviewPage() {
  const info = useInfo();
  const metrics = useMetrics(10_000);
  const { settings } = useSettings();
  const clusters = info.data?.clusters ?? [];
  const cals = useQueries({ queries: clusters.map((c) => ({ queryKey: ["calibration", c], queryFn: () => api.calibration(c), staleTime: Infinity })) });
  const all = cals.map((c) => c.data).filter(Boolean) as ClusterCalibration[];
  const order = ["deck", "pavement", "wall", "pooled"];
  all.sort((a, b) => order.indexOf(a.cluster) - order.indexOf(b.cluster));
  const e = settings.epsilon;
  const m = metrics.data;

  return (
    <div className="space-y-6">
      {/* Hero */}
      <Card className="relative overflow-hidden">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,rgb(var(--accent)/0.14),transparent_55%)]" />
        <div className="pointer-events-none absolute -right-16 -top-16 h-64 w-64 rounded-full border border-accent/10" />
        <div className="pointer-events-none absolute -right-4 -top-4 h-40 w-40 rounded-full border border-accent/10" />
        <CardBody className="relative p-6 md:p-8">
          <Badge tone="accent" className="mb-4">Conformal risk control · frozen vision-language models</Badge>
          <h1 className="max-w-3xl text-[28px] font-semibold leading-tight tracking-tight md:text-[34px]">
            Which inspection images can skip a human, and how many real defects will that miss?
          </h1>
          <p className="mt-3 max-w-2xl text-[14.5px] text-subtle">
            Certus turns a model's score into an auto-clear threshold λ̂ with a finite-sample bound:
            on average, at most ε of real defects are auto-cleared, assuming new images look like the calibration images.
            The model is never retrained; the score only sets the price, measured as the human-review fraction.
          </p>
          <div className="mt-6 rounded-xl border border-border bg-surface/80 px-4 py-3 font-mono text-[13px] backdrop-blur md:inline-block">
            E[ P(auto-cleared | defect) ] ≤ ε
          </div>
        </CardBody>
      </Card>

      {info.isError && <ErrorState error={info.error} onRetry={() => info.refetch()} />}

      {/* Actions */}
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {ACTIONS.map(({ to, icon: Icon, title, text }) => (
          <Link key={to} to={to} className="group">
            <Card className="h-full p-5 transition-all duration-200 group-hover:-translate-y-0.5 group-hover:border-accent/40 group-hover:shadow-pop">
              <div className="mb-3 grid h-9 w-9 place-items-center rounded-lg bg-accent/10 text-accent">
                <Icon className="h-4.5 w-4.5" />
              </div>
              <div className="flex items-center gap-1.5 text-[14.5px] font-semibold">
                {title}
                <ArrowRight className="h-3.5 w-3.5 -translate-x-1 opacity-0 transition-all group-hover:translate-x-0 group-hover:opacity-100" />
              </div>
              <p className="mt-1 text-[13px] text-subtle">{text}</p>
            </Card>
          </Link>
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader
            title={<span className="flex items-center gap-2"><Gauge className="h-4 w-4 text-subtle" /> Thresholds at ε = {eps(e)}</span>}
            description="The default ε is set in Settings. Auto-clear iff s > λ̂."
            action={<Link to="/calibration" className="text-[13px] font-medium text-accent hover:underline">All thresholds</Link>}
          />
          <CardBody className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {all.length === 0
              ? [0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-28" />)
              : all.map((c) => {
                  const t = c.thresholds.find((x) => x.epsilon === e);
                  return (
                    <div key={c.cluster} className="rounded-xl border border-border bg-muted/40 p-4">
                      <div className="label">{clusterLabel(c.cluster)}</div>
                      <div className="num mt-2 text-[24px] font-semibold">{t ? lambda(t.lambda_hat) : "—"}</div>
                      <div className="mt-1 text-[12px] text-subtle">λ̂ · {int(c.n_defects)} defects / {int(c.n_cal)}</div>
                    </div>
                  );
                })}
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Service" description="Live from /v1/info and /v1/metrics." action={<Link to="/monitoring" className="text-[13px] font-medium text-accent hover:underline">Monitoring</Link>} />
          <CardBody className="pt-3">
            {info.data ? (
              <dl className="divide-y divide-border">
                <KV k="Calibration" v={info.data.calibration_version} />
                <KV k="Model" v={info.data.model_version} />
                <KV k="Clusters" v={info.data.clusters.map(clusterLabel).join(" · ")} mono={false} />
                <KV k="Shift check" v={info.data.escalate_on_shift ? "escalates" : "advisory"} mono={false} />
                <KV k="Audit sample" v={pct(info.data.audit_rate)} />
                <KV k="Inspections (this process)" v={m ? int(m.total) : "—"} />
                <KV k="Human-review share" v={m && m.total ? pct((m.decisions.FLAG + m.decisions.ESCALATE) / m.total) : "—"} />
              </dl>
            ) : (
              <Skeleton className="h-48" />
            )}
          </CardBody>
        </Card>
      </div>

      <Card className="overflow-hidden">
        <CardHeader title="How it works" description="Figure F1 from the paper." />
        <CardBody>
          <div className="rounded-lg bg-white p-3">
            <img src={api.figureUrl("F1_system")} alt="System overview: score, calibrate, triage" className="mx-auto max-h-[420px] w-full object-contain" />
          </div>
        </CardBody>
      </Card>

      <Card className="border-flag/30 bg-flag/5">
        <CardBody className="text-[13px] leading-relaxed">
          <span className="font-semibold">Limits of the guarantee.</span>{" "}
          <span className="text-subtle">
            It bounds the miss rate in expectation over calibration draws, not for a single batch or image. It assumes the
            deployment images are exchangeable with the calibration set; leave-one-site-out showed miss rates of 2–3× ε on a new
            structure type, which is why the service escalates images that fail its shift check. All scores are off-the-shelf
            (best AUC 0.79), so the review costs describe these scores, not the method.
          </span>
        </CardBody>
      </Card>
    </div>
  );
}
