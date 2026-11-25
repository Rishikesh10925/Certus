import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ClipboardCheck, RefreshCw } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "@/lib/api";
import { clusterLabel, fixed, pct } from "@/lib/format";
import { useInfo, useMetrics } from "@/lib/queries";
import { Badge, Button, Card, CardBody, CardHeader, PageHeader, Skeleton, Stat } from "@/components/ui/primitives";
import { FeedbackForm, KV } from "@/components/domain";
import { EmptyState, ErrorState } from "@/components/States";

export default function AuditPage() {
  const qc = useQueryClient();
  const info = useInfo();
  const metrics = useMetrics(10_000);
  const next = useQuery({ queryKey: ["audit-next"], queryFn: api.auditNext, refetchOnWindowFocus: true });
  const rate = info.data?.audit_rate;

  const advance = () => {
    qc.invalidateQueries({ queryKey: ["metrics"] });
    next.refetch();
  };

  return (
    <>
      <PageHeader
        eyebrow="Operate"
        title="Audit queue"
        description={`The service holds a random ${rate != null ? pct(rate) : "share"} of auto-cleared images for a human to label. Labels on this sample are how a deployment notices a broken guarantee: a defect here is a miss the model let through.`}
        actions={
          <Button variant="outline" size="sm" onClick={() => next.refetch()} loading={next.isFetching}>
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </Button>
        }
      />

      <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Held for audit" value={metrics.data?.auditSampled ?? "—"} hint="Since the service started" />
        <Stat label="Labels received" value={metrics.data?.feedback ?? "—"} hint="All inspections, not only audits" />
        <Stat label="Missed defects found" value={metrics.data?.feedbackMissed ?? "—"} tone={metrics.data?.feedbackMissed ? "defect" : undefined} hint="Auto-cleared, labelled defective" />
        <Stat label="Audit rate" value={rate != null ? pct(rate) : "—"} hint="Of auto-cleared images" />
      </div>

      {next.isError ? (
        <ErrorState error={next.error} onRetry={() => next.refetch()} />
      ) : next.isPending ? (
        <div className="grid gap-4 lg:grid-cols-2"><Skeleton className="aspect-square" /><Skeleton className="h-64" /></div>
      ) : next.data === null ? (
        <Card>
          <EmptyState
            icon={<ClipboardCheck className="h-5 w-5" />}
            title="Nothing waiting for a label"
            action={<div className="flex gap-2"><Link to="/triage"><Button variant="outline">Run a batch</Button></Link><Link to="/inspect"><Button>Inspect an image</Button></Link></div>}
          >
            Audit images appear here when the service auto-clears an image and samples it for review. Every labelled audit image leaves the queue.
          </EmptyState>
        </Card>
      ) : (
        <div className="grid gap-4 animate-fade-in lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]" key={next.data.inspection_id}>
          <Card className="overflow-hidden">
            <img src={`data:image/jpeg;base64,${next.data.image_b64}`} alt={`Audit image for inspection ${next.data.inspection_id}`} className="max-h-[560px] w-full bg-muted object-contain" />
          </Card>
          <Card>
            <CardHeader title={`Inspection #${next.data.inspection_id}`} action={<Badge tone="clear">auto-cleared</Badge>} description="Is there a defect the model missed?" />
            <CardBody className="space-y-5">
              <dl className="divide-y divide-border">
                <KV k="Site" v={next.data.site_id} />
                <KV k="Cluster" v={clusterLabel(next.data.cluster)} />
                <KV k="Score s" v={fixed(next.data.score)} />
                <KV k="Inspected" v={new Date(next.data.created_at * 1000).toLocaleString()} />
              </dl>
              <FeedbackForm inspectionId={next.data.inspection_id} submitLabel="Submit and load next" onDone={() => setTimeout(advance, 900)} />
            </CardBody>
          </Card>
        </div>
      )}
    </>
  );
}
