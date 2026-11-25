import { AlertTriangle, Inbox, PlugZap, RefreshCw } from "lucide-react";
import type { ReactNode } from "react";
import { ApiError } from "@/lib/api";
import { Button, Card } from "./ui/primitives";
import { cn } from "@/lib/cn";

export function EmptyState({
  icon,
  title,
  children,
  action,
  className,
}: {
  icon?: ReactNode;
  title: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center justify-center px-6 py-14 text-center", className)}>
      <div className="mb-4 grid h-11 w-11 place-items-center rounded-xl bg-muted text-subtle ring-1 ring-border">
        {icon ?? <Inbox className="h-5 w-5" />}
      </div>
      <h3 className="text-[15px] font-semibold">{title}</h3>
      {children && <div className="mt-1.5 max-w-md text-[13px] text-subtle">{children}</div>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function ErrorState({
  error,
  onRetry,
  className,
  compact,
}: {
  error: unknown;
  onRetry?: () => void;
  className?: string;
  compact?: boolean;
}) {
  const unreachable = error instanceof ApiError && error.unreachable;
  const message = error instanceof Error ? error.message : String(error);
  if (compact) {
    return (
      <div className={cn("flex items-start gap-2 rounded-lg border border-escalate/30 bg-escalate/5 p-3 text-[13px] text-escalate", className)}>
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
        <span className="min-w-0 break-words">{message}</span>
      </div>
    );
  }
  return (
    <Card className={cn("border-escalate/30", className)}>
      <EmptyState
        icon={unreachable ? <PlugZap className="h-5 w-5 text-escalate" /> : <AlertTriangle className="h-5 w-5 text-escalate" />}
        title={unreachable ? "Service not reachable" : "Something went wrong"}
        action={
          onRetry && (
            <Button variant="outline" size="sm" onClick={onRetry}>
              <RefreshCw className="h-3.5 w-3.5" /> Retry
            </Button>
          )
        }
      >
        {unreachable ? (
          <>
            Start the triage service, then retry:
            <code className="mt-2 block rounded-md bg-muted px-3 py-2 text-left font-mono text-[12px] text-fg">
              uvicorn certus.service.app:build --factory --port 8000
            </code>
          </>
        ) : (
          message
        )}
      </EmptyState>
    </Card>
  );
}
