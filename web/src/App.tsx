import { lazy, Suspense } from "react";
import { createBrowserRouter, Link, RouterProvider, useRouteError } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { Button, Skeleton } from "./components/ui/primitives";
import { EmptyState } from "./components/States";
import OverviewPage from "./pages/Overview";

const TriagePage = lazy(() => import("./pages/Triage"));
const InspectPage = lazy(() => import("./pages/Inspect"));
const CalibrationPage = lazy(() => import("./pages/Calibration"));
const AuditPage = lazy(() => import("./pages/Audit"));
const MonitoringPage = lazy(() => import("./pages/Monitoring"));
const ResearchPage = lazy(() => import("./pages/Research"));
const SettingsPage = lazy(() => import("./pages/Settings"));

function Page({ children }: { children: React.ReactNode }) {
  return (
    <Suspense
      fallback={
        <div className="space-y-4">
          <Skeleton className="h-10 w-72" />
          <Skeleton className="h-64" />
        </div>
      }
    >
      <div className="animate-fade-in">{children}</div>
    </Suspense>
  );
}

function NotFound() {
  return (
    <EmptyState title="Page not found" action={<Link to="/"><Button>Back to overview</Button></Link>}>
      There is no page at this address.
    </EmptyState>
  );
}

function RouteError() {
  const err = useRouteError() as Error;
  return (
    <EmptyState title="This page crashed" action={<Button onClick={() => location.reload()}>Reload</Button>}>
      {err?.message ?? String(err)}
    </EmptyState>
  );
}

const router = createBrowserRouter([
  {
    element: <AppShell />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <Page><OverviewPage /></Page> },
      { path: "triage", element: <Page><TriagePage /></Page> },
      { path: "inspect", element: <Page><InspectPage /></Page> },
      { path: "calibration", element: <Page><CalibrationPage /></Page> },
      { path: "audit", element: <Page><AuditPage /></Page> },
      { path: "monitoring", element: <Page><MonitoringPage /></Page> },
      { path: "research", element: <Page><ResearchPage /></Page> },
      { path: "settings", element: <Page><SettingsPage /></Page> },
      { path: "*", element: <NotFound /> },
    ],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
