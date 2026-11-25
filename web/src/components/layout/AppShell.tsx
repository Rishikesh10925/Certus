import { NavLink, Outlet, useLocation } from "react-router-dom";
import { Menu, Monitor, Moon, Sun } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { cn } from "@/lib/cn";
import { useInfo } from "@/lib/queries";
import { useSettings, type Theme } from "@/lib/settings";
import { Sheet, Tip } from "../ui/radix";

/* 16px line icons drawn to the instrument's 1.6 stroke. */
const I = ({ children }: { children: ReactNode }) => (
  <svg viewBox="0 0 16 16" className="h-4 w-4 shrink-0 fill-none stroke-current" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    {children}
  </svg>
);
const ICONS = {
  overview: (
    <I>
      <rect x="2" y="2" width="5" height="5" rx="1" />
      <rect x="9" y="2" width="5" height="5" rx="1" />
      <rect x="2" y="9" width="5" height="5" rx="1" />
      <rect x="9" y="9" width="5" height="5" rx="1" />
    </I>
  ),
  triage: (
    <I>
      <path d="M2 12.5h12" />
      <path d="M10.5 2.5v11" />
      <circle cx="4.5" cy="9" r="1.3" />
      <circle cx="7.5" cy="7" r="1.3" />
      <circle cx="13" cy="9" r="1.3" />
    </I>
  ),
  inspect: (
    <I>
      <circle cx="7" cy="7" r="4.5" />
      <path d="M10.5 10.5L14 14" />
    </I>
  ),
  audit: (
    <I>
      <path d="M6 4h8M6 8h8M6 12h8" />
      <path d="M2 4l1 1 1.5-2M2 8l1 1 1.5-2" />
      <circle cx="3" cy="12" r=".6" />
    </I>
  ),
  calibration: (
    <I>
      <circle cx="8" cy="8" r="6" />
      <circle cx="8" cy="8" r="2.5" />
      <path d="M8 1v2.5M8 12.5V15M1 8h2.5M12.5 8H15" />
    </I>
  ),
  monitoring: (
    <I>
      <path d="M1.5 8.5h3l2-5 3 9 2-4h3" />
    </I>
  ),
  research: (
    <I>
      <path d="M6 1.5h4M6.8 1.5v4.2L2.6 13a1 1 0 00.9 1.5h9a1 1 0 00.9-1.5L9.2 5.7V1.5" />
      <path d="M4.3 10h7.4" />
    </I>
  ),
  settings: (
    <I>
      <circle cx="8" cy="8" r="2.2" />
      <path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M3.4 12.6l1.4-1.4M11.2 4.8l1.4-1.4" />
    </I>
  ),
};

const NAV = [
  { group: "Operate", items: [
    { to: "/", label: "Overview", icon: ICONS.overview, end: true },
    { to: "/triage", label: "Batch triage", icon: ICONS.triage },
    { to: "/inspect", label: "Inspect image", icon: ICONS.inspect },
    { to: "/audit", label: "Audit queue", icon: ICONS.audit },
  ]},
  { group: "Model", items: [
    { to: "/calibration", label: "Calibration", icon: ICONS.calibration },
    { to: "/monitoring", label: "Monitoring", icon: ICONS.monitoring },
    { to: "/research", label: "Research results", icon: ICONS.research },
  ]},
  { group: "Console", items: [{ to: "/settings", label: "Settings", icon: ICONS.settings }] },
];

/** A graphite square with the lime λ̂ cut through a baseline. */
export function BrandMark({ className }: { className?: string }) {
  return (
    <span className={cn("relative block h-[26px] w-[26px] shrink-0 overflow-hidden rounded-[7px] border border-border-strong bg-sunken", className)} aria-hidden>
      <span className="absolute inset-x-1 bottom-[7px] h-px bg-border-strong" />
      <span className="absolute bottom-1 left-[15px] top-1 w-0.5 rounded-sm bg-signal" />
    </span>
  );
}

function Brand() {
  return (
    <div className="flex items-center gap-2.5 px-1.5 pb-5 pt-0.5">
      <BrandMark />
      <div className="leading-tight">
        <div className="text-[15px] font-semibold tracking-[-0.01em]">Certus</div>
        <div className="font-mono text-[10.5px] tracking-[0.04em] text-faint">inspection triage</div>
      </div>
    </div>
  );
}

function RailNav({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav aria-label="Primary">
      {NAV.map((g) => (
        <div key={g.group}>
          <div className="px-2 pb-1.5 pt-3 text-[10.5px] uppercase tracking-[0.1em] text-faint">{g.group}</div>
          <ul className="grid gap-px">
            {g.items.map(({ to, label, icon, end }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end={end}
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    cn(
                      "relative flex items-center gap-2.5 rounded-[5px] px-2 py-[7px] text-[13.5px] transition-colors",
                      isActive
                        ? "bg-accent-soft font-medium text-fg before:absolute before:-left-3 before:bottom-2 before:top-2 before:w-0.5 before:rounded-sm before:bg-signal [&_svg]:text-accent"
                        : "text-subtle hover:bg-sunken hover:text-fg",
                    )
                  }
                >
                  {icon}
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}

function Pulse({ state }: { state: "ok" | "down" | "pending" }) {
  return (
    <span
      className={cn("h-[7px] w-[7px] shrink-0 rounded-full", state === "ok" ? "animate-pulse-ring bg-clear" : state === "down" ? "bg-danger" : "bg-faint")}
      aria-hidden
    />
  );
}

function StatusCard() {
  const info = useInfo();
  const ok = info.isSuccess;
  const state = info.isPending ? "pending" : ok ? "ok" : "down";
  return (
    <div className="mt-auto grid gap-1.5 rounded-lg border border-border bg-muted p-2.5" role="status" aria-label="Service status">
      <div className="flex items-center gap-2 text-[12.5px]">
        <Pulse state={state} />
        {info.isPending ? "Connecting…" : ok ? "Service online" : "Service offline"}
      </div>
      {ok ? (
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-2 gap-y-0.5 text-[11.5px]">
          <dt className="text-faint">calib</dt>
          <dd className="truncate font-mono text-subtle">{info.data.calibration_version}</dd>
          <dt className="text-faint">model</dt>
          <dd className="truncate font-mono text-subtle" title={info.data.model_version}>{info.data.model_version}</dd>
          <dt className="text-faint">api</dt>
          <dd className="truncate font-mono text-subtle">v{info.data.version}</dd>
        </dl>
      ) : !info.isPending ? (
        <p className="text-[11.5px] leading-snug text-faint">
          Start it with <code className="break-all font-mono">uvicorn certus.service.app:build --factory --port 8000</code>
        </p>
      ) : null}
    </div>
  );
}

function ThemeToggle({ withLabel }: { withLabel?: boolean }) {
  const { settings, update } = useSettings();
  const next: Record<Theme, Theme> = { light: "dark", dark: "system", system: "light" };
  const Icon = settings.theme === "light" ? Sun : settings.theme === "dark" ? Moon : Monitor;
  return (
    <Tip content={`Theme: ${settings.theme}. Click for ${next[settings.theme]}.`}>
      <button
        type="button"
        onClick={() => update({ theme: next[settings.theme] })}
        className={cn(
          "inline-flex h-8 shrink-0 items-center justify-center gap-2 rounded-[5px] border border-border-strong bg-surface text-[12.5px] text-fg transition-colors hover:bg-sunken",
          withLabel ? "px-2.5" : "w-8",
        )}
        aria-label={`Theme: ${settings.theme}. Switch to ${next[settings.theme]}`}
      >
        <Icon className="h-4 w-4" strokeWidth={1.6} />
        {withLabel && <span className="capitalize">{settings.theme}</span>}
      </button>
    </Tip>
  );
}

export function AppShell() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const info = useInfo();
  useEffect(() => {
    setOpen(false);
    window.scrollTo({ top: 0 });
  }, [location.pathname]);

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[212px_minmax(0,1fr)]">
      <a
        href="#main"
        className="sr-only z-50 rounded-[5px] bg-accent px-3 py-2 text-accent-ink focus:not-sr-only focus:fixed focus:left-3 focus:top-3"
      >
        Skip to content
      </a>

      {/* Left rail (desktop) */}
      <aside className="sticky top-0 hidden h-screen flex-col border-r border-border bg-surface/[.86] px-3 py-4 lg:flex">
        <Brand />
        <RailNav />
        <div className="mt-auto grid gap-2 pt-4">
          <StatusCard />
          <ThemeToggle withLabel />
        </div>
      </aside>

      <div className="min-w-0">
        {/* Top bar (mobile and tablet) */}
        <div className="sticky top-0 z-40 flex items-center gap-2.5 border-b border-border bg-bg/90 px-4 py-2.5 backdrop-blur-md lg:hidden">
          <button
            type="button"
            className="grid h-8 w-8 place-items-center rounded-[5px] border border-border-strong bg-surface hover:bg-sunken"
            onClick={() => setOpen(true)}
            aria-label="Open navigation"
            aria-expanded={open}
          >
            <Menu className="h-4 w-4" strokeWidth={1.6} />
          </button>
          <BrandMark />
          <b className="font-semibold">Certus</b>
          <span className="flex-1" />
          <span className="flex min-w-0 items-center gap-1.5 font-mono text-[10.5px] text-subtle" role="status">
            <Pulse state={info.isPending ? "pending" : info.isSuccess ? "ok" : "down"} />
            <span className="truncate">{info.isSuccess ? info.data.calibration_version : info.isPending ? "connecting" : "offline"}</span>
          </span>
          <ThemeToggle />
        </div>

        <Sheet
          open={open}
          onOpenChange={setOpen}
          side="left"
          title={
            <span className="flex items-center gap-2.5">
              <BrandMark /> Certus
            </span>
          }
          description="inspection triage"
        >
          <div className="flex min-h-full flex-col gap-4 px-3 pb-4">
            <RailNav onNavigate={() => setOpen(false)} />
            <StatusCard />
          </div>
        </Sheet>

        <main id="main" tabIndex={-1} className="w-full px-4 pb-10 pt-4 focus:outline-none md:px-6 md:pt-5">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
