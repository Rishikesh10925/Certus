import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
  Activity,
  BookOpen,
  ClipboardCheck,
  Gauge,
  LayoutDashboard,
  Menu,
  Monitor,
  Moon,
  ScanSearch,
  Settings as SettingsIcon,
  SlidersHorizontal,
  Sun,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/cn";
import { useInfo } from "@/lib/queries";
import { useSettings, type Theme } from "@/lib/settings";
import { Tip } from "../ui/radix";

const NAV = [
  { group: "Operate", items: [
    { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
    { to: "/triage", label: "Batch triage", icon: SlidersHorizontal },
    { to: "/inspect", label: "Inspect image", icon: ScanSearch },
    { to: "/audit", label: "Audit queue", icon: ClipboardCheck },
  ]},
  { group: "Understand", items: [
    { to: "/calibration", label: "Calibration", icon: Gauge },
    { to: "/monitoring", label: "Monitoring", icon: Activity },
    { to: "/research", label: "Research results", icon: BookOpen },
  ]},
];

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="grid h-8 w-8 place-items-center rounded-lg bg-accent text-accent-fg shadow-sm">
        <svg viewBox="0 0 32 32" className="h-5 w-5" aria-hidden>
          <path d="M21.5 11.2A7 7 0 1 0 21.5 20.8" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" />
          <circle cx="22.2" cy="16" r="2.2" fill="currentColor" />
        </svg>
      </div>
      <div className="leading-tight">
        <div className="text-[15px] font-semibold tracking-tight">Certus</div>
        <div className="hidden text-[11px] text-subtle sm:block">Risk-controlled triage</div>
      </div>
    </div>
  );
}

function ServiceStatus() {
  const info = useInfo();
  const ok = info.isSuccess && !info.isError;
  const label = info.isPending ? "Connecting…" : ok ? "Service online" : "Service offline";
  return (
    <Tip
      content={
        ok
          ? `${info.data.calibration_version} · ${info.data.model_version}`
          : "Start it with: uvicorn certus.service.app:build --factory --port 8000"
      }
    >
      <div className="flex items-center gap-2 rounded-lg border border-border bg-surface px-2.5 py-1.5 text-[12px]">
        <span className="relative flex h-2 w-2">
          {ok && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-clear opacity-50" />}
          <span className={cn("relative inline-flex h-2 w-2 rounded-full", info.isPending ? "bg-subtle" : ok ? "bg-clear" : "bg-escalate")} />
        </span>
        <span className="hidden whitespace-nowrap font-medium sm:inline">{label}</span>
        {ok && <span className="num hidden text-subtle sm:inline">{info.data.calibration_version}</span>}
      </div>
    </Tip>
  );
}

function ThemeToggle() {
  const { settings, update } = useSettings();
  const next: Record<Theme, Theme> = { light: "dark", dark: "system", system: "light" };
  const Icon = settings.theme === "light" ? Sun : settings.theme === "dark" ? Moon : Monitor;
  return (
    <Tip content={`Theme: ${settings.theme} (click to change)`}>
      <button
        onClick={() => update({ theme: next[settings.theme] })}
        className="grid h-8 w-8 place-items-center rounded-lg border border-border bg-surface text-subtle transition-colors hover:text-fg"
        aria-label={`Theme: ${settings.theme}`}
      >
        <Icon className="h-4 w-4" />
      </button>
    </Tip>
  );
}

function SideNav({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="flex flex-1 flex-col gap-6 px-3 py-4">
      {NAV.map((g) => (
        <div key={g.group}>
          <div className="label mb-1.5 px-2.5">{g.group}</div>
          <ul className="space-y-0.5">
            {g.items.map(({ to, label, icon: Icon, end }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end={end}
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    cn(
                      "group flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13.5px] font-medium transition-colors",
                      isActive ? "bg-accent/10 text-accent" : "text-subtle hover:bg-muted hover:text-fg",
                    )
                  }
                >
                  <Icon className="h-4 w-4 shrink-0" />
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}
      <div className="mt-auto">
        <NavLink
          to="/settings"
          onClick={onNavigate}
          className={({ isActive }) =>
            cn(
              "flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13.5px] font-medium transition-colors",
              isActive ? "bg-accent/10 text-accent" : "text-subtle hover:bg-muted hover:text-fg",
            )
          }
        >
          <SettingsIcon className="h-4 w-4" /> Settings
        </NavLink>
      </div>
    </nav>
  );
}

export function AppShell() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  useEffect(() => {
    setOpen(false);
    window.scrollTo({ top: 0 });
  }, [location.pathname]);

  return (
    <div className="min-h-screen">
      {/* Desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-60 flex-col border-r border-border bg-surface lg:flex">
        <div className="flex h-16 items-center px-5">
          <Logo />
        </div>
        <SideNav />
      </aside>

      {/* Mobile drawer */}
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 flex w-64 flex-col border-r border-border bg-surface animate-fade-in">
            <div className="flex h-16 items-center justify-between px-5">
              <Logo />
              <button onClick={() => setOpen(false)} aria-label="Close menu" className="rounded-md p-1 text-subtle">
                <X className="h-5 w-5" />
              </button>
            </div>
            <SideNav onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}

      <div className="lg:pl-60">
        <header className="sticky top-0 z-20 flex h-16 items-center gap-3 border-b border-border bg-bg/80 px-4 backdrop-blur-md md:px-8">
          <button
            className="grid h-8 w-8 place-items-center rounded-lg border border-border bg-surface lg:hidden"
            onClick={() => setOpen(true)}
            aria-label="Open menu"
          >
            <Menu className="h-4 w-4" />
          </button>
          <div className="lg:hidden">
            <Logo />
          </div>
          <div className="ml-auto flex items-center gap-2">
            <ServiceStatus />
            <ThemeToggle />
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1400px] px-4 py-6 md:px-8 md:py-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
