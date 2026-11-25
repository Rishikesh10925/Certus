import { NavLink, Outlet, useLocation, Link } from "react-router-dom";
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
} from "lucide-react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { cn } from "@/lib/cn";
import { useInfo } from "@/lib/queries";
import { useSettings, type Theme } from "@/lib/settings";
import { Sheet, Tip } from "../ui/radix";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true, group: "Operate" },
  { to: "/triage", label: "Batch triage", icon: SlidersHorizontal, group: "Operate" },
  { to: "/inspect", label: "Inspect image", icon: ScanSearch, group: "Operate" },
  { to: "/audit", label: "Audit queue", icon: ClipboardCheck, group: "Operate" },
  { to: "/calibration", label: "Calibration", icon: Gauge, group: "Understand" },
  { to: "/monitoring", label: "Monitoring", icon: Activity, group: "Understand" },
  { to: "/research", label: "Research results", icon: BookOpen, group: "Understand" },
  { to: "/settings", label: "Settings", icon: SettingsIcon, group: "Console" },
];

export function BrandMark({ className }: { className?: string }) {
  return (
    <span className={cn("grid h-[30px] w-[30px] shrink-0 place-items-center rounded-lg bg-accent", className)} aria-hidden>
      <svg viewBox="0 0 32 32" className="h-5 w-5">
        <path d="M5 22h22" stroke="rgb(var(--accent-ink) / 0.55)" strokeWidth="2.6" strokeLinecap="round" />
        <path d="M9 13l3.6 3.6L23 6.5" fill="none" stroke="rgb(var(--accent-ink))" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </span>
  );
}

function Brand() {
  return (
    <Link to="/" className="flex shrink-0 items-center gap-2.5 rounded-md text-fg" aria-label="Certus overview">
      <BrandMark />
      <span className="leading-none">
        <b className="block text-[18px] font-bold tracking-[-0.02em]">Certus</b>
        <small className="mt-0.5 block text-[10.5px] font-medium uppercase tracking-[0.08em] text-faint">Triage console</small>
      </span>
    </Link>
  );
}

function ServiceStatus() {
  const info = useInfo();
  const ok = info.isSuccess && !info.isError;
  const label = info.isPending ? "Connecting" : ok ? "Online" : "Offline";
  return (
    <Tip
      content={
        ok
          ? `${info.data.calibration_version} · ${info.data.model_version}`
          : "Start it with: uvicorn certus.service.app:build --factory --port 8000"
      }
    >
      <div
        className="flex h-[34px] items-center gap-2 rounded-md border border-border bg-muted px-2.5 text-[12.5px] text-subtle"
        role="status"
        aria-label={`Service ${label.toLowerCase()}${ok ? `, calibration ${info.data.calibration_version}` : ""}`}
        tabIndex={0}
      >
        <span
          className={cn(
            "h-2 w-2 shrink-0 rounded-full",
            info.isPending ? "bg-faint" : ok ? "animate-pulse-ring bg-clear" : "bg-danger",
          )}
        />
        <b className="font-semibold text-fg">{label}</b>
        {ok && (
          <>
            <span className="hidden h-3.5 w-px bg-border-strong sm:block" aria-hidden />
            <span className="hidden whitespace-nowrap sm:inline">
              cal <span className="num">{info.data.calibration_version}</span>
            </span>
          </>
        )}
      </div>
    </Tip>
  );
}

function ThemeToggle() {
  const { settings, update } = useSettings();
  const next: Record<Theme, Theme> = { light: "dark", dark: "system", system: "light" };
  const Icon = settings.theme === "light" ? Sun : settings.theme === "dark" ? Moon : Monitor;
  return (
    <Tip content={`Theme: ${settings.theme} (click for ${next[settings.theme]})`}>
      <button
        type="button"
        onClick={() => update({ theme: next[settings.theme] })}
        className="grid h-[38px] w-[38px] shrink-0 place-items-center rounded-md border border-border bg-surface text-fg transition-colors hover:border-border-strong hover:bg-muted"
        aria-label={`Theme: ${settings.theme}. Switch to ${next[settings.theme]}`}
      >
        <Icon className="h-[18px] w-[18px]" />
      </button>
    </Tip>
  );
}

/** Header tabs. They scroll when the row is too narrow; the edge with hidden tabs fades. */
function Tabs() {
  const ref = useRef<HTMLDivElement>(null);
  const location = useLocation();
  const [edges, setEdges] = useState({ start: false, end: false });

  const measure = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    setEdges({ start: el.scrollLeft > 2, end: el.scrollLeft + el.clientWidth < el.scrollWidth - 2 });
  }, []);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [measure]);

  useEffect(() => {
    const active = ref.current?.querySelector<HTMLElement>('[aria-current="page"]');
    active?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [location.pathname]);

  return (
    <nav aria-label="Primary" className="hidden h-full min-w-0 flex-1 lg:block">
      <div
        ref={ref}
        onScroll={measure}
        className={cn("no-scrollbar flex h-full gap-0.5 overflow-x-auto", edges.start && "fade-x-start", edges.end && "fade-x-end")}
      >
        {NAV.map(({ to, label, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              cn(
                "relative flex h-full shrink-0 items-center whitespace-nowrap px-3 text-[14px] transition-colors focus-visible:ring-inset focus-visible:ring-offset-0",
                "after:absolute after:inset-x-2.5 after:bottom-0 after:h-[3px] after:rounded-t-[3px]",
                isActive ? "font-bold text-fg after:bg-accent" : "font-medium text-subtle hover:bg-muted hover:text-fg",
              )
            }
          >
            {label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}

function DrawerNav({ onNavigate }: { onNavigate: () => void }) {
  const groups = [...new Set(NAV.map((n) => n.group))];
  return (
    <nav aria-label="Primary" className="space-y-5 px-3 py-4">
      {groups.map((g) => (
        <div key={g}>
          <div className="kpi-label mb-1.5 px-2.5">{g}</div>
          <ul className="space-y-0.5">
            {NAV.filter((n) => n.group === g).map(({ to, label, icon: Icon, end }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end={end}
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    cn(
                      "relative flex items-center gap-2.5 rounded-md px-2.5 py-2.5 text-[14.5px] transition-colors",
                      isActive
                        ? "bg-accent-soft font-bold text-fg before:absolute before:inset-y-1.5 before:left-0 before:w-[3px] before:rounded-r before:bg-accent"
                        : "font-medium text-subtle hover:bg-muted hover:text-fg",
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
      <a
        href="#main"
        className="sr-only z-50 rounded-md bg-accent px-3 py-2 text-accent-ink focus:not-sr-only focus:fixed focus:left-3 focus:top-3"
      >
        Skip to content
      </a>
      <header className="border-b border-border bg-surface">
        <div className="flex h-[60px] items-center gap-3 px-4 md:px-6 lg:gap-6 xl:px-8">
          <button
            type="button"
            className="grid h-[38px] w-[38px] shrink-0 place-items-center rounded-md border border-border bg-surface hover:bg-muted lg:hidden"
            onClick={() => setOpen(true)}
            aria-label="Open navigation"
            aria-expanded={open}
          >
            <Menu className="h-[18px] w-[18px]" />
          </button>
          <Brand />
          <Tabs />
          <div className="ml-auto flex shrink-0 items-center gap-2 sm:gap-3">
            <ServiceStatus />
            <ThemeToggle />
          </div>
        </div>
      </header>

      <Sheet
        open={open}
        onOpenChange={setOpen}
        side="left"
        title={
          <span className="flex items-center gap-2.5">
            <BrandMark className="h-7 w-7" /> Certus
          </span>
        }
        description="Risk-controlled triage console"
      >
        <DrawerNav onNavigate={() => setOpen(false)} />
      </Sheet>

      <main id="main" tabIndex={-1} className="w-full px-4 pb-12 pt-5 focus:outline-none md:px-6 md:pt-6 xl:px-8">
        <Outlet />
      </main>
    </div>
  );
}
