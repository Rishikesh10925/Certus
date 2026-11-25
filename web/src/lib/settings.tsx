import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export type Theme = "light" | "dark" | "system";

/** Per-browser preferences. They change what is shown, never a threshold. */
export interface Settings {
  theme: Theme;
  /** Default guaranteed miss rate for new inspections and the batch console. */
  epsilon: number;
  imagesPerDay: number;
  minutesPerReview: number;
  batchSize: number;
  showTruth: boolean;
  /** Parallel /v1/inspect requests while a batch is inspected. */
  concurrency: number;
}

export const DEFAULTS: Settings = {
  theme: "system",
  epsilon: 0.05,
  imagesPerDay: 2000,
  minutesPerReview: 1.5,
  batchSize: 48,
  showTruth: true,
  concurrency: 4,
};

const KEY = "certus.settings";

function load(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    const theme = (localStorage.getItem("certus.theme") as Theme | null) ?? DEFAULTS.theme;
    return { ...DEFAULTS, ...(raw ? JSON.parse(raw) : {}), theme };
  } catch {
    return DEFAULTS;
  }
}

interface Ctx {
  settings: Settings;
  update: (patch: Partial<Settings>) => void;
  reset: () => void;
  dark: boolean;
}

const SettingsContext = createContext<Ctx | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(load);
  const [systemDark, setSystemDark] = useState(
    () => typeof matchMedia !== "undefined" && matchMedia("(prefers-color-scheme: dark)").matches,
  );

  useEffect(() => {
    const mq = matchMedia("(prefers-color-scheme: dark)");
    const on = (e: MediaQueryListEvent) => setSystemDark(e.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);

  const dark = settings.theme === "dark" || (settings.theme === "system" && systemDark);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  useEffect(() => {
    try {
      const { theme, ...rest } = settings;
      localStorage.setItem(KEY, JSON.stringify(rest));
      localStorage.setItem("certus.theme", theme);
    } catch {
      /* storage blocked: settings still work for this session */
    }
  }, [settings]);

  const update = useCallback((patch: Partial<Settings>) => setSettings((s) => ({ ...s, ...patch })), []);
  const reset = useCallback(() => setSettings({ ...DEFAULTS, theme: settings.theme }), [settings.theme]);

  const value = useMemo(() => ({ settings, update, reset, dark }), [settings, update, reset, dark]);
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings outside SettingsProvider");
  return ctx;
}
