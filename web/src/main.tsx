import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import { SettingsProvider } from "./lib/settings";
import { TooltipProvider } from "./components/ui/radix";
import { ApiError } from "./lib/api";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      // Client errors (4xx) will not fix themselves; retry only transient failures.
      retry: (count, err) => !(err instanceof ApiError && err.status >= 400 && err.status < 500) && count < 2,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <SettingsProvider>
        <TooltipProvider>
          <App />
        </TooltipProvider>
      </SettingsProvider>
    </QueryClientProvider>
  </StrictMode>,
);
