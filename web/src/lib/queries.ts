import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { readMetrics } from "./metrics";

export const useInfo = () =>
  useQuery({ queryKey: ["info"], queryFn: api.info, refetchInterval: 15_000, retry: 1 });

export const useCalibration = (cluster: string | undefined) =>
  useQuery({
    queryKey: ["calibration", cluster],
    queryFn: () => api.calibration(cluster!),
    enabled: !!cluster,
    staleTime: Infinity, // the artifact is loaded once at service startup
  });

export const useMetrics = (intervalMs = 5000) =>
  useQuery({
    queryKey: ["metrics"],
    queryFn: async () => readMetrics(await api.metrics()),
    refetchInterval: intervalMs,
  });

export const useResearch = <T,>(name: string) =>
  useQuery({ queryKey: ["research", name], queryFn: () => api.research<T>(name), staleTime: Infinity });

export const useResearchIndex = () =>
  useQuery({ queryKey: ["research-index"], queryFn: api.researchIndex, staleTime: Infinity });

export const useRiskStatement = (p: {
  cluster?: string;
  epsilon: number;
  decision?: string;
  shifted?: boolean;
  defect_type?: string;
}) =>
  useQuery({
    queryKey: ["risk", p],
    queryFn: () => api.riskStatement({ ...p, cluster: p.cluster! }),
    enabled: !!p.cluster,
    staleTime: Infinity,
  });
