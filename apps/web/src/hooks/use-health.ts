"use client";

import useSWR from "swr";

import {
  fetchHealth,
  HEALTH_ENDPOINT,
  type HealthResponse,
} from "@/lib/api/health";

const POLL_INTERVAL_MS = 15_000;

export type HealthState =
  | { status: "checking" }
  | { status: "online"; health: HealthResponse }
  | { status: "offline" };

export function useHealth(): HealthState {
  const { data, error, isLoading } = useSWR<HealthResponse>(
    HEALTH_ENDPOINT,
    fetchHealth,
    {
      refreshInterval: POLL_INTERVAL_MS,
      // Keep the red state visible while the API is down instead of
      // retrying on SWR's exponential backoff schedule.
      shouldRetryOnError: false,
    },
  );

  if (error) return { status: "offline" };
  if (data) return { status: "online", health: data };
  return isLoading ? { status: "checking" } : { status: "offline" };
}
