"use client";

import useSWR from "swr";

import {
  API_ENDPOINTS,
  getTelemetryLogs,
  type TelemetryRecord,
} from "@/lib/api/client";

export interface TelemetryLogsState {
  records: TelemetryRecord[];
  /** True only until the first response (or failure) for the current limit. */
  isLoading: boolean;
  /** True while any request is in flight, including background polls. */
  isRefreshing: boolean;
  /** Already normalised by `request()` via `normalizeApiError`. */
  errorMessage: string | null;
  refresh: () => Promise<void>;
}

export function useTelemetryLogs(
  limit: number,
  autoRefreshMs: number,
): TelemetryLogsState {
  const { data, error, isLoading, isValidating, mutate } = useSWR(
    [API_ENDPOINTS.analyticsRecent, limit],
    ([, currentLimit]) => getTelemetryLogs(currentLimit),
    {
      refreshInterval: autoRefreshMs,
      // Switching the limit keeps the current rows on screen until the
      // larger or smaller page arrives, instead of flashing the skeleton.
      keepPreviousData: true,
      // A manual refresh is the retry path; backoff retries would hide it.
      shouldRetryOnError: false,
    },
  );

  return {
    records: data?.records ?? [],
    isLoading,
    isRefreshing: isValidating,
    errorMessage: error instanceof Error ? error.message : null,
    refresh: async () => {
      await mutate();
    },
  };
}
