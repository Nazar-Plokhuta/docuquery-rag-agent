export const LIMIT_OPTIONS = [25, 50, 100, 250] as const;
export const DEFAULT_LIMIT = 50;
export const AUTO_REFRESH_INTERVAL_MS = 10_000;

export type LatencyTier = "fast" | "moderate" | "high";

const FAST_BELOW_MS = 1_000;
const MODERATE_UP_TO_MS = 3_000;

export function isLimitOption(value: number): value is (typeof LIMIT_OPTIONS)[number] {
  return (LIMIT_OPTIONS as readonly number[]).includes(value);
}

export function latencyTier(ms: number): LatencyTier {
  if (ms < FAST_BELOW_MS) return "fast";
  return ms <= MODERATE_UP_TO_MS ? "moderate" : "high";
}

export const LATENCY_TIER_LABEL: Record<LatencyTier, string> = {
  fast: "Fast",
  moderate: "Moderate",
  high: "High",
};

export function formatLatency(ms: number): string {
  return `${Math.round(ms).toLocaleString("en-US")} ms`;
}

const HAS_TIME_ZONE = /(?:Z|[+-]\d{2}:?\d{2})$/i;

function parseTimestamp(iso: string): Date {
  // A zone-less value would be read as local time; the API contract is UTC.
  return new Date(HAS_TIME_ZONE.test(iso) ? iso : `${iso}Z`);
}

/** Renders an ISO-8601 instant as `HH:mm:ss YYYY-MM-DD` in UTC. */
export function formatTimestampUtc(iso: string): string {
  const date = parseTimestamp(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const time = date.toISOString().slice(11, 19);
  const day = date.toISOString().slice(0, 10);
  return `${time} ${day}`;
}

/** Share of the total taken by the prompt, as a 0–100 percentage. */
export function promptSharePercent(
  promptTokens: number,
  completionTokens: number,
): number {
  const total = promptTokens + completionTokens;
  return total > 0 ? (promptTokens / total) * 100 : 0;
}
