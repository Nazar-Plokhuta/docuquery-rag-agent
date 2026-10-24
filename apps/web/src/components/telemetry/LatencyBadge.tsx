import { Badge } from "@/components/ui/badge";
import {
  formatLatency,
  LATENCY_TIER_LABEL,
  latencyTier,
  type LatencyTier,
} from "@/lib/telemetry";
import { cn } from "@/lib/utils";

const TIER_CLASS: Record<LatencyTier, string> = {
  fast: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  moderate: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  high: "border-rose-500/30 bg-rose-500/10 text-rose-300",
};

export function LatencyBadge({ ms }: { ms: number }) {
  const tier = latencyTier(ms);
  return (
    <Badge
      variant="outline"
      title={`${LATENCY_TIER_LABEL[tier]} latency`}
      className={cn("font-mono tabular-nums", TIER_CLASS[tier])}
    >
      {formatLatency(ms)}
    </Badge>
  );
}
