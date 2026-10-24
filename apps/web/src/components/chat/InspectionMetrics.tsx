import type { ReactNode } from "react";

import type { MessageMetrics } from "@/lib/chat";

interface InspectionMetricsProps {
  metrics: MessageMetrics | null;
  citationCount: number;
}

function formatDuration(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-mono text-xs tabular-nums">{children}</dd>
    </div>
  );
}

export function InspectionMetrics({
  metrics,
  citationCount,
}: InspectionMetricsProps) {
  if (metrics === null) {
    return (
      <p className="text-sm text-muted-foreground">
        Metrics for the latest answer appear here.
      </p>
    );
  }

  return (
    <dl className="space-y-2 text-sm">
      <Row label="Mode">{metrics.mode}</Row>
      <Row label="Latency">{formatDuration(metrics.latencyMs)}</Row>
      {metrics.firstTokenMs !== null && (
        <Row label="First token">{formatDuration(metrics.firstTokenMs)}</Row>
      )}
      {metrics.deltaCount !== null && (
        <Row label="Token events">{metrics.deltaCount}</Row>
      )}
      <Row label="Total tokens">
        {/* Streamed completions carry no usage object. */}
        {metrics.totalTokens ?? "n/a"}
      </Row>
      <Row label="Citations">{citationCount}</Row>
      <Row label="top_k">{metrics.params.topK}</Row>
      <Row label="score_threshold">{metrics.params.scoreThreshold.toFixed(2)}</Row>
    </dl>
  );
}
