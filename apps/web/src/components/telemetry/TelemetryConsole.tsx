"use client";

import { AlertCircle, Activity } from "lucide-react";
import { useState } from "react";

import { TelemetryTable } from "@/components/telemetry/TelemetryTable";
import { TelemetryToolbar } from "@/components/telemetry/TelemetryToolbar";
import { TraceInspectorModal } from "@/components/telemetry/TraceInspectorModal";
import { useTelemetryLogs } from "@/hooks/use-telemetry-logs";
import type { TelemetryRecord } from "@/lib/api/client";
import { AUTO_REFRESH_INTERVAL_MS, DEFAULT_LIMIT } from "@/lib/telemetry";

const SKELETON_ROWS = 6;

function LoadingRows() {
  return (
    <div
      role="status"
      aria-label="Loading telemetry records"
      className="space-y-2 rounded-lg border p-4"
    >
      {Array.from({ length: SKELETON_ROWS }, (_, index) => (
        <div key={index} className="h-9 animate-pulse rounded-md bg-muted/60" />
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed px-6 py-16 text-center">
      <Activity aria-hidden className="size-8 text-muted-foreground" />
      <p className="max-w-md text-sm text-muted-foreground">
        No telemetry records found. Execute queries in the Chat Studio to
        populate audit trails.
      </p>
    </div>
  );
}

export function TelemetryConsole() {
  const [limit, setLimit] = useState<number>(DEFAULT_LIMIT);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [selected, setSelected] = useState<TelemetryRecord | null>(null);
  const [inspectorOpen, setInspectorOpen] = useState(false);

  const { records, isLoading, isRefreshing, errorMessage, refresh } =
    useTelemetryLogs(limit, autoRefresh ? AUTO_REFRESH_INTERVAL_MS : 0);

  // The selected record is a snapshot: audit rows are immutable, and a poll
  // or limit change must not yank an open trace out from under the reader.
  const handleInspect = (record: TelemetryRecord) => {
    setSelected(record);
    setInspectorOpen(true);
  };

  return (
    <div className="space-y-4">
      <TelemetryToolbar
        limit={limit}
        onLimitChange={setLimit}
        recordCount={records.length}
        isRefreshing={isRefreshing}
        onRefresh={() => void refresh()}
        autoRefresh={autoRefresh}
        onAutoRefreshChange={setAutoRefresh}
      />

      {errorMessage !== null && (
        <div
          role="alert"
          className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-foreground"
        >
          <AlertCircle aria-hidden className="mt-0.5 size-4 shrink-0" />
          <div>
            <p className="font-medium">Could not load telemetry.</p>
            <p className="text-muted-foreground">
              {errorMessage}
              {records.length > 0 && " Showing the last successful result."}
            </p>
          </div>
        </div>
      )}

      {isLoading ? (
        <LoadingRows />
      ) : records.length > 0 ? (
        <TelemetryTable records={records} onInspect={handleInspect} />
      ) : (
        errorMessage === null && <EmptyState />
      )}

      <TraceInspectorModal
        record={selected}
        open={inspectorOpen}
        onOpenChange={setInspectorOpen}
      />
    </div>
  );
}
