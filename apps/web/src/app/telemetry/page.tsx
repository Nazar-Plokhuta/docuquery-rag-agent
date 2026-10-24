import type { Metadata } from "next";

import { TelemetryConsole } from "@/components/telemetry/TelemetryConsole";

export const metadata: Metadata = { title: "Telemetry" };

export default function TelemetryPage() {
  return (
    <div className="mx-auto w-full max-w-7xl space-y-6 px-4 py-8 sm:px-6">
      <header className="space-y-1">
        <h1 className="text-xl font-semibold tracking-tight">
          Telemetry &amp; Observability Console
        </h1>
        <p className="text-sm text-muted-foreground">
          Audit trail of every RAG request: latency, token usage, and the full
          query and response behind each trace.
        </p>
      </header>
      <TelemetryConsole />
    </div>
  );
}
