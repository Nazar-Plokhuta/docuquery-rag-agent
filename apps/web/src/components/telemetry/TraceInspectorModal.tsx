"use client";

import { useState, type ReactNode } from "react";

import { AnswerMarkdown } from "@/components/chat/AnswerMarkdown";
import { CitationSheet } from "@/components/chat/CitationSheet";
import { CopyButton } from "@/components/telemetry/CopyButton";
import { LatencyBadge } from "@/components/telemetry/LatencyBadge";
import { TokenBar } from "@/components/telemetry/TokenBar";
import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import type { TelemetryRecord } from "@/lib/api/client";
import type { ChatCitation } from "@/lib/chat";
import { formatTimestampUtc } from "@/lib/telemetry";

interface TraceInspectorModalProps {
  record: TelemetryRecord | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function Metric({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="space-y-1 rounded-md border p-3">
      <dt className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
        {label}
      </dt>
      <dd className="font-mono text-sm tabular-nums">{children}</dd>
    </div>
  );
}

function Section({
  title,
  action,
  children,
}: {
  title: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-medium">{title}</h3>
        {action}
      </div>
      {children}
    </section>
  );
}

function TraceDetails({ record }: { record: TelemetryRecord }) {
  const [citation, setCitation] = useState<ChatCitation | null>(null);
  const payload = JSON.stringify(record, null, 2);
  const number = (value: number) => value.toLocaleString("en-US");

  return (
    <>
      <div className="space-y-6 overflow-y-auto px-4 pb-6">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <code className="rounded bg-muted px-2 py-1 font-mono text-xs break-all">
              {record.request_id}
            </code>
            <CopyButton
              text={record.request_id}
              label="Copy request ID"
              iconOnly
              size="icon-sm"
            />
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Badge variant="outline" className="font-mono">
              {record.model_name}
            </Badge>
            <span className="font-mono tabular-nums">
              {formatTimestampUtc(record.created_at)} UTC
            </span>
          </div>
        </div>

        <Section title="Performance">
          <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Metric label="Latency">
              <LatencyBadge ms={record.latency_ms} />
            </Metric>
            <Metric label="Prompt">{number(record.prompt_tokens)}</Metric>
            <Metric label="Completion">{number(record.completion_tokens)}</Metric>
            <Metric label="Total">{number(record.total_tokens)}</Metric>
          </dl>
          <TokenBar
            promptTokens={record.prompt_tokens}
            completionTokens={record.completion_tokens}
          />
          <p className="flex gap-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span aria-hidden className="size-2 rounded-full bg-sky-400" />
              Prompt
            </span>
            <span className="flex items-center gap-1.5">
              <span aria-hidden className="size-2 rounded-full bg-violet-400" />
              Completion
            </span>
          </p>
        </Section>

        <Section title="Query">
          <p className="max-h-48 overflow-y-auto rounded-md border bg-muted/30 p-3 text-sm break-words whitespace-pre-wrap">
            {record.query_text}
          </p>
        </Section>

        <Section title="Response">
          <div className="max-h-96 overflow-y-auto rounded-md border bg-muted/30 p-3 text-sm break-words">
            <AnswerMarkdown
              content={record.response_text}
              citations={null}
              onCitationSelect={setCitation}
            />
          </div>
        </Section>

        <Section
          title="Raw payload"
          action={<CopyButton text={payload} label="Copy Payload" />}
        >
          <pre className="max-h-80 overflow-auto rounded-md border bg-muted/50 p-3 font-mono text-xs">
            <code>{payload}</code>
          </pre>
        </Section>
      </div>

      <CitationSheet
        citation={citation}
        open={citation !== null}
        onOpenChange={(open) => {
          if (!open) setCitation(null);
        }}
      />
    </>
  );
}

export function TraceInspectorModal({
  record,
  open,
  onOpenChange,
}: TraceInspectorModalProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full gap-0 sm:max-w-2xl">
        <SheetHeader>
          <SheetTitle>Trace inspector</SheetTitle>
          <SheetDescription>
            Full audit record for one RAG request.
          </SheetDescription>
        </SheetHeader>
        {/* Keyed so per-trace state (the open citation) never leaks across records. */}
        {record && <TraceDetails key={record.request_id} record={record} />}
      </SheetContent>
    </Sheet>
  );
}
