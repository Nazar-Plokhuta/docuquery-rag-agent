import { CircleAlert, Loader2, OctagonX, Unplug } from "lucide-react";
import { memo, type ReactNode } from "react";

import { AnswerMarkdown } from "@/components/chat/AnswerMarkdown";
import { CitationBadge } from "@/components/chat/CitationBadge";
import { GroundingShieldBanner } from "@/components/chat/GroundingShieldBanner";
import {
  evaluateGuardrail,
  extractCitations,
  hidePartialCitation,
  type AssistantMessage,
  type ChatCitation,
  type ChatMessage,
} from "@/lib/chat";

interface MessageItemProps {
  message: ChatMessage;
  onCitationSelect: (citation: ChatCitation) => void;
}

function Notice({
  icon,
  tone,
  children,
}: {
  icon: ReactNode;
  tone: "muted" | "error" | "warning";
  children: ReactNode;
}) {
  const toneClass = {
    muted: "text-muted-foreground",
    error: "text-rose-300",
    warning: "text-amber-300",
  }[tone];
  return (
    <p className={`flex items-center gap-1.5 text-xs ${toneClass}`}>
      {icon}
      {children}
    </p>
  );
}

function AssistantBody({
  message,
  onCitationSelect,
}: {
  message: AssistantMessage;
  onCitationSelect: (citation: ChatCitation) => void;
}) {
  const verdict = evaluateGuardrail(message);
  const streaming = message.state === "streaming";
  const content = streaming
    ? hidePartialCitation(message.content)
    : message.content;
  const sources = message.citations ?? extractCitations(message.content);

  return (
    <div className="min-w-0 flex-1 space-y-3 text-sm">
      {message.state === "retrieving" && (
        <Notice
          tone="muted"
          icon={<Loader2 aria-hidden className="size-3.5 animate-spin" />}
        >
          Retrieving context…
        </Notice>
      )}

      {verdict && <GroundingShieldBanner verdict={verdict} />}

      {content !== "" && (
        <div
          className={
            verdict?.kind === "refusal" ? "text-muted-foreground" : undefined
          }
        >
          <AnswerMarkdown
            content={content}
            citations={message.citations}
            onCitationSelect={onCitationSelect}
          />
          {streaming && (
            <span
              aria-hidden
              className="ml-0.5 inline-block h-4 w-1.5 translate-y-0.5 animate-pulse bg-foreground/70"
            />
          )}
        </div>
      )}

      {message.state === "complete" && message.content.trim() === "" && (
        <Notice tone="muted" icon={<CircleAlert aria-hidden className="size-3.5" />}>
          The model returned an empty response.
        </Notice>
      )}
      {message.truncated && (
        <Notice tone="warning" icon={<Unplug aria-hidden className="size-3.5" />}>
          The connection closed before the stream finished; this answer may be
          incomplete.
        </Notice>
      )}
      {message.state === "stopped" && (
        <Notice tone="muted" icon={<OctagonX aria-hidden className="size-3.5" />}>
          Generation stopped.
        </Notice>
      )}
      {message.state === "error" && (
        <div role="alert">
          <Notice
            tone="error"
            icon={<CircleAlert aria-hidden className="size-3.5" />}
          >
            {message.errorMessage ?? "Something went wrong."}
          </Notice>
        </div>
      )}

      {sources.length > 0 && !streaming && (
        <div className="flex flex-wrap items-center gap-1 border-t border-border pt-2">
          <span className="mr-1 text-xs text-muted-foreground">Sources</span>
          {sources.map((citation) => (
            <CitationBadge
              key={`${citation.source}|${citation.section}|${citation.chunkId ?? ""}`}
              citation={citation}
              onSelect={onCitationSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export const MessageItem = memo(function MessageItem({
  message,
  onCitationSelect,
}: MessageItemProps) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <p className="max-w-[85%] rounded-2xl rounded-br-sm bg-secondary px-3.5 py-2 text-sm whitespace-pre-wrap break-words">
          {message.content}
        </p>
      </div>
    );
  }

  return (
    <div className="flex gap-3">
      <span
        aria-hidden
        className="grid size-6 shrink-0 place-items-center rounded-md bg-foreground font-mono text-xs text-background"
      >
        D
      </span>
      <AssistantBody message={message} onCitationSelect={onCitationSelect} />
    </div>
  );
});
