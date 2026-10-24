"use client";

import { MessagesSquare, SlidersHorizontal, Trash2 } from "lucide-react";
import Link from "next/link";
import {
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type UIEvent,
} from "react";

import { ChatInput } from "@/components/chat/ChatInput";
import { CitationSheet } from "@/components/chat/CitationSheet";
import { InspectionMetrics } from "@/components/chat/InspectionMetrics";
import { MessageItem } from "@/components/chat/MessageItem";
import { RagControls } from "@/components/chat/RagControls";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { useChatSession, type ChatPhase } from "@/hooks/use-chat-session";
import {
  extractCitations,
  type AssistantMessage,
  type ChatCitation,
  type ChatMessage,
} from "@/lib/chat";

// Within this distance of the bottom the view keeps following new output;
// scrolling further up pauses it so the user can read while tokens arrive.
const STICK_TO_BOTTOM_PX = 96;

const PHASE_LABEL: Record<ChatPhase, string> = {
  idle: "",
  retrieving: "Retrieving context…",
  streaming: "Streaming answer…",
};

function latestMetrics(messages: ChatMessage[]): AssistantMessage | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant" && message.metrics !== null) {
      return message;
    }
  }
  return null;
}

function EmptyState() {
  return (
    <div className="m-auto max-w-sm space-y-3 text-center">
      <MessagesSquare
        aria-hidden
        className="mx-auto size-8 text-muted-foreground"
      />
      <h2 className="text-base font-medium">Ask your documents</h2>
      <p className="text-sm text-muted-foreground">
        Answers are grounded in indexed documents and cite their sources. If
        nothing relevant is found, the assistant refuses rather than guesses.
        Nothing indexed yet?{" "}
        <Link href="/ingest" className="text-foreground underline">
          Ingest a document
        </Link>
        .
      </p>
    </div>
  );
}

export function ChatStudio() {
  const session = useChatSession();
  const { messages, mode, params, phase } = session;
  const isGenerating = phase !== "idle";

  const [activeCitation, setActiveCitation] = useState<ChatCitation | null>(
    null,
  );
  const [sheetOpen, setSheetOpen] = useState(false);
  const [tuningOpen, setTuningOpen] = useState(false);

  const handleCitationSelect = useCallback((citation: ChatCitation) => {
    setActiveCitation(citation);
    setSheetOpen(true);
  }, []);

  const scrollerRef = useRef<HTMLDivElement>(null);
  const followRef = useRef(true);

  const handleScroll = (event: UIEvent<HTMLDivElement>) => {
    const { scrollHeight, scrollTop, clientHeight } = event.currentTarget;
    followRef.current =
      scrollHeight - scrollTop - clientHeight < STICK_TO_BOTTOM_PX;
  };

  useLayoutEffect(() => {
    const scroller = scrollerRef.current;
    if (scroller && followRef.current) {
      scroller.scrollTop = scroller.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = (text: string) => {
    followRef.current = true;
    session.send(text);
  };

  const metricsSource = useMemo(() => latestMetrics(messages), [messages]);
  const sidebar = (
    <div className="space-y-6">
      <section aria-labelledby="rag-tuning-heading" className="space-y-4">
        <h2 id="rag-tuning-heading" className="sr-only">
          Retrieval tuning
        </h2>
        <RagControls
          mode={mode}
          onModeChange={session.setMode}
          params={params}
          onParamsChange={session.setParams}
          disabled={isGenerating}
        />
      </section>
      <section
        aria-labelledby="inspection-heading"
        className="space-y-3 border-t border-border pt-5"
      >
        <h2 id="inspection-heading" className="text-sm font-medium">
          Latest answer
        </h2>
        <InspectionMetrics
          metrics={metricsSource?.metrics ?? null}
          citationCount={
            metricsSource
              ? (
                  metricsSource.citations ??
                  extractCitations(metricsSource.content)
                ).length
              : 0
          }
        />
      </section>
    </div>
  );

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-1 gap-6 px-4 py-4 sm:px-6 lg:h-[calc(100dvh-3.5rem)] lg:flex-none">
      <section
        aria-label="Conversation"
        className="flex h-[calc(100dvh-5.5rem)] min-w-0 flex-1 flex-col gap-3 lg:h-auto"
      >
        <header className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <h1 className="text-xl font-semibold tracking-tight">
              Retrieval &amp; Chat Studio
            </h1>
            <p
              aria-live="polite"
              className="h-4 text-xs text-muted-foreground"
            >
              {PHASE_LABEL[phase]}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Sheet open={tuningOpen} onOpenChange={setTuningOpen}>
              <SheetTrigger asChild>
                <Button variant="outline" size="sm" className="lg:hidden">
                  <SlidersHorizontal />
                  Tuning
                </Button>
              </SheetTrigger>
              <SheetContent side="right">
                <SheetHeader>
                  <SheetTitle>Retrieval tuning</SheetTitle>
                  <SheetDescription className="sr-only">
                    Adjust retrieval parameters and review metrics
                  </SheetDescription>
                </SheetHeader>
                <div className="overflow-y-auto px-4 pb-4">{sidebar}</div>
              </SheetContent>
            </Sheet>
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={messages.length === 0}
              onClick={session.clear}
            >
              <Trash2 />
              Clear
            </Button>
          </div>
        </header>

        <div
          ref={scrollerRef}
          onScroll={handleScroll}
          role="log"
          aria-label="Messages"
          aria-busy={isGenerating}
          className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto rounded-xl border bg-card/40 p-4"
        >
          {messages.length === 0 ? (
            <EmptyState />
          ) : (
            messages.map((message) => (
              <MessageItem
                key={message.id}
                message={message}
                onCitationSelect={handleCitationSelect}
              />
            ))
          )}
        </div>

        <ChatInput
          isGenerating={isGenerating}
          onSubmit={handleSubmit}
          onStop={session.stop}
        />
      </section>

      <aside
        aria-label="Retrieval tuning and metrics"
        className="hidden w-80 shrink-0 overflow-y-auto rounded-xl border bg-card/40 p-4 lg:block"
      >
        {sidebar}
      </aside>

      <CitationSheet
        citation={activeCitation}
        open={sheetOpen}
        onOpenChange={setSheetOpen}
      />
    </div>
  );
}
