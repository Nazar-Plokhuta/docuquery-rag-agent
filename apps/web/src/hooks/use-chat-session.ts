"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiError,
  isAbortError,
  queryDocuments,
  type QueryRequest,
} from "@/lib/api/client";
import {
  citationsFromResponse,
  DEFAULT_RAG_PARAMS,
  type AssistantMessage,
  type ChatMessage,
  type ChatMode,
  type RagParams,
} from "@/lib/chat";
import { useStreamQuery, type StreamResult } from "@/hooks/use-stream-query";

export type ChatPhase = "idle" | "retrieving" | "streaming";

export interface ChatSession {
  messages: ChatMessage[];
  mode: ChatMode;
  setMode: (mode: ChatMode) => void;
  params: RagParams;
  setParams: (params: RagParams) => void;
  phase: ChatPhase;
  send: (question: string) => void;
  stop: () => void;
  clear: () => void;
}

const UNEXPECTED_ERROR_MESSAGE = "Unexpected error while querying the API.";

function assistantMessage(
  id: string,
  patch: Partial<AssistantMessage>,
): AssistantMessage {
  return {
    id,
    role: "assistant",
    content: "",
    state: "complete",
    citations: null,
    metrics: null,
    truncated: false,
    errorMessage: null,
    ...patch,
  };
}

function fromStreamResult(
  id: string,
  result: StreamResult,
  params: RagParams,
): AssistantMessage {
  switch (result.outcome) {
    case "complete":
    case "truncated":
      return assistantMessage(id, {
        content: result.text,
        truncated: result.outcome === "truncated",
        metrics: {
          mode: "stream",
          params,
          latencyMs: result.elapsedMs,
          firstTokenMs: result.firstTokenMs,
          deltaCount: result.deltaCount,
          totalTokens: null,
        },
      });
    case "aborted":
      return assistantMessage(id, { content: result.text, state: "stopped" });
    case "error":
      return assistantMessage(id, {
        content: result.text,
        state: "error",
        errorMessage: result.errorMessage,
      });
  }
}

/**
 * Owns the conversation transcript and routes each question to the streaming
 * or the inspection endpoint. Messages live in component state only; a page
 * reload starts a fresh session by design.
 */
export function useChatSession(): ChatSession {
  const {
    status: streamStatus,
    text: streamText,
    streamQuery,
    abort: abortStream,
    reset: resetStream,
  } = useStreamQuery();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [mode, setMode] = useState<ChatMode>("stream");
  const [params, setParams] = useState<RagParams>(DEFAULT_RAG_PARAMS);
  const [pending, setPending] = useState<{
    id: string;
    mode: ChatMode;
  } | null>(null);

  // `pending` lags a render behind a rapid double submit; the ref does not.
  const busyRef = useRef(false);
  // Aborted runs still resolve; a run that was cleared away must not append.
  const runIdRef = useRef(0);
  const inspectControllerRef = useRef<AbortController | null>(null);

  const append = useCallback((message: ChatMessage) => {
    setMessages((current) => [...current, message]);
  }, []);

  const send = useCallback(
    (question: string) => {
      const trimmed = question.trim();
      if (trimmed === "" || busyRef.current) return;
      busyRef.current = true;
      const runId = ++runIdRef.current;

      const assistantId = crypto.randomUUID();
      const request: QueryRequest = {
        query: trimmed,
        top_k: params.topK,
        score_threshold: params.scoreThreshold,
      };

      append({ id: crypto.randomUUID(), role: "user", content: trimmed });
      setPending({ id: assistantId, mode });

      const settle = (message: AssistantMessage) => {
        if (runIdRef.current !== runId) return;
        busyRef.current = false;
        append(message);
        setPending(null);
      };

      if (mode === "stream") {
        void streamQuery(request).then((result) => {
          settle(fromStreamResult(assistantId, result, params));
        });
        return;
      }

      const controller = new AbortController();
      inspectControllerRef.current = controller;
      void queryDocuments(request, controller.signal)
        .then((response) =>
          assistantMessage(assistantId, {
            content: response.answer,
            citations: citationsFromResponse(response),
            metrics: {
              mode: "inspect",
              params,
              latencyMs: response.latency_ms,
              firstTokenMs: null,
              deltaCount: null,
              totalTokens: response.total_tokens,
            },
          }),
        )
        .catch((error: unknown) =>
          isAbortError(error)
            ? assistantMessage(assistantId, { state: "stopped" })
            : assistantMessage(assistantId, {
                state: "error",
                errorMessage:
                  error instanceof ApiError
                    ? error.message
                    : UNEXPECTED_ERROR_MESSAGE,
              }),
        )
        .then((message) => {
          inspectControllerRef.current = null;
          settle(message);
        });
    },
    [append, mode, params, streamQuery],
  );

  const stop = useCallback(() => {
    abortStream();
    inspectControllerRef.current?.abort();
  }, [abortStream]);

  const clear = useCallback(() => {
    runIdRef.current += 1;
    resetStream();
    inspectControllerRef.current?.abort();
    inspectControllerRef.current = null;
    busyRef.current = false;
    setPending(null);
    setMessages([]);
  }, [resetStream]);

  useEffect(() => {
    return () => inspectControllerRef.current?.abort();
  }, []);

  const phase: ChatPhase =
    pending === null
      ? "idle"
      : pending.mode === "stream" && streamStatus === "streaming"
        ? "streaming"
        : "retrieving";

  const visibleMessages = useMemo<ChatMessage[]>(() => {
    if (pending === null) return messages;
    return [
      ...messages,
      assistantMessage(pending.id, {
        content: pending.mode === "stream" ? streamText : "",
        state: phase === "streaming" ? "streaming" : "retrieving",
      }),
    ];
  }, [messages, pending, phase, streamText]);

  return {
    messages: visibleMessages,
    mode,
    setMode,
    params,
    setParams,
    phase,
    send,
    stop,
    clear,
  };
}
