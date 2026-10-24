"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  API_ENDPOINTS,
  isAbortError,
  NETWORK_ERROR_MESSAGE,
  type QueryRequest,
} from "@/lib/api/client";
import { normalizeApiError } from "@/lib/api/errors";
import { createSseParser, type SseEvent } from "@/lib/sse";

export type StreamStatus =
  | "idle"
  | "retrieving"
  | "streaming"
  | "complete"
  | "error";

export type StreamOutcome =
  | "complete"
  // The connection closed without `[DONE]`; whatever arrived is kept.
  | "truncated"
  | "aborted"
  | "error";

export interface StreamResult {
  outcome: StreamOutcome;
  text: string;
  errorMessage: string | null;
  firstTokenMs: number | null;
  elapsedMs: number;
  deltaCount: number;
}

export interface StreamQuery {
  status: StreamStatus;
  /** Answer accumulated so far, refreshed at most once per animation frame. */
  text: string;
  error: string | null;
  streamQuery: (request: QueryRequest) => Promise<StreamResult>;
  abort: () => void;
  reset: () => void;
}

/** How a stream that reached its terminator ended. */
type StreamEnd = { failed: false } | { failed: true; message: string };

const STREAM_INTERRUPTED_MESSAGE = "The connection was lost while streaming.";
const UNEXPECTED_ERROR_MESSAGE = "Unexpected error while reading the stream.";
const EMPTY_STREAM_MESSAGE = "The stream ended before any content arrived.";

/**
 * Streams a RAG answer from `POST /api/v1/query/stream`.
 *
 * Token deltas land in a ref and are published to React state once per
 * animation frame: a fast model emits far more deltas than the display can
 * paint, and a state update per delta would re-parse the markdown each time.
 */
export function useStreamQuery(): StreamQuery {
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);

  const controllerRef = useRef<AbortController | null>(null);
  const accumulatedRef = useRef("");
  const frameRef = useRef<number | null>(null);

  const cancelFrame = useCallback(() => {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
  }, []);

  const publishNow = useCallback(() => {
    cancelFrame();
    setText(accumulatedRef.current);
  }, [cancelFrame]);

  const schedulePublish = useCallback(() => {
    if (frameRef.current !== null) return;
    frameRef.current = requestAnimationFrame(() => {
      frameRef.current = null;
      setText(accumulatedRef.current);
    });
  }, []);

  const streamQuery = useCallback(
    async (request: QueryRequest): Promise<StreamResult> => {
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;
      // A superseded run may still unwind after a newer one has started and
      // must not overwrite that run's state.
      const isCurrent = () => controllerRef.current === controller;

      cancelFrame();
      accumulatedRef.current = "";
      setText("");
      setError(null);
      setStatus("retrieving");

      const startedAt = performance.now();
      let firstTokenMs: number | null = null;
      let deltaCount = 0;

      const result = (
        outcome: StreamOutcome,
        errorMessage: string | null = null,
      ): StreamResult => ({
        outcome,
        text: accumulatedRef.current,
        errorMessage,
        firstTokenMs,
        elapsedMs: performance.now() - startedAt,
        deltaCount,
      });

      const fail = (message: string): StreamResult => {
        if (isCurrent()) {
          publishNow();
          setError(message);
          setStatus("error");
        }
        return result("error", message);
      };

      // Applies parsed events; returns how the stream ended once it did: the
      // `[DONE]` sentinel, or an in-band error frame (the server sends one when
      // it fails after the 200 status line is already committed).
      const apply = (events: SseEvent[]): StreamEnd | null => {
        for (const event of events) {
          if (event.type === "done") return { failed: false };
          if (event.type === "error") {
            return { failed: true, message: event.message };
          }
          if (firstTokenMs === null) {
            firstTokenMs = performance.now() - startedAt;
            if (isCurrent()) setStatus("streaming");
          }
          deltaCount += 1;
          accumulatedRef.current += event.token;
          if (isCurrent()) schedulePublish();
        }
        return null;
      };

      try {
        let response: Response;
        try {
          response = await fetch(API_ENDPOINTS.queryStream, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "text/event-stream",
            },
            body: JSON.stringify(request),
            cache: "no-store",
            signal: controller.signal,
          });
        } catch (fetchError) {
          if (isAbortError(fetchError)) throw fetchError;
          return fail(NETWORK_ERROR_MESSAGE);
        }

        if (!response.ok) return fail(await normalizeApiError(response));
        if (response.body === null) return fail(EMPTY_STREAM_MESSAGE);

        const reader = response.body.getReader();
        // `stream: true` buffers a multi-byte character split across chunks.
        const decoder = new TextDecoder();
        const parser = createSseParser();

        let end: StreamEnd | null = null;
        try {
          while (end === null) {
            const { done, value } = await reader.read();
            if (done) break;
            end = apply(parser.push(decoder.decode(value, { stream: true })));
          }
        } catch (readError) {
          if (isAbortError(readError)) throw readError;
          return fail(STREAM_INTERRUPTED_MESSAGE);
        }

        end ??= apply([...parser.push(decoder.decode()), ...parser.flush()]);

        if (end?.failed) {
          // Surface the server's own message; `fail` keeps the partial answer.
          await reader.cancel().catch(() => undefined);
          return fail(end.message);
        }

        publishNow();
        const finished = end !== null;
        if (finished) {
          // Release the connection; the server has nothing left to send.
          await reader.cancel().catch(() => undefined);
        } else if (accumulatedRef.current === "") {
          return fail(EMPTY_STREAM_MESSAGE);
        }

        if (isCurrent()) setStatus("complete");
        return result(finished ? "complete" : "truncated");
      } catch (caught) {
        if (!isAbortError(caught)) return fail(UNEXPECTED_ERROR_MESSAGE);
        if (isCurrent()) {
          publishNow();
          setStatus("idle");
        }
        return result("aborted");
      } finally {
        if (isCurrent()) controllerRef.current = null;
      }
    },
    [cancelFrame, publishNow, schedulePublish],
  );

  const abort = useCallback(() => controllerRef.current?.abort(), []);

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    cancelFrame();
    accumulatedRef.current = "";
    setText("");
    setError(null);
    setStatus("idle");
  }, [cancelFrame]);

  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
      cancelFrame();
    };
  }, [cancelFrame]);

  return { status, text, error, streamQuery, abort, reset };
}
