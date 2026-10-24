"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { ApiError, ingestFile, isAbortError } from "@/lib/api/client";
import {
  MAX_CONCURRENT_UPLOADS,
  validateFile,
  type IngestItem,
  type RejectedFile,
} from "@/lib/ingest";

type QueueAction =
  | { type: "enqueue"; items: IngestItem[] }
  | { type: "start"; id: string }
  | { type: "settle"; id: string; patch: Partial<IngestItem> }
  | { type: "retry-failed" }
  | { type: "clear-completed" };

function queueReducer(state: IngestItem[], action: QueueAction): IngestItem[] {
  switch (action.type) {
    case "enqueue":
      return [...state, ...action.items];
    case "start":
      return state.map((item) =>
        item.id === action.id ? { ...item, status: "uploading" } : item,
      );
    case "settle":
      return state.map((item) =>
        item.id === action.id ? { ...item, ...action.patch } : item,
      );
    case "retry-failed":
      return state.map((item) =>
        item.status === "error"
          ? { ...item, status: "pending", errorMessage: null }
          : item,
      );
    case "clear-completed":
      return state.filter(
        (item) => item.status !== "success" && item.status !== "warning",
      );
  }
}

export interface IngestQueue {
  items: IngestItem[];
  rejections: RejectedFile[];
  addFiles: (files: File[]) => void;
  retryFailed: () => void;
  clearCompleted: () => void;
  dismissRejections: () => void;
}

/**
 * Client-side upload queue with a fixed concurrency ceiling.
 *
 * `File` handles live in a ref rather than in reducer state so the state stays
 * plain, serialisable data and retries can re-send the original bytes.
 */
export function useIngestQueue(): IngestQueue {
  const [items, dispatch] = useReducer(queueReducer, []);
  const [rejections, setRejections] = useState<RejectedFile[]>([]);

  const files = useRef(new Map<string, File>());
  const inFlight = useRef(new Map<string, AbortController>());

  useEffect(() => {
    // Ids are claimed in `inFlight` synchronously, so a re-run of this effect
    // before React commits the "uploading" status cannot double-start a file.
    for (const item of items) {
      if (inFlight.current.size >= MAX_CONCURRENT_UPLOADS) return;
      if (item.status !== "pending" || inFlight.current.has(item.id)) continue;

      const file = files.current.get(item.id);
      if (!file) continue;

      const controller = new AbortController();
      inFlight.current.set(item.id, controller);
      dispatch({ type: "start", id: item.id });

      ingestFile(file, controller.signal)
        .then((response) => {
          inFlight.current.delete(item.id);
          if (response.status !== "success") {
            dispatch({
              type: "settle",
              id: item.id,
              patch: {
                status: "error",
                errorMessage: `Unexpected ingestion status "${response.status}"`,
              },
            });
            return;
          }
          dispatch({
            type: "settle",
            id: item.id,
            patch: {
              // An empty or whitespace-only document is accepted by the API
              // but contributes nothing retrievable; surface it distinctly.
              status: response.chunks_ingested === 0 ? "warning" : "success",
              chunksIngested: response.chunks_ingested,
              errorMessage: null,
            },
          });
        })
        .catch((error: unknown) => {
          inFlight.current.delete(item.id);
          if (isAbortError(error)) return;
          dispatch({
            type: "settle",
            id: item.id,
            patch: {
              status: "error",
              errorMessage:
                error instanceof ApiError
                  ? error.message
                  : "Unexpected error during upload",
            },
          });
        });
    }
  }, [items]);

  useEffect(() => {
    // Release `File` handles for items that were cleared from the queue.
    const live = new Set(items.map((item) => item.id));
    for (const id of files.current.keys()) {
      if (!live.has(id)) files.current.delete(id);
    }
  }, [items]);

  useEffect(() => {
    const pending = inFlight.current;
    return () => {
      for (const controller of pending.values()) controller.abort();
    };
  }, []);

  const addFiles = useCallback((incoming: File[]) => {
    const accepted: IngestItem[] = [];
    const rejected: RejectedFile[] = [];

    for (const file of incoming) {
      const id = crypto.randomUUID();
      const reason = validateFile(file);
      if (reason) {
        rejected.push({ id, filename: file.name, reason });
        continue;
      }
      files.current.set(id, file);
      accepted.push({
        id,
        filename: file.name,
        size: file.size,
        status: "pending",
        chunksIngested: null,
        errorMessage: null,
      });
    }

    setRejections(rejected);
    if (accepted.length > 0) dispatch({ type: "enqueue", items: accepted });
  }, []);

  const clearCompleted = useCallback(
    () => dispatch({ type: "clear-completed" }),
    [],
  );
  const retryFailed = useCallback(() => dispatch({ type: "retry-failed" }), []);
  const dismissRejections = useCallback(() => setRejections([]), []);

  return {
    items,
    rejections,
    addFiles,
    retryFailed,
    clearCompleted,
    dismissRejections,
  };
}
