import type { QueryResponse } from "@/lib/api/client";

/** Mirrors `FALLBACK_REFUSAL_MESSAGE` in `apps/api/src/core/rag/prompts.py`. */
export const FALLBACK_REFUSAL_PREFIX =
  "I am sorry, but the provided documentation does not contain sufficient information";

export const TOP_K_RANGE = { min: 1, max: 20, default: 6 } as const;
export const SCORE_THRESHOLD_RANGE = {
  min: 0,
  max: 1,
  step: 0.05,
  default: 0.2,
} as const;

export type ChatMode = "stream" | "inspect";

export interface RagParams {
  topK: number;
  scoreThreshold: number;
}

export const DEFAULT_RAG_PARAMS: RagParams = {
  topK: TOP_K_RANGE.default,
  scoreThreshold: SCORE_THRESHOLD_RANGE.default,
};

export interface ChatCitation {
  source: string;
  section: string;
  /** Only known in inspect mode; inline markers carry no chunk identity. */
  chunkId: string | null;
}

export interface MessageMetrics {
  mode: ChatMode;
  params: RagParams;
  /** End-to-end duration: server-reported when inspecting, client-measured when streaming. */
  latencyMs: number;
  /** Streaming only: time from submit to the first token delta. */
  firstTokenMs: number | null;
  /** Streaming only: number of SSE token events received. */
  deltaCount: number | null;
  /** Inspect only: the API does not report usage for streamed completions. */
  totalTokens: number | null;
}

export type AssistantState =
  | "retrieving"
  | "streaming"
  | "complete"
  | "stopped"
  | "error";

export interface UserMessage {
  id: string;
  role: "user";
  content: string;
}

export interface AssistantMessage {
  id: string;
  role: "assistant";
  content: string;
  state: AssistantState;
  /** `null` until known; streamed answers derive citations from the text. */
  citations: ChatCitation[] | null;
  metrics: MessageMetrics | null;
  /** The stream closed without `[DONE]`, so the answer may be cut short. */
  truncated: boolean;
  errorMessage: string | null;
}

export type ChatMessage = UserMessage | AssistantMessage;

export function citationsFromResponse(response: QueryResponse): ChatCitation[] {
  return response.citations.map((citation) => ({
    source: citation.source,
    section: citation.section,
    chunkId: citation.chunk_id,
  }));
}

export function citationKey(citation: ChatCitation): string {
  return `${citation.source}\u0000${citation.section}`;
}

/** Fresh instance per call: global regexes carry `lastIndex` state. */
function citationPattern(): RegExp {
  return /\[Source:\s*([^,\]]+)(?:,\s*Section:\s*([^\]]*))?\]/g;
}

export interface CitationMatch {
  start: number;
  end: number;
  citation: ChatCitation;
}

export function findCitationMarkers(text: string): CitationMatch[] {
  const matches: CitationMatch[] = [];
  for (const match of text.matchAll(citationPattern())) {
    matches.push({
      start: match.index,
      end: match.index + match[0].length,
      citation: {
        source: match[1].trim(),
        section: (match[2] ?? "").trim(),
        chunkId: null,
      },
    });
  }
  return matches;
}

/** Unique citations in order of first appearance. */
export function extractCitations(text: string): ChatCitation[] {
  const seen = new Map<string, ChatCitation>();
  for (const { citation } of findCitationMarkers(text)) {
    const key = citationKey(citation);
    if (!seen.has(key)) seen.set(key, citation);
  }
  return [...seen.values()];
}

const MARKER_OPENING = "[Source:";

/**
 * Drops a citation marker that is still being typed out by the model, so the
 * UI never flashes raw `[Sour` text before it resolves into a badge.
 */
export function hidePartialCitation(text: string): string {
  const open = text.lastIndexOf("[");
  if (open === -1 || text.indexOf("]", open) !== -1) return text;
  const tail = text.slice(open);
  const couldBeMarker = tail.startsWith(MARKER_OPENING)
    ? true
    : MARKER_OPENING.startsWith(tail);
  return couldBeMarker ? text.slice(0, open) : text;
}

export function isRefusal(content: string): boolean {
  return content.replace(/\s+/g, " ").includes(FALLBACK_REFUSAL_PREFIX);
}

export type GuardrailVerdict =
  | { kind: "refusal"; shortCircuited: boolean }
  | { kind: "uncited" }
  | null;

/**
 * Decides which grounding banner, if any, an answer deserves.
 *
 * `shortCircuited` is only claimed when it is evidenced: inspect mode reports
 * zero tokens, and a streamed refusal arrives as one single delta (the engine
 * yields the fallback phrase whole, whereas a model refusal streams in pieces).
 */
export function evaluateGuardrail(message: AssistantMessage): GuardrailVerdict {
  if (message.state !== "complete") return null;

  if (isRefusal(message.content)) {
    const { metrics } = message;
    const shortCircuited =
      metrics?.mode === "inspect"
        ? metrics.totalTokens === 0
        : metrics?.deltaCount === 1;
    return { kind: "refusal", shortCircuited };
  }

  const citations = message.citations ?? extractCitations(message.content);
  return citations.length === 0 && message.content.trim() !== ""
    ? { kind: "uncited" }
    : null;
}
