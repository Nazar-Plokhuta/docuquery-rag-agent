export type SseEvent =
  | { type: "token"; token: string }
  // A failure after the status line was committed: `{ error, message }`.
  | { type: "error"; error: string; message: string }
  | { type: "done" };

export interface SseParser {
  /** Feeds decoded text and returns every event completed by it. */
  push: (text: string) => SseEvent[];
  /** Drains an unterminated trailing event once the byte stream has closed. */
  flush: () => SseEvent[];
}

const DATA_FIELD = "data:";
const DONE_SENTINEL = "[DONE]";

function parseEvent(block: string): SseEvent | null {
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    // Comment lines (": keep-alive") and non-data fields carry no payload.
    if (!line.startsWith(DATA_FIELD)) continue;
    const value = line.slice(DATA_FIELD.length);
    // Per the SSE spec exactly one leading space after the colon is stripped.
    dataLines.push(value.startsWith(" ") ? value.slice(1) : value);
  }
  if (dataLines.length === 0) return null;

  const payload = dataLines.join("\n");
  if (payload === DONE_SENTINEL) return { type: "done" };

  try {
    const parsed: unknown = JSON.parse(payload);
    if (typeof parsed === "object" && parsed !== null) {
      if ("token" in parsed && typeof parsed.token === "string") {
        return { type: "token", token: parsed.token };
      }
      if (
        "error" in parsed &&
        typeof parsed.error === "string" &&
        "message" in parsed &&
        typeof parsed.message === "string"
      ) {
        return { type: "error", error: parsed.error, message: parsed.message };
      }
    }
  } catch {
    // A single malformed frame must not discard the answer streamed so far.
  }
  return null;
}

/**
 * Incremental parser for the `data: <json>\n\n` frames emitted by
 * `POST /api/v1/query/stream`: token deltas, an optional terminal error
 * frame, and the `[DONE]` sentinel.
 *
 * It is transport-agnostic: callers decode bytes themselves (with
 * `TextDecoder({ stream: true })`, so multi-byte characters split across
 * network chunks stay intact) and hand over strings.
 */
export function createSseParser(): SseParser {
  let buffer = "";

  const drain = (final: boolean): SseEvent[] => {
    // CRLF is a legal SSE line terminator; normalise so one split rule works.
    // A lone trailing "\r" may be the first half of a "\r\n" split across
    // chunks, so it stays buffered until more data (or the final flush).
    let normalised = buffer.replace(/\r\n/g, "\n");
    let carry = "";
    if (!final && normalised.endsWith("\r")) {
      carry = "\r";
      normalised = normalised.slice(0, -1);
    }
    normalised = normalised.replace(/\r/g, "\n");

    const blocks = normalised.split("\n\n");
    const remainder = final ? "" : (blocks.pop() ?? "");
    buffer = remainder + carry;

    const events: SseEvent[] = [];
    for (const block of blocks) {
      const event = parseEvent(block);
      if (event) events.push(event);
    }
    return events;
  };

  return {
    push(text) {
      buffer += text;
      return drain(false);
    },
    flush() {
      return drain(true);
    },
  };
}
