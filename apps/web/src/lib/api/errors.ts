const MAX_PLAIN_TEXT_LENGTH = 200;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.trim() !== "" ? value : null;
}

// Pydantic reports `loc` as ["body", "file"]; the leading transport segment
// ("body" | "query" | ...) is noise for end users, the field path is not.
function formatValidationIssue(issue: unknown): string | null {
  if (!isRecord(issue)) return null;
  const message = nonEmptyString(issue.msg);
  if (message === null) return null;
  const field = Array.isArray(issue.loc) ? issue.loc.slice(1).join(".") : "";
  return field ? `${field}: ${message}` : message;
}

function fromJsonBody(body: unknown): string | null {
  if (!isRecord(body)) return null;

  // Domain `AppException` envelope: { error, message }.
  const domainMessage = nonEmptyString(body.message);
  if (domainMessage !== null) return domainMessage;

  const { detail } = body;
  // FastAPI `HTTPException` (415/422 raised by routes): { detail: string }.
  const detailMessage = nonEmptyString(detail);
  if (detailMessage !== null) return detailMessage;

  // FastAPI request validation (422): { detail: [{ loc, msg, type }, ...] }.
  if (Array.isArray(detail)) {
    const issues = detail
      .map(formatValidationIssue)
      .filter((issue): issue is string => issue !== null);
    if (issues.length > 0) return issues.join("; ");
  }
  return null;
}

function fromPlainText(text: string): string | null {
  const trimmed = text.trim();
  if (trimmed === "") return null;
  // HTML bodies come from intermediaries (proxy error pages), not from the API.
  if (trimmed.startsWith("<")) return null;
  return trimmed.length > MAX_PLAIN_TEXT_LENGTH
    ? `${trimmed.slice(0, MAX_PLAIN_TEXT_LENGTH)}…`
    : trimmed;
}

/**
 * Reduces any non-2xx API response to one human-readable line.
 *
 * Never throws: an unreadable or unrecognised body degrades to the HTTP
 * status line so callers can always surface something to the user.
 */
export async function normalizeApiError(res: Response): Promise<string> {
  const statusLine = `HTTP ${res.status}${res.statusText ? ` ${res.statusText}` : ""}`;

  let text: string;
  try {
    text = await res.text();
  } catch {
    return statusLine;
  }

  try {
    const message = fromJsonBody(JSON.parse(text));
    if (message !== null) return message;
  } catch {
    // Not JSON (e.g. an unhandled 500 serves plain text); fall through.
  }

  const plain = fromPlainText(text);
  // Servers often echo the reason phrase as the body; don't repeat it.
  return plain === null || plain === res.statusText
    ? statusLine
    : `${statusLine}: ${plain}`;
}
