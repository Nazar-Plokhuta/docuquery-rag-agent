import { normalizeApiError } from "./errors";
import type { components, paths } from "./schema";

export type HealthResponse = components["schemas"]["HealthResponse"];
export type IngestResponse = components["schemas"]["IngestResponse"];
export type QueryRequest = components["schemas"]["QueryRequest"];
export type QueryResponse = components["schemas"]["QueryResponse"];

// Keys are checked against the generated OpenAPI `paths`, so a route renamed
// in the backend fails the type check here. The literal trailing slashes are
// load-bearing: next.config.ts disables slash normalisation, and a slashless
// request makes FastAPI 307 to its own origin, bypassing the Next.js proxy.
export const API_ENDPOINTS = {
  health: "/api/v1/health/",
  ingestFile: "/api/v1/ingest/file",
  query: "/api/v1/query/",
  queryStream: "/api/v1/query/stream",
  analyticsRecent: "/api/v1/analytics/recent",
} as const satisfies Record<string, keyof paths>;

const HEALTH_TIMEOUT_MS = 5_000;

export class ApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

async function request<T>(url: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, { cache: "no-store", ...init });
  } catch (error) {
    // Caller-initiated aborts are control flow, not failures to report.
    if (isAbortError(error)) throw error;
    throw new ApiError(
      "Cannot reach the API. Check that the backend is running.",
    );
  }
  if (!response.ok) {
    throw new ApiError(await normalizeApiError(response), response.status);
  }
  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>(API_ENDPOINTS.health, {
    signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
  });
}

export function ingestFile(
  file: File,
  signal?: AbortSignal,
): Promise<IngestResponse> {
  // No explicit Content-Type: the browser must set the multipart boundary.
  const body = new FormData();
  body.append("file", file, file.name);
  return request<IngestResponse>(API_ENDPOINTS.ingestFile, {
    method: "POST",
    body,
    signal,
  });
}

export function queryDocuments(
  payload: QueryRequest,
  signal?: AbortSignal,
): Promise<QueryResponse> {
  return request<QueryResponse>(API_ENDPOINTS.query, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
}
