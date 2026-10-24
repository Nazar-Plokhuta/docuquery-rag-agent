export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  environment: "development" | "staging" | "production";
  active_model: string;
  embedding_model: string;
}

// The trailing slash is deliberate: FastAPI registers the route as `/health/`
// and next.config.ts disables Next.js's slash-normalising redirect.
export const HEALTH_ENDPOINT = "/api/v1/health/";

const REQUEST_TIMEOUT_MS = 5_000;

export async function fetchHealth(url: string): Promise<HealthResponse> {
  const response = await fetch(url, {
    cache: "no-store",
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
  if (!response.ok) {
    throw new Error(`Health check failed with HTTP ${response.status}`);
  }
  return (await response.json()) as HealthResponse;
}
