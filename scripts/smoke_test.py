"""End-to-end smoke test script for the DocuQuery RAG Agent.

Usage:
    python scripts/smoke_test.py [--base-url http://localhost:8000]

Runs an automated sanity check against a running server instance:
    1. Liveness probe  — GET  /api/v1/health/
    2. Document ingest — POST /api/v1/ingest/file  (data/sample_docs/sla_policy.md)
    3. Grounded query  — POST /api/v1/query        (verifies citations are returned)
    4. Out-of-scope query — POST /api/v1/query     (verifies FALLBACK_REFUSAL_MESSAGE is returned)
    5. SSE streaming   — POST /api/v1/query/stream (validates SSE event format and [DONE] sentinel)
    6. Analytics       — GET  /api/v1/analytics/recent (verifies telemetry records are logged)

Exit code:
    0 — all probes passed.
    1 — one or more probes failed; full error details are printed to stderr.

The script does not import any application source modules; it interacts with the
service exclusively through its HTTP API, making it suitable for post-deployment
smoke testing in CI and Docker Compose environments.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL_DEFAULT: str = "http://localhost:8000"
_SAMPLE_DOCS_DIR: Path = Path(__file__).parent.parent / "data" / "sample_docs"
_SLA_POLICY_PATH: Path = _SAMPLE_DOCS_DIR / "sla_policy.md"

# The canonical grounded query must reference content present in sla_policy.md.
_GROUNDED_QUERY: str = (
    "What are the initial response times for a P0 service outage under Tier 1 Enterprise Platinum?"
)
_UNGROUNDED_QUERY: str = (
    "What is the quantum entanglement coefficient of the DocuQuery photon accelerator?"
)

# The deterministic fallback refusal phrase as defined in src/core/rag/prompts.py.
# Copied here to avoid importing application source code from the smoke script.
_FALLBACK_REFUSAL_PREFIX: str = "I am sorry, but the provided documentation does not contain"

_GREEN: str = "\033[92m"
_RED: str = "\033[91m"
_YELLOW: str = "\033[93m"
_CYAN: str = "\033[96m"
_BOLD: str = "\033[1m"
_RESET: str = "\033[0m"

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _pass(label: str) -> None:
    print(f"  {_GREEN}✔ PASS{_RESET}  {label}")


def _fail(label: str, detail: str) -> None:
    print(f"  {_RED}✘ FAIL{_RESET}  {label}", file=sys.stderr)
    print(f"          {_YELLOW}{detail}{_RESET}", file=sys.stderr)


def _section(title: str) -> None:
    print(f"\n{_BOLD}{_CYAN}▶ {title}{_RESET}")


def _info(msg: str) -> None:
    print(f"    {msg}")


# ---------------------------------------------------------------------------
# Individual probe implementations
# ---------------------------------------------------------------------------


async def probe_health(client: httpx.AsyncClient) -> bool:
    """Probe 1: Verify the liveness endpoint returns HTTP 200 with status='ok'.

    Args:
        client: Shared httpx async client.

    Returns:
        True if the probe passed, False otherwise.
    """
    _section("Probe 1 — GET /api/v1/health/")
    try:
        response = await client.get("/api/v1/health/")
        if response.status_code != 200:
            _fail("HTTP 200 expected", f"Got HTTP {response.status_code}")
            return False
        _pass("HTTP 200 received")

        payload: dict[str, Any] = response.json()
        if payload.get("status") != "ok":
            _fail("status == 'ok'", f"Got: {payload.get('status')!r}")
            return False
        _pass(
            f"status='ok'  app_name='{payload.get('app_name')}'  version='{payload.get('version')}'"
        )
        return True
    except httpx.ConnectError as exc:
        _fail(
            "Connection to server",
            f"Could not reach {client.base_url} — is the server running? ({exc})",
        )
        return False


async def probe_ingest(client: httpx.AsyncClient) -> int:
    """Probe 2: Upload sla_policy.md and verify it is indexed successfully.

    Args:
        client: Shared httpx async client.

    Returns:
        Number of chunks ingested on success, or -1 on failure.
    """
    _section("Probe 2 — POST /api/v1/ingest/file  (sla_policy.md)")
    if not _SLA_POLICY_PATH.exists():
        _fail("Sample document exists", f"File not found: {_SLA_POLICY_PATH}")
        return -1
    _info(f"Uploading: {_SLA_POLICY_PATH.name}  ({_SLA_POLICY_PATH.stat().st_size:,} bytes)")

    with _SLA_POLICY_PATH.open("rb") as fh:
        content = fh.read()

    response = await client.post(
        "/api/v1/ingest/file",
        files={"file": (_SLA_POLICY_PATH.name, content, "text/markdown")},
        timeout=120.0,  # Ingestion involves embedding API calls; allow generous timeout.
    )

    if response.status_code != 200:
        _fail("HTTP 200 expected", f"Got HTTP {response.status_code}: {response.text[:200]}")
        return -1

    payload = response.json()
    status = payload.get("status")
    chunks = payload.get("chunks_ingested", 0)

    if status != "success":
        _fail("status == 'success'", f"Got: {status!r}")
        return -1

    _pass(f"status='success'  filename='{payload.get('filename')}'  chunks_ingested={chunks}")
    return chunks


async def probe_grounded_query(client: httpx.AsyncClient) -> bool:
    """Probe 3: Execute a query grounded in the ingested SLA policy document.

    Verifies that the response contains a non-empty answer and at least one
    citation, confirming that the retrieval and LLM pipeline executed correctly.

    Args:
        client: Shared httpx async client.

    Returns:
        True if the probe passed, False otherwise.
    """
    _section("Probe 3 — POST /api/v1/query  (grounded query)")
    _info(f"Query: {_GROUNDED_QUERY!r}")

    response = await client.post(
        "/api/v1/query/",
        json={"query": _GROUNDED_QUERY, "top_k": 5, "score_threshold": 0.25},
        timeout=60.0,
    )

    if response.status_code != 200:
        _fail("HTTP 200 expected", f"Got HTTP {response.status_code}: {response.text[:200]}")
        return False

    payload = response.json()
    answer: str = payload.get("answer", "")
    citations: list[dict[str, str]] = payload.get("citations", [])
    latency_ms: float = payload.get("latency_ms", 0.0)
    total_tokens: int = payload.get("total_tokens", 0)

    _pass(f"HTTP 200  latency_ms={latency_ms:.0f}  total_tokens={total_tokens}")

    if not answer or answer.startswith(_FALLBACK_REFUSAL_PREFIX):
        _fail(
            "Grounded answer expected",
            "Got fallback refusal — embeddings may not be ready yet.",
        )
        return False
    _pass(f"Grounded answer received ({len(answer)} chars)")

    if not citations:
        _fail("Citations expected", "No citations returned — grounding may be incomplete.")
        return False
    _pass(f"{len(citations)} citation(s) returned:")
    for i, citation in enumerate(citations, start=1):
        _info(f"  [{i}] source='{citation.get('source')}'  section='{citation.get('section')}'")

    return True


async def probe_ungrounded_query(client: httpx.AsyncClient) -> bool:
    """Probe 4: Execute a query with no possible grounding in the indexed corpus.

    Verifies that the deterministic fallback refusal phrase is returned and
    that no hallucinated answer is generated.

    Args:
        client: Shared httpx async client.

    Returns:
        True if the probe passed (refusal detected), False otherwise.
    """
    _section("Probe 4 — POST /api/v1/query  (out-of-scope query, must trigger refusal)")
    _info(f"Query: {_UNGROUNDED_QUERY!r}")

    response = await client.post(
        "/api/v1/query/",
        json={"query": _UNGROUNDED_QUERY, "top_k": 5, "score_threshold": 0.3},
        timeout=60.0,
    )

    if response.status_code != 200:
        _fail("HTTP 200 expected", f"Got HTTP {response.status_code}: {response.text[:200]}")
        return False

    payload = response.json()
    answer: str = payload.get("answer", "")
    citations: list[dict[str, str]] = payload.get("citations", [])

    if not answer.startswith(_FALLBACK_REFUSAL_PREFIX):
        _fail(
            "Fallback refusal expected",
            f"Got non-refusal answer — potential hallucination: {answer[:200]!r}",
        )
        return False
    _pass("Fallback refusal phrase returned — anti-hallucination guardrail active")

    if citations:
        _fail("Zero citations expected on refusal", f"Got {len(citations)} citation(s)")
        return False
    _pass("Zero citations returned — correct behaviour on out-of-scope query")

    return True


async def probe_sse_stream(client: httpx.AsyncClient) -> bool:
    """Probe 5: Consume the SSE streaming endpoint and validate the event format.

    Verifies that:
    - The response Content-Type is text/event-stream.
    - All data: lines (except [DONE]) contain valid JSON with a 'token' key.
    - The [DONE] termination sentinel is present at the end of the stream.
    - At least one token is emitted (stream is not empty).

    Args:
        client: Shared httpx async client.

    Returns:
        True if the probe passed, False otherwise.
    """
    _section("Probe 5 — POST /api/v1/query/stream  (SSE streaming)")
    _info(f"Query: {_GROUNDED_QUERY!r}")

    all_passed = True
    token_count = 0
    done_seen = False
    full_text_tokens: list[str] = []

    async with client.stream(
        "POST",
        "/api/v1/query/stream",
        json={"query": _GROUNDED_QUERY, "top_k": 5, "score_threshold": 0.25},
        timeout=90.0,
    ) as response:
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" not in content_type:
            _fail("Content-Type: text/event-stream", f"Got: {content_type!r}")
            all_passed = False
        else:
            _pass("Content-Type: text/event-stream confirmed")

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload_raw = line[len("data:") :].strip()

            if payload_raw == "[DONE]":
                done_seen = True
                continue

            # Every non-sentinel data: line must contain JSON with a 'token' key.
            try:
                parsed = json.loads(payload_raw)
            except json.JSONDecodeError:
                _fail("Valid JSON in data: line", f"Could not parse: {payload_raw[:80]!r}")
                all_passed = False
                continue

            token_delta: str | None = parsed.get("token")
            if token_delta is None:
                _fail("'token' key in SSE event", f"Got keys: {list(parsed.keys())}")
                all_passed = False
                continue

            token_count += 1
            full_text_tokens.append(token_delta)

    if not done_seen:
        _fail("[DONE] sentinel present", "Stream ended without the [DONE] termination sentinel")
        all_passed = False
    else:
        _pass("[DONE] termination sentinel received")

    if token_count == 0:
        _fail("At least one token emitted", "Stream was empty — retrieval may have failed")
        all_passed = False
    else:
        reconstructed = "".join(full_text_tokens)
        _pass(f"{token_count} token(s) received  ({len(reconstructed)} chars total)")
        _info(f"  Preview: {reconstructed[:120]!r}{'...' if len(reconstructed) > 120 else ''}")

    return all_passed


async def probe_analytics(client: httpx.AsyncClient) -> bool:
    """Probe 6: Fetch recent telemetry records and verify the response structure.

    After probes 3 and 4, at least 2 telemetry records should be present in
    the audit log. This probe verifies the analytics endpoint and confirms
    that audit logging is functioning end-to-end.

    Args:
        client: Shared httpx async client.

    Returns:
        True if the probe passed, False otherwise.
    """
    _section("Probe 6 — GET /api/v1/analytics/recent  (telemetry audit log)")

    response = await client.get("/api/v1/analytics/recent?limit=10")

    if response.status_code != 200:
        _fail("HTTP 200 expected", f"Got HTTP {response.status_code}: {response.text[:200]}")
        return False

    payload = response.json()
    total_count: int = payload.get("total_count", 0)
    records: list[dict[str, Any]] = payload.get("records", [])

    _pass(f"HTTP 200  total_count={total_count}")

    if total_count == 0:
        _fail(
            "At least 1 telemetry record expected",
            "Audit log is empty — queries may not have been logged",
        )
        return False

    # Validate the structure of the most recent record.
    record = records[0]
    required_fields = [
        "request_id",
        "query_text",
        "response_text",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "model_name",
        "created_at",
    ]
    missing = [f for f in required_fields if f not in record]
    if missing:
        _fail("All TelemetryRecord fields present", f"Missing fields: {missing}")
        return False

    _pass(f"{total_count} record(s) in audit log — most recent:")
    _info(f"  request_id  : {record['request_id']}")
    _info(f"  query_text  : {record['query_text'][:80]!r}")
    _info(f"  latency_ms  : {record['latency_ms']:.1f}")
    _info(f"  total_tokens: {record['total_tokens']}")
    _info(f"  model_name  : {record['model_name']}")
    _info(f"  created_at  : {record['created_at']}")

    return True


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def run_smoke_test(base_url: str) -> int:
    """Execute all probes sequentially against the running server.

    Args:
        base_url: Base URL of the DocuQuery server (e.g. 'http://localhost:8000').

    Returns:
        Exit code: 0 if all probes passed, 1 if any probe failed.
    """
    print(f"\n{_BOLD}{'=' * 70}{_RESET}")
    print(f"{_BOLD}  DocuQuery RAG Agent — End-to-End Smoke Test{_RESET}")
    print(f"{_BOLD}  Target: {base_url}{_RESET}")
    print(f"{_BOLD}{'=' * 70}{_RESET}")

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        results: list[tuple[str, bool]] = []

        # Probe 1: Health check — must pass before proceeding with ingest.
        health_ok = await probe_health(client)
        results.append(("Health probe", health_ok))
        if not health_ok:
            print(
                f"\n{_RED}Server is not reachable or not healthy. "
                f"Aborting remaining probes.{_RESET}"
            )
            _print_summary(results)
            return 1

        # Probe 2: Ingest the SLA policy document.
        chunks_ingested = await probe_ingest(client)
        ingest_ok = chunks_ingested >= 0
        results.append(("Document ingest", ingest_ok))

        # Probes 3–6 run regardless of ingest outcome (the corpus may already be
        # populated from a previous smoke test run), but we note the ingest result.
        grounded_ok = await probe_grounded_query(client)
        results.append(("Grounded query", grounded_ok))

        ungrounded_ok = await probe_ungrounded_query(client)
        results.append(("Ungrounded query (refusal check)", ungrounded_ok))

        sse_ok = await probe_sse_stream(client)
        results.append(("SSE streaming", sse_ok))

        analytics_ok = await probe_analytics(client)
        results.append(("Analytics / telemetry", analytics_ok))

    return _print_summary(results)


def _print_summary(results: list[tuple[str, bool]]) -> int:
    """Print the final pass/fail summary table and return the exit code.

    Args:
        results: List of (probe_name, passed) tuples.

    Returns:
        0 if all probes passed, 1 otherwise.
    """
    print(f"\n{_BOLD}{'=' * 70}{_RESET}")
    print(f"{_BOLD}  Summary{_RESET}")
    print(f"{_BOLD}{'=' * 70}{_RESET}")

    all_passed = True
    for label, passed in results:
        status = f"{_GREEN}PASS{_RESET}" if passed else f"{_RED}FAIL{_RESET}"
        print(f"  {status}  {label}")
        if not passed:
            all_passed = False

    print(f"{_BOLD}{'=' * 70}{_RESET}")
    if all_passed:
        print(f"\n{_GREEN}{_BOLD}All probes passed. Service is healthy.{_RESET}\n")
        return 0
    else:
        failed_count = sum(1 for _, p in results if not p)
        print(
            f"\n{_RED}{_BOLD}{failed_count} probe(s) failed. "
            f"Review output above for details.{_RESET}\n",
            file=sys.stderr,
        )
        return 1


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DocuQuery RAG Agent — End-to-End Smoke Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-url",
        default=_BASE_URL_DEFAULT,
        metavar="URL",
        help=f"Base URL of the running DocuQuery server (default: {_BASE_URL_DEFAULT})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exit_code = asyncio.run(run_smoke_test(args.base_url))
    sys.exit(exit_code)
