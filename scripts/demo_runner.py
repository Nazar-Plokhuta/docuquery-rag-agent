"""Terminal showcase for the DocuQuery RAG pipeline.

Renders a paced, human-readable execution log against the live HTTP API.
When nothing is listening on port 8000, the script starts Uvicorn headless and
points that process at an ephemeral Chroma and SQLite directory so the
recording does not depend on, or mutate, a developer's existing index
(re-ingesting ``sla_policy`` would collide with chunk ids already stored).

Usage:
    python scripts/demo_runner.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from rich import box
from rich.console import Console
from rich.json import JSON
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SLA_PATH = _REPO_ROOT / "data" / "sample_docs" / "sla_policy.docx"
_BASE_URL = "http://localhost:8000"
_HEALTH_PATH = "/api/v1/health/"
_HEALTH_TIMEOUT_S = 5.0

_GROUNDED_QUERY = "What is the P0 incident response time for Tier 1 Platinum?"
_UNGROUNDED_QUERY = "Who won the 2026 World Cup?"

# Copied from src/core/rag/prompts.py so this script stays a pure HTTP client.
_FALLBACK_REFUSAL_PREFIX = "I am sorry, but the provided documentation does not contain"

# Loosely related SLA chunks sit below this bar, so the out-of-scope question
# takes the engine's deterministic no-evidence path (exact refusal, no citations).
_REFUSAL_SCORE_THRESHOLD = 0.45

_PAUSE_AFTER_HEALTH_S = 4.0
_PAUSE_AFTER_INGEST_S = 4.5
_PAUSE_AFTER_GROUNDED_S = 7.0
_PAUSE_AFTER_REFUSAL_S = 6.0

_DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# The model emits ``[Source: ..., Section: <heading>]`` on each factual claim.
_INLINE_SECTION_RE = re.compile(r"\[Source:\s*[^,\]]+,\s*Section:\s*(?P<section>[^\]]+?)\s*\]")

console = Console(highlight=False)


class DemoError(Exception):
    """A showcase step could not be completed against the live API."""


def _banner() -> Panel:
    """Opening banner. Printed before any server work so the recording is never blank."""
    subtitle = Text(
        "Grounded retrieval   ·   exact citations   ·   deterministic refusal",
        style="dim",
        justify="center",
    )
    return Panel(
        subtitle,
        title="[bold cyan]DocuQuery Enterprise RAG — Pipeline Showcase[/]",
        title_align="center",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 2),
    )


def _print_step(number: str, title: str, detail: str) -> None:
    console.print()
    console.print(f"[bold cyan]{number}[/]  [bold]{title}[/]")
    console.print(f"    [dim]{detail}[/]")


def _print_json(payload: dict[str, Any]) -> None:
    console.print(Padding(JSON.from_data(payload), (0, 0, 0, 4)))


def _is_healthy(client: httpx.Client) -> bool:
    try:
        response = client.get(_HEALTH_PATH, timeout=1.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _spawn_server(workspace: tempfile.TemporaryDirectory[str]) -> subprocess.Popen[bytes]:
    """Start Uvicorn with stdout and stderr discarded so the showcase stays quiet.

    The child inherits the current environment (including the API key from the
    process environment and ``.env`` resolved via the repo-root working
    directory) but overrides persistence paths to the ephemeral workspace.
    """
    root = Path(workspace.name)
    env = os.environ.copy()
    env["CHROMA_PERSIST_DIRECTORY"] = str(root / "chroma")
    env["SQLITE_DATABASE_PATH"] = str(root / "telemetry.db")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "warning",
        ],
        cwd=_REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_until_ready(client: httpx.Client, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + _HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise DemoError(
                f"API process exited before becoming ready (code {process.returncode})."
            )
        if _is_healthy(client):
            return
        time.sleep(0.2)
    raise DemoError("API did not respond at http://localhost:8000/api/v1/health/ within 5 seconds.")


def _stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _ensure_server(
    client: httpx.Client,
) -> tuple[subprocess.Popen[bytes] | None, tempfile.TemporaryDirectory[str] | None]:
    """Reuse a healthy server, or start a headless one and wait for liveness."""
    if _is_healthy(client):
        return None, None

    workspace: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
        prefix="docuquery-demo-"
    )
    process = _spawn_server(workspace)
    try:
        _wait_until_ready(client, process)
    except DemoError:
        _stop_server(process)
        workspace.cleanup()
        raise
    return process, workspace


def _read_json(response: httpx.Response, action: str) -> dict[str, Any]:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise DemoError(
            f"{action} failed with HTTP {response.status_code}: {response.text[:300]}"
        ) from exc
    try:
        payload: Any = response.json()
    except ValueError as exc:
        raise DemoError(f"{action} returned a non-JSON body.") from exc
    if not isinstance(payload, dict):
        raise DemoError(f"{action} returned an unexpected payload.")
    return payload


def _step_health(client: httpx.Client) -> None:
    _print_step("01", "Service Health Check", "GET /api/v1/health/")
    payload = _read_json(client.get(_HEALTH_PATH), "Health check")
    _print_json(
        {
            "status": payload.get("status"),
            "version": payload.get("version"),
            "environment": payload.get("environment"),
        }
    )


def _step_ingest(client: httpx.Client) -> None:
    if not _SLA_PATH.is_file():
        raise DemoError(f"Sample document not found: {_SLA_PATH}")

    _print_step(
        "02",
        "Document Ingestion",
        "POST /api/v1/ingest/file    data/sample_docs/sla_policy.docx",
    )
    with _SLA_PATH.open("rb") as handle:
        response = client.post(
            "/api/v1/ingest/file",
            files={"file": (_SLA_PATH.name, handle, _DOCX_MEDIA_TYPE)},
            timeout=120.0,
        )
    payload = _read_json(response, "Document ingestion")
    if payload.get("status") != "success":
        raise DemoError(f"Ingestion did not succeed: {payload!r}")
    _print_json(
        {
            "status": payload.get("status"),
            "filename": payload.get("filename"),
            "chunks_ingested": payload.get("chunks_ingested"),
        }
    )


def _source_label(citations: list[Any]) -> str:
    """Present the uploaded filename when the index stores only the file stem."""
    for item in citations:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        if source in {"", _SLA_PATH.stem, _SLA_PATH.name}:
            return _SLA_PATH.name
        if source:
            return source
    return _SLA_PATH.name


def _section_title(answer: str, citations: list[Any]) -> str:
    """Return the section attached to the factual claim.

    Indexed chunk metadata records the last Markdown heading in a window, so a
    chunk that contains the Platinum table and ends in a later tier is labeled
    with that later heading. The inline citation is the section the answer
    actually used.
    """
    match = _INLINE_SECTION_RE.search(answer)
    if match:
        section = match.group("section").strip()
        if section:
            return section
    for item in citations:
        if isinstance(item, dict):
            section = str(item.get("section", "")).strip()
            if section:
                return section
    return "—"


def _meta_line(label: str, value: str, accent: str) -> Text:
    line = Text()
    line.append(f"{label}: ", style=f"bold {accent}")
    line.append(value)
    return line


def _step_grounded(client: httpx.Client) -> None:
    _print_step("03", "Grounded Query", "POST /api/v1/query/")
    console.print(f'    [italic]"{_GROUNDED_QUERY}"[/]')
    payload = _read_json(
        client.post(
            "/api/v1/query/",
            json={"query": _GROUNDED_QUERY, "top_k": 5, "score_threshold": 0.25},
            timeout=90.0,
        ),
        "Grounded query",
    )
    answer = str(payload.get("answer", "")).strip()
    citations_raw = payload.get("citations")
    citations: list[Any] = citations_raw if isinstance(citations_raw, list) else []
    if not answer or answer.startswith(_FALLBACK_REFUSAL_PREFIX) or not citations:
        raise DemoError(
            "Grounded query did not return a cited answer from the SLA policy. "
            f"Answer: {answer[:240]}"
        )

    section = _section_title(answer, citations)
    latency_ms = float(payload.get("latency_ms", 0.0))
    body = Text(answer)
    body.append("\n\n")
    body.append(_meta_line("Source", _source_label(citations), "green"))
    body.append("\n")
    body.append(_meta_line("Section", section, "green"))
    body.append("\n")
    body.append(_meta_line("Retrieval latency", f"{latency_ms:,.0f} ms", "green"))
    console.print()
    console.print(
        Panel(
            body,
            title="[bold green]Ground Truth Verification[/]",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 2),
        )
    )


def _step_refusal(client: httpx.Client) -> None:
    _print_step("04", "Zero-Hallucination Guardrail", "POST /api/v1/query/")
    console.print(f'    [italic]"{_UNGROUNDED_QUERY}"[/]')
    payload = _read_json(
        client.post(
            "/api/v1/query/",
            json={
                "query": _UNGROUNDED_QUERY,
                "top_k": 5,
                "score_threshold": _REFUSAL_SCORE_THRESHOLD,
            },
            timeout=90.0,
        ),
        "Out-of-scope query",
    )
    answer = str(payload.get("answer", "")).strip()
    citations_raw = payload.get("citations")
    citations: list[Any] = citations_raw if isinstance(citations_raw, list) else []
    if not answer.startswith(_FALLBACK_REFUSAL_PREFIX):
        raise DemoError(f"Out-of-scope query was not refused. Answer: {answer[:240]}")
    if citations:
        raise DemoError(f"Out-of-scope query returned {len(citations)} citation(s); expected 0.")

    body = Text(answer)
    body.append("\n\n")
    body.append(_meta_line("Citations", "0", "magenta"))
    console.print()
    console.print(
        Panel(
            body,
            title="[bold magenta]Anti-Hallucination Guardrail Triggered[/]",
            border_style="red",
            box=box.ROUNDED,
            padding=(0, 2),
        )
    )


def _close_showcase() -> None:
    console.print()
    console.print(
        "[bold green]Pipeline complete[/]  [dim]· citation verified · refusal guardrail held[/]"
    )


def main() -> int:
    """Run the four-step showcase and always tear down a server this process started."""
    console.print(_banner())
    server: subprocess.Popen[bytes] | None = None
    workspace: tempfile.TemporaryDirectory[str] | None = None
    try:
        with httpx.Client(base_url=_BASE_URL, timeout=120.0) as client:
            server, workspace = _ensure_server(client)
            _step_health(client)
            time.sleep(_PAUSE_AFTER_HEALTH_S)
            _step_ingest(client)
            time.sleep(_PAUSE_AFTER_INGEST_S)
            _step_grounded(client)
            time.sleep(_PAUSE_AFTER_GROUNDED_S)
            _step_refusal(client)
            _close_showcase()
            time.sleep(_PAUSE_AFTER_REFUSAL_S)
        return 0
    except DemoError as exc:
        console.print()
        console.print(Panel(str(exc), title="[bold red]Showcase aborted[/]", border_style="red"))
        return 1
    finally:
        if server is not None:
            _stop_server(server)
        if workspace is not None:
            workspace.cleanup()


if __name__ == "__main__":
    sys.exit(main())
