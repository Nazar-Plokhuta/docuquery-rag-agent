"""Integration tests for ``POST /api/v1/query/stream`` (SSE streaming endpoint).

Coverage:
- Response carries ``Content-Type: text/event-stream``.
- Response body contains the explicit ``[DONE]`` termination sentinel.
- Each token delta is wrapped in the canonical SSE envelope
  ``data: {"token": "..."}\n\n``.
- The fallback refusal phrase is yielded as a single SSE event when the
  engine emits it (no qualifying chunks retrieved).
- Input validation (empty query) still returns HTTP 422 before any streaming
  begins.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from src.core.rag.prompts import FALLBACK_REFUSAL_MESSAGE


def _make_token_stream(*tokens: str):
    """Return an async generator factory that yields the provided token strings."""

    async def _gen(*args, **kwargs) -> AsyncGenerator[str, None]:
        for token in tokens:
            yield token

    return _gen


@pytest.mark.asyncio
async def test_stream_returns_text_event_stream_content_type(
    integration_client: AsyncClient,
    mock_rag_engine: AsyncMock,
) -> None:
    """The ``/stream`` endpoint must respond with ``Content-Type: text/event-stream``."""
    mock_rag_engine.stream_query.side_effect = _make_token_stream("Hello")

    response = await integration_client.post(
        "/api/v1/query/stream",
        json={"query": "Test query"},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_stream_contains_done_sentinel(
    integration_client: AsyncClient,
    mock_rag_engine: AsyncMock,
) -> None:
    """The stream body must include the ``[DONE]`` termination sentinel."""
    mock_rag_engine.stream_query.side_effect = _make_token_stream("token1", "token2")

    response = await integration_client.post(
        "/api/v1/query/stream",
        json={"query": "Sentinel test"},
    )

    assert "data: [DONE]" in response.text


@pytest.mark.asyncio
async def test_stream_sse_token_format(
    integration_client: AsyncClient,
    mock_rag_engine: AsyncMock,
) -> None:
    """Each token delta must be wrapped in the canonical SSE envelope."""
    mock_rag_engine.stream_query.side_effect = _make_token_stream("Hello", " world")

    response = await integration_client.post(
        "/api/v1/query/stream",
        json={"query": "Format test"},
    )

    body = response.text
    # Verify at least one well-formed SSE data line is present.
    assert 'data: {"token": "Hello"}' in body
    assert 'data: {"token": " world"}' in body


@pytest.mark.asyncio
async def test_stream_sse_events_are_valid_json(
    integration_client: AsyncClient,
    mock_rag_engine: AsyncMock,
) -> None:
    """All ``data:`` lines (except ``[DONE]``) must contain parseable JSON."""
    mock_rag_engine.stream_query.side_effect = _make_token_stream("A", "B", "C")

    response = await integration_client.post(
        "/api/v1/query/stream",
        json={"query": "JSON validation test"},
    )

    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data: "):]
        if payload.strip() == "[DONE]":
            continue
        # Must be valid JSON with a "token" key.
        parsed = json.loads(payload)
        assert "token" in parsed


@pytest.mark.asyncio
async def test_stream_fallback_refusal_is_yielded(
    integration_client: AsyncClient,
    mock_rag_engine: AsyncMock,
) -> None:
    """When the engine emits only the fallback refusal, it must appear in the stream."""
    mock_rag_engine.stream_query.side_effect = _make_token_stream(FALLBACK_REFUSAL_MESSAGE)

    response = await integration_client.post(
        "/api/v1/query/stream",
        json={"query": "Unanswerable question"},
    )

    assert response.status_code == 200
    assert FALLBACK_REFUSAL_MESSAGE in response.text
    assert "data: [DONE]" in response.text


@pytest.mark.asyncio
async def test_stream_rejects_empty_query(
    integration_client: AsyncClient,
) -> None:
    """Empty ``query`` string must be rejected with HTTP 422 before any streaming."""
    response = await integration_client.post(
        "/api/v1/query/stream",
        json={"query": ""},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_stream_engine_called_with_correct_parameters(
    integration_client: AsyncClient,
    mock_rag_engine: AsyncMock,
) -> None:
    """The streaming route must forward all validated parameters to ``stream_query``."""
    mock_rag_engine.stream_query.side_effect = _make_token_stream("ok")

    await integration_client.post(
        "/api/v1/query/stream",
        json={"query": "Param test", "top_k": 6, "score_threshold": 0.4},
    )

    mock_rag_engine.stream_query.assert_called_once_with(
        query_text="Param test",
        top_k=6,
        score_threshold=0.4,
    )
