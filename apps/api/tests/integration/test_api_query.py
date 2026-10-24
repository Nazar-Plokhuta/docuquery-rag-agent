"""Integration tests for ``POST /api/v1/query`` (non-streaming RAG endpoint).

Coverage:
- HTTP 200 with a well-formed ``QueryResponse`` payload when the engine
  returns a valid ``RAGResult``.
- HTTP 422 on empty query string (Pydantic ``min_length`` enforcement).
- HTTP 422 on missing required ``query`` field.
- HTTP 422 on ``top_k`` out-of-range values.
- Citation list is faithfully projected from the engine result.
- Token usage and latency figures pass through without mutation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from src.core.models import Citation, RAGResult


def _make_rag_result(
    query: str = "What is RAG?",
    answer: str = "RAG stands for Retrieval-Augmented Generation.",
    citations: list[Citation] | None = None,
    latency_ms: float = 123.4,
    prompt_tokens: int = 50,
    completion_tokens: int = 20,
    total_tokens: int = 70,
    retrieved_chunks_count: int = 1,
) -> RAGResult:
    """Construct a minimal valid ``RAGResult`` for test fixtures."""
    return RAGResult(
        query=query,
        answer=answer,
        citations=citations or [],
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        retrieved_chunks_count=retrieved_chunks_count,
    )


@pytest.mark.asyncio
async def test_query_returns_200_with_valid_response(
    integration_client: AsyncClient,
    mock_rag_engine: AsyncMock,
) -> None:
    """``POST /api/v1/query/`` must return HTTP 200 with a valid ``QueryResponse``."""
    mock_rag_engine.query.return_value = _make_rag_result()

    response = await integration_client.post(
        "/api/v1/query/",
        json={"query": "What is RAG?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "What is RAG?"
    assert payload["answer"] == "RAG stands for Retrieval-Augmented Generation."
    assert isinstance(payload["citations"], list)
    assert isinstance(payload["latency_ms"], float)
    assert isinstance(payload["total_tokens"], int)


@pytest.mark.asyncio
async def test_query_engine_called_with_correct_parameters(
    integration_client: AsyncClient,
    mock_rag_engine: AsyncMock,
) -> None:
    """The route must forward all validated request parameters to ``RAGEngine.query``."""
    mock_rag_engine.query.return_value = _make_rag_result()

    await integration_client.post(
        "/api/v1/query/",
        json={"query": "Explain chunking.", "top_k": 8, "score_threshold": 0.5},
    )

    mock_rag_engine.query.assert_called_once_with(
        query_text="Explain chunking.",
        top_k=8,
        score_threshold=0.5,
    )


@pytest.mark.asyncio
async def test_query_response_includes_citations(
    integration_client: AsyncClient,
    mock_rag_engine: AsyncMock,
) -> None:
    """Citations returned by the engine must appear verbatim in the response."""
    citation = Citation(
        source="architecture.md",
        section="Core Interfaces",
        chunk_id="architecture#c0001",
    )
    mock_rag_engine.query.return_value = _make_rag_result(citations=[citation])

    response = await integration_client.post(
        "/api/v1/query/",
        json={"query": "What are the core interfaces?"},
    )

    assert response.status_code == 200
    citations = response.json()["citations"]
    assert len(citations) == 1
    assert citations[0]["source"] == "architecture.md"
    assert citations[0]["section"] == "Core Interfaces"
    assert citations[0]["chunk_id"] == "architecture#c0001"


@pytest.mark.asyncio
async def test_query_rejects_empty_query_string(
    integration_client: AsyncClient,
) -> None:
    """Empty ``query`` string must be rejected with HTTP 422 (``min_length=1``)."""
    response = await integration_client.post(
        "/api/v1/query/",
        json={"query": ""},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_rejects_missing_query_field(
    integration_client: AsyncClient,
) -> None:
    """Omitting the required ``query`` field must be rejected with HTTP 422."""
    response = await integration_client.post(
        "/api/v1/query/",
        json={"top_k": 4},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_rejects_top_k_below_minimum(
    integration_client: AsyncClient,
) -> None:
    """``top_k`` of 0 violates ``ge=1`` and must be rejected with HTTP 422."""
    response = await integration_client.post(
        "/api/v1/query/",
        json={"query": "test", "top_k": 0},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_rejects_top_k_above_maximum(
    integration_client: AsyncClient,
) -> None:
    """``top_k`` of 21 violates ``le=20`` and must be rejected with HTTP 422."""
    response = await integration_client.post(
        "/api/v1/query/",
        json={"query": "test", "top_k": 21},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_rejects_extra_fields(
    integration_client: AsyncClient,
) -> None:
    """Unknown fields must be rejected with HTTP 422 (``extra='forbid'``)."""
    response = await integration_client.post(
        "/api/v1/query/",
        json={"query": "test", "unknown_field": "value"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_token_usage_is_preserved(
    integration_client: AsyncClient,
    mock_rag_engine: AsyncMock,
) -> None:
    """``total_tokens`` from the engine result must be echoed without modification."""
    mock_rag_engine.query.return_value = _make_rag_result(total_tokens=999)

    response = await integration_client.post(
        "/api/v1/query/",
        json={"query": "Token test"},
    )

    assert response.status_code == 200
    assert response.json()["total_tokens"] == 999
