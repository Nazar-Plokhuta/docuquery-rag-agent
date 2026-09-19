"""Query router: non-streaming and SSE streaming RAG endpoints.

Both endpoints delegate entirely to ``RAGEngine`` — no vector store or LLM
calls are permitted inside this module.  The streaming endpoint converts the
engine's async token generator into RFC-compliant Server-Sent Events so any
SSE-capable client can consume incremental output without polling.

SSE format used throughout:
    data: {"token": "<token_delta>"}\n\n   — incremental content chunk
    data: [DONE]\n\n                       — explicit stream termination sentinel
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_rag_engine
from src.api.schemas.query import QueryRequest, QueryResponse
from src.core.rag.engine import RAGEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "/",
    response_model=QueryResponse,
    summary="RAG query — synchronous",
    description=(
        "Retrieve relevant document chunks, assemble a token-budgeted prompt, "
        "and return the grounded LLM answer with source citations in a single JSON response."
    ),
)
async def query_endpoint(
    payload: QueryRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
) -> QueryResponse:
    """Execute a non-streaming RAG query and return the full response at once.

    The entire retrieval, token budgeting, and LLM generation pipeline completes
    before the response is serialised.  Prefer ``POST /query/stream`` for
    large completions where perceived latency matters.

    Args:
        payload: Validated ``QueryRequest`` carrying the user question and
            retrieval tuning parameters.
        rag_engine: Injected ``RAGEngine`` resolved from application state.

    Returns:
        ``QueryResponse`` with the grounded answer, citations, latency, and
        token usage figures.
    """
    result = await rag_engine.query(
        query_text=payload.query,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
    )
    logger.info(
        "Query completed. latency_ms=%.1f total_tokens=%d citations=%d",
        result.latency_ms,
        result.total_tokens,
        len(result.citations),
    )
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        citations=list(result.citations),
        latency_ms=result.latency_ms,
        total_tokens=result.total_tokens,
    )


@router.post(
    "/stream",
    summary="RAG query — SSE streaming",
    description=(
        "Stream the grounded LLM response token-by-token via Server-Sent Events. "
        "Each event carries a single token delta. The stream terminates with ``[DONE]``."
    ),
    response_class=StreamingResponse,
)
async def stream_query_endpoint(
    payload: QueryRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
) -> StreamingResponse:
    """Execute a streaming RAG query and emit incremental SSE token events.

    The retrieval and token budgeting pipeline runs to completion before the
    first token is emitted.  Each token delta from the OpenAI streaming
    completion is immediately forwarded to the client, reducing time-to-first-
    token for long completions.

    SSE event format:
        ``data: {"token": "<delta>"}\n\n``
        ``data: [DONE]\n\n``

    Args:
        payload: Validated ``QueryRequest`` carrying the user question and
            retrieval tuning parameters.
        rag_engine: Injected ``RAGEngine`` resolved from application state.

    Returns:
        ``StreamingResponse`` with ``media_type="text/event-stream"``.
    """

    async def _event_generator() -> AsyncGenerator[str, None]:
        """Convert the engine's token stream into SSE-formatted strings."""
        async for token_delta in rag_engine.stream_query(
            query_text=payload.query,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
        ):
            yield f"data: {json.dumps({'token': token_delta})}\n\n"
        # Explicit termination sentinel required by the SSE standard so clients
        # know when the stream has ended rather than inferring it from a timeout.
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            # Prevent proxies from buffering the event stream.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
