"""Query router: non-streaming and SSE streaming RAG endpoints.

Both endpoints delegate entirely to ``RAGEngine`` — no vector store or LLM
calls are permitted inside this module.  The streaming endpoint converts the
engine's async token generator into RFC-compliant Server-Sent Events so any
SSE-capable client can consume incremental output without polling.

SSE format used throughout:
    data: {"token": "<token_delta>"}\n\n   — incremental content chunk
    data: {"error": "<Type>", "message": "<text>"}\n\n
                                           — terminal failure after the stream began
    data: [DONE]\n\n                       — explicit stream termination sentinel

Failure transport depends on *when* the fault occurs. Retrieval and the opening
of the completion happen before the first token exists, so they surface as an
ordinary HTTP error (domain status plus ``{error, message}`` JSON). Once a token
has been sent the status line is already committed, so a later fault can only be
reported in-band as the error event above, always followed by ``[DONE]``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_rag_engine
from src.api.schemas.query import QueryRequest, QueryResponse
from src.core.exceptions import AppException
from src.core.rag.engine import RAGEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])

_UNEXPECTED_STREAM_ERROR_MESSAGE = "An unexpected error interrupted the response stream."


def _sse_frame(payload: dict[str, str]) -> str:
    """Serialise ``payload`` as one standard ``data: <json>`` SSE frame."""
    return f"data: {json.dumps(payload)}\n\n"


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
        "Each event carries a single token delta. The stream terminates with ``[DONE]``. "
        "A fault before the first token is an ordinary HTTP error with an "
        "``{error, message}`` body; a fault after streaming began is sent as a final "
        '``{"error": ..., "message": ...}`` event, still followed by ``[DONE]``.'
    ),
    response_class=StreamingResponse,
)
async def stream_query_endpoint(
    payload: QueryRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
) -> StreamingResponse:
    """Execute a streaming RAG query and emit incremental SSE token events.

    The retrieval and token budgeting pipeline runs to completion before the
    first token is emitted.  Each token delta from the streaming completion is
    immediately forwarded to the client, reducing time-to-first-token for long
    completions.

    The first token is awaited *before* the response object is built. A
    ``StreamingResponse`` commits its 200 status as soon as it starts, so a fault
    in embedding, retrieval or the completion handshake would otherwise be
    unreportable and would tear the connection down mid-body; reverse proxies
    turn that into an empty reply that clients can only read as "unreachable".
    Raising here instead routes the fault through the domain exception handler,
    which answers with its proper status and ``{error, message}`` body.

    SSE event format:
        ``data: {"token": "<delta>"}\n\n``
        ``data: {"error": "<Type>", "message": "<text>"}\n\n``  (terminal)
        ``data: [DONE]\n\n``

    Args:
        payload: Validated ``QueryRequest`` carrying the user question and
            retrieval tuning parameters.
        rag_engine: Injected ``RAGEngine`` resolved from application state.

    Returns:
        ``StreamingResponse`` with ``media_type="text/event-stream"``.

    Raises:
        AppException: Any domain fault raised before the first token, e.g.
            ``LLMProviderError`` or ``StorageError``.
    """
    token_stream = rag_engine.stream_query(
        query_text=payload.query,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
    )
    first_token: str | None = await anext(token_stream, None)

    async def _event_generator() -> AsyncGenerator[str, None]:
        """Convert the engine's token stream into SSE-formatted strings."""
        try:
            if first_token is not None:
                yield _sse_frame({"token": first_token})
                async for token_delta in token_stream:
                    yield _sse_frame({"token": token_delta})
        except AppException as exc:
            logger.error(
                "Response stream aborted after it began [%s]: %s", type(exc).__name__, exc.message
            )
            yield _sse_frame({"error": type(exc).__name__, "message": exc.message})
        except Exception:
            # The status line is already committed, so the only way left to tell the
            # client is an error frame; the traceback stays in the server log.
            logger.exception("Unexpected failure while streaming a RAG response.")
            yield _sse_frame(
                {"error": "InternalServerError", "message": _UNEXPECTED_STREAM_ERROR_MESSAGE}
            )
        finally:
            await token_stream.aclose()
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
