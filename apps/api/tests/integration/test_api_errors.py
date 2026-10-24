"""Integration tests for the HTTP failure contract of the query and ingestion endpoints.

Every fault must reach the client as a parseable ``{error, message}`` JSON body
with a meaningful status — never as a bare ``text/plain`` 500 and never as a
connection torn down mid-body. The streaming endpoint is the subtle case: a fault
before the first token is a real HTTP error, a fault after it is an in-band
``error`` frame followed by ``[DONE]``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.exceptions import (
    DocumentParsingError,
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
    StorageError,
)
from src.main import app


def _frames(body: str) -> list[str]:
    """Return the payload of every ``data:`` line of an SSE body, in order."""
    return [line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data: ")]


def _stream_that_fails_immediately(error: Exception) -> Callable[..., AsyncGenerator[str, None]]:
    async def _gen(*_args: object, **_kwargs: object) -> AsyncGenerator[str, None]:
        raise error
        yield  # pragma: no cover - makes this an async generator

    return _gen


def _stream_that_fails_after(
    tokens: list[str], error: Exception
) -> Callable[..., AsyncGenerator[str, None]]:
    async def _gen(*_args: object, **_kwargs: object) -> AsyncGenerator[str, None]:
        for token in tokens:
            yield token
        raise error

    return _gen


_LLM_FAULTS = [
    (LLMAuthenticationError("bad key"), 502),
    (LLMProviderError("rejected"), 502),
    (LLMRateLimitError("slow down"), 429),
    (LLMUnavailableError("down"), 503),
    (LLMTimeoutError("too slow"), 504),
]


class TestNonStreamingQueryErrors:
    @pytest.mark.parametrize(("fault", "status"), _LLM_FAULTS, ids=lambda v: type(v).__name__)
    async def test_provider_fault_maps_to_status_and_envelope(
        self,
        integration_client: AsyncClient,
        mock_rag_engine: AsyncMock,
        fault: LLMProviderError,
        status: int,
    ) -> None:
        mock_rag_engine.query.side_effect = fault

        response = await integration_client.post("/api/v1/query/", json={"query": "q"})

        assert response.status_code == status
        assert response.json() == {"error": type(fault).__name__, "message": fault.message}

    async def test_storage_fault_maps_to_503_envelope(
        self, integration_client: AsyncClient, mock_rag_engine: AsyncMock
    ) -> None:
        mock_rag_engine.query.side_effect = StorageError("chroma unavailable")

        response = await integration_client.post("/api/v1/query/", json={"query": "q"})

        assert response.status_code == 503
        assert response.json()["error"] == "StorageError"


class TestIngestionErrors:
    async def test_embedding_credential_failure_is_a_502_envelope(
        self, integration_client: AsyncClient, mock_ingestion_pipeline: AsyncMock
    ) -> None:
        mock_ingestion_pipeline.ingest_document.side_effect = LLMAuthenticationError(
            "The openai provider rejected the credentials used for embedding."
        )

        response = await integration_client.post(
            "/api/v1/ingest/file",
            files={"file": ("report.pdf", b"%PDF-1.4", "application/pdf")},
        )

        assert response.status_code == 502
        body = response.json()
        assert body["error"] == "LLMAuthenticationError"
        assert "rejected the credentials" in body["message"]

    async def test_unparseable_document_is_a_422_envelope(
        self, integration_client: AsyncClient, mock_ingestion_pipeline: AsyncMock
    ) -> None:
        mock_ingestion_pipeline.ingest_document.side_effect = DocumentParsingError(
            "Unable to read PDF 'x.pdf': the file may be corrupted or invalid."
        )

        response = await integration_client.post(
            "/api/v1/ingest/file",
            files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")},
        )

        assert response.status_code == 422
        assert response.json()["error"] == "DocumentParsingError"


class TestStreamingErrors:
    async def test_fault_before_first_token_is_a_real_http_error(
        self, integration_client: AsyncClient, mock_rag_engine: AsyncMock
    ) -> None:
        mock_rag_engine.stream_query.side_effect = _stream_that_fails_immediately(
            LLMAuthenticationError("embedding key rejected")
        )

        response = await integration_client.post("/api/v1/query/stream", json={"query": "q"})

        assert response.status_code == 502
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {
            "error": "LLMAuthenticationError",
            "message": "embedding key rejected",
        }

    async def test_fault_after_tokens_is_an_in_band_error_frame_then_done(
        self, integration_client: AsyncClient, mock_rag_engine: AsyncMock
    ) -> None:
        mock_rag_engine.stream_query.side_effect = _stream_that_fails_after(
            ["Hel", "lo"], LLMUnavailableError("provider dropped the stream")
        )

        response = await integration_client.post("/api/v1/query/stream", json={"query": "q"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = _frames(response.text)
        assert [json.loads(f) for f in frames[:-1]] == [
            {"token": "Hel"},
            {"token": "lo"},
            {"error": "LLMUnavailableError", "message": "provider dropped the stream"},
        ]
        assert frames[-1] == "[DONE]"

    async def test_unexpected_fault_mid_stream_is_reported_without_leaking_details(
        self, integration_client: AsyncClient, mock_rag_engine: AsyncMock
    ) -> None:
        mock_rag_engine.stream_query.side_effect = _stream_that_fails_after(
            ["Hi"], RuntimeError("secret-internal-detail")
        )

        response = await integration_client.post("/api/v1/query/stream", json={"query": "q"})

        assert "secret-internal-detail" not in response.text
        frames = _frames(response.text)
        assert json.loads(frames[1])["error"] == "InternalServerError"
        assert frames[-1] == "[DONE]"

    async def test_stream_with_no_tokens_still_terminates_with_done(
        self, integration_client: AsyncClient, mock_rag_engine: AsyncMock
    ) -> None:
        # An engine that yields nothing must not be mistaken for a failure.
        async def _empty(*_args: object, **_kwargs: object) -> AsyncGenerator[str, None]:
            return
            yield  # pragma: no cover

        mock_rag_engine.stream_query.side_effect = _empty

        response = await integration_client.post("/api/v1/query/stream", json={"query": "q"})

        assert response.status_code == 200
        assert _frames(response.text) == ["[DONE]"]


class TestUnhandledExceptions:
    async def test_unexpected_fault_returns_json_envelope_not_plain_text(
        self, integration_client: AsyncClient, mock_rag_engine: AsyncMock
    ) -> None:
        """Faults outside the domain hierarchy are still answered in the standard shape."""
        mock_rag_engine.query.side_effect = RuntimeError("secret-internal-detail")
        # The shared client re-raises server exceptions into the test; a real client
        # would just see the response, which is what this test needs to observe.
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as client:
            response = await client.post("/api/v1/query/", json={"query": "q"})

        assert response.status_code == 500
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert body["error"] == "InternalServerError"
        assert "secret-internal-detail" not in response.text
