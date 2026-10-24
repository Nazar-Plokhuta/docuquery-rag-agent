"""Unit tests for RAGEngine.

All LiteLLM calls (``litellm.acompletion`` / ``litellm.aembedding``) and storage
operations are isolated via ``unittest.mock.AsyncMock`` and ``MagicMock``.  No live
network connections, ChromaDB instances, or relational databases are required.

Test groups:
- TestRAGEngineQuery  — non-streaming query() method.
- TestRAGEngineStream — streaming stream_query() async generator.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import httpx
import litellm
import pytest

from src.config.settings import Settings
from src.core.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMUnavailableError,
    StorageError,
)
from src.core.models import Citation, DocumentChunk, RAGResult, RetrievalResult, TelemetryRecord
from src.core.rag.engine import DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K, RAGEngine
from src.core.rag.prompts import FALLBACK_REFUSAL_MESSAGE
from src.core.rag.token_counter import TokenBudgetManager

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_settings() -> Settings:
    """Minimal Settings instance safe for unit tests (no filesystem access)."""
    return Settings(
        app_env="development",
        openai_api_key="sk-test-placeholder",
        openai_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        chroma_persist_directory="/tmp/test_chroma_db",
        database_url="sqlite+aiosqlite:///:memory:",
    )


def _make_retrieval_result(
    document_id: str = "tech-spec",
    content: str = "The system uses async I/O throughout.",
    section: str = "Architecture",
    score: float = 0.85,
    chunk_index: int = 0,
) -> RetrievalResult:
    """Construct a realistic RetrievalResult for injection into mock returns."""
    chunk = DocumentChunk(
        chunk_id=f"{document_id}#c{chunk_index:04d}",
        document_id=document_id,
        content=content,
        metadata={
            "document_id": document_id,
            "section": section,
            "chunk_index": chunk_index,
            "token_count": 10,
        },
        token_count=10,
    )
    return RetrievalResult(chunk=chunk, score=score)


def _make_mock_embedding_response(embedding: list[float] | None = None) -> MagicMock:
    """Build a mock ``litellm.aembedding`` response (items are plain dicts)."""
    response = MagicMock()
    response.data = [{"object": "embedding", "index": 0, "embedding": embedding or [0.1, 0.2, 0.3]}]
    return response


def _make_mock_completion_response(
    content: str = "The system uses async I/O. [Source: tech-spec, Section: Architecture]",
    prompt_tokens: int = 120,
    completion_tokens: int = 30,
) -> MagicMock:
    """Build a mock ``litellm.acompletion`` response."""
    response = MagicMock()
    response.choices[0].message.content = content
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens
    return response


@dataclass(frozen=True)
class LiteLLMMocks:
    """Handles on the patched LiteLLM entry points."""

    acompletion: AsyncMock
    aembedding: AsyncMock


@pytest.fixture
def litellm_mocks(monkeypatch: pytest.MonkeyPatch) -> LiteLLMMocks:
    """Patch ``litellm.acompletion`` and ``litellm.aembedding`` with ``AsyncMock``s.

    The embedding mock is pre-armed with a default vector; completion behaviour is
    configured per test.
    """
    acompletion = AsyncMock()
    aembedding = AsyncMock(return_value=_make_mock_embedding_response())
    monkeypatch.setattr("litellm.acompletion", acompletion)
    monkeypatch.setattr("litellm.aembedding", aembedding)
    return LiteLLMMocks(acompletion=acompletion, aembedding=aembedding)


def _delta_chunk(text: str | None) -> MagicMock:
    """Build a streaming chunk carrying a content delta and no usage block."""
    chunk = MagicMock()
    chunk.choices[0].delta.content = text
    chunk.usage = None
    return chunk


def _usage_chunk(prompt: int, completion: int, total: int) -> MagicMock:
    """Build the usage-only trailer chunk emitted under ``include_usage``."""
    chunk = MagicMock()
    chunk.choices = []
    chunk.usage.prompt_tokens = prompt
    chunk.usage.completion_tokens = completion
    chunk.usage.total_tokens = total
    return chunk


def _build_engine(
    *,
    vector_store: MagicMock | None = None,
    audit_repo: MagicMock | None = None,
    token_budget: MagicMock | None = None,
    settings: Settings | None = None,
) -> tuple[RAGEngine, MagicMock, MagicMock, MagicMock]:
    """Construct a RAGEngine with all dependencies mocked.

    Returns:
        Tuple of (engine, mock_vector_store, mock_audit_repo, mock_token_budget).
    """
    mock_vs = vector_store or MagicMock()
    mock_ar = audit_repo or MagicMock()
    if audit_repo is None:
        mock_ar.log_query = AsyncMock()
    mock_tb = token_budget or MagicMock(spec=TokenBudgetManager)
    cfg = settings or _make_settings()

    engine = RAGEngine(
        vector_store=mock_vs,
        audit_repo=mock_ar,
        token_budget=mock_tb,
        settings=cfg,
    )
    return engine, mock_vs, mock_ar, mock_tb


# ---------------------------------------------------------------------------
# TestRAGEngineQuery
# ---------------------------------------------------------------------------


class TestRAGEngineQuery:
    """Tests for the non-streaming RAGEngine.query() method."""

    async def test_below_threshold_returns_fallback_without_completion_call(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        """When no chunk meets the score threshold, chat completions must NOT be called."""
        engine, mock_vs, mock_ar, _ = _build_engine()

        # Embedding succeeds but similarity search returns a below-threshold result.
        mock_vs.similarity_search = AsyncMock(
            return_value=[_make_retrieval_result(score=0.1)]  # below 0.3 threshold
        )
        mock_ar.log_query = AsyncMock()

        result: RAGResult = await engine.query("What is the SLA?", score_threshold=0.3)

        assert result.answer == FALLBACK_REFUSAL_MESSAGE
        assert result.citations == []
        assert result.retrieved_chunks_count == 0
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0
        # The LLM completion endpoint must never be called on the fast path.
        litellm_mocks.acompletion.assert_not_called()

    async def test_below_threshold_logs_telemetry_with_zero_tokens(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        """Fallback path must still persist a TelemetryRecord with zero token counts."""
        engine, mock_vs, mock_ar, _ = _build_engine()

        mock_vs.similarity_search = AsyncMock(return_value=[])
        mock_ar.log_query = AsyncMock()

        await engine.query("Unanswerable question", score_threshold=0.3)

        mock_ar.log_query.assert_awaited_once()
        logged_record: TelemetryRecord = mock_ar.log_query.call_args[0][0]
        assert logged_record.prompt_tokens == 0
        assert logged_record.completion_tokens == 0
        assert logged_record.total_tokens == 0
        assert logged_record.response_text == FALLBACK_REFUSAL_MESSAGE

    async def test_successful_retrieval_invokes_chat_completion(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        """When chunks pass the threshold, chat completions must be called exactly once."""
        retrieval_result = _make_retrieval_result(score=0.9)
        mock_tb = MagicMock(spec=TokenBudgetManager)
        mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]

        engine, mock_vs, mock_ar, _ = _build_engine(token_budget=mock_tb)
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])
        litellm_mocks.acompletion.return_value = _make_mock_completion_response()
        mock_ar.log_query = AsyncMock()

        await engine.query("How does async I/O work?", score_threshold=0.3)

        litellm_mocks.acompletion.assert_awaited_once()
        call_kwargs = litellm_mocks.acompletion.await_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.0
        assert not call_kwargs.get("stream")
        litellm_mocks.aembedding.assert_awaited_once()
        assert litellm_mocks.aembedding.await_args.kwargs["input"] == ["How does async I/O work?"]

    async def test_successful_retrieval_returns_correct_citations(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        """Citations must be extracted from the metadata of budgeted chunks."""
        retrieval_result = _make_retrieval_result(
            document_id="api-guide",
            section="Endpoints",
            score=0.92,
        )
        mock_tb = MagicMock(spec=TokenBudgetManager)
        mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]

        engine, mock_vs, mock_ar, _ = _build_engine(token_budget=mock_tb)
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])
        litellm_mocks.acompletion.return_value = _make_mock_completion_response()
        mock_ar.log_query = AsyncMock()

        result: RAGResult = await engine.query("List all endpoints.")

        assert len(result.citations) == 1
        citation: Citation = result.citations[0]
        assert citation.source == "api-guide"
        assert citation.section == "Endpoints"
        assert citation.chunk_id == "api-guide#c0000"

    async def test_successful_retrieval_logs_audit_record(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        """A TelemetryRecord must be persisted after a successful completion."""
        retrieval_result = _make_retrieval_result(score=0.88)
        mock_tb = MagicMock(spec=TokenBudgetManager)
        mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]
        completion = _make_mock_completion_response(prompt_tokens=200, completion_tokens=50)

        engine, mock_vs, mock_ar, _ = _build_engine(token_budget=mock_tb)
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])
        litellm_mocks.acompletion.return_value = completion
        mock_ar.log_query = AsyncMock()

        await engine.query("Explain the chunking strategy.")

        mock_ar.log_query.assert_awaited_once()
        logged_record: TelemetryRecord = mock_ar.log_query.call_args[0][0]
        assert logged_record.prompt_tokens == 200
        assert logged_record.completion_tokens == 50
        assert logged_record.total_tokens == 250
        assert logged_record.model_name == "gpt-4o-mini"

    async def test_successful_retrieval_populates_rag_result_fields(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        """RAGResult fields must reflect the completion response and budgeted chunks."""
        retrieval_result = _make_retrieval_result(score=0.75)
        mock_tb = MagicMock(spec=TokenBudgetManager)
        mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]
        answer_text = (
            "Async I/O is achieved via asyncio. [Source: tech-spec, Section: Architecture]"
        )
        completion = _make_mock_completion_response(
            content=answer_text,
            prompt_tokens=100,
            completion_tokens=20,
        )

        engine, mock_vs, mock_ar, _ = _build_engine(token_budget=mock_tb)
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])
        litellm_mocks.acompletion.return_value = completion
        mock_ar.log_query = AsyncMock()

        result: RAGResult = await engine.query("Explain async I/O.")

        assert result.answer == answer_text
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 20
        assert result.total_tokens == 120
        assert result.retrieved_chunks_count == 1
        assert result.latency_ms > 0.0

    async def test_empty_similarity_search_returns_fallback(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        """An empty similarity search result (no results at all) triggers refusal."""
        engine, mock_vs, mock_ar, _ = _build_engine()
        mock_vs.similarity_search = AsyncMock(return_value=[])
        mock_ar.log_query = AsyncMock()

        result: RAGResult = await engine.query("Any question")

        assert result.answer == FALLBACK_REFUSAL_MESSAGE
        assert result.citations == []
        litellm_mocks.acompletion.assert_not_called()

    async def test_llm_refusal_returns_empty_citations(self, litellm_mocks: LiteLLMMocks) -> None:
        """Citations MUST be empty when the LLM itself returns the refusal phrase.

        Scenario: chunks pass the score threshold and are budgeted, but the LLM
        answers with FALLBACK_REFUSAL_MESSAGE (e.g. the system prompt guard fires
        inside the model).  Attaching citations to a refusal implies false
        provenance and violates the anti-hallucination contract.
        """
        retrieval_result = _make_retrieval_result(
            document_id="classified-doc",
            section="Redacted",
            score=0.91,
        )
        mock_tb = MagicMock(spec=TokenBudgetManager)
        mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]

        # LLM explicitly refuses despite receiving context.
        refusal_completion = _make_mock_completion_response(
            content=FALLBACK_REFUSAL_MESSAGE,
            prompt_tokens=80,
            completion_tokens=15,
        )

        engine, mock_vs, mock_ar, _ = _build_engine(token_budget=mock_tb)
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])
        litellm_mocks.acompletion.return_value = refusal_completion
        mock_ar.log_query = AsyncMock()

        result: RAGResult = await engine.query("Tell me the classified details.")

        assert result.answer == FALLBACK_REFUSAL_MESSAGE
        assert result.citations == [], (
            "A refusal answer must never carry source citations — "
            "doing so implies false provenance."
        )
        # The LLM *was* called (chunks passed threshold); confirm that.
        litellm_mocks.acompletion.assert_awaited_once()

    async def test_token_budget_is_applied_before_completion(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        """token_budget.fit_contexts_to_budget must be called with filtered chunks."""
        retrieval_result = _make_retrieval_result(score=0.8)
        mock_tb = MagicMock(spec=TokenBudgetManager)
        mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]

        engine, mock_vs, mock_ar, _ = _build_engine(token_budget=mock_tb)
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])
        litellm_mocks.acompletion.return_value = _make_mock_completion_response()
        mock_ar.log_query = AsyncMock()

        await engine.query("What is the architecture?")

        mock_tb.fit_contexts_to_budget.assert_called_once()
        call_args = mock_tb.fit_contexts_to_budget.call_args[0][0]
        assert retrieval_result in call_args


# ---------------------------------------------------------------------------
# TestRAGEngineStream
# ---------------------------------------------------------------------------


class TestRAGEngineStream:
    """Tests for the streaming RAGEngine.stream_query() async generator."""

    async def test_stream_query_yields_fallback_when_no_chunks_pass_threshold(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        """When no chunks meet the score threshold, exactly the refusal phrase is yielded."""
        engine, mock_vs, mock_ar, _ = _build_engine()
        mock_vs.similarity_search = AsyncMock(
            return_value=[_make_retrieval_result(score=0.05)]  # well below threshold
        )

        tokens: list[str] = []
        async for token in engine.stream_query("Unanswerable?", score_threshold=0.3):
            tokens.append(token)

        assert tokens == [FALLBACK_REFUSAL_MESSAGE]
        litellm_mocks.acompletion.assert_not_called()

    async def test_stream_query_yields_tokens_sequentially(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        """Streaming tokens must be yielded in the order they arrive from the LLM."""
        retrieval_result = _make_retrieval_result(score=0.9)
        mock_tb = MagicMock(spec=TokenBudgetManager)
        mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]

        engine, mock_vs, mock_ar, _ = _build_engine(token_budget=mock_tb)
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])

        # Build an async generator that simulates the LiteLLM stream.
        expected_tokens = ["The ", "answer ", "is ", "async."]

        async def _mock_stream() -> AsyncGenerator[MagicMock, None]:
            for token_text in expected_tokens:
                yield _delta_chunk(token_text)

        litellm_mocks.acompletion.return_value = _mock_stream()

        received: list[str] = []
        async for token in engine.stream_query("Explain async.", score_threshold=0.3):
            received.append(token)

        assert received == expected_tokens
        call_kwargs = litellm_mocks.acompletion.await_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}
        assert call_kwargs["temperature"] == 0.0

    async def test_stream_query_skips_none_delta_content(self, litellm_mocks: LiteLLMMocks) -> None:
        """Chunks with None delta content (e.g. the first streaming chunk) must be skipped."""
        retrieval_result = _make_retrieval_result(score=0.9)
        mock_tb = MagicMock(spec=TokenBudgetManager)
        mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]

        engine, mock_vs, mock_ar, _ = _build_engine(token_budget=mock_tb)
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])

        async def _mock_stream_with_none() -> AsyncGenerator[MagicMock, None]:
            # First chunk has None content (role announcement in streaming format).
            yield _delta_chunk(None)
            # Subsequent chunks carry actual text.
            yield _delta_chunk("Hello")

        litellm_mocks.acompletion.return_value = _mock_stream_with_none()

        received: list[str] = []
        async for token in engine.stream_query("Hello?", score_threshold=0.3):
            received.append(token)

        assert received == ["Hello"]

    async def test_stream_query_skips_chunks_without_choices(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        """Usage-only trailing chunks (empty ``choices``) must not raise."""
        retrieval_result = _make_retrieval_result(score=0.9)
        mock_tb = MagicMock(spec=TokenBudgetManager)
        mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]

        engine, mock_vs, _, _ = _build_engine(token_budget=mock_tb)
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])

        async def _stream() -> AsyncGenerator[MagicMock, None]:
            yield _delta_chunk("Hi")
            yield _usage_chunk(10, 2, 12)

        litellm_mocks.acompletion.return_value = _stream()

        received = [t async for t in engine.stream_query("Hi?", score_threshold=0.3)]

        assert received == ["Hi"]

    async def test_stream_query_empty_similarity_result_yields_fallback(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        """An empty similarity search result triggers the fallback in stream_query too."""
        engine, mock_vs, mock_ar, _ = _build_engine()
        mock_vs.similarity_search = AsyncMock(return_value=[])

        tokens: list[str] = []
        async for token in engine.stream_query("Any question"):
            tokens.append(token)

        assert tokens == [FALLBACK_REFUSAL_MESSAGE]
        litellm_mocks.acompletion.assert_not_called()


# ---------------------------------------------------------------------------
# TestRAGEngineProviderFailures
# ---------------------------------------------------------------------------


def _provider_fault(error_class: type[Exception]) -> Exception:
    """Build a LiteLLM exception as a failing provider call would raise it."""
    return error_class(message="provider said no", model="m", llm_provider="openai")  # type: ignore[call-arg]


def _engine_with_one_chunk() -> tuple[RAGEngine, MagicMock]:
    """Engine whose retrieval succeeds, so only the LLM calls can fail."""
    retrieval_result = _make_retrieval_result(score=0.9)
    mock_tb = MagicMock(spec=TokenBudgetManager)
    mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]
    engine, mock_vs, mock_ar, _ = _build_engine(token_budget=mock_tb)
    mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])
    mock_ar.log_query = AsyncMock()
    return engine, mock_ar


class TestRAGEngineProviderFailures:
    """LiteLLM faults must surface as domain errors, never as raw provider exceptions."""

    async def test_query_embedding_auth_failure_skips_vector_search(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        litellm_mocks.aembedding.side_effect = _provider_fault(litellm.AuthenticationError)
        engine, mock_vs, _, _ = _build_engine()
        mock_vs.similarity_search = AsyncMock()

        with pytest.raises(LLMAuthenticationError, match="EMBEDDING_MODEL"):
            await engine.query("What is the SLA?")

        mock_vs.similarity_search.assert_not_called()

    async def test_query_completion_rate_limit_is_translated_and_not_audited(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        engine, mock_ar = _engine_with_one_chunk()
        litellm_mocks.acompletion.side_effect = _provider_fault(litellm.RateLimitError)

        with pytest.raises(LLMRateLimitError):
            await engine.query("What is the SLA?")

        mock_ar.log_query.assert_not_called()

    async def test_stream_failure_before_first_token_is_translated(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        engine, _ = _engine_with_one_chunk()
        litellm_mocks.acompletion.side_effect = _provider_fault(litellm.BadGatewayError)

        with pytest.raises(LLMUnavailableError):
            await anext(engine.stream_query("What is the SLA?"))

    async def test_stream_failure_between_tokens_keeps_earlier_tokens(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        engine, _ = _engine_with_one_chunk()

        async def _stream_that_dies() -> AsyncGenerator[MagicMock, None]:
            yield _delta_chunk("Partial")
            raise litellm.APIConnectionError(
                message="connection reset",
                model="m",
                llm_provider="openai",
                request=httpx.Request("POST", "https://provider.test/v1"),
            )

        litellm_mocks.acompletion.return_value = _stream_that_dies()

        received: list[str] = []
        with pytest.raises(LLMUnavailableError):
            async for token in engine.stream_query("What is the SLA?"):
                received.append(token)

        assert received == ["Partial"]


# ---------------------------------------------------------------------------
# Streaming telemetry, prompt provenance and retrieval defaults
# ---------------------------------------------------------------------------


def _streaming_engine(*, estimated_tokens: int = 7) -> tuple[RAGEngine, MagicMock, MagicMock]:
    """Engine with one qualifying chunk; returns (engine, audit_repo, token_budget)."""
    retrieval_result = _make_retrieval_result(score=0.9)
    mock_tb = MagicMock(spec=TokenBudgetManager)
    mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]
    mock_tb.count_tokens.return_value = estimated_tokens
    engine, mock_vs, mock_ar, _ = _build_engine(token_budget=mock_tb)
    mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])
    return engine, mock_ar, mock_tb


class TestStreamTelemetry:
    """``stream_query`` must audit every completed stream, like ``query`` does."""

    async def test_logs_provider_usage_after_last_token(self, litellm_mocks: LiteLLMMocks) -> None:
        engine, mock_ar, _ = _streaming_engine()

        async def _stream() -> AsyncGenerator[MagicMock, None]:
            yield _delta_chunk("Four ")
            yield _delta_chunk("nines.")
            yield _usage_chunk(120, 5, 125)

        litellm_mocks.acompletion.return_value = _stream()

        received = [t async for t in engine.stream_query("What is the SLA?")]

        assert received == ["Four ", "nines."]
        mock_ar.log_query.assert_awaited_once()
        record: TelemetryRecord = mock_ar.log_query.call_args[0][0]
        assert record.query_text == "What is the SLA?"
        assert record.response_text == "Four nines."
        assert (record.prompt_tokens, record.completion_tokens, record.total_tokens) == (
            120,
            5,
            125,
        )
        assert record.latency_ms > 0
        assert record.model_name == "gpt-4o-mini"

    async def test_estimates_tokens_when_provider_omits_usage(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        engine, mock_ar, _ = _streaming_engine(estimated_tokens=7)

        async def _stream() -> AsyncGenerator[MagicMock, None]:
            yield _delta_chunk("Answer.")

        litellm_mocks.acompletion.return_value = _stream()

        _ = [t async for t in engine.stream_query("Q?")]

        record: TelemetryRecord = mock_ar.log_query.call_args[0][0]
        # Two prompt messages (system + user) plus the completion, 7 tokens each.
        assert record.prompt_tokens == 14
        assert record.completion_tokens == 7
        assert record.total_tokens == 21

    async def test_tolerates_null_usage_fields(self, litellm_mocks: LiteLLMMocks) -> None:
        engine, mock_ar, _ = _streaming_engine(estimated_tokens=3)

        async def _stream() -> AsyncGenerator[MagicMock, None]:
            yield _delta_chunk("Answer.")
            yield _usage_chunk(None, None, None)  # type: ignore[arg-type]

        litellm_mocks.acompletion.return_value = _stream()

        _ = [t async for t in engine.stream_query("Q?")]

        record: TelemetryRecord = mock_ar.log_query.call_args[0][0]
        assert record.total_tokens == record.prompt_tokens + record.completion_tokens > 0

    async def test_refusal_fast_path_logs_zero_token_record(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        engine, mock_vs, mock_ar, _ = _build_engine()
        mock_vs.similarity_search = AsyncMock(return_value=[_make_retrieval_result(score=0.01)])

        received = [t async for t in engine.stream_query("Unanswerable?")]

        assert received == [FALLBACK_REFUSAL_MESSAGE]
        record: TelemetryRecord = mock_ar.log_query.call_args[0][0]
        assert record.response_text == FALLBACK_REFUSAL_MESSAGE
        assert record.total_tokens == 0

    async def test_storage_failure_does_not_break_a_delivered_stream(
        self, litellm_mocks: LiteLLMMocks, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, mock_ar, _ = _streaming_engine()
        mock_ar.log_query = AsyncMock(side_effect=StorageError("db down"))

        async def _stream() -> AsyncGenerator[MagicMock, None]:
            yield _delta_chunk("Done.")

        litellm_mocks.acompletion.return_value = _stream()

        received = [t async for t in engine.stream_query("Q?")]

        assert received == ["Done."]
        assert "Telemetry for a streamed response was not persisted" in caplog.text


class TestPromptProvenanceAndDefaults:
    """Chunks reach the model with real provenance; defaults favour recall."""

    async def test_prompt_labels_each_chunk_with_source_and_section(
        self, litellm_mocks: LiteLLMMocks
    ) -> None:
        engine, _, _ = _streaming_engine()

        async def _stream() -> AsyncGenerator[MagicMock, None]:
            yield _delta_chunk("ok")

        litellm_mocks.acompletion.return_value = _stream()

        _ = [t async for t in engine.stream_query("Q?")]

        user_message = litellm_mocks.acompletion.await_args.kwargs["messages"][1]["content"]
        assert (
            "[Source: tech-spec, Section: Architecture]\nThe system uses async I/O" in user_message
        )

    def test_defaults_are_recall_oriented(self) -> None:
        assert DEFAULT_TOP_K == 6
        assert DEFAULT_SCORE_THRESHOLD == 0.2
