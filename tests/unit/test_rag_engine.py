"""Unit tests for RAGEngine.

All OpenAI API calls and storage operations are isolated via
``unittest.mock.AsyncMock`` and ``unittest.mock.MagicMock``.  No live
network connections, ChromaDB instances, or SQLite databases are required.

Test groups:
- TestRAGEngineQuery  — non-streaming query() method.
- TestRAGEngineStream — streaming stream_query() async generator.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from src.config.settings import Settings
from src.core.models import Citation, DocumentChunk, RAGResult, RetrievalResult, TelemetryRecord
from src.core.rag.engine import RAGEngine
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
        sqlite_database_path="/tmp/test_telemetry.db",
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
    """Build a mock OpenAI embeddings response."""
    response = MagicMock()
    response.data[0].embedding = embedding or [0.1, 0.2, 0.3]
    return response


def _make_mock_completion_response(
    content: str = "The system uses async I/O. [Source: tech-spec, Section: Architecture]",
    prompt_tokens: int = 120,
    completion_tokens: int = 30,
) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    response = MagicMock()
    response.choices[0].message.content = content
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens
    return response


def _build_engine(
    *,
    vector_store: MagicMock | None = None,
    audit_repo: MagicMock | None = None,
    openai_client: MagicMock | None = None,
    token_budget: MagicMock | None = None,
    settings: Settings | None = None,
) -> tuple[RAGEngine, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Construct a RAGEngine with all dependencies mocked.

    Returns:
        Tuple of (engine, mock_vector_store, mock_audit_repo,
                  mock_openai_client, mock_token_budget).
    """
    mock_vs = vector_store or MagicMock()
    mock_ar = audit_repo or MagicMock()
    mock_oc = openai_client or MagicMock()
    mock_tb = token_budget or MagicMock(spec=TokenBudgetManager)
    cfg = settings or _make_settings()

    engine = RAGEngine(
        vector_store=mock_vs,
        audit_repo=mock_ar,
        openai_client=mock_oc,
        token_budget=mock_tb,
        settings=cfg,
    )
    return engine, mock_vs, mock_ar, mock_oc, mock_tb


# ---------------------------------------------------------------------------
# TestRAGEngineQuery
# ---------------------------------------------------------------------------


class TestRAGEngineQuery:
    """Tests for the non-streaming RAGEngine.query() method."""

    async def test_below_threshold_returns_fallback_without_completion_call(self) -> None:
        """When no chunk meets the score threshold, chat completions must NOT be called."""
        engine, mock_vs, mock_ar, mock_oc, _ = _build_engine()

        # Embedding succeeds but similarity search returns a below-threshold result.
        mock_oc.embeddings.create = AsyncMock(return_value=_make_mock_embedding_response())
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
        mock_oc.chat.completions.create.assert_not_called()

    async def test_below_threshold_logs_telemetry_with_zero_tokens(self) -> None:
        """Fallback path must still persist a TelemetryRecord with zero token counts."""
        engine, mock_vs, mock_ar, mock_oc, _ = _build_engine()

        mock_oc.embeddings.create = AsyncMock(return_value=_make_mock_embedding_response())
        mock_vs.similarity_search = AsyncMock(return_value=[])
        mock_ar.log_query = AsyncMock()

        await engine.query("Unanswerable question", score_threshold=0.3)

        mock_ar.log_query.assert_awaited_once()
        logged_record: TelemetryRecord = mock_ar.log_query.call_args[0][0]
        assert logged_record.prompt_tokens == 0
        assert logged_record.completion_tokens == 0
        assert logged_record.total_tokens == 0
        assert logged_record.response_text == FALLBACK_REFUSAL_MESSAGE

    async def test_successful_retrieval_invokes_chat_completion(self) -> None:
        """When chunks pass the threshold, chat completions must be called exactly once."""
        retrieval_result = _make_retrieval_result(score=0.9)
        mock_tb = MagicMock(spec=TokenBudgetManager)
        mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]

        engine, mock_vs, mock_ar, mock_oc, _ = _build_engine(token_budget=mock_tb)
        mock_oc.embeddings.create = AsyncMock(return_value=_make_mock_embedding_response())
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])
        mock_oc.chat.completions.create = AsyncMock(return_value=_make_mock_completion_response())
        mock_ar.log_query = AsyncMock()

        await engine.query("How does async I/O work?", score_threshold=0.3)

        mock_oc.chat.completions.create.assert_awaited_once()

    async def test_successful_retrieval_returns_correct_citations(self) -> None:
        """Citations must be extracted from the metadata of budgeted chunks."""
        retrieval_result = _make_retrieval_result(
            document_id="api-guide",
            section="Endpoints",
            score=0.92,
        )
        mock_tb = MagicMock(spec=TokenBudgetManager)
        mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]

        engine, mock_vs, mock_ar, mock_oc, _ = _build_engine(token_budget=mock_tb)
        mock_oc.embeddings.create = AsyncMock(return_value=_make_mock_embedding_response())
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])
        mock_oc.chat.completions.create = AsyncMock(return_value=_make_mock_completion_response())
        mock_ar.log_query = AsyncMock()

        result: RAGResult = await engine.query("List all endpoints.")

        assert len(result.citations) == 1
        citation: Citation = result.citations[0]
        assert citation.source == "api-guide"
        assert citation.section == "Endpoints"
        assert citation.chunk_id == "api-guide#c0000"

    async def test_successful_retrieval_logs_audit_record(self) -> None:
        """A TelemetryRecord must be persisted after a successful completion."""
        retrieval_result = _make_retrieval_result(score=0.88)
        mock_tb = MagicMock(spec=TokenBudgetManager)
        mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]
        completion = _make_mock_completion_response(prompt_tokens=200, completion_tokens=50)

        engine, mock_vs, mock_ar, mock_oc, _ = _build_engine(token_budget=mock_tb)
        mock_oc.embeddings.create = AsyncMock(return_value=_make_mock_embedding_response())
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])
        mock_oc.chat.completions.create = AsyncMock(return_value=completion)
        mock_ar.log_query = AsyncMock()

        await engine.query("Explain the chunking strategy.")

        mock_ar.log_query.assert_awaited_once()
        logged_record: TelemetryRecord = mock_ar.log_query.call_args[0][0]
        assert logged_record.prompt_tokens == 200
        assert logged_record.completion_tokens == 50
        assert logged_record.total_tokens == 250
        assert logged_record.model_name == "gpt-4o-mini"

    async def test_successful_retrieval_populates_rag_result_fields(self) -> None:
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

        engine, mock_vs, mock_ar, mock_oc, _ = _build_engine(token_budget=mock_tb)
        mock_oc.embeddings.create = AsyncMock(return_value=_make_mock_embedding_response())
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])
        mock_oc.chat.completions.create = AsyncMock(return_value=completion)
        mock_ar.log_query = AsyncMock()

        result: RAGResult = await engine.query("Explain async I/O.")

        assert result.answer == answer_text
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 20
        assert result.total_tokens == 120
        assert result.retrieved_chunks_count == 1
        assert result.latency_ms > 0.0

    async def test_empty_similarity_search_returns_fallback(self) -> None:
        """An empty similarity search result (no results at all) triggers refusal."""
        engine, mock_vs, mock_ar, mock_oc, _ = _build_engine()
        mock_oc.embeddings.create = AsyncMock(return_value=_make_mock_embedding_response())
        mock_vs.similarity_search = AsyncMock(return_value=[])
        mock_ar.log_query = AsyncMock()

        result: RAGResult = await engine.query("Any question")

        assert result.answer == FALLBACK_REFUSAL_MESSAGE
        assert result.citations == []
        mock_oc.chat.completions.create.assert_not_called()

    async def test_llm_refusal_returns_empty_citations(self) -> None:
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

        engine, mock_vs, mock_ar, mock_oc, _ = _build_engine(token_budget=mock_tb)
        mock_oc.embeddings.create = AsyncMock(return_value=_make_mock_embedding_response())
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])
        mock_oc.chat.completions.create = AsyncMock(return_value=refusal_completion)
        mock_ar.log_query = AsyncMock()

        result: RAGResult = await engine.query("Tell me the classified details.")

        assert result.answer == FALLBACK_REFUSAL_MESSAGE
        assert result.citations == [], (
            "A refusal answer must never carry source citations — "
            "doing so implies false provenance."
        )
        # The LLM *was* called (chunks passed threshold); confirm that.
        mock_oc.chat.completions.create.assert_awaited_once()

    async def test_token_budget_is_applied_before_completion(self) -> None:
        """token_budget.fit_contexts_to_budget must be called with filtered chunks."""
        retrieval_result = _make_retrieval_result(score=0.8)
        mock_tb = MagicMock(spec=TokenBudgetManager)
        mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]

        engine, mock_vs, mock_ar, mock_oc, _ = _build_engine(token_budget=mock_tb)
        mock_oc.embeddings.create = AsyncMock(return_value=_make_mock_embedding_response())
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])
        mock_oc.chat.completions.create = AsyncMock(return_value=_make_mock_completion_response())
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

    async def test_stream_query_yields_fallback_when_no_chunks_pass_threshold(self) -> None:
        """When no chunks meet the score threshold, exactly the refusal phrase is yielded."""
        engine, mock_vs, mock_ar, mock_oc, _ = _build_engine()
        mock_oc.embeddings.create = AsyncMock(return_value=_make_mock_embedding_response())
        mock_vs.similarity_search = AsyncMock(
            return_value=[_make_retrieval_result(score=0.05)]  # well below threshold
        )

        tokens: list[str] = []
        async for token in engine.stream_query("Unanswerable?", score_threshold=0.3):
            tokens.append(token)

        assert tokens == [FALLBACK_REFUSAL_MESSAGE]
        mock_oc.chat.completions.create.assert_not_called()

    async def test_stream_query_yields_tokens_sequentially(self) -> None:
        """Streaming tokens must be yielded in the order they arrive from the LLM."""
        retrieval_result = _make_retrieval_result(score=0.9)
        mock_tb = MagicMock(spec=TokenBudgetManager)
        mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]

        engine, mock_vs, mock_ar, mock_oc, _ = _build_engine(token_budget=mock_tb)
        mock_oc.embeddings.create = AsyncMock(return_value=_make_mock_embedding_response())
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])

        # Build an async generator that simulates the OpenAI stream.
        expected_tokens = ["The ", "answer ", "is ", "async."]

        async def _mock_stream() -> AsyncGenerator[MagicMock, None]:
            for token_text in expected_tokens:
                chunk = MagicMock()
                chunk.choices[0].delta.content = token_text
                yield chunk

        mock_oc.chat.completions.create = AsyncMock(return_value=_mock_stream())

        received: list[str] = []
        async for token in engine.stream_query("Explain async.", score_threshold=0.3):
            received.append(token)

        assert received == expected_tokens

    async def test_stream_query_skips_none_delta_content(self) -> None:
        """Chunks with None delta content (e.g. the first streaming chunk) must be skipped."""
        retrieval_result = _make_retrieval_result(score=0.9)
        mock_tb = MagicMock(spec=TokenBudgetManager)
        mock_tb.fit_contexts_to_budget.return_value = [retrieval_result]

        engine, mock_vs, mock_ar, mock_oc, _ = _build_engine(token_budget=mock_tb)
        mock_oc.embeddings.create = AsyncMock(return_value=_make_mock_embedding_response())
        mock_vs.similarity_search = AsyncMock(return_value=[retrieval_result])

        async def _mock_stream_with_none() -> AsyncGenerator[MagicMock, None]:
            # First chunk has None content (role announcement in streaming format).
            none_chunk = MagicMock()
            none_chunk.choices[0].delta.content = None
            yield none_chunk
            # Subsequent chunks carry actual text.
            real_chunk = MagicMock()
            real_chunk.choices[0].delta.content = "Hello"
            yield real_chunk

        mock_oc.chat.completions.create = AsyncMock(return_value=_mock_stream_with_none())

        received: list[str] = []
        async for token in engine.stream_query("Hello?", score_threshold=0.3):
            received.append(token)

        assert received == ["Hello"]

    async def test_stream_query_empty_similarity_result_yields_fallback(self) -> None:
        """An empty similarity search result triggers the fallback in stream_query too."""
        engine, mock_vs, mock_ar, mock_oc, _ = _build_engine()
        mock_oc.embeddings.create = AsyncMock(return_value=_make_mock_embedding_response())
        mock_vs.similarity_search = AsyncMock(return_value=[])

        tokens: list[str] = []
        async for token in engine.stream_query("Any question"):
            tokens.append(token)

        assert tokens == [FALLBACK_REFUSAL_MESSAGE]
        mock_oc.chat.completions.create.assert_not_called()
