"""RAG orchestration engine: retrieval, token budgeting, LLM generation, and audit logging.

``RAGEngine`` is the domain-layer orchestrator that wires together every
upstream capability (vector search, token budgeting, prompt construction,
LLM inference, telemetry) into a single coherent query pipeline.

Design constraints respected here:
- No direct imports of ChromaDB, aiosqlite, or any storage driver.
- All dependencies are injected via the constructor (Dependency Inversion).
- The event loop is never blocked: embedding and completion calls are
  already async; ChromaDB blocking calls are hidden behind VectorStoreInterface.
- Token budgeting is enforced before every LLM call — no unbounded context
  is ever forwarded to the LLM provider.
- Anti-hallucination refusal fires in the fast path (before the LLM call)
  when retrieved context is empty, and is enforced by prompt design otherwise.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator, Sequence
from datetime import UTC, datetime

import litellm

from src.config.settings import Settings
from src.core.interfaces import AuditRepositoryInterface, VectorStoreInterface
from src.core.llm import embed_texts, provider_credentials, translate_litellm_errors
from src.core.models import Citation, RAGResult, RetrievalResult, TelemetryRecord
from src.core.rag.prompts import FALLBACK_REFUSAL_MESSAGE, build_rag_prompt
from src.core.rag.token_counter import TokenBudgetManager

logger = logging.getLogger(__name__)


class RAGEngine:
    """Orchestrates the full retrieval-augmented generation pipeline.

    Responsibilities:
    - Embed the user query using the configured embedding model (routed via LiteLLM).
    - Retrieve semantically similar chunks from the vector store.
    - Filter low-confidence chunks by score threshold.
    - Apply token budget constraints to the surviving candidate set.
    - Build a grounded, citation-enforcing prompt via ``build_rag_prompt``.
    - Call the chat model through ``litellm.acompletion`` (blocking or streaming).
    - Persist a ``TelemetryRecord`` to the audit repository.
    - Return a fully structured ``RAGResult`` with citations and usage stats.

    All public methods are ``async``.  The engine itself performs no blocking
    I/O; blocking concerns are delegated to injected interface implementations.
    """

    def __init__(
        self,
        vector_store: VectorStoreInterface,
        audit_repo: AuditRepositoryInterface,
        token_budget: TokenBudgetManager,
        settings: Settings,
    ) -> None:
        """Initialise the engine with its injected dependencies.

        Args:
            vector_store: Abstract vector store adapter for similarity search.
            audit_repo: Abstract audit repository for telemetry persistence.
            token_budget: Token budget manager that measures and bounds context.
            settings: Application configuration (model names, etc.).
        """
        self._vector_store = vector_store
        self._audit_repo = audit_repo
        self._token_budget = token_budget
        self._settings = settings

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _embed_query(self, query_text: str) -> list[float]:
        """Produce a dense embedding vector for the user query.

        Args:
            query_text: Raw user query string.

        Returns:
            Dense float vector from the configured embedding model.
        """
        embeddings = await embed_texts([query_text], self._settings)
        return embeddings[0]

    async def _retrieve_and_filter(
        self,
        query_text: str,
        top_k: int,
        score_threshold: float,
    ) -> list[RetrievalResult]:
        """Embed query, retrieve top-k chunks, and filter by similarity threshold.

        Args:
            query_text: Raw user query.
            top_k: Maximum number of candidates to retrieve from the vector store.
            score_threshold: Minimum acceptable similarity score; chunks scoring
                below this value are discarded before token budgeting.

        Returns:
            Filtered list of ``RetrievalResult`` instances with score ≥ threshold.
        """
        query_embedding = await self._embed_query(query_text)
        raw_results: Sequence[RetrievalResult] = await self._vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
        return [r for r in raw_results if r.score >= score_threshold]

    def _extract_citations(self, budgeted_chunks: list[RetrievalResult]) -> list[Citation]:
        """Derive structured Citation objects from chunk metadata.

        The ``document_id`` and ``section`` fields in ``chunk.metadata`` are
        the authoritative provenance fields set by the ingestion pipeline at
        index time.  Falling back to ``chunk.document_id`` ensures the source
        field is always non-empty even if metadata is sparse.

        Args:
            budgeted_chunks: Token-budgeted retrieval results used in the prompt.

        Returns:
            One ``Citation`` per chunk, preserving input order.
        """
        return [
            Citation(
                source=str(result.chunk.metadata.get("document_id", result.chunk.document_id)),
                section=str(result.chunk.metadata.get("section", "")),
                chunk_id=result.chunk.chunk_id,
            )
            for result in budgeted_chunks
        ]

    async def _log_telemetry(
        self,
        query_text: str,
        response_text: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        """Construct and persist a TelemetryRecord to the audit repository.

        Args:
            query_text: The raw user query.
            response_text: The generated answer or fallback refusal phrase.
            latency_ms: End-to-end wall-clock duration in milliseconds.
            prompt_tokens: Tokens consumed by the prompt.
            completion_tokens: Tokens generated in the completion.
            total_tokens: Sum of prompt and completion tokens.
        """
        record = TelemetryRecord(
            request_id=str(uuid.uuid4()),
            query_text=query_text,
            response_text=response_text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model_name=self._settings.openai_model,
            created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        await self._audit_repo.log_query(record)

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    async def query(
        self,
        query_text: str,
        top_k: int = 4,
        score_threshold: float = 0.3,
    ) -> RAGResult:
        """Execute a non-streaming RAG query and return a structured ``RAGResult``.

        Pipeline stages:
        1. Embed the query and retrieve top-k candidates from the vector store.
        2. Filter candidates by ``score_threshold``; fast-path fallback if none survive.
        3. Apply token budgeting to surviving candidates.
        4. Build the grounded prompt and call chat completions.
        5. Extract citations, log telemetry, and return ``RAGResult``.

        Args:
            query_text: The raw user query.
            top_k: Maximum number of candidates to retrieve from the vector store.
            score_threshold: Minimum similarity score for a chunk to be considered.

        Returns:
            A fully populated ``RAGResult`` carrying the grounded answer,
            source citations, latency, and token usage figures.

        Raises:
            LLMProviderError: If the embedding or chat provider fails.
            StorageError: If the vector store query fails.
        """
        start_time = time.perf_counter()

        filtered_chunks = await self._retrieve_and_filter(query_text, top_k, score_threshold)

        if not filtered_chunks:
            # Fast-path refusal: no evidence context survives the threshold.
            # Log telemetry with zero tokens and return the deterministic refusal phrase.
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "RAG query produced no qualifying chunks — returning fallback refusal. "
                "query=%r score_threshold=%.2f",
                query_text,
                score_threshold,
            )
            await self._log_telemetry(
                query_text=query_text,
                response_text=FALLBACK_REFUSAL_MESSAGE,
                latency_ms=latency_ms,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )
            return RAGResult(
                query=query_text,
                answer=FALLBACK_REFUSAL_MESSAGE,
                citations=[],
                latency_ms=latency_ms,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                retrieved_chunks_count=0,
            )

        budgeted_chunks = self._token_budget.fit_contexts_to_budget(filtered_chunks)
        contexts = [chunk.chunk.content for chunk in budgeted_chunks]
        messages = build_rag_prompt(query=query_text, retrieved_contexts=contexts)

        with translate_litellm_errors(self._settings.openai_model, "chat completion"):
            completion = await litellm.acompletion(
                model=self._settings.openai_model,
                messages=messages,
                temperature=0.0,
                **provider_credentials(self._settings.openai_model, self._settings),
            )

        answer: str = completion.choices[0].message.content or FALLBACK_REFUSAL_MESSAGE
        usage = completion.usage
        prompt_tokens: int = usage.prompt_tokens if usage else 0
        completion_tokens: int = usage.completion_tokens if usage else 0
        total_tokens: int = usage.total_tokens if usage else 0

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # A refusal answer must never carry source attributions — the LLM was
        # unable (or instructed) to ground its response in the retrieved context,
        # so attaching citations would imply false provenance.
        citations = (
            [] if FALLBACK_REFUSAL_MESSAGE in answer else self._extract_citations(budgeted_chunks)
        )

        await self._log_telemetry(
            query_text=query_text,
            response_text=answer,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        logger.info(
            "RAG query completed. chunks=%d prompt_tokens=%d completion_tokens=%d latency_ms=%.1f",
            len(budgeted_chunks),
            prompt_tokens,
            completion_tokens,
            latency_ms,
        )

        return RAGResult(
            query=query_text,
            answer=answer,
            citations=citations,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            retrieved_chunks_count=len(budgeted_chunks),
        )

    async def stream_query(
        self,
        query_text: str,
        top_k: int = 4,
        score_threshold: float = 0.3,
    ) -> AsyncGenerator[str, None]:
        """Execute a streaming RAG query, yielding token deltas as they arrive.

        The retrieval, filtering, and token budgeting stages are identical to
        ``query()``.  When no chunks pass the score threshold, the deterministic
        fallback refusal phrase is yielded as a single string and the generator
        terminates immediately — no LLM call is made.

        Telemetry is not logged for streaming queries because provider
        streams do not return a usage object by default.
        Streaming usage accounting is deferred to a future sprint.

        Args:
            query_text: The raw user query.
            top_k: Maximum number of candidates to retrieve from the vector store.
            score_threshold: Minimum similarity score for a chunk to be considered.

        Yields:
            Individual token delta strings from the LLM completion stream.
            Yields exactly ``FALLBACK_REFUSAL_MESSAGE`` and returns immediately
            if no chunks survive threshold filtering.

        Raises:
            LLMProviderError: If the embedding or chat provider fails, before the
                first token or between tokens.
            StorageError: If the vector store query fails.
        """
        filtered_chunks = await self._retrieve_and_filter(query_text, top_k, score_threshold)

        if not filtered_chunks:
            logger.info(
                "Stream RAG query produced no qualifying chunks — yielding fallback refusal. "
                "query=%r score_threshold=%.2f",
                query_text,
                score_threshold,
            )
            yield FALLBACK_REFUSAL_MESSAGE
            return

        budgeted_chunks = self._token_budget.fit_contexts_to_budget(filtered_chunks)
        contexts = [chunk.chunk.content for chunk in budgeted_chunks]
        messages = build_rag_prompt(query=query_text, retrieved_contexts=contexts)

        # The guard spans the consuming loop as well: a provider can fail after the
        # stream opened, and that fault must surface as a domain error too.
        with translate_litellm_errors(self._settings.openai_model, "chat completion"):
            stream = await litellm.acompletion(
                model=self._settings.openai_model,
                messages=messages,
                temperature=0.0,
                stream=True,
                **provider_credentials(self._settings.openai_model, self._settings),
            )

            async for chunk in stream:
                # Providers may emit trailing chunks without choices (e.g. usage-only).
                if not chunk.choices:
                    continue
                delta_content: str | None = chunk.choices[0].delta.content
                if delta_content:
                    yield delta_content
