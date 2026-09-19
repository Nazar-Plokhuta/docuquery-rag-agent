"""Immutable domain models for the DocuQuery RAG Agent.

These Pydantic v2 frozen models serve as the shared data contract between
the ingestion, storage, and presentation layers.  Freezing prevents accidental
mutation across async call boundaries and enables safe use as dict keys / set
members.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DocumentChunk(BaseModel):
    """A bounded text segment derived from a source document.

    Attributes:
        chunk_id: Deterministic identifier in the form ``{document_id}#c{index:04d}``.
        document_id: Identifier of the parent document (typically the file stem).
        content: Raw text content of this chunk after sliding-window extraction.
        metadata: Arbitrary scalar key-value pairs for source attribution —
            filename, section header, chunk index, token count, etc.
        token_count: Number of tokens in ``content`` as measured by tiktoken;
            stored explicitly so consumers avoid re-tokenising on every access.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, str | int]
    token_count: int


class RetrievalResult(BaseModel):
    """A scored document chunk returned from a vector similarity search.

    Attributes:
        chunk: The retrieved ``DocumentChunk`` containing text and provenance.
        score: Similarity score in [0, 1]; higher values indicate greater
            semantic relevance to the query.
    """

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    score: float


class TelemetryRecord(BaseModel):
    """Audit log entry capturing the full lifecycle of a single RAG query.

    Every completed RAG request produces one ``TelemetryRecord``, which is
    persisted to the SQLite audit store for observability and cost attribution.

    Attributes:
        request_id: UUID-based trace identifier for cross-service correlation.
        query_text: The raw user query submitted to the RAG pipeline.
        response_text: The grounded LLM response returned to the caller.
        latency_ms: End-to-end wall-clock duration in milliseconds.
        prompt_tokens: Tokens consumed by the system and user prompt.
        completion_tokens: Tokens generated in the LLM completion.
        total_tokens: Sum of prompt and completion token counts.
        model_name: LLM identifier used to generate the response (e.g. ``gpt-4o-mini``).
        created_at: ISO-8601 UTC timestamp of the query (e.g. ``2026-09-19T09:00:00Z``).
    """

    model_config = ConfigDict(frozen=True)

    request_id: str
    query_text: str
    response_text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_name: str
    created_at: str
