"""Request and response schemas for the /query resource.

``QueryRequest`` enforces strict input validation at the presentation layer so
that the RAG engine never receives malformed payloads.  ``QueryResponse``
projects the domain ``RAGResult`` into a stable, versioned API contract that
decouples consumers from internal model changes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.core.models import Citation


class QueryRequest(BaseModel):
    """Validated inbound payload for a RAG query.

    Attributes:
        query: The user question. Must be a non-empty string.
        top_k: Maximum number of candidate chunks to retrieve from the vector
            store before score filtering. Bounded to [1, 20] to prevent
            runaway retrieval costs and context window overflow.
        score_threshold: Minimum cosine similarity score a retrieved chunk must
            achieve to be injected into the LLM prompt. Chunks below this
            threshold are treated as irrelevant and discarded.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Non-empty user question string.")
    top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum candidate chunks to retrieve (1–20).",
    )
    score_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score for chunk inclusion (0.0–1.0).",
    )


class QueryResponse(BaseModel):
    """Structured response returned by the non-streaming RAG query endpoint.

    Attributes:
        query: The original user question, echoed for client-side correlation.
        answer: Grounded LLM response text, or the deterministic fallback
            refusal phrase when retrieved context is insufficient.
        citations: Source attributions for every chunk injected into the prompt.
        latency_ms: End-to-end wall-clock duration in milliseconds.
        total_tokens: Combined prompt and completion token count for cost attribution.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    answer: str
    citations: list[Citation]
    latency_ms: float
    total_tokens: int
