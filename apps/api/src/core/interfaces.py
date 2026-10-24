"""Abstract interface definitions for storage adapters.

Separating interfaces (Dependency Inversion Principle) from concrete
implementations ensures the domain and ingestion layers remain fully
decoupled from storage drivers (ChromaDB, SQLite).  All concrete adapters
must implement these ABCs, and all consumers must depend only on these types,
enabling straightforward substitution of stubs in unit tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.core.models import DocumentChunk, RetrievalResult, TelemetryRecord


class VectorStoreInterface(ABC):
    """Contract for vector database adapters.

    Implementations are responsible for persisting document chunk embeddings
    and performing approximate nearest-neighbour retrieval.  All I/O must be
    async; blocking driver calls must be wrapped by the implementation (e.g.
    via ``asyncio.to_thread``).
    """

    @abstractmethod
    async def add_documents(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Persist a batch of document chunks with their pre-computed embeddings.

        Args:
            chunks: Ordered sequence of ``DocumentChunk`` instances to index.
            embeddings: Dense embedding vectors, one per chunk, in the same
                order as ``chunks``.  When ``None``, the implementation may
                delegate embedding generation to a configured embedding function.
        """
        ...

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> Sequence[RetrievalResult]:
        """Retrieve the top-k most semantically similar chunks.

        Args:
            query_embedding: Dense vector representation of the user query,
                produced by the same embedding model used during ingestion.
            top_k: Maximum number of results to return.

        Returns:
            Sequence of ``RetrievalResult`` sorted by descending similarity score.
        """
        ...


class AuditRepositoryInterface(ABC):
    """Contract for telemetry and audit log persistence adapters.

    Implementations store a ``TelemetryRecord`` for every completed RAG
    request, enabling cost tracking, latency analysis, and compliance audits.
    """

    @abstractmethod
    async def log_query(self, record: TelemetryRecord) -> None:
        """Persist a single telemetry record to the audit store.

        Args:
            record: Fully-populated ``TelemetryRecord`` for the completed query.
        """
        ...

    @abstractmethod
    async def get_recent_logs(self, limit: int = 50) -> Sequence[TelemetryRecord]:
        """Retrieve the most recent telemetry records ordered by creation time.

        Args:
            limit: Maximum number of records to return.

        Returns:
            Sequence of ``TelemetryRecord`` ordered descending by ``created_at``.
        """
        ...
