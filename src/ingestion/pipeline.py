"""End-to-end document ingestion pipeline.

Orchestrates the full ingestion flow:
    read file → chunk text → batch embed via OpenAI → index in vector store.

The pipeline is stateless beyond its constructor-injected dependencies, making
it trivially testable in isolation with stub implementations of each interface.
"""

from __future__ import annotations

import logging
from pathlib import Path

from openai import AsyncOpenAI

from src.config.settings import Settings
from src.core.interfaces import VectorStoreInterface
from src.ingestion.chunker import TokenSlidingWindowChunker

logger = logging.getLogger(__name__)

# OpenAI embeddings endpoint accepts at most 2048 inputs per call; we cap at 100
# to stay well within rate-limit budgets and keep individual request latencies low.
_EMBEDDING_BATCH_SIZE: int = 100


class IngestionPipeline:
    """Orchestrates read → chunk → embed → index for a single document.

    Args:
        chunker: Configured ``TokenSlidingWindowChunker`` instance.
        vector_store: Concrete vector store adapter implementing
            ``VectorStoreInterface``.
        openai_client: Async OpenAI client used for embedding generation.
        settings: Application settings (provides ``embedding_model`` identifier).
    """

    def __init__(
        self,
        chunker: TokenSlidingWindowChunker,
        vector_store: VectorStoreInterface,
        openai_client: AsyncOpenAI,
        settings: Settings,
    ) -> None:
        self._chunker = chunker
        self._vector_store = vector_store
        self._openai = openai_client
        self._settings = settings

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for up to ``_EMBEDDING_BATCH_SIZE`` text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of dense embedding vectors in the same order as ``texts``.
        """
        response = await self._openai.embeddings.create(
            model=self._settings.embedding_model,
            input=texts,
        )
        # The API guarantees items are returned in the same order as the input.
        return [item.embedding for item in response.data]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_document(self, file_path: Path) -> int:
        """Ingest a single document end-to-end.

        Processing steps:
        1. Read the document as UTF-8 text.
        2. Segment into overlapping token-bounded chunks.
        3. Call the OpenAI Embeddings API in batches of at most
           ``_EMBEDDING_BATCH_SIZE`` chunks.
        4. Persist each batch of (chunk, embedding) pairs via the vector store.

        Args:
            file_path: Path to the source document (plain text or Markdown).

        Returns:
            Total number of ``DocumentChunk`` objects successfully ingested.
            Returns 0 for empty documents without raising.
        """
        document_id: str = file_path.stem
        text: str = file_path.read_text(encoding="utf-8")

        chunks = self._chunker.chunk_document(document_id, text)
        if not chunks:
            logger.warning(
                "Document %r produced zero chunks — nothing ingested.", file_path
            )
            return 0

        total_batches = (len(chunks) + _EMBEDDING_BATCH_SIZE - 1) // _EMBEDDING_BATCH_SIZE
        logger.info(
            "Ingesting %r: %d chunks across %d embedding batch(es).",
            document_id,
            len(chunks),
            total_batches,
        )

        for batch_start in range(0, len(chunks), _EMBEDDING_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + _EMBEDDING_BATCH_SIZE]
            texts = [c.content for c in batch]
            embeddings = await self._embed_batch(texts)

            await self._vector_store.add_documents(batch, embeddings)
            logger.debug(
                "Batch %d/%d indexed (%d chunks).",
                batch_start // _EMBEDDING_BATCH_SIZE + 1,
                total_batches,
                len(batch),
            )

        logger.info(
            "Document %r ingestion complete: %d chunks indexed from %s.",
            document_id,
            len(chunks),
            file_path,
        )
        return len(chunks)
