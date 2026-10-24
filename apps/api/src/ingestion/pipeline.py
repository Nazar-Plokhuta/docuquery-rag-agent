"""End-to-end document ingestion pipeline.

Orchestrates the full ingestion flow:
    read file → chunk text → batch embed via LiteLLM → index in vector store.

The pipeline is stateless beyond its constructor-injected dependencies, making
it trivially testable in isolation with stub implementations of each interface.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from src.config.settings import Settings
from src.core.interfaces import VectorStoreInterface
from src.core.llm import embed_texts
from src.ingestion.chunker import TokenSlidingWindowChunker
from src.ingestion.loaders.factory import DocumentLoaderFactory

logger = logging.getLogger(__name__)

# Embedding endpoints accept at most 2048 inputs per call (OpenAI); we cap at 100
# to stay well within provider rate-limit budgets and keep individual request latencies low.
_EMBEDDING_BATCH_SIZE: int = 100


class IngestionPipeline:
    """Orchestrates read → chunk → embed → index for a single document.

    Args:
        chunker: Configured ``TokenSlidingWindowChunker`` instance.
        vector_store: Concrete vector store adapter implementing
            ``VectorStoreInterface``.
        settings: Application settings (provides ``embedding_model`` identifier
            and provider credentials for LiteLLM).
    """

    def __init__(
        self,
        chunker: TokenSlidingWindowChunker,
        vector_store: VectorStoreInterface,
        settings: Settings,
    ) -> None:
        self._chunker = chunker
        self._vector_store = vector_store
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
        # LiteLLM preserves input order in the returned embedding items.
        return await embed_texts(texts, self._settings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_document(self, file_path: Path) -> int:
        """Ingest a single document end-to-end.

        Processing steps:
        1. Read raw bytes and decode/extract text via a format-specific loader.
        2. Segment into overlapping token-bounded chunks.
        3. Call the embedding model (via LiteLLM) in batches of at most
           ``_EMBEDDING_BATCH_SIZE`` chunks.
        4. Persist each batch of (chunk, embedding) pairs via the vector store.

        Args:
            file_path: Path to the source document (``.txt``, ``.md``, or ``.pdf``).

        Returns:
            Total number of ``DocumentChunk`` objects successfully ingested.
            Returns 0 for empty documents without raising.
        """
        document_id: str = file_path.stem
        file_bytes: bytes = await asyncio.to_thread(file_path.read_bytes)
        loader = DocumentLoaderFactory.get_loader(file_path.suffix)
        loaded = await loader.load(file_bytes, file_path.name)
        text: str = loaded.content

        chunks = self._chunker.chunk_document(document_id, text)
        if not chunks:
            logger.warning("Document %r produced zero chunks — nothing ingested.", file_path)
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
