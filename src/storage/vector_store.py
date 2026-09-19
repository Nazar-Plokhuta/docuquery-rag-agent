"""ChromaDB-backed vector store adapter.

All blocking Chroma calls are isolated inside ``asyncio.to_thread()`` to
prevent the synchronous ChromaDB client from stalling the FastAPI event loop.

Design decisions:
- ``chromadb.PersistentClient`` is instantiated lazily on first access so the
  constructor never performs I/O, keeping FastAPI startup fast.
- The collection uses ``cosine`` distance (``hnsw:space = "cosine"``) which
  aligns with OpenAI embedding model output (unit-normalised vectors).
- A single ``Collection`` handle is cached on the instance after first access.
  It is safe to share across coroutines because all mutations run through the
  thread executor and Chroma handles its own internal locking.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import chromadb
from chromadb import Collection

from src.config.settings import Settings
from src.core.exceptions import StorageError
from src.core.interfaces import VectorStoreInterface
from src.core.models import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)

_DEFAULT_COLLECTION_NAME = "enterprise_knowledge_base"

# Cosine distance metadata consumed by Chroma's HNSW index builder.
_COSINE_COLLECTION_METADATA: dict[str, str] = {"hnsw:space": "cosine"}


class ChromaVectorStore(VectorStoreInterface):
    """Persistent ChromaDB vector store adapter.

    Args:
        settings: Application settings providing the ChromaDB persistence path.
        collection_name: Chroma collection to use (defaults to
            ``enterprise_knowledge_base``).
    """

    def __init__(
        self,
        settings: Settings,
        collection_name: str = _DEFAULT_COLLECTION_NAME,
    ) -> None:
        self._persist_dir: str = settings.chroma_persist_directory
        self._collection_name: str = collection_name
        # Lazily initialised; populated on first call to ``_get_collection()``.
        self._client: chromadb.PersistentClient | None = None
        self._collection: Collection | None = None

    # ------------------------------------------------------------------
    # Internal — lazy initialisation (all blocking, run in thread)
    # ------------------------------------------------------------------

    def _init_collection_sync(self) -> Collection:
        """Create (or open) the Chroma client and collection synchronously.

        This method is intentionally blocking and must only be invoked from
        inside ``asyncio.to_thread()``.
        """
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata=_COSINE_COLLECTION_METADATA,
        )

    async def _get_collection(self) -> Collection:
        """Return the cached collection handle, initialising lazily if needed."""
        if self._collection is None:
            self._collection = await asyncio.to_thread(self._init_collection_sync)
        return self._collection

    # ------------------------------------------------------------------
    # VectorStoreInterface implementation
    # ------------------------------------------------------------------

    async def add_documents(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Persist chunks and their pre-computed embeddings to Chroma.

        Chroma metadata is restricted to scalar types (``str``, ``int``,
        ``float``, ``bool``).  The ``DocumentChunk.metadata`` dict already
        conforms to this constraint (``dict[str, str | int]``), so it is
        passed through verbatim.

        Args:
            chunks: Ordered sequence of ``DocumentChunk`` instances to index.
            embeddings: Pre-computed dense vectors, one per chunk, in the
                same order as ``chunks``.  If ``None``, Chroma will attempt to
                use a configured embedding function (not recommended here).

        Raises:
            StorageError: If the Chroma ``add`` call fails.
        """
        if not chunks:
            return

        ids: list[str] = [c.chunk_id for c in chunks]
        documents: list[str] = [c.content for c in chunks]
        metadatas: list[dict[str, str | int]] = [dict(c.metadata) for c in chunks]

        def _add() -> None:
            collection = asyncio.get_event_loop().run_until_complete(self._get_collection())
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,  # type: ignore[arg-type]
                embeddings=embeddings,
            )

        # Re-acquire the collection handle safely outside the thread, then
        # dispatch only the blocking ``.add()`` call into a thread.
        collection = await self._get_collection()

        try:
            await asyncio.to_thread(
                collection.add,
                ids=ids,
                documents=documents,
                metadatas=metadatas,  # type: ignore[arg-type]
                embeddings=embeddings,
            )
            logger.info(
                "Persisted %d chunks to Chroma collection %r.",
                len(chunks),
                self._collection_name,
            )
        except Exception as exc:
            raise StorageError(f"ChromaDB add_documents failed: {exc}") from exc

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> Sequence[RetrievalResult]:
        """Retrieve the top-k most semantically similar chunks.

        Chroma returns cosine *distance* in [0, 2] (0 = identical).
        We convert to a similarity score via ``score = 1.0 - distance``
        (valid for unit-normalised OpenAI embeddings where distance ∈ [0, 1]).

        Args:
            query_embedding: Dense vector from the same embedding model used
                during ingestion.
            top_k: Number of nearest neighbours to retrieve.

        Returns:
            Sequence of ``RetrievalResult`` sorted by descending score.

        Raises:
            StorageError: If the Chroma ``query`` call fails.
        """
        collection = await self._get_collection()

        try:
            raw: dict = await asyncio.to_thread(  # type: ignore[type-arg]
                collection.query,
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise StorageError(f"ChromaDB similarity_search failed: {exc}") from exc

        # Chroma returns nested lists indexed by query; we issued a single query.
        ids: list[str] = (raw.get("ids") or [[]])[0]
        documents: list[str] = (raw.get("documents") or [[]])[0]
        metadatas: list[dict[str, str | int]] = (raw.get("metadatas") or [[]])[0]
        distances: list[float] = (raw.get("distances") or [[]])[0]

        results: list[RetrievalResult] = []
        for chunk_id, doc_text, meta, dist in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            # Clamp to [0, 1] to guard against floating-point edge cases.
            score: float = max(0.0, 1.0 - float(dist))
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=str(meta.get("document_id", "")),
                content=doc_text,
                metadata={k: v for k, v in meta.items() if isinstance(v, (str, int))},
                token_count=int(meta.get("token_count", 0)),
            )
            results.append(RetrievalResult(chunk=chunk, score=score))

        return results
