"""Unit tests for IngestionPipeline embedding via ``litellm.aembedding``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.settings import Settings
from src.ingestion.chunker import TokenSlidingWindowChunker
from src.ingestion.pipeline import _EMBEDDING_BATCH_SIZE, IngestionPipeline


def _settings() -> Settings:
    return Settings(
        openai_api_key="sk-test",
        openai_base_url=None,
        embedding_model="text-embedding-3-small",
        database_url="sqlite+aiosqlite:///:memory:",
    )


def _embedding_response(count: int) -> MagicMock:
    response = MagicMock()
    response.data = [{"index": i, "embedding": [float(i)]} for i in range(count)]
    return response


def _stub_chunker(chunk_count: int) -> MagicMock:
    chunker = MagicMock(spec=TokenSlidingWindowChunker)
    chunker.chunk_document.return_value = [
        MagicMock(content=f"chunk {i}") for i in range(chunk_count)
    ]
    return chunker


class TestIngestionPipelineEmbedding:
    async def test_embeds_via_litellm_and_indexes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("hello")
        aembedding = AsyncMock(return_value=_embedding_response(3))
        monkeypatch.setattr("litellm.aembedding", aembedding)
        vector_store = MagicMock()
        vector_store.add_documents = AsyncMock()
        pipeline = IngestionPipeline(_stub_chunker(3), vector_store, _settings())

        ingested = await pipeline.ingest_document(doc)

        assert ingested == 3
        aembedding.assert_awaited_once()
        assert aembedding.await_args.kwargs["model"] == "text-embedding-3-small"
        assert aembedding.await_args.kwargs["input"] == ["chunk 0", "chunk 1", "chunk 2"]
        embeddings = vector_store.add_documents.await_args.args[1]
        assert embeddings == [[0.0], [1.0], [2.0]]

    async def test_batches_by_embedding_batch_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("hello")
        total = _EMBEDDING_BATCH_SIZE + 5

        async def _fake_aembedding(**kwargs: object) -> MagicMock:
            return _embedding_response(len(kwargs["input"]))  # type: ignore[arg-type]

        aembedding = AsyncMock(side_effect=_fake_aembedding)
        monkeypatch.setattr("litellm.aembedding", aembedding)
        vector_store = MagicMock()
        vector_store.add_documents = AsyncMock()
        pipeline = IngestionPipeline(_stub_chunker(total), vector_store, _settings())

        await pipeline.ingest_document(doc)

        assert aembedding.await_count == 2
        sizes = [len(call.kwargs["input"]) for call in aembedding.await_args_list]
        assert sizes == [_EMBEDDING_BATCH_SIZE, 5]
        assert vector_store.add_documents.await_count == 2

    async def test_empty_document_skips_embedding(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("")
        aembedding = AsyncMock()
        monkeypatch.setattr("litellm.aembedding", aembedding)
        pipeline = IngestionPipeline(_stub_chunker(0), MagicMock(), _settings())

        assert await pipeline.ingest_document(doc) == 0
        aembedding.assert_not_called()
