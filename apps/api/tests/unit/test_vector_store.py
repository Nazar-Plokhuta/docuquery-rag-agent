"""Unit tests for ``ChromaVectorStore`` client selection.

Both Chroma client constructors are patched, so no test touches the filesystem
or a live Chroma server.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config.settings import Settings
from src.core.models import DocumentChunk
from src.storage.vector_store import ChromaVectorStore


def _settings(**overrides: str | int | None) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        openai_api_key="sk-test",
        database_url="sqlite+aiosqlite:///:memory:",
        chroma_persist_directory="/tmp/test_chroma_db",
        **overrides,  # type: ignore[arg-type]
    )


@pytest.fixture
def chroma_clients(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Patch both client factories and return ``(persistent, http)`` mocks."""
    persistent = MagicMock(name="PersistentClient")
    http = MagicMock(name="HttpClient")
    monkeypatch.setattr("chromadb.PersistentClient", persistent)
    monkeypatch.setattr("chromadb.HttpClient", http)
    return persistent, http


class TestClientSelection:
    async def test_local_mode_uses_persistent_client(
        self, chroma_clients: tuple[MagicMock, MagicMock]
    ) -> None:
        persistent, http = chroma_clients
        store = ChromaVectorStore(_settings())

        await store.add_documents([_chunk()], embeddings=[[0.1, 0.2]])

        persistent.assert_called_once_with(path="/tmp/test_chroma_db")
        http.assert_not_called()

    async def test_host_set_uses_http_client(
        self, chroma_clients: tuple[MagicMock, MagicMock]
    ) -> None:
        persistent, http = chroma_clients
        store = ChromaVectorStore(_settings(chroma_host="chroma", chroma_port=9000))

        await store.add_documents([_chunk()], embeddings=[[0.1, 0.2]])

        http.assert_called_once_with(host="chroma", port=9000)
        persistent.assert_not_called()

    async def test_empty_host_falls_back_to_local_mode(
        self, chroma_clients: tuple[MagicMock, MagicMock]
    ) -> None:
        persistent, http = chroma_clients
        store = ChromaVectorStore(_settings(chroma_host=""))

        await store.add_documents([_chunk()], embeddings=[[0.1, 0.2]])

        persistent.assert_called_once()
        http.assert_not_called()

    async def test_remote_collection_receives_cosine_metadata_and_chunks(
        self, chroma_clients: tuple[MagicMock, MagicMock]
    ) -> None:
        _, http = chroma_clients
        collection = http.return_value.get_or_create_collection.return_value
        store = ChromaVectorStore(_settings(chroma_host="chroma"))

        await store.add_documents([_chunk()], embeddings=[[0.1, 0.2]])

        http.return_value.get_or_create_collection.assert_called_once_with(
            name="enterprise_knowledge_base", metadata={"hnsw:space": "cosine"}
        )
        collection.add.assert_called_once()
        assert collection.add.call_args.kwargs["ids"] == ["c1"]


def _chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="c1",
        document_id="d1",
        content="hello",
        metadata={"document_id": "d1"},
        token_count=1,
    )
