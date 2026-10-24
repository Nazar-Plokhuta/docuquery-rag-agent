"""Shared fixtures for integration tests of the DocuQuery RAG Agent API.

Each fixture in this module provides a mock of a specific application service.
The ``integration_client`` fixture assembles them all and installs
``app.dependency_overrides`` so that route handlers receive mocks instead of
the real service instances stored in ``app.state``.

Design principles:
- Overrides are installed and removed at fixture teardown, never leaking
  state across unrelated tests or into the session-scoped settings override.
- Mock instances are created with ``AsyncMock(spec=...)`` so type errors in
  calling code surface as ``AttributeError`` at test time, not silently.
- Real I/O (ChromaDB, OpenAI, SQLite) never occurs during integration tests
  because every DI provider function is replaced by a lambda returning a mock.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import (
    get_audit_repository,
    get_ingestion_pipeline,
    get_rag_engine,
    get_vector_store,
)
from src.core.interfaces import AuditRepositoryInterface, VectorStoreInterface
from src.core.rag.engine import RAGEngine
from src.ingestion.pipeline import IngestionPipeline
from src.main import app


@pytest_asyncio.fixture
async def mock_rag_engine() -> AsyncMock:
    """Return a fully specced ``AsyncMock`` of ``RAGEngine``."""
    return AsyncMock(spec=RAGEngine)


@pytest_asyncio.fixture
async def mock_ingestion_pipeline() -> AsyncMock:
    """Return a fully specced ``AsyncMock`` of ``IngestionPipeline``."""
    return AsyncMock(spec=IngestionPipeline)


@pytest_asyncio.fixture
async def mock_audit_repository() -> AsyncMock:
    """Return a fully specced ``AsyncMock`` of ``AuditRepositoryInterface``."""
    return AsyncMock(spec=AuditRepositoryInterface)


@pytest_asyncio.fixture
async def mock_vector_store() -> AsyncMock:
    """Return a fully specced ``AsyncMock`` of ``VectorStoreInterface``."""
    return AsyncMock(spec=VectorStoreInterface)


@pytest_asyncio.fixture
async def integration_client(
    mock_rag_engine: AsyncMock,
    mock_ingestion_pipeline: AsyncMock,
    mock_audit_repository: AsyncMock,
    mock_vector_store: AsyncMock,
) -> AsyncGenerator[AsyncClient, None]:
    """Yield an ``AsyncClient`` with all service DI providers replaced by mocks.

    The client is bound to the FastAPI ASGI app via an in-process transport
    so no real TCP socket is opened.  Lifespan startup still runs (provisioning
    directories and creating a real SQLiteAuditRepository), but every request
    that reaches a route handler receives the mock services via DI override.

    The session-scoped ``get_settings`` override from the top-level conftest
    is preserved; only the service-layer overrides are added and removed here.
    """
    # Install service-layer DI overrides.  Lambda wrappers capture the mock
    # fixtures by closure so the exact same instance is injected on every call.
    app.dependency_overrides[get_rag_engine] = lambda: mock_rag_engine
    app.dependency_overrides[get_ingestion_pipeline] = lambda: mock_ingestion_pipeline
    app.dependency_overrides[get_audit_repository] = lambda: mock_audit_repository
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client

    # Remove only the overrides added by this fixture, leaving the session-
    # scoped settings override intact for subsequent tests.
    app.dependency_overrides.pop(get_rag_engine, None)
    app.dependency_overrides.pop(get_ingestion_pipeline, None)
    app.dependency_overrides.pop(get_audit_repository, None)
    app.dependency_overrides.pop(get_vector_store, None)
