"""FastAPI dependency providers for shared application services.

Each provider function extracts a pre-initialised service from
``request.app.state``, which is populated once during the lifespan startup
hook in ``src/main.py``.  This strategy ensures every request reuses the
same long-lived connection or service instance without reconnecting, while
remaining trivially overridable in tests via ``app.dependency_overrides``.

Design invariant: no provider function performs I/O or constructs new service
instances.  Construction belongs exclusively in the lifespan handler.
"""

from __future__ import annotations

from fastapi import Request

from src.core.interfaces import AuditRepositoryInterface, VectorStoreInterface
from src.core.rag.engine import RAGEngine
from src.ingestion.pipeline import IngestionPipeline


def get_audit_repository(request: Request) -> AuditRepositoryInterface:
    """Resolve the application-scoped audit repository from request state.

    Args:
        request: Injected FastAPI ``Request`` object carrying ``app.state``.

    Returns:
        The ``AuditRepositoryInterface`` instance initialised at startup.
    """
    repository: AuditRepositoryInterface = request.app.state.audit_repo
    return repository


def get_vector_store(request: Request) -> VectorStoreInterface:
    """Resolve the application-scoped vector store adapter from request state.

    Args:
        request: Injected FastAPI ``Request`` object carrying ``app.state``.

    Returns:
        The ``VectorStoreInterface`` instance initialised at startup.
    """
    store: VectorStoreInterface = request.app.state.vector_store
    return store


def get_rag_engine(request: Request) -> RAGEngine:
    """Resolve the application-scoped RAG orchestration engine from request state.

    Args:
        request: Injected FastAPI ``Request`` object carrying ``app.state``.

    Returns:
        The ``RAGEngine`` instance initialised at startup.
    """
    engine: RAGEngine = request.app.state.rag_engine
    return engine


def get_ingestion_pipeline(request: Request) -> IngestionPipeline:
    """Resolve the application-scoped ingestion pipeline from request state.

    Args:
        request: Injected FastAPI ``Request`` object carrying ``app.state``.

    Returns:
        The ``IngestionPipeline`` instance initialised at startup.
    """
    pipeline: IngestionPipeline = request.app.state.ingestion_pipeline
    return pipeline
