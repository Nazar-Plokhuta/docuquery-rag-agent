"""Storage adapters for the DocuQuery RAG Agent.

This package contains concrete implementations of the storage interfaces
defined in ``src.core.interfaces``:

- ``orm_models``: SQLAlchemy declarative mappings (Alembic's metadata target).
- ``audit_db``: SQLAlchemy async telemetry / audit log repository
  (PostgreSQL in production, in-memory SQLite in tests).
- ``vector_store``: ChromaDB-backed vector store for semantic retrieval.
"""
