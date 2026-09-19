"""Application entry point for the DocuQuery RAG Agent FastAPI service.

Responsibilities:
- Boot-time directory provisioning (idempotent ``mkdir``).
- FastAPI application instantiation with metadata and OpenAPI config.
- Global exception handler translating ``AppException`` subclasses to clean
  JSON error envelopes with appropriate HTTP status codes.
- Full service graph construction and ``app.state`` population inside the
  lifespan context manager so every request reuses pre-initialised, long-lived
  service instances without per-request reconnection overhead.
- Router registration under the versioned ``/api/v1`` prefix.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from src.api.routes import analytics as analytics_router
from src.api.routes import health as health_router
from src.api.routes import ingestion as ingestion_router
from src.api.routes import query as query_router
from src.config.settings import get_settings
from src.core.exceptions import AppException, ResourceNotFoundError, StorageError
from src.core.rag.engine import RAGEngine
from src.core.rag.token_counter import TokenBudgetManager
from src.ingestion.chunker import TokenSlidingWindowChunker
from src.ingestion.pipeline import IngestionPipeline
from src.storage.audit_db import SQLiteAuditRepository
from src.storage.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boot-time infrastructure provisioning
# ---------------------------------------------------------------------------


async def _provision_directories() -> None:
    """Ensure all required data directories exist before accepting traffic.

    Runs once inside the lifespan startup hook.  Using ``exist_ok=True``
    makes the operation idempotent across hot-reloads and container restarts.
    """
    settings = get_settings()
    directories: list[Path] = [
        Path(settings.chroma_persist_directory),
        Path(settings.sqlite_database_path).parent,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Provisioned directory: %s", directory.resolve())


# ---------------------------------------------------------------------------
# Lifespan context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and graceful shutdown.

    Startup sequence:
    1. Configure root log level from settings.
    2. Provision required filesystem directories.
    3. Initialise and connect the SQLite audit repository.
    4. Construct the ChromaDB vector store adapter (lazy I/O — no network call
       yet; the collection is opened on first actual read/write).
    5. Construct the async OpenAI client.
    6. Construct the token budget manager (loads tiktoken BPE vocabulary).
    7. Assemble the ``RAGEngine`` with all injected dependencies.
    8. Assemble the ``IngestionPipeline`` with chunker, vector store, and
       OpenAI client.
    9. Attach all services to ``app.state`` for DI resolution per request.

    Shutdown:
    - Close the SQLite connection, draining any in-flight writes.
    - ChromaDB, OpenAI client, and tiktoken hold no persistent connections
      requiring explicit teardown.
    """
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    logger.info(
        "DocuQuery RAG Agent starting — env=%s port=%d",
        settings.app_env,
        settings.app_port,
    )

    await _provision_directories()

    # ── Storage layer ────────────────────────────────────────────────────────
    # SQLite: open a single persistent connection for the application's entire
    # lifetime (see ADR-02 in docs/internal/architecture.md).
    audit_repo = SQLiteAuditRepository(settings)
    await audit_repo.initialize_db()
    app.state.audit_repo = audit_repo

    # ChromaDB: lazy-initialised adapter; no blocking I/O occurs here.
    vector_store = ChromaVectorStore(settings)
    app.state.vector_store = vector_store

    # ── OpenAI client ────────────────────────────────────────────────────────
    # A single ``AsyncOpenAI`` instance is reused across all requests to share
    # the underlying httpx connection pool and avoid per-request TLS handshakes.
    # ``base_url`` is conditionally forwarded only when explicitly configured to
    # avoid passing ``None`` to the SDK, which some versions treat as an error.
    openai_kwargs: dict[str, str] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url is not None:
        openai_kwargs["base_url"] = settings.openai_base_url
        logger.info("OpenAI client using custom base URL: %s", settings.openai_base_url)
    openai_client = AsyncOpenAI(**openai_kwargs)  # type: ignore[arg-type]

    # ── Domain / core layer ──────────────────────────────────────────────────
    token_budget = TokenBudgetManager()

    rag_engine = RAGEngine(
        vector_store=vector_store,
        audit_repo=audit_repo,
        openai_client=openai_client,
        token_budget=token_budget,
        settings=settings,
    )
    app.state.rag_engine = rag_engine

    # ── Ingestion layer ──────────────────────────────────────────────────────
    chunker = TokenSlidingWindowChunker()
    ingestion_pipeline = IngestionPipeline(
        chunker=chunker,
        vector_store=vector_store,
        openai_client=openai_client,
        settings=settings,
    )
    app.state.ingestion_pipeline = ingestion_pipeline

    logger.info("All services initialised — application is ready to accept traffic.")

    yield  # Application is live and handling requests.

    logger.info("DocuQuery RAG Agent shutting down.")
    await audit_repo.close()


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DocuQuery RAG Agent API",
    version="0.1.0",
    description=(
        "Enterprise-grade Retrieval-Augmented Generation service providing "
        "document ingestion, semantic search, and grounded LLM responses."
    ),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Global domain-exception → HTTP error handler
# ---------------------------------------------------------------------------


def _exception_to_status(exc: AppException) -> int:
    """Map domain exception types to canonical HTTP status codes.

    Using an explicit mapping instead of ``isinstance`` chains keeps the
    translation logic declarative and trivially extensible.
    """
    mapping: dict[type[AppException], int] = {
        ResourceNotFoundError: 404,
        StorageError: 503,
    }
    return mapping.get(type(exc), 500)


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Translate any ``AppException`` subclass into a structured JSON error response.

    The response envelope deliberately omits stack traces and internal paths
    to prevent information leakage in non-development environments.
    """
    status_code = _exception_to_status(exc)
    logger.error(
        "Domain exception [%s] on %s %s: %s",
        exc.__class__.__name__,
        request.method,
        request.url.path,
        exc.message,
    )
    return JSONResponse(
        status_code=status_code,
        content={"error": exc.__class__.__name__, "message": exc.message},
    )


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

app.include_router(health_router.router, prefix="/api/v1")
app.include_router(query_router.router, prefix="/api/v1")
app.include_router(ingestion_router.router, prefix="/api/v1")
app.include_router(analytics_router.router, prefix="/api/v1")
