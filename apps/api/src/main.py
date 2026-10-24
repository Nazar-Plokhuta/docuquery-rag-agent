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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from rich.logging import RichHandler
from sqlalchemy.ext.asyncio import create_async_engine

from src import __version__
from src.api.routes import analytics as analytics_router
from src.api.routes import health as health_router
from src.api.routes import ingestion as ingestion_router
from src.api.routes import query as query_router
from src.config.settings import Settings, get_settings
from src.core.exceptions import (
    AppException,
    DocumentParsingError,
    ResourceNotFoundError,
    StorageError,
    UnsupportedFileTypeError,
)
from src.core.llm import configure_litellm
from src.core.rag.engine import RAGEngine
from src.core.rag.token_counter import TokenBudgetManager
from src.ingestion.chunker import TokenSlidingWindowChunker
from src.ingestion.pipeline import IngestionPipeline
from src.storage.audit_db import SQLAlchemyAuditRepository
from src.storage.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

_UVICORN_LOGGERS: tuple[str, ...] = ("uvicorn", "uvicorn.error", "uvicorn.access")


def _configure_logging(level: str) -> None:
    """Route application and Uvicorn logs through a single ``RichHandler``.

    Uvicorn attaches its own handlers and disables propagation before the
    lifespan runs. Clearing those handlers and propagating to the root logger
    removes the concatenated ``INFO:logger:`` prefix so every line shares one
    timestamped, colored layout. ``force=True`` replaces any handler already
    installed on the root logger.
    """
    handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=True,
        log_time_format="[%X]",
    )
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )
    for logger_name in _UVICORN_LOGGERS:
        named_logger = logging.getLogger(logger_name)
        named_logger.handlers.clear()
        named_logger.propagate = True
        named_logger.setLevel(level)


# ---------------------------------------------------------------------------
# Boot-time infrastructure provisioning
# ---------------------------------------------------------------------------


async def _provision_directories() -> None:
    """Ensure all required data directories exist before accepting traffic.

    Runs once inside the lifespan startup hook.  Using ``exist_ok=True``
    makes the operation idempotent across hot-reloads and container restarts.
    """
    settings = get_settings()
    if settings.chroma_host:
        # Remote Chroma owns its own storage; no local directory is needed.
        return
    directory = Path(settings.chroma_persist_directory)
    directory.mkdir(parents=True, exist_ok=True)
    logger.info("Provisioned directory: %s", directory.resolve())


# ---------------------------------------------------------------------------
# Lifespan context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and graceful shutdown.

    Startup sequence:
    1. Configure root and Uvicorn logging through ``RichHandler``.
    2. Provision required filesystem directories.
    3. Create the async SQLAlchemy engine and bootstrap the audit repository.
    4. Construct the ChromaDB vector store adapter (lazy I/O — no network call
       yet; the collection is opened on first actual read/write).
    5. Apply global LiteLLM flags (``drop_params``, telemetry off).
    6. Construct the token budget manager (loads tiktoken BPE vocabulary).
    7. Assemble the ``RAGEngine`` with all injected dependencies.
    8. Assemble the ``IngestionPipeline`` with chunker, vector store, and
       settings.
    9. Attach all services to ``app.state`` for DI resolution per request.

    Shutdown:
    - Dispose the audit engine and its connection pool.
    - ChromaDB, LiteLLM, and tiktoken hold no persistent connections
      requiring explicit teardown.
    """
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger.info(
        "DocuQuery RAG Agent starting — env=%s port=%d",
        settings.app_env,
        settings.app_port,
    )

    await _provision_directories()

    # ── Storage layer ────────────────────────────────────────────────────────
    # Relational audit store: one pooled async engine for the application's
    # lifetime. ``pool_pre_ping`` discards connections a database restart or
    # idle timeout has silently killed; ``hide_parameters`` keeps bound values
    # (user queries and model responses) out of SQLAlchemy error messages.
    audit_engine = create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        hide_parameters=True,
    )
    audit_repo = SQLAlchemyAuditRepository(audit_engine)
    await audit_repo.initialize_db()
    app.state.audit_repo = audit_repo

    # ChromaDB: lazy-initialised adapter; no blocking I/O occurs here.
    vector_store = ChromaVectorStore(settings)
    app.state.vector_store = vector_store

    # ── LLM abstraction ──────────────────────────────────────────────────────
    # LiteLLM keeps no client object: flags are process-global and provider
    # connections are managed per call, so there is nothing to inject or close.
    configure_litellm(settings)

    # ── Domain / core layer ──────────────────────────────────────────────────
    token_budget = TokenBudgetManager()

    rag_engine = RAGEngine(
        vector_store=vector_store,
        audit_repo=audit_repo,
        token_budget=token_budget,
        settings=settings,
    )
    app.state.rag_engine = rag_engine

    # ── Ingestion layer ──────────────────────────────────────────────────────
    chunker = TokenSlidingWindowChunker()
    ingestion_pipeline = IngestionPipeline(
        chunker=chunker,
        vector_store=vector_store,
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
    version=__version__,
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
# CORS (opt-in)
# ---------------------------------------------------------------------------


def parse_cors_origins(settings: Settings) -> list[str]:
    """Split the comma-separated ``CORS_ALLOW_ORIGINS`` value into clean origins."""
    return [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]


def install_cors(target: FastAPI, origins: list[str]) -> None:
    """Attach ``CORSMiddleware`` to ``target`` only when origins are configured.

    The API is a same-origin service by default; the web console is the sole
    cross-origin consumer. Credentials stay disabled (no cookies or auth headers
    are used) and methods/headers are limited to what the console actually sends,
    so a misconfigured origin list widens exposure as little as possible.
    """
    if not origins:
        return
    target.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )


install_cors(app, parse_cors_origins(get_settings()))


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
        DocumentParsingError: 422,
        UnsupportedFileTypeError: 415,
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


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Redirect root requests to interactive OpenAPI documentation."""
    return RedirectResponse(url="/api/docs")


app.include_router(health_router.router, prefix="/api/v1")
app.include_router(query_router.router, prefix="/api/v1")
app.include_router(ingestion_router.router, prefix="/api/v1")
app.include_router(analytics_router.router, prefix="/api/v1")
