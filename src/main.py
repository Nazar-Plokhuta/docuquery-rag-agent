"""Application entry point for the DocuQuery RAG Agent FastAPI service.

Responsibilities:
- Boot-time directory provisioning (idempotent ``mkdir``).
- FastAPI application instantiation with metadata and OpenAPI config.
- Global exception handler translating ``AppException`` subclasses to clean
  JSON error envelopes with appropriate HTTP status codes.
- Router registration under the versioned ``/api/v1`` prefix.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes import health as health_router
from src.config.settings import get_settings
from src.core.exceptions import AppException, ResourceNotFoundError, StorageError
from src.storage.audit_db import SQLiteAuditRepository

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

    Shutdown is a no-op at this stage; teardown hooks for DB connections will
    be added in subsequent sprints.
    """
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    logger.info(
        "DocuQuery RAG Agent starting — env=%s port=%d",
        settings.app_env,
        settings.app_port,
    )

    await _provision_directories()

    # Initialise the SQLite audit schema and hold the connection for the
    # application's lifetime.  Stored on ``app.state`` so Sprint 3 DI can
    # resolve ``SQLiteAuditRepository`` from the request context without
    # reopening the connection on every call.
    audit_repo = SQLiteAuditRepository(settings)
    await audit_repo.initialize_db()
    app.state.audit_repo = audit_repo

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
