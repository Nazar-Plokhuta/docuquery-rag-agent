"""SQLAlchemy-backed audit repository for RAG query telemetry.

Persists a ``TelemetryRecord`` for every RAG query processed by this service.
The repository is dialect-agnostic: production runs on PostgreSQL through
``postgresql+asyncpg://``, while the test suite uses
``sqlite+aiosqlite:///:memory:`` to stay offline without a database server.

Design note — session per operation:
    Every operation opens a short-lived ``AsyncSession`` and returns its
    connection to the engine pool on exit, so no connection is pinned for the
    application's lifetime and concurrent requests do not queue behind one
    another. For ``:memory:`` SQLite, SQLAlchemy backs the engine with a
    ``StaticPool`` that shares the single connection, so the schema created by
    ``initialize_db()`` stays visible to later sessions.

Design note — error reporting:
    ``StorageError.message`` is surfaced to API clients and logs. The caught
    exception's text is therefore never interpolated into it: ``str()`` of a
    SQLAlchemy error embeds the statement's bound parameters, which would leak
    user queries and model responses. Only exception type names are reported;
    the original exception stays on ``__cause__``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src.core.exceptions import StorageError
from src.core.interfaces import AuditRepositoryInterface
from src.core.models import TelemetryRecord
from src.storage.orm_models import Base, QueryTelemetryRow

logger = logging.getLogger(__name__)

# asyncpg reports an unreachable server as a raw ``OSError`` (for example
# ``ConnectionRefusedError``) that SQLAlchemy does not wrap, so catching
# ``SQLAlchemyError`` alone would let a database outage escape as an untyped 500.
_DATABASE_ERRORS: tuple[type[Exception], ...] = (SQLAlchemyError, OSError)


def _error_kind(exc: Exception) -> str:
    """Name a database failure without echoing the exception text.

    DBAPI errors carry the driver exception as ``orig``; including its type
    distinguishes e.g. ``IntegrityError(UniqueViolationError)`` from
    ``OperationalError(ConnectionDoesNotExistError)``.
    """
    orig = getattr(exc, "orig", None)
    if orig is None:
        return type(exc).__name__
    return f"{type(exc).__name__}({type(orig).__name__})"


class SQLAlchemyAuditRepository(AuditRepositoryInterface):
    """Concrete audit repository backed by SQLAlchemy 2.0 async sessions.

    Exactly one of ``engine`` or ``session_factory`` must be supplied.

    Args:
        engine: Async engine to build sessions from. The repository owns it and
            disposes it in ``close()``.
        session_factory: Pre-configured session factory, for callers that need
            custom session options. The caller retains ownership of the
            underlying engine, so ``close()`` leaves it untouched. The factory
            should set ``expire_on_commit=False``.

    Raises:
        ValueError: If neither or both of ``engine`` and ``session_factory`` are given.
    """

    def __init__(
        self,
        engine: AsyncEngine | None = None,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        if engine is not None and session_factory is None:
            self._engine: AsyncEngine | None = engine
            self._session_factory = async_sessionmaker(engine, expire_on_commit=False)
        elif session_factory is not None and engine is None:
            self._engine = None
            self._session_factory = session_factory
        else:
            raise ValueError("Provide exactly one of 'engine' or 'session_factory'.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize_db(self) -> None:
        """Create the audit schema if it does not exist yet.

        Idempotent: ``create_all`` checks for each table first, so repeated
        calls across restarts are safe. It is a bootstrap and test convenience
        that never alters an existing table; evolve production schemas with
        Alembic.

        Raises:
            StorageError: If the database is unreachable or the DDL fails.
        """
        try:
            async with self._session_factory() as session:
                connection = await session.connection()
                await connection.run_sync(Base.metadata.create_all)
                await session.commit()
        except _DATABASE_ERRORS as exc:
            raise StorageError(
                f"Failed to initialise audit database schema: {_error_kind(exc)}"
            ) from exc
        logger.info("Audit database schema is ready.")

    async def close(self) -> None:
        """Dispose the engine and its connection pool.

        Should be called from the FastAPI lifespan shutdown hook. A no-op for
        the engine when the repository was built from a caller-owned
        ``session_factory``.
        """
        if self._engine is not None:
            await self._engine.dispose()
            logger.info("Audit database engine disposed.")

    # ------------------------------------------------------------------
    # AuditRepositoryInterface implementation
    # ------------------------------------------------------------------

    async def log_query(self, record: TelemetryRecord) -> None:
        """Persist a single telemetry record in its own transaction.

        Args:
            record: Populated ``TelemetryRecord`` for the completed RAG request.

        Raises:
            StorageError: If the INSERT or commit fails (e.g. duplicate
                ``request_id`` or an unreachable database). The transaction is
                rolled back first.
        """
        async with self._session_factory() as session:
            try:
                session.add(QueryTelemetryRow(**record.model_dump()))
                await session.commit()
            except _DATABASE_ERRORS as exc:
                await session.rollback()
                raise StorageError(
                    f"Failed to log telemetry record {record.request_id!r}: {_error_kind(exc)}"
                ) from exc
        logger.debug("Telemetry record %r persisted.", record.request_id)

    async def get_recent_logs(self, limit: int = 50) -> Sequence[TelemetryRecord]:
        """Retrieve the most recent telemetry records.

        Args:
            limit: Maximum number of rows to return (default 50).

        Returns:
            Records sorted descending by ``created_at``.

        Raises:
            StorageError: If the SELECT fails.
        """
        statement = (
            select(QueryTelemetryRow).order_by(QueryTelemetryRow.created_at.desc()).limit(limit)
        )
        try:
            async with self._session_factory() as session:
                rows = (await session.scalars(statement)).all()
        except _DATABASE_ERRORS as exc:
            raise StorageError(
                f"Failed to retrieve recent telemetry logs: {_error_kind(exc)}"
            ) from exc
        return [TelemetryRecord.model_validate(row, from_attributes=True) for row in rows]
