"""SQLite-backed audit repository using aiosqlite.

Stores a ``TelemetryRecord`` for every RAG query processed by this service.
The schema is intentionally minimal (no FK constraints, single table) to keep
startup migrations instant and the file fully portable.

Design note — persistent connection:
    A single ``aiosqlite.Connection`` is opened in ``initialize_db()`` and
    reused by all subsequent operations.  This is intentional:

    1. It is the only strategy compatible with ``:memory:`` databases (each
       new ``aiosqlite.connect(":memory:")`` call creates an isolated DB).
    2. For file-based databases it avoids per-operation connect/disconnect
       overhead under moderate request rates.
    3. The connection is closed explicitly in ``close()``, which should be
       called from the FastAPI lifespan shutdown hook.
"""

from __future__ import annotations

import logging

import aiosqlite

from src.config.settings import Settings
from src.core.exceptions import StorageError
from src.core.interfaces import AuditRepositoryInterface
from src.core.models import TelemetryRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL statements
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS query_telemetry (
    request_id        TEXT PRIMARY KEY,
    query_text        TEXT NOT NULL,
    response_text     TEXT NOT NULL,
    latency_ms        REAL NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens      INTEGER NOT NULL,
    model_name        TEXT NOT NULL,
    created_at        TEXT NOT NULL
);
"""

_INSERT_SQL = """
INSERT INTO query_telemetry (
    request_id, query_text, response_text, latency_ms,
    prompt_tokens, completion_tokens, total_tokens, model_name, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_RECENT_SQL = """
SELECT
    request_id, query_text, response_text, latency_ms,
    prompt_tokens, completion_tokens, total_tokens, model_name, created_at
FROM query_telemetry
ORDER BY created_at DESC
LIMIT ?;
"""


class SQLiteAuditRepository(AuditRepositoryInterface):
    """Concrete audit repository backed by SQLite via ``aiosqlite``.

    Args:
        settings: Application settings providing the SQLite DB file path.
            Pass ``sqlite_database_path=":memory:"`` in tests for a hermetic,
            in-process store that requires no filesystem access.
    """

    def __init__(self, settings: Settings) -> None:
        self._db_path: str = settings.sqlite_database_path
        self._connection: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize_db(self) -> None:
        """Open the persistent connection and create the audit schema.

        Idempotent — ``CREATE TABLE IF NOT EXISTS`` ensures repeated calls on
        a file-based DB are safe across restarts.

        Raises:
            StorageError: If the connection cannot be opened or the DDL fails.
        """
        try:
            conn = await aiosqlite.connect(self._db_path)
            conn.row_factory = aiosqlite.Row
            await conn.execute(_CREATE_TABLE_SQL)
            await conn.commit()
            self._connection = conn
            logger.info("SQLite audit DB initialised at %r", self._db_path)
        except Exception as exc:
            raise StorageError(
                f"Failed to initialise audit database at {self._db_path!r}: {exc}"
            ) from exc

    async def close(self) -> None:
        """Close the persistent connection.

        Should be called from the FastAPI lifespan shutdown hook.
        """
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            logger.info("SQLite audit DB connection closed.")

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    async def _get_connection(self) -> aiosqlite.Connection:
        """Return the active connection, auto-initialising if not yet open.

        Auto-init is a convenience for ad-hoc usage (e.g. CLI scripts);
        the canonical path is explicit ``initialize_db()`` at startup.
        """
        if self._connection is None:
            await self.initialize_db()
        # initialize_db() always sets self._connection; assertion for type narrowing.
        assert self._connection is not None
        return self._connection

    # ------------------------------------------------------------------
    # AuditRepositoryInterface implementation
    # ------------------------------------------------------------------

    async def log_query(self, record: TelemetryRecord) -> None:
        """Persist a single telemetry record.

        Args:
            record: Populated ``TelemetryRecord`` for the completed RAG request.

        Raises:
            StorageError: If the INSERT fails (e.g. duplicate ``request_id``).
        """
        conn = await self._get_connection()
        try:
            await conn.execute(
                _INSERT_SQL,
                (
                    record.request_id,
                    record.query_text,
                    record.response_text,
                    record.latency_ms,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.model_name,
                    record.created_at,
                ),
            )
            await conn.commit()
            logger.debug("Telemetry record %r persisted.", record.request_id)
        except Exception as exc:
            raise StorageError(
                f"Failed to log telemetry record {record.request_id!r}: {exc}"
            ) from exc

    async def get_recent_logs(self, limit: int = 50) -> list[TelemetryRecord]:
        """Retrieve the most recent telemetry records.

        Args:
            limit: Maximum number of rows to return (default 50).

        Returns:
            List of ``TelemetryRecord`` sorted descending by ``created_at``.

        Raises:
            StorageError: If the SELECT query fails.
        """
        conn = await self._get_connection()
        try:
            async with conn.execute(_SELECT_RECENT_SQL, (limit,)) as cursor:
                rows = await cursor.fetchall()
        except Exception as exc:
            raise StorageError(f"Failed to retrieve recent telemetry logs: {exc}") from exc

        return [
            TelemetryRecord(
                request_id=row["request_id"],
                query_text=row["query_text"],
                response_text=row["response_text"],
                latency_ms=row["latency_ms"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                total_tokens=row["total_tokens"],
                model_name=row["model_name"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
