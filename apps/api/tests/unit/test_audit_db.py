"""Unit tests for SQLAlchemyAuditRepository.

Uses ``sqlite+aiosqlite:///:memory:`` so every test gets a fresh, isolated
schema with no filesystem side-effects and no PostgreSQL server. SQLAlchemy
backs in-memory SQLite engines with a ``StaticPool``, so the schema created by
``initialize_db()`` stays visible to the short-lived sessions of later calls.
Disposing the engine in the fixture teardown discards the database.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.exceptions import StorageError
from src.core.models import TelemetryRecord
from src.storage.audit_db import SQLAlchemyAuditRepository

_IN_MEMORY_URL = "sqlite+aiosqlite:///:memory:"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo() -> SQLAlchemyAuditRepository:
    """Return an uninitialised repository over a fresh in-memory SQLite engine."""
    return SQLAlchemyAuditRepository(create_async_engine(_IN_MEMORY_URL))


def _make_record(request_id: str = "req-001") -> TelemetryRecord:
    """Construct a fully-populated ``TelemetryRecord`` for testing."""
    return TelemetryRecord(
        request_id=request_id,
        query_text="What is the return policy?",
        response_text=(
            "[Source: policy.pdf, Section: Returns] Items may be returned within 30 days."
        ),
        latency_ms=312.5,
        prompt_tokens=120,
        completion_tokens=80,
        total_tokens=200,
        model_name="gpt-4o-mini",
        created_at="2026-09-19T09:00:00Z",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def repo() -> AsyncGenerator[SQLAlchemyAuditRepository, None]:
    """Yield a fully-initialised repository backed by an in-memory SQLite DB.

    The engine is disposed after each test to release the in-memory database.
    """
    r = _make_repo()
    await r.initialize_db()
    yield r
    await r.close()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    async def test_requires_exactly_one_of_engine_or_session_factory(self) -> None:
        """Supplying neither or both collaborators is a programming error."""
        engine = create_async_engine(_IN_MEMORY_URL)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        try:
            with pytest.raises(ValueError, match="exactly one"):
                SQLAlchemyAuditRepository()
            with pytest.raises(ValueError, match="exactly one"):
                SQLAlchemyAuditRepository(engine, session_factory=factory)
        finally:
            await engine.dispose()

    async def test_session_factory_is_caller_owned(self) -> None:
        """A repository built from a session factory works and ``close()`` spares the engine."""
        engine = create_async_engine(_IN_MEMORY_URL)
        try:
            r = SQLAlchemyAuditRepository(
                session_factory=async_sessionmaker(engine, expire_on_commit=False)
            )
            await r.initialize_db()
            await r.log_query(_make_record())

            await r.close()  # Must not dispose the caller's engine.

            logs = await r.get_recent_logs()
            assert [log.request_id for log in logs] == ["req-001"]
        finally:
            await engine.dispose()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


class TestInitialiseDb:
    async def test_initialize_db_does_not_raise(self) -> None:
        """``initialize_db()`` must complete without raising on a fresh DB."""
        r = _make_repo()
        await r.initialize_db()  # Must not raise.
        await r.close()

    async def test_initialize_db_is_idempotent(self, repo: SQLAlchemyAuditRepository) -> None:
        """A repeated ``initialize_db()`` must neither raise nor discard existing rows."""
        await repo.log_query(_make_record("survivor"))

        await repo.initialize_db()

        logs = await repo.get_recent_logs()
        assert [log.request_id for log in logs] == ["survivor"]


# ---------------------------------------------------------------------------
# log_query
# ---------------------------------------------------------------------------


class TestLogQuery:
    async def test_log_query_persists_record(self, repo: SQLAlchemyAuditRepository) -> None:
        """A logged record must be retrievable via ``get_recent_logs``."""
        record = _make_record()
        await repo.log_query(record)

        logs = await repo.get_recent_logs(limit=10)
        assert len(logs) == 1
        assert logs[0].request_id == record.request_id

    async def test_log_query_all_fields_round_trip(self, repo: SQLAlchemyAuditRepository) -> None:
        """Every field of a ``TelemetryRecord`` must survive a write-read round-trip."""
        original = _make_record("round-trip-001")
        await repo.log_query(original)

        logs = await repo.get_recent_logs(limit=1)
        assert len(logs) == 1
        result = logs[0]

        assert result.request_id == original.request_id
        assert result.query_text == original.query_text
        assert result.response_text == original.response_text
        assert result.latency_ms == original.latency_ms
        assert result.prompt_tokens == original.prompt_tokens
        assert result.completion_tokens == original.completion_tokens
        assert result.total_tokens == original.total_tokens
        assert result.model_name == original.model_name
        assert result.created_at == original.created_at

    async def test_log_multiple_records(self, repo: SQLAlchemyAuditRepository) -> None:
        """Multiple distinct records must all be persisted and retrievable."""
        for i in range(5):
            await repo.log_query(_make_record(f"req-{i:03d}"))

        logs = await repo.get_recent_logs(limit=10)
        assert len(logs) == 5


# ---------------------------------------------------------------------------
# get_recent_logs
# ---------------------------------------------------------------------------


class TestGetRecentLogs:
    async def test_empty_table_returns_empty_list(self, repo: SQLAlchemyAuditRepository) -> None:
        """``get_recent_logs`` on an empty table must return an empty list."""
        logs = await repo.get_recent_logs()
        assert logs == []

    async def test_limit_is_respected(self, repo: SQLAlchemyAuditRepository) -> None:
        """``get_recent_logs(limit=N)`` must return at most N records."""
        for i in range(10):
            await repo.log_query(_make_record(f"req-limit-{i:03d}"))

        logs = await repo.get_recent_logs(limit=3)
        assert len(logs) == 3

    async def test_default_limit_is_fifty(self, repo: SQLAlchemyAuditRepository) -> None:
        """Default ``limit=50`` must cap results at 50 when more records exist."""
        for i in range(60):
            await repo.log_query(_make_record(f"req-fifty-{i:03d}"))

        logs = await repo.get_recent_logs()  # default limit=50
        assert len(logs) == 50

    async def test_ordering_is_descending_by_created_at(
        self, repo: SQLAlchemyAuditRepository
    ) -> None:
        """Records must be returned in descending ``created_at`` order."""
        timestamps = [
            "2026-09-19T07:00:00Z",
            "2026-09-19T08:00:00Z",
            "2026-09-19T09:00:00Z",
        ]
        for i, ts in enumerate(timestamps):
            record = TelemetryRecord(
                request_id=f"order-{i:03d}",
                query_text="query",
                response_text="response",
                latency_ms=100.0,
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20,
                model_name="gpt-4o-mini",
                created_at=ts,
            )
            await repo.log_query(record)

        logs = await repo.get_recent_logs(limit=3)
        returned_timestamps = [log.created_at for log in logs]
        # Descending order: latest first.
        assert returned_timestamps == sorted(returned_timestamps, reverse=True), (
            f"Expected descending order, got: {returned_timestamps!r}"
        )

    async def test_returns_list_of_telemetry_records(self, repo: SQLAlchemyAuditRepository) -> None:
        """``get_recent_logs`` must return a list of ``TelemetryRecord`` instances."""
        await repo.log_query(_make_record())
        logs = await repo.get_recent_logs()
        assert all(isinstance(log, TelemetryRecord) for log in logs)


# ---------------------------------------------------------------------------
# Typed error handling
# ---------------------------------------------------------------------------


class TestStorageErrors:
    async def test_duplicate_request_id_raises_storage_error(
        self, repo: SQLAlchemyAuditRepository
    ) -> None:
        """A primary-key collision must surface as ``StorageError`` with the cause chained."""
        await repo.log_query(_make_record("dup-001"))

        with pytest.raises(StorageError, match="dup-001") as exc_info:
            await repo.log_query(_make_record("dup-001"))

        assert isinstance(exc_info.value.__cause__, SQLAlchemyError)

    async def test_error_message_does_not_leak_row_data(
        self, repo: SQLAlchemyAuditRepository
    ) -> None:
        """``StorageError.message`` reaches API clients, so it must not echo bound values."""
        record = _make_record("leak-001")
        await repo.log_query(record)

        with pytest.raises(StorageError) as exc_info:
            await repo.log_query(record)

        assert record.query_text not in exc_info.value.message
        assert record.response_text not in exc_info.value.message
        assert "IntegrityError" in exc_info.value.message

    async def test_repository_remains_usable_after_failed_insert(
        self, repo: SQLAlchemyAuditRepository
    ) -> None:
        """A failed insert is rolled back and must not poison later operations."""
        await repo.log_query(_make_record("ok-001"))
        with pytest.raises(StorageError):
            await repo.log_query(_make_record("ok-001"))

        await repo.log_query(_make_record("ok-002"))

        logs = await repo.get_recent_logs()
        assert sorted(log.request_id for log in logs) == ["ok-001", "ok-002"]

    async def test_query_failure_raises_storage_error(self) -> None:
        """Reading before the schema exists must raise ``StorageError``, not a driver error."""
        r = _make_repo()
        try:
            with pytest.raises(StorageError, match="retrieve recent telemetry logs") as exc_info:
                await r.get_recent_logs()
            assert isinstance(exc_info.value.__cause__, SQLAlchemyError)
        finally:
            await r.close()

    async def test_os_error_from_driver_raises_storage_error(
        self, repo: SQLAlchemyAuditRepository, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """asyncpg reports an unreachable server as a raw ``OSError``; it must be wrapped too."""

        async def refuse_connection(self: AsyncSession) -> None:
            raise ConnectionRefusedError("database unreachable")

        monkeypatch.setattr(AsyncSession, "commit", refuse_connection)

        with pytest.raises(StorageError) as exc_info:
            await repo.log_query(_make_record())

        assert isinstance(exc_info.value.__cause__, ConnectionRefusedError)
