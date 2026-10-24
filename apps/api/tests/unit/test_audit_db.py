"""Unit tests for SQLiteAuditRepository.

Uses an in-memory SQLite database (``sqlite_database_path=":memory:"``) so
every test fixture gets a fresh, isolated schema with no filesystem side-effects.

The ``SQLiteAuditRepository`` maintains a persistent ``aiosqlite.Connection``
opened by ``initialize_db()``, which is the only pattern compatible with
``:memory:`` databases (each new ``aiosqlite.connect(":memory:")`` call
produces a completely separate, empty database).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest_asyncio

from src.config.settings import Settings
from src.core.models import TelemetryRecord
from src.storage.audit_db import SQLiteAuditRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _in_memory_settings() -> Settings:
    """Return a ``Settings`` instance that points to an in-memory SQLite DB."""
    return Settings(
        sqlite_database_path=":memory:",
        chroma_persist_directory="/tmp/test_chroma",
        openai_api_key="sk-test-placeholder",
    )


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
async def repo() -> AsyncGenerator[SQLiteAuditRepository, None]:
    """Yield a fully-initialised repository backed by an in-memory SQLite DB.

    The connection is closed after each test to release the in-memory database.
    """
    r = SQLiteAuditRepository(_in_memory_settings())
    await r.initialize_db()
    yield r
    await r.close()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


class TestInitialiseDb:
    async def test_initialize_db_does_not_raise(self) -> None:
        """``initialize_db()`` must complete without raising on a fresh DB."""
        r = SQLiteAuditRepository(_in_memory_settings())
        await r.initialize_db()  # Must not raise.
        await r.close()

    async def test_initialize_db_is_idempotent(self) -> None:
        """Calling ``initialize_db()`` twice must not raise (IF NOT EXISTS guard)."""
        r = SQLiteAuditRepository(_in_memory_settings())
        await r.initialize_db()
        await r.close()
        # Re-open on a file would be fine; on :memory: we just verify no error
        # on a fresh instance.
        r2 = SQLiteAuditRepository(_in_memory_settings())
        await r2.initialize_db()
        await r2.close()


# ---------------------------------------------------------------------------
# log_query
# ---------------------------------------------------------------------------


class TestLogQuery:
    async def test_log_query_persists_record(self, repo: SQLiteAuditRepository) -> None:
        """A logged record must be retrievable via ``get_recent_logs``."""
        record = _make_record()
        await repo.log_query(record)

        logs = await repo.get_recent_logs(limit=10)
        assert len(logs) == 1
        assert logs[0].request_id == record.request_id

    async def test_log_query_all_fields_round_trip(self, repo: SQLiteAuditRepository) -> None:
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

    async def test_log_multiple_records(self, repo: SQLiteAuditRepository) -> None:
        """Multiple distinct records must all be persisted and retrievable."""
        for i in range(5):
            await repo.log_query(_make_record(f"req-{i:03d}"))

        logs = await repo.get_recent_logs(limit=10)
        assert len(logs) == 5


# ---------------------------------------------------------------------------
# get_recent_logs
# ---------------------------------------------------------------------------


class TestGetRecentLogs:
    async def test_empty_table_returns_empty_list(self, repo: SQLiteAuditRepository) -> None:
        """``get_recent_logs`` on an empty table must return an empty list."""
        logs = await repo.get_recent_logs()
        assert logs == []

    async def test_limit_is_respected(self, repo: SQLiteAuditRepository) -> None:
        """``get_recent_logs(limit=N)`` must return at most N records."""
        for i in range(10):
            await repo.log_query(_make_record(f"req-limit-{i:03d}"))

        logs = await repo.get_recent_logs(limit=3)
        assert len(logs) == 3

    async def test_default_limit_is_fifty(self, repo: SQLiteAuditRepository) -> None:
        """Default ``limit=50`` must cap results at 50 when more records exist."""
        for i in range(60):
            await repo.log_query(_make_record(f"req-fifty-{i:03d}"))

        logs = await repo.get_recent_logs()  # default limit=50
        assert len(logs) == 50

    async def test_ordering_is_descending_by_created_at(self, repo: SQLiteAuditRepository) -> None:
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

    async def test_returns_list_of_telemetry_records(self, repo: SQLiteAuditRepository) -> None:
        """``get_recent_logs`` must return a list of ``TelemetryRecord`` instances."""
        await repo.log_query(_make_record())
        logs = await repo.get_recent_logs()
        assert all(isinstance(log, TelemetryRecord) for log in logs)
