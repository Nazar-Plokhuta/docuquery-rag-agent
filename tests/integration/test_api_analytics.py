"""Integration tests for ``GET /api/v1/analytics/recent`` (telemetry endpoint).

Coverage:
- HTTP 200 with a valid ``AnalyticsResponse`` containing the mocked records.
- ``total_count`` equals the number of records returned by the repository.
- The ``limit`` query parameter is forwarded to ``get_recent_logs``.
- An empty audit log returns HTTP 200 with an empty ``records`` list and
  ``total_count`` of 0.
- Invalid ``limit`` values (0, negative, > 1000) are rejected with HTTP 422.
- Each record in the response contains all ``TelemetryRecord`` fields.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from src.core.models import TelemetryRecord


def _make_telemetry_record(
    request_id: str = "req-001",
    query_text: str = "What is RAG?",
    response_text: str = "RAG is Retrieval-Augmented Generation.",
    latency_ms: float = 150.0,
    prompt_tokens: int = 40,
    completion_tokens: int = 15,
    total_tokens: int = 55,
    model_name: str = "gpt-4o-mini",
    created_at: str | None = None,
) -> TelemetryRecord:
    """Construct a minimal valid ``TelemetryRecord`` for test fixtures."""
    return TelemetryRecord(
        request_id=request_id,
        query_text=query_text,
        response_text=response_text,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model_name=model_name,
        created_at=created_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


@pytest.mark.asyncio
async def test_analytics_returns_200(
    integration_client: AsyncClient,
    mock_audit_repository: AsyncMock,
) -> None:
    """``GET /api/v1/analytics/recent`` must return HTTP 200."""
    mock_audit_repository.get_recent_logs.return_value = []

    response = await integration_client.get("/api/v1/analytics/recent")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_analytics_returns_recent_records(
    integration_client: AsyncClient,
    mock_audit_repository: AsyncMock,
) -> None:
    """Records returned by the repository must appear verbatim in the response."""
    records = [
        _make_telemetry_record(request_id="req-001", query_text="Query A"),
        _make_telemetry_record(request_id="req-002", query_text="Query B"),
    ]
    mock_audit_repository.get_recent_logs.return_value = records

    response = await integration_client.get("/api/v1/analytics/recent")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_count"] == 2
    assert len(payload["records"]) == 2
    assert payload["records"][0]["request_id"] == "req-001"
    assert payload["records"][1]["request_id"] == "req-002"


@pytest.mark.asyncio
async def test_analytics_total_count_equals_records_length(
    integration_client: AsyncClient,
    mock_audit_repository: AsyncMock,
) -> None:
    """``total_count`` must always equal ``len(records)``."""
    records = [_make_telemetry_record(request_id=f"req-{i:03d}") for i in range(5)]
    mock_audit_repository.get_recent_logs.return_value = records

    response = await integration_client.get("/api/v1/analytics/recent")

    payload = response.json()
    assert payload["total_count"] == len(payload["records"])


@pytest.mark.asyncio
async def test_analytics_empty_audit_log_returns_empty_list(
    integration_client: AsyncClient,
    mock_audit_repository: AsyncMock,
) -> None:
    """An empty audit log must return HTTP 200 with empty ``records`` and zero count."""
    mock_audit_repository.get_recent_logs.return_value = []

    response = await integration_client.get("/api/v1/analytics/recent")

    assert response.status_code == 200
    payload = response.json()
    assert payload["records"] == []
    assert payload["total_count"] == 0


@pytest.mark.asyncio
async def test_analytics_limit_parameter_is_forwarded(
    integration_client: AsyncClient,
    mock_audit_repository: AsyncMock,
) -> None:
    """The ``limit`` query parameter must be forwarded to ``get_recent_logs``."""
    mock_audit_repository.get_recent_logs.return_value = []

    await integration_client.get("/api/v1/analytics/recent?limit=25")

    mock_audit_repository.get_recent_logs.assert_called_once_with(limit=25)


@pytest.mark.asyncio
async def test_analytics_default_limit_is_fifty(
    integration_client: AsyncClient,
    mock_audit_repository: AsyncMock,
) -> None:
    """Omitting ``limit`` must default to 50, matching the repository default."""
    mock_audit_repository.get_recent_logs.return_value = []

    await integration_client.get("/api/v1/analytics/recent")

    mock_audit_repository.get_recent_logs.assert_called_once_with(limit=50)


@pytest.mark.asyncio
async def test_analytics_rejects_limit_zero(
    integration_client: AsyncClient,
) -> None:
    """``limit=0`` violates ``ge=1`` and must be rejected with HTTP 422."""
    response = await integration_client.get("/api/v1/analytics/recent?limit=0")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analytics_rejects_limit_above_maximum(
    integration_client: AsyncClient,
) -> None:
    """``limit=1001`` violates ``le=1000`` and must be rejected with HTTP 422."""
    response = await integration_client.get("/api/v1/analytics/recent?limit=1001")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analytics_record_fields_are_complete(
    integration_client: AsyncClient,
    mock_audit_repository: AsyncMock,
) -> None:
    """Every ``TelemetryRecord`` field must be present in the serialised response."""
    record = _make_telemetry_record(
        request_id="full-field-test",
        query_text="Full field query",
        response_text="Full field response",
        latency_ms=99.9,
        prompt_tokens=30,
        completion_tokens=10,
        total_tokens=40,
        model_name="gpt-4o-mini",
    )
    mock_audit_repository.get_recent_logs.return_value = [record]

    response = await integration_client.get("/api/v1/analytics/recent")

    record_json = response.json()["records"][0]
    assert record_json["request_id"] == "full-field-test"
    assert record_json["query_text"] == "Full field query"
    assert record_json["response_text"] == "Full field response"
    assert record_json["latency_ms"] == pytest.approx(99.9)
    assert record_json["prompt_tokens"] == 30
    assert record_json["completion_tokens"] == 10
    assert record_json["total_tokens"] == 40
    assert record_json["model_name"] == "gpt-4o-mini"
    assert "created_at" in record_json
