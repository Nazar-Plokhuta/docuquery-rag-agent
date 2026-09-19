"""Unit tests for the health-check endpoint.

Covers:
- HTTP 200 response on ``GET /api/v1/health/``.
- Correct content-type header.
- Presence and type correctness of all ``HealthResponse`` fields.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_200(async_client: AsyncClient) -> None:
    """``GET /api/v1/health/`` must respond with HTTP 200 OK."""
    response = await async_client.get("/api/v1/health/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_content_type(async_client: AsyncClient) -> None:
    """Response must carry a JSON content-type header."""
    response = await async_client.get("/api/v1/health/")
    assert "application/json" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_health_response_payload(async_client: AsyncClient) -> None:
    """All ``HealthResponse`` fields must be present with expected values."""
    response = await async_client.get("/api/v1/health/")
    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["app_name"] == "DocuQuery RAG Agent"
    assert payload["version"] == "0.1.0"
    # Environment is driven by the test ``Settings`` override.
    assert payload["environment"] == "development"


@pytest.mark.asyncio
async def test_health_response_field_types(async_client: AsyncClient) -> None:
    """All fields in the health payload must be non-empty strings."""
    response = await async_client.get("/api/v1/health/")
    payload = response.json()

    for field in ("status", "app_name", "version", "environment"):
        assert isinstance(payload[field], str), f"Field '{field}' must be a string"
        assert payload[field], f"Field '{field}' must not be empty"
