"""Integration tests for PDF uploads on ``POST /api/v1/ingest/file``."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_ingest_processes_pdf_file(
    integration_client: AsyncClient,
    mock_ingestion_pipeline: AsyncMock,
) -> None:
    """``POST /api/v1/ingest/file`` must accept ``.pdf`` files and return HTTP 200."""
    mock_ingestion_pipeline.ingest_document.return_value = 5

    response = await integration_client.post(
        "/api/v1/ingest/file",
        files={"file": ("manual.pdf", b"%PDF-1.4 sample", "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["filename"] == "manual.pdf"
    assert payload["chunks_ingested"] == 5
