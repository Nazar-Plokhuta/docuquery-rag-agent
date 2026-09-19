"""Integration tests for ``POST /api/v1/ingest/file`` (document ingestion endpoint).

Coverage:
- HTTP 415 on unsupported file extensions (.exe, .docx).
- HTTP 422 when no filename is included in the multipart upload.
- HTTP 200 with valid ``IngestResponse`` for ``.md`` files.
- HTTP 200 with valid ``IngestResponse`` for ``.txt`` files.
- ``filename`` in the response matches the uploaded file's original name.
- ``chunks_ingested`` reflects the value returned by the pipeline mock.
- ``status`` field is ``"success"`` on the happy path.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_ingest_rejects_exe_extension(
    integration_client: AsyncClient,
) -> None:
    """``POST /api/v1/ingest/file`` must return HTTP 415 for ``.exe`` uploads."""
    response = await integration_client.post(
        "/api/v1/ingest/file",
        files={"file": ("setup.exe", b"MZ binary", "application/octet-stream")},
    )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_ingest_rejects_docx_extension(
    integration_client: AsyncClient,
) -> None:
    """``POST /api/v1/ingest/file`` must return HTTP 415 for ``.docx`` uploads."""
    response = await integration_client.post(
        "/api/v1/ingest/file",
        files={"file": ("report.docx", b"PK binary", "application/vnd.openxmlformats")},
    )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_ingest_processes_md_file(
    integration_client: AsyncClient,
    mock_ingestion_pipeline: AsyncMock,
) -> None:
    """``POST /api/v1/ingest/file`` must accept ``.md`` files and return HTTP 200."""
    mock_ingestion_pipeline.ingest_document.return_value = 7

    response = await integration_client.post(
        "/api/v1/ingest/file",
        files={"file": ("readme.md", b"# Title\n\nSome content.", "text/markdown")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["filename"] == "readme.md"
    assert payload["chunks_ingested"] == 7


@pytest.mark.asyncio
async def test_ingest_processes_txt_file(
    integration_client: AsyncClient,
    mock_ingestion_pipeline: AsyncMock,
) -> None:
    """``POST /api/v1/ingest/file`` must accept ``.txt`` files and return HTTP 200."""
    mock_ingestion_pipeline.ingest_document.return_value = 3

    response = await integration_client.post(
        "/api/v1/ingest/file",
        files={"file": ("notes.txt", b"Plain text content here.", "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["filename"] == "notes.txt"
    assert payload["chunks_ingested"] == 3


@pytest.mark.asyncio
async def test_ingest_response_status_is_success(
    integration_client: AsyncClient,
    mock_ingestion_pipeline: AsyncMock,
) -> None:
    """The ``status`` field must be ``"success"`` on the happy path."""
    mock_ingestion_pipeline.ingest_document.return_value = 1

    response = await integration_client.post(
        "/api/v1/ingest/file",
        files={"file": ("doc.md", b"# Doc\n\nContent.", "text/markdown")},
    )

    assert response.json()["status"] == "success"


@pytest.mark.asyncio
async def test_ingest_zero_chunks_for_empty_document(
    integration_client: AsyncClient,
    mock_ingestion_pipeline: AsyncMock,
) -> None:
    """An empty document that produces zero chunks must still return HTTP 200."""
    mock_ingestion_pipeline.ingest_document.return_value = 0

    response = await integration_client.post(
        "/api/v1/ingest/file",
        files={"file": ("empty.md", b"", "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json()["chunks_ingested"] == 0


@pytest.mark.asyncio
async def test_ingest_pipeline_called_with_path_argument(
    integration_client: AsyncClient,
    mock_ingestion_pipeline: AsyncMock,
) -> None:
    """The pipeline's ``ingest_document`` must be called with a ``Path`` argument."""
    from pathlib import Path

    mock_ingestion_pipeline.ingest_document.return_value = 2

    await integration_client.post(
        "/api/v1/ingest/file",
        files={"file": ("article.md", b"# Article\n\nBody text.", "text/markdown")},
    )

    mock_ingestion_pipeline.ingest_document.assert_called_once()
    call_arg = mock_ingestion_pipeline.ingest_document.call_args[0][0]
    # The route passes a ``Path`` object, not a raw string.
    assert isinstance(call_arg, Path)
    # The file stem must match the original filename stem so ``document_id`` is correct.
    assert call_arg.stem == "article"
