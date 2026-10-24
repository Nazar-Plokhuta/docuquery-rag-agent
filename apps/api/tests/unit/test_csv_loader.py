"""Unit tests for the CSV document loader."""

from __future__ import annotations

import pytest

from src.core.exceptions import DocumentParsingError
from src.ingestion.loaders import CsvDocumentLoader as CsvLoaderExport
from src.ingestion.loaders.csv_loader import CsvDocumentLoader
from src.ingestion.loaders.factory import DocumentLoaderFactory


def test_public_exports_include_csv_loader() -> None:
    assert CsvLoaderExport is CsvDocumentLoader


@pytest.mark.asyncio
async def test_csv_loader_renders_markdown_table() -> None:
    payload = b"name,score\nAlice,10\nBob,20\n"
    loaded = await CsvDocumentLoader().load(payload, "scores.csv")

    assert "## CSV: scores" in loaded.content
    assert "| name | score |" in loaded.content
    assert "| --- | --- |" in loaded.content
    assert "| Alice | 10 |" in loaded.content
    assert "| Bob | 20 |" in loaded.content
    assert loaded.source == "scores"
    assert loaded.metadata["format"] == "csv"
    assert loaded.metadata["row_count"] == 3


@pytest.mark.asyncio
async def test_csv_loader_handles_semicolon_delimiter() -> None:
    payload = b"col_a;col_b\n1;2\n"
    loaded = await CsvDocumentLoader().load(payload, "semicolon.csv")

    assert "| col_a | col_b |" in loaded.content
    assert "| 1 | 2 |" in loaded.content


@pytest.mark.asyncio
async def test_csv_loader_rejects_empty_file() -> None:
    with pytest.raises(DocumentParsingError, match="empty"):
        await CsvDocumentLoader().load(b"   ", "empty.csv")


@pytest.mark.asyncio
async def test_csv_loader_rejects_file_with_no_rows() -> None:
    with pytest.raises(DocumentParsingError, match="no rows or columns"):
        await CsvDocumentLoader().load(b",,\n", "blank.csv")


def test_factory_resolves_csv_from_filename() -> None:
    assert isinstance(DocumentLoaderFactory.get_loader("data.csv"), CsvDocumentLoader)
    assert isinstance(DocumentLoaderFactory.get_loader(".csv"), CsvDocumentLoader)
