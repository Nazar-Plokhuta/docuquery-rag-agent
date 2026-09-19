"""Unit tests for PDF and text document loaders and the loader factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import DocumentParsingError, UnsupportedFileTypeError
from src.ingestion.chunker import TokenSlidingWindowChunker
from src.ingestion.loaders import PdfDocumentLoader as PdfLoaderExport
from src.ingestion.loaders.factory import DocumentLoaderFactory
from src.ingestion.loaders.pdf import PdfDocumentLoader
from src.ingestion.loaders.text import TextDocumentLoader


def test_public_exports_include_pdf_loader() -> None:
    assert PdfLoaderExport is PdfDocumentLoader


@pytest.mark.asyncio
async def test_text_loader_decodes_utf8() -> None:
    loader = TextDocumentLoader()
    loaded = await loader.load(b"Hello", "notes.txt")
    assert loaded.content == "Hello"
    assert loaded.source == "notes"


@pytest.mark.asyncio
async def test_text_loader_replaces_invalid_utf8_sequences() -> None:
    loader = TextDocumentLoader()
    loaded = await loader.load(b"\xff\xfe", "bad.txt")
    assert loaded.content  # decodes without raising


@pytest.mark.asyncio
async def test_pdf_loader_injects_page_headings() -> None:
    page_one = MagicMock()
    page_one.extract_text.return_value = "Alpha content"
    page_two = MagicMock()
    page_two.extract_text.return_value = "Beta content"
    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = [page_one, page_two]

    with patch("src.ingestion.loaders.pdf.PdfReader", return_value=mock_reader):
        loader = PdfDocumentLoader()
        loaded = await loader.load(b"%PDF-1.4 fake", "report.pdf")

    assert "## Page 1" in loaded.content
    assert "## Page 2" in loaded.content
    assert "Alpha content" in loaded.content
    assert loaded.source == "report"
    assert loaded.metadata["page_count"] == 2


@pytest.mark.asyncio
async def test_pdf_loader_skips_empty_pages() -> None:
    empty_page = MagicMock()
    empty_page.extract_text.return_value = "   "
    content_page = MagicMock()
    content_page.extract_text.return_value = "Only page"
    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = [empty_page, content_page]

    with patch("src.ingestion.loaders.pdf.PdfReader", return_value=mock_reader):
        loaded = await PdfDocumentLoader().load(b"%PDF", "doc.pdf")

    assert "## Page 2" in loaded.content
    assert "## Page 1" not in loaded.content
    assert loaded.metadata["non_empty_page_count"] == 1


@pytest.mark.asyncio
async def test_pdf_loader_rejects_encrypted_pdf() -> None:
    mock_reader = MagicMock()
    mock_reader.is_encrypted = True
    mock_reader.pages = []

    with (
        patch("src.ingestion.loaders.pdf.PdfReader", return_value=mock_reader),
        pytest.raises(DocumentParsingError, match="encrypted"),
    ):
        await PdfDocumentLoader().load(b"%PDF", "secret.pdf")


@pytest.mark.asyncio
async def test_pdf_loader_corrupted_pdf_maps_to_document_parsing_error() -> None:
    from pypdf.errors import PdfReadError

    with (
        patch(
            "src.ingestion.loaders.pdf.PdfReader",
            side_effect=PdfReadError("invalid"),
        ),
        pytest.raises(DocumentParsingError, match="corrupted"),
    ):
        await PdfDocumentLoader().load(b"bad", "broken.pdf")


@pytest.mark.asyncio
async def test_pdf_loader_rejects_all_empty_pages() -> None:
    empty_page = MagicMock()
    empty_page.extract_text.return_value = ""
    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = [empty_page]

    with (
        patch("src.ingestion.loaders.pdf.PdfReader", return_value=mock_reader),
        pytest.raises(DocumentParsingError, match="no readable page content"),
    ):
        await PdfDocumentLoader().load(b"%PDF", "blank.pdf")


@pytest.mark.asyncio
async def test_pdf_loader_rejects_empty_file_bytes() -> None:
    with pytest.raises(DocumentParsingError, match="empty"):
        await PdfDocumentLoader().load(b"   ", "empty.pdf")


@pytest.mark.asyncio
async def test_pdf_page_headings_map_to_chunk_sections() -> None:
    page_one = MagicMock()
    page_one.extract_text.return_value = "First page body " * 50
    page_two = MagicMock()
    page_two.extract_text.return_value = "Second page body " * 50
    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = [page_one, page_two]

    with patch("src.ingestion.loaders.pdf.PdfReader", return_value=mock_reader):
        loaded = await PdfDocumentLoader().load(b"%PDF", "sections.pdf")

    chunker = TokenSlidingWindowChunker(chunk_size=40, chunk_overlap=5)
    chunks = chunker.chunk_document("sections", loaded.content)
    sections = {chunk.metadata["section"] for chunk in chunks if chunk.metadata["section"]}

    assert "Page 1" in sections
    assert "Page 2" in sections


def test_factory_resolves_supported_extensions() -> None:
    assert isinstance(DocumentLoaderFactory.get_loader(".pdf"), PdfDocumentLoader)
    assert isinstance(DocumentLoaderFactory.get_loader(".MD"), TextDocumentLoader)
    assert isinstance(DocumentLoaderFactory.get_loader(".txt"), TextDocumentLoader)


def test_factory_raises_for_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError, match=".docx"):
        DocumentLoaderFactory.get_loader(".docx")
