"""Unit tests for the HTML document loader."""

from __future__ import annotations

import pytest

from src.core.exceptions import DocumentParsingError
from src.ingestion.chunker import TokenSlidingWindowChunker
from src.ingestion.loaders import HtmlDocumentLoader as HtmlLoaderExport
from src.ingestion.loaders.factory import DocumentLoaderFactory
from src.ingestion.loaders.html import HtmlDocumentLoader


def _sample_html_bytes() -> bytes:
    html = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Ignored When H1 Exists</title>
        <style>body { color: red; }</style>
        <script>alert("noise");</script>
      </head>
      <body>
        <nav>Skip me</nav>
        <h1>Main Topic</h1>
        <p>Intro paragraph under the main heading.</p>
        <h2>Subsection</h2>
        <ul>
          <li>First item</li>
          <li>Second item</li>
        </ul>
        <table>
          <tr><th>Key</th><th>Value</th></tr>
          <tr><td>Alpha</td><td>1</td></tr>
        </table>
      </body>
    </html>
    """
    return html.encode("utf-8")


def _empty_html_bytes() -> bytes:
    return b"<html><head><style>.x{}</style></head><body><script></script></body></html>"


def _title_only_html_bytes() -> bytes:
    return b"<html><head><title>Page Title</title></head><body><p>Body copy.</p></body></html>"


def test_public_exports_include_html_loader() -> None:
    assert HtmlLoaderExport is HtmlDocumentLoader


@pytest.mark.asyncio
async def test_html_loader_strips_script_style_and_preserves_headings() -> None:
    loaded = await HtmlDocumentLoader().load(_sample_html_bytes(), "page.html")

    assert "# Main Topic" in loaded.content
    assert "## Subsection" in loaded.content
    assert "Intro paragraph under the main heading." in loaded.content
    assert "- First item" in loaded.content
    assert "Key | Value" in loaded.content
    assert "Alpha | 1" in loaded.content
    assert "alert" not in loaded.content
    assert "color: red" not in loaded.content
    assert "Skip me" not in loaded.content
    assert "Ignored When H1 Exists" not in loaded.content
    assert loaded.source == "page"
    assert loaded.metadata["format"] == "html"


def _sectioned_html_bytes() -> bytes:
    body = "Body under the first heading. " * 50
    html = f"""
    <html><body>
      <h1>Main Topic</h1>
      <p>{body}</p>
      <h2>Subsection</h2>
      <p>{"Body under the second heading. " * 50}</p>
    </body></html>
    """
    return html.encode("utf-8")


@pytest.mark.asyncio
async def test_html_heading_maps_to_chunk_section_metadata() -> None:
    loaded = await HtmlDocumentLoader().load(_sectioned_html_bytes(), "sections.html")

    chunker = TokenSlidingWindowChunker(chunk_size=40, chunk_overlap=5)
    chunks = chunker.chunk_document("sections", loaded.content)
    sections = {chunk.metadata["section"] for chunk in chunks if chunk.metadata["section"]}

    assert "Main Topic" in sections
    assert "Subsection" in sections


@pytest.mark.asyncio
async def test_html_loader_uses_title_when_no_h1() -> None:
    loaded = await HtmlDocumentLoader().load(_title_only_html_bytes(), "title.html")

    assert loaded.content.startswith("# Page Title")
    assert "Body copy." in loaded.content


@pytest.mark.asyncio
async def test_html_loader_rejects_empty_file_bytes() -> None:
    with pytest.raises(DocumentParsingError, match="empty"):
        await HtmlDocumentLoader().load(b"   ", "empty.html")


@pytest.mark.asyncio
async def test_html_loader_rejects_whitespace_only_html() -> None:
    with pytest.raises(DocumentParsingError, match="no extractable body text"):
        await HtmlDocumentLoader().load(_empty_html_bytes(), "blank.html")


def test_factory_resolves_html_and_htm_from_filename() -> None:
    assert isinstance(DocumentLoaderFactory.get_loader("page.html"), HtmlDocumentLoader)
    assert isinstance(DocumentLoaderFactory.get_loader("legacy.htm"), HtmlDocumentLoader)
    assert isinstance(DocumentLoaderFactory.get_loader(".html"), HtmlDocumentLoader)
