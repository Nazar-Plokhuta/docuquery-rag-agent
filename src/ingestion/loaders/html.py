"""HTML document loader with Markdown heading markers for citation provenance."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

from src.core.exceptions import DocumentParsingError
from src.ingestion.loaders.base import BaseDocumentLoader, LoadedDocument

logger = logging.getLogger(__name__)

_STRIP_TAGS = frozenset({"script", "style", "noscript", "header", "footer", "nav", "svg"})
_HEADING_PREFIX_BY_TAG: dict[str, str] = {
    "h1": "# ",
    "h2": "## ",
    "h3": "### ",
    "h4": "#### ",
    "h5": "##### ",
    "h6": "###### ",
}
_CONTAINER_TAGS = frozenset(
    {
        "html",
        "body",
        "main",
        "article",
        "section",
        "div",
        "aside",
        "figure",
        "figcaption",
        "details",
        "summary",
    }
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _decode_html_bytes(file_bytes: bytes) -> tuple[str, str]:
    """Decode HTML bytes with UTF-8 first, then common legacy encodings."""
    try:
        return file_bytes.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        for encoding in ("latin-1", "cp1251"):
            try:
                return file_bytes.decode(encoding), encoding
            except UnicodeDecodeError:
                continue
        return file_bytes.decode("utf-8", errors="replace"), "utf-8-replace"


def _strip_non_content(soup: BeautifulSoup) -> None:
    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()


def _table_to_markdown(table: Tag) -> str:
    rows: list[str] = []
    for row in table.find_all("tr"):
        cells = [
            _normalize_whitespace(cell.get_text(separator=" ", strip=True)).replace("|", "\\|")
            for cell in row.find_all(["th", "td"])
        ]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _list_to_markdown(list_tag: Tag) -> str:
    ordered = list_tag.name == "ol"
    lines: list[str] = []
    for index, item in enumerate(list_tag.find_all("li", recursive=False), start=1):
        text = _normalize_whitespace(item.get_text(separator=" ", strip=True))
        if not text:
            continue
        prefix = f"{index}. " if ordered else "- "
        lines.append(f"{prefix}{text}")
    return "\n".join(lines)


def _append_block(blocks: list[str], block: str) -> None:
    normalized = block.strip()
    if normalized:
        blocks.append(normalized)


def _process_node(node: Tag | NavigableString, blocks: list[str]) -> None:
    if isinstance(node, NavigableString):
        return

    name = node.name
    if name is None:
        return

    if name in _HEADING_PREFIX_BY_TAG:
        text = _normalize_whitespace(node.get_text(separator=" ", strip=True))
        if text:
            _append_block(blocks, f"{_HEADING_PREFIX_BY_TAG[name]}{text}")
        return

    if name == "p":
        text = _normalize_whitespace(node.get_text(separator=" ", strip=True))
        _append_block(blocks, text)
        return

    if name in ("ul", "ol"):
        _append_block(blocks, _list_to_markdown(node))
        return

    if name == "table":
        _append_block(blocks, _table_to_markdown(node))
        return

    if name == "br":
        blocks.append("")
        return

    if name in _CONTAINER_TAGS:
        for child in node.children:
            if isinstance(child, Tag):
                _process_node(child, blocks)
            elif isinstance(child, NavigableString):
                text = _normalize_whitespace(str(child))
                _append_block(blocks, text)
        return

    if name == "li":
        return

    text = _normalize_whitespace(node.get_text(separator=" ", strip=True))
    _append_block(blocks, text)


def _document_has_h1(soup: BeautifulSoup) -> bool:
    return soup.find("h1") is not None


def _title_heading(soup: BeautifulSoup) -> str | None:
    title_tag = soup.find("title")
    if title_tag is None:
        return None
    title_text = _normalize_whitespace(title_tag.get_text(separator=" ", strip=True))
    if not title_text:
        return None
    return f"# {title_text}"


def _extract_html_text(file_bytes: bytes, filename: str) -> LoadedDocument:
    """Synchronous HTML parse — intended to run inside ``asyncio.to_thread``."""
    source = Path(filename).stem
    html_text, encoding = _decode_html_bytes(file_bytes)
    soup = BeautifulSoup(html_text, "html.parser")

    title_heading = None if _document_has_h1(soup) else _title_heading(soup)
    _strip_non_content(soup)

    root = soup.body if soup.body is not None else soup
    blocks: list[str] = []
    if title_heading is not None:
        blocks.append(title_heading)

    for child in root.children:
        if isinstance(child, Tag):
            _process_node(child, blocks)

    content = "\n\n".join(blocks).strip()
    if not content:
        raise DocumentParsingError(
            f"Unable to extract text from HTML {filename!r}: no extractable body text found."
        )

    metadata: dict[str, Any] = {
        "format": "html",
        "filename": filename,
        "encoding": encoding,
    }

    return LoadedDocument(content=content, source=source, metadata=metadata)


class HtmlDocumentLoader(BaseDocumentLoader):
    """Extracts HTML text with Markdown headings for chunk section metadata."""

    async def load(self, file_bytes: bytes, filename: str) -> LoadedDocument:
        """Parse ``file_bytes`` off the event loop and return heading-marked text."""
        if not file_bytes.strip():
            raise DocumentParsingError(f"Unable to read HTML {filename!r}: file is empty.")

        return await asyncio.to_thread(_extract_html_text, file_bytes, filename)
