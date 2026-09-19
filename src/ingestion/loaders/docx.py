"""DOCX document loader with Markdown heading markers for citation provenance."""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.core.exceptions import DocumentParsingError
from src.ingestion.loaders.base import BaseDocumentLoader, LoadedDocument

logger = logging.getLogger(__name__)

_HEADING_PREFIX_BY_STYLE: dict[str, str] = {
    "Title": "# ",
    "Heading 1": "# ",
    "Heading 2": "## ",
    "Heading 3": "### ",
    "Heading 4": "#### ",
    "Heading 5": "##### ",
    "Heading 6": "###### ",
}


def _iter_block_items(parent: DocxDocument) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in document order (not separate flat lists)."""
    for child in parent.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def _paragraph_style_name(paragraph: Paragraph) -> str:
    style = paragraph.style
    if style is None or style.name is None:
        return ""
    return style.name


def _paragraph_to_markdown(paragraph: Paragraph) -> str:
    text = (paragraph.text or "").strip()
    if not text:
        return ""
    prefix = _HEADING_PREFIX_BY_STYLE.get(_paragraph_style_name(paragraph), "")
    return f"{prefix}{text}" if prefix else text


def _table_to_text(table: Table) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells = [(cell.text or "").strip().replace("|", "\\|") for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extract_docx_text(file_bytes: bytes, filename: str) -> LoadedDocument:
    """Synchronous DOCX parse — intended to run inside ``asyncio.to_thread``."""
    source = Path(filename).stem
    try:
        document = Document(io.BytesIO(file_bytes))
    except BadZipFile as exc:
        raise DocumentParsingError(
            f"Unable to read DOCX {filename!r}: the file may be corrupted or not a valid "
            f"Word document."
        ) from exc
    except (KeyError, ValueError, OSError) as exc:
        raise DocumentParsingError(
            f"Unable to read DOCX {filename!r}: the file may be corrupted or invalid."
        ) from exc

    blocks: list[str] = []
    paragraph_count = 0
    table_count = 0

    for block in _iter_block_items(document):
        if isinstance(block, Paragraph):
            paragraph_count += 1
            markdown_line = _paragraph_to_markdown(block)
            if markdown_line:
                blocks.append(markdown_line)
        else:
            table_count += 1
            table_text = _table_to_text(block)
            if table_text:
                blocks.append(table_text)

    if not blocks:
        raise DocumentParsingError(
            f"Unable to extract text from DOCX {filename!r}: no extractable content found."
        )

    metadata: dict[str, Any] = {
        "format": "docx",
        "filename": filename,
        "paragraph_count": paragraph_count,
        "table_count": table_count,
    }

    return LoadedDocument(
        content="\n\n".join(blocks),
        source=source,
        metadata=metadata,
    )


class DocxDocumentLoader(BaseDocumentLoader):
    """Extracts DOCX text with Markdown headings for chunk section metadata."""

    async def load(self, file_bytes: bytes, filename: str) -> LoadedDocument:
        """Parse ``file_bytes`` off the event loop and return heading-marked text."""
        if not file_bytes.strip():
            raise DocumentParsingError(f"Unable to read DOCX {filename!r}: file is empty.")

        return await asyncio.to_thread(_extract_docx_text, file_bytes, filename)
