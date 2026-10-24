"""PDF document loader with page-level section markers for citation provenance."""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from src.core.exceptions import DocumentParsingError
from src.ingestion.loaders.base import BaseDocumentLoader, LoadedDocument

logger = logging.getLogger(__name__)


def _extract_pdf_text(file_bytes: bytes, filename: str) -> LoadedDocument:
    """Synchronous PDF parse — intended to run inside ``asyncio.to_thread``."""
    source = Path(filename).stem
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            raise DocumentParsingError(
                f"Unable to read PDF {filename!r}: encrypted documents are not supported."
            )

        page_sections: list[str] = []
        for page_index, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                page_sections.append(f"## Page {page_index}\n\n{page_text}")
        page_count = len(reader.pages)
    except DocumentParsingError:
        raise
    except Exception as exc:
        # pypdf parses lazily, so a damaged stream fails during page iteration or
        # text extraction, not at construction, and not always with ``PdfReadError``
        # (it also raises ``PdfStreamError``, ``LimitReachedError``, ``zlib.error``,
        # ``KeyError``, ...). Untrusted input must never surface as a 500, so every
        # parser fault is folded into one client-facing domain error.
        logger.warning("pypdf failed on %r: %s: %s", filename, type(exc).__name__, exc)
        raise DocumentParsingError(
            f"Unable to read PDF {filename!r}: the file may be corrupted or invalid."
        ) from exc

    if not page_sections:
        raise DocumentParsingError(
            f"Unable to extract text from PDF {filename!r}: no readable page content found."
        )

    metadata: dict[str, Any] = {
        "format": "pdf",
        "filename": filename,
        "page_count": page_count,
        "non_empty_page_count": len(page_sections),
    }

    return LoadedDocument(
        content="\n\n".join(page_sections),
        source=source,
        metadata=metadata,
    )


class PdfDocumentLoader(BaseDocumentLoader):
    """Extracts PDF text page-by-page with Markdown headings for chunk section metadata."""

    async def load(self, file_bytes: bytes, filename: str) -> LoadedDocument:
        """Parse ``file_bytes`` off the event loop and return page-marked text."""
        if not file_bytes.strip():
            raise DocumentParsingError(f"Unable to read PDF {filename!r}: file is empty.")

        return await asyncio.to_thread(_extract_pdf_text, file_bytes, filename)
