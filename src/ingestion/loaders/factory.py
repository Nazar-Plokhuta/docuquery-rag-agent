"""Factory for resolving format-specific document loaders by file extension."""

from __future__ import annotations

from pathlib import Path

from src.core.exceptions import UnsupportedFileTypeError
from src.ingestion.loaders.base import BaseDocumentLoader
from src.ingestion.loaders.csv_loader import CsvDocumentLoader
from src.ingestion.loaders.docx import DocxDocumentLoader
from src.ingestion.loaders.excel import ExcelDocumentLoader
from src.ingestion.loaders.html import HtmlDocumentLoader
from src.ingestion.loaders.pdf import PdfDocumentLoader
from src.ingestion.loaders.text import TextDocumentLoader

_SUPPORTED_EXTENSIONS: dict[str, type[BaseDocumentLoader]] = {
    ".txt": TextDocumentLoader,
    ".md": TextDocumentLoader,
    ".pdf": PdfDocumentLoader,
    ".docx": DocxDocumentLoader,
    ".csv": CsvDocumentLoader,
    ".xlsx": ExcelDocumentLoader,
    ".html": HtmlDocumentLoader,
    ".htm": HtmlDocumentLoader,
}


class DocumentLoaderFactory:
    """Selects a ``BaseDocumentLoader`` implementation from a file extension."""

    @staticmethod
    def get_loader(extension: str) -> BaseDocumentLoader:
        """Return a loader instance for ``extension`` (including the leading dot).

        Args:
            extension: Lowercase or mixed-case suffix such as ``".pdf"``.

        Raises:
            UnsupportedFileTypeError: When no loader is registered for ``extension``.
        """
        normalized = extension.lower()
        if not normalized.startswith("."):
            normalized = Path(normalized).suffix.lower()
        loader_cls = _SUPPORTED_EXTENSIONS.get(normalized)
        if loader_cls is None:
            supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{normalized}'. Accepted extensions: {supported}."
            )
        return loader_cls()
