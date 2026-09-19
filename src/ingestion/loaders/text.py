"""Plain-text and Markdown document loader."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.loaders.base import BaseDocumentLoader, LoadedDocument


class TextDocumentLoader(BaseDocumentLoader):
    """Decodes ``.txt`` and ``.md`` uploads as UTF-8 text."""

    async def load(self, file_bytes: bytes, filename: str) -> LoadedDocument:
        """Decode ``file_bytes`` as UTF-8, replacing undecodable sequences when needed."""
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = file_bytes.decode("utf-8", errors="replace")

        source = Path(filename).stem
        return LoadedDocument(
            content=content,
            source=source,
            metadata={"format": "text", "filename": filename},
        )
