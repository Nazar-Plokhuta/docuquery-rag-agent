"""Abstract document loader contract and loaded-document value object."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LoadedDocument:
    """Normalized output of a format-specific document loader.

    Attributes:
        content: Full extracted or decoded text ready for chunking.
        source: Human-readable source identifier (typically the file stem).
        metadata: Format-specific provenance fields (page count, encoding, etc.).
    """

    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDocumentLoader(ABC):
    """Strategy interface for turning raw file bytes into ``LoadedDocument`` instances."""

    @abstractmethod
    async def load(self, file_bytes: bytes, filename: str) -> LoadedDocument:
        """Extract or decode document text from ``file_bytes``.

        Args:
            file_bytes: Raw uploaded or on-disk file contents.
            filename: Original filename (used for source attribution).

        Returns:
            A ``LoadedDocument`` whose ``content`` is ready for the chunker.
        """
