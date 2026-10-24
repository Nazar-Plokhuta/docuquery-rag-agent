"""Token-bounded sliding-window text chunker for document ingestion.

Uses tiktoken to measure token counts precisely, ensuring each chunk stays
within the configured budget.  A configurable overlap preserves semantic
context across chunk boundaries, reducing the risk of splitting coherent
passages mid-thought.

Algorithm:
1. Encode the full document text into a flat token list using cl100k_base.
2. Slide a window of ``chunk_size`` tokens forward by
   ``chunk_size - chunk_overlap`` tokens per step.
3. Decode each window back to UTF-8 text.
4. Derive the character offset of each window start by decoding the prefix
   token slice, then scan backwards through the original text for the nearest
   Markdown heading — this is stored in the chunk metadata as ``section``.
5. Assign a deterministic ID: ``{document_id}#c{chunk_index:04d}``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import tiktoken

from src.core.models import DocumentChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# cl100k_base is used by text-embedding-3-small and GPT-4 family models.
_ENCODING_NAME: str = "cl100k_base"

# ATX heading pattern (levels 1–6) anchored to line starts.
_HEADING_RE: re.Pattern[str] = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


@dataclass
class TokenSlidingWindowChunker:
    """Splits a document into overlapping token-bounded ``DocumentChunk`` objects.

    Attributes:
        chunk_size: Target maximum token count per chunk (default 400).
        chunk_overlap: Number of tokens shared between adjacent chunks (default 50).
            Must be strictly less than ``chunk_size``.
    """

    chunk_size: int = 400
    chunk_overlap: int = 50
    # Populated in __post_init__; excluded from dataclass __repr__ for brevity.
    _encoding: tiktoken.Encoding = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than "
                f"chunk_size ({self.chunk_size})."
            )
        self._encoding = tiktoken.get_encoding(_ENCODING_NAME)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _nearest_containing_section(self, text: str, chunk_end_offset: int) -> str:
        """Return the nearest Markdown heading that precedes or begins this chunk.

        Scans the slice ``text[:chunk_end_offset]`` (i.e. everything up to and
        including the chunk's last character) so that a heading which *opens*
        the chunk — rather than appearing strictly before it — is still captured.
        This correctly attributes the very first chunk of a document when the
        document begins with a heading.

        Args:
            text: Full document text.
            chunk_end_offset: Character position immediately after the chunk's
                last character in ``text``.

        Returns:
            Heading text (without ``#`` prefix), or ``""`` if none is found.
        """
        window = text[:chunk_end_offset]
        matches = _HEADING_RE.findall(window)
        return matches[-1] if matches else ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, document_id: str, text: str) -> list[DocumentChunk]:
        """Split ``text`` into overlapping, token-bounded ``DocumentChunk`` objects.

        Args:
            document_id: Unique identifier for the parent document
                (typically the source file stem).
            text: Full plain-text or Markdown document content.

        Returns:
            Ordered list of ``DocumentChunk`` instances.  Returns ``[]`` for
            empty or whitespace-only input without raising.
        """
        if not text.strip():
            return []

        tokens: list[int] = self._encoding.encode(text)
        step: int = self.chunk_size - self.chunk_overlap
        chunks: list[DocumentChunk] = []

        for chunk_index, start_token in enumerate(range(0, len(tokens), step)):
            end_token: int = min(start_token + self.chunk_size, len(tokens))
            window_tokens: list[int] = tokens[start_token:end_token]
            chunk_text: str = self._encoding.decode(window_tokens)

            # Reconstruct the character window ending at this chunk's last character
            # so that headings which *open* the chunk are captured (e.g. the very
            # first chunk of a document that starts with an ATX heading).
            prefix_text: str = (
                self._encoding.decode(tokens[:start_token]) if start_token > 0 else ""
            )
            chunk_end_offset: int = len(prefix_text) + len(chunk_text)
            section: str = self._nearest_containing_section(text, chunk_end_offset)

            chunk_id: str = f"{document_id}#c{chunk_index:04d}"
            token_count: int = len(window_tokens)

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=chunk_text,
                    metadata={
                        "document_id": document_id,
                        "section": section,
                        "chunk_index": chunk_index,
                        "token_count": token_count,
                    },
                    token_count=token_count,
                )
            )

        logger.debug(
            "Chunked document %r → %d chunks (size=%d, overlap=%d).",
            document_id,
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks
