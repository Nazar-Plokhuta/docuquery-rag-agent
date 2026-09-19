"""Unit tests for TokenSlidingWindowChunker.

Validates:
- Chunk ID format and sequential ordering.
- Token-size boundary adherence (no chunk exceeds chunk_size).
- Overlap correctness between adjacent chunks.
- Markdown section header capture in metadata.
- Graceful handling of empty / sub-threshold documents.
- Agreement between ``chunk.token_count`` and ``chunk.metadata["token_count"]``.
- ValueError on invalid overlap configuration.
"""

from __future__ import annotations

import re

import pytest
import tiktoken

from src.ingestion.chunker import TokenSlidingWindowChunker

_ENC: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    """Return the cl100k_base token count of ``text``."""
    return len(_ENC.encode(text))


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_chunker(chunk_size: int = 100, chunk_overlap: int = 20) -> TokenSlidingWindowChunker:
    return TokenSlidingWindowChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _long_text(n_words: int = 300) -> str:
    """Generate a simple space-separated word sequence for deterministic tests."""
    return " ".join(f"word{i}" for i in range(n_words))


# ---------------------------------------------------------------------------
# Chunk ID format
# ---------------------------------------------------------------------------


class TestChunkIdFormat:
    def test_ids_match_pattern(self) -> None:
        """Chunk IDs must follow ``{document_id}#c{index:04d}``."""
        chunker = _make_chunker()
        chunks = chunker.chunk_document("doc-001", _long_text())
        pattern = re.compile(r"^doc-001#c\d{4}$")
        for chunk in chunks:
            assert pattern.match(chunk.chunk_id), (
                f"ID {chunk.chunk_id!r} does not match expected pattern."
            )

    def test_ids_are_zero_based_and_sequential(self) -> None:
        """Chunk indices must start at 0 and increment by 1 with no gaps."""
        chunker = _make_chunker()
        chunks = chunker.chunk_document("seq-doc", _long_text())
        for expected_idx, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"seq-doc#c{expected_idx:04d}", (
                f"Expected seq-doc#c{expected_idx:04d}, got {chunk.chunk_id!r}."
            )

    def test_document_id_embedded_in_chunk_id(self) -> None:
        """The document ID prefix must match exactly the supplied ``document_id``."""
        chunker = _make_chunker()
        doc_id = "annual-report-2026"
        chunks = chunker.chunk_document(doc_id, _long_text())
        assert all(c.chunk_id.startswith(f"{doc_id}#c") for c in chunks)


# ---------------------------------------------------------------------------
# Chunk size boundary
# ---------------------------------------------------------------------------


class TestChunkSizeBoundary:
    def test_no_chunk_exceeds_chunk_size(self) -> None:
        """Every chunk's token_count must be ≤ configured chunk_size."""
        chunk_size = 80
        chunker = TokenSlidingWindowChunker(chunk_size=chunk_size, chunk_overlap=15)
        text = "The quick brown fox jumps over the lazy dog. " * 100

        for chunk in chunker.chunk_document("size-test", text):
            assert chunk.token_count <= chunk_size, (
                f"Chunk {chunk.chunk_id} has {chunk.token_count} tokens; "
                f"limit is {chunk_size}."
            )

    def test_token_count_matches_actual_encoding(self) -> None:
        """``chunk.token_count`` must equal tiktoken's count of the chunk text."""
        chunker = _make_chunker(chunk_size=80, chunk_overlap=10)
        for chunk in chunker.chunk_document("encode-check", _long_text()):
            actual = _token_count(chunk.content)
            assert chunk.token_count == actual, (
                f"Declared token_count={chunk.token_count} but tiktoken says {actual}."
            )


# ---------------------------------------------------------------------------
# Overlap correctness
# ---------------------------------------------------------------------------


class TestChunkOverlap:
    def test_overlap_tokens_shared_between_adjacent_chunks(self) -> None:
        """Adjacent chunks must share ≈ chunk_overlap tokens at their boundary."""
        chunk_size = 60
        chunk_overlap = 12
        chunker = TokenSlidingWindowChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = chunker.chunk_document("overlap-test", _long_text(400))

        assert len(chunks) >= 2, "Need at least 2 chunks to verify overlap."

        tokens_0 = _ENC.encode(chunks[0].content)
        tokens_1 = _ENC.encode(chunks[1].content)

        # The tail of chunk 0 should equal the head of chunk 1 for overlap tokens.
        tail = tokens_0[-chunk_overlap:]
        head = tokens_1[:chunk_overlap]
        shared = sum(a == b for a, b in zip(tail, head, strict=True))

        # Allow ±2 tolerance for tiktoken boundary effects.
        assert shared >= chunk_overlap - 2, (
            f"Expected ≥{chunk_overlap - 2} shared tokens, found {shared}."
        )

    def test_multiple_chunk_overlaps_are_consistent(self) -> None:
        """Overlap should be consistent across all consecutive chunk pairs."""
        chunk_size = 60
        chunk_overlap = 10
        chunker = TokenSlidingWindowChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = chunker.chunk_document("multi-overlap", _long_text(500))

        for prev, nxt in zip(chunks, chunks[1:], strict=False):
            tokens_prev = _ENC.encode(prev.content)
            tokens_next = _ENC.encode(nxt.content)
            tail = tokens_prev[-chunk_overlap:]
            head = tokens_next[:chunk_overlap]
            shared = sum(a == b for a, b in zip(tail, head, strict=True))
            assert shared >= chunk_overlap - 2, (
                f"Inconsistent overlap between {prev.chunk_id} and {nxt.chunk_id}: "
                f"shared={shared}, expected≥{chunk_overlap - 2}."
            )


# ---------------------------------------------------------------------------
# Metadata — section headers
# ---------------------------------------------------------------------------


class TestMarkdownSectionMetadata:
    def test_first_chunk_captures_h2_section(self) -> None:
        """Chunks following an H2 heading must record its text as ``section``."""
        chunker = _make_chunker(chunk_size=60, chunk_overlap=5)
        text = "## Introduction\n" + "Content sentence. " * 40
        chunks = chunker.chunk_document("md-doc", text)

        # The first chunk starts after the Introduction heading.
        assert chunks[0].metadata.get("section") == "Introduction", (
            f"Expected section='Introduction', got {chunks[0].metadata.get('section')!r}."
        )

    def test_section_transitions_across_chunks(self) -> None:
        """Chunks must reflect the section heading that precedes them."""
        chunker = _make_chunker(chunk_size=40, chunk_overlap=5)
        intro_body = "Intro sentence. " * 10
        methods_body = "Methods sentence. " * 10
        text = (
            "## Introduction\n"
            + intro_body
            + "\n## Methods\n"
            + methods_body
        )
        chunks = chunker.chunk_document("sections-doc", text)

        # At least some chunks should be in 'Methods' after the heading.
        sections = {c.metadata.get("section") for c in chunks}
        assert "Methods" in sections, (
            f"Expected 'Methods' section to appear in metadata; got {sections!r}."
        )

    def test_no_heading_yields_empty_section(self) -> None:
        """Documents without headings should have ``section=''`` on every chunk."""
        chunker = _make_chunker(chunk_size=60, chunk_overlap=5)
        text = "Plain paragraph. " * 100
        chunks = chunker.chunk_document("no-heading", text)

        for chunk in chunks:
            assert chunk.metadata.get("section") == "", (
                f"Expected empty section, got {chunk.metadata.get('section')!r}."
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string_returns_empty_list(self) -> None:
        """Empty input must return an empty list without raising."""
        assert _make_chunker().chunk_document("empty", "") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        """Whitespace-only input must return an empty list without raising."""
        assert _make_chunker().chunk_document("ws", "   \n\t  ") == []

    def test_short_document_yields_single_chunk(self) -> None:
        """A document shorter than ``chunk_size`` must produce exactly one chunk."""
        chunker = TokenSlidingWindowChunker(chunk_size=400, chunk_overlap=50)
        chunks = chunker.chunk_document("short", "Hello world.")
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "short#c0000"

    def test_document_exactly_chunk_size_yields_single_chunk(self) -> None:
        """A document of exactly ``chunk_size`` tokens must produce one chunk."""
        chunk_size = 50
        chunker = TokenSlidingWindowChunker(chunk_size=chunk_size, chunk_overlap=5)
        # Construct text that tokenises to exactly ``chunk_size`` tokens.
        tokens = list(range(chunk_size))
        text = _ENC.decode(tokens)
        chunks = chunker.chunk_document("exact", text)
        assert len(chunks) == 1

    def test_invalid_overlap_raises_value_error(self) -> None:
        """``chunk_overlap >= chunk_size`` must raise ``ValueError`` on construction."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            TokenSlidingWindowChunker(chunk_size=50, chunk_overlap=50)

    def test_overlap_greater_than_size_raises_value_error(self) -> None:
        """``chunk_overlap > chunk_size`` must also raise ``ValueError``."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            TokenSlidingWindowChunker(chunk_size=50, chunk_overlap=60)


# ---------------------------------------------------------------------------
# Metadata field consistency
# ---------------------------------------------------------------------------


class TestMetadataConsistency:
    def test_token_count_agrees_between_field_and_metadata(self) -> None:
        """``chunk.token_count`` and ``chunk.metadata['token_count']`` must agree."""
        chunker = _make_chunker(chunk_size=80, chunk_overlap=10)
        for chunk in chunker.chunk_document("meta-check", _long_text()):
            assert chunk.token_count == chunk.metadata["token_count"], (
                f"Mismatch on {chunk.chunk_id}: field={chunk.token_count}, "
                f"metadata={chunk.metadata['token_count']}."
            )

    def test_chunk_index_in_metadata_matches_id(self) -> None:
        """``metadata['chunk_index']`` must equal the numeric suffix in ``chunk_id``."""
        chunker = _make_chunker()
        for chunk in chunker.chunk_document("idx-check", _long_text()):
            suffix = int(chunk.chunk_id.split("#c")[1])
            assert chunk.metadata["chunk_index"] == suffix, (
                f"chunk_id={chunk.chunk_id!r} but metadata chunk_index="
                f"{chunk.metadata['chunk_index']}."
            )

    def test_document_id_in_metadata_matches_argument(self) -> None:
        """``metadata['document_id']`` must equal the ``document_id`` argument."""
        doc_id = "my-document"
        chunker = _make_chunker()
        for chunk in chunker.chunk_document(doc_id, _long_text()):
            assert chunk.metadata["document_id"] == doc_id
