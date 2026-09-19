"""Unit tests for TokenBudgetManager.

Covers:
- Token counting accuracy against a known tiktoken encoding.
- Budget-fitting: all chunks admitted when budget is ample.
- Budget-fitting: overflow chunks dropped once the ceiling is reached.
- Budget-fitting: chunks are selected in descending score order.
"""

from __future__ import annotations

import tiktoken

from src.core.models import DocumentChunk, RetrievalResult
from src.core.rag.token_counter import TokenBudgetManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(
    document_id: str,
    content: str,
    chunk_index: int = 0,
    score: float = 0.9,
) -> RetrievalResult:
    """Construct a minimal RetrievalResult for use in budget tests."""
    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoding.encode(content))
    chunk = DocumentChunk(
        chunk_id=f"{document_id}#c{chunk_index:04d}",
        document_id=document_id,
        content=content,
        metadata={
            "document_id": document_id,
            "section": "Introduction",
            "chunk_index": chunk_index,
            "token_count": token_count,
        },
        token_count=token_count,
    )
    return RetrievalResult(chunk=chunk, score=score)


# ---------------------------------------------------------------------------
# TestCountTokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    """Verify that count_tokens produces the same count as a direct tiktoken call."""

    def test_count_tokens_matches_tiktoken(self) -> None:
        """Token count must equal the reference tiktoken cl100k_base encode length."""
        text = "The quick brown fox jumps over the lazy dog."
        encoding = tiktoken.get_encoding("cl100k_base")
        expected = len(encoding.encode(text))

        manager = TokenBudgetManager()
        assert manager.count_tokens(text) == expected

    def test_count_tokens_empty_string_returns_zero(self) -> None:
        """Empty string encodes to zero tokens."""
        manager = TokenBudgetManager()
        assert manager.count_tokens("") == 0

    def test_count_tokens_single_word(self) -> None:
        """Single common word should produce exactly one token under cl100k_base."""
        manager = TokenBudgetManager()
        # "hello" is a single BPE token in cl100k_base
        assert manager.count_tokens("hello") == 1

    def test_count_tokens_custom_encoding(self) -> None:
        """Providing an alternative encoding name must load that encoding."""
        # p50k_base is a valid tiktoken encoding; the exact count may differ
        # from cl100k_base, but it must be a positive integer for non-empty text.
        manager = TokenBudgetManager(encoding_name="p50k_base")
        text = "enterprise RAG pipeline"
        assert manager.count_tokens(text) > 0


# ---------------------------------------------------------------------------
# TestFitContextsToBudget
# ---------------------------------------------------------------------------


class TestFitContextsToBudget:
    """Verify the greedy budget-fitting algorithm."""

    def test_all_chunks_admitted_when_budget_is_ample(self) -> None:
        """All chunks are included when their combined token count fits the budget."""
        manager = TokenBudgetManager()
        chunks = [
            _make_chunk("doc-a", "short text", chunk_index=0, score=0.9),
            _make_chunk("doc-b", "another short text", chunk_index=0, score=0.8),
        ]
        result = manager.fit_contexts_to_budget(chunks, max_context_tokens=2500)
        assert len(result) == 2

    def test_overflow_chunk_is_dropped(self) -> None:
        """A chunk that would push cumulative tokens over the budget is dropped."""
        manager = TokenBudgetManager()
        # 50-token content — one token per "word" approximation is an over-estimate;
        # use a string whose token count we know exceeds the micro-budget.
        large_content = " ".join(["token"] * 60)  # ~60 tokens
        small_content = "tiny"  # 1 token

        high_score_chunk = _make_chunk("doc-a", large_content, chunk_index=0, score=0.95)
        low_score_chunk = _make_chunk("doc-b", small_content, chunk_index=0, score=0.80)

        # Budget of 65 tokens admits the large chunk (60 tokens) but not both if
        # combined (61 tokens).  We set budget to exactly 60 to force the drop.
        budget = manager.count_tokens(large_content)
        result = manager.fit_contexts_to_budget(
            [high_score_chunk, low_score_chunk],
            max_context_tokens=budget,
        )
        # Only the high-score chunk fits; low-score is dropped.
        assert len(result) == 1
        assert result[0].chunk.document_id == "doc-a"

    def test_chunks_selected_in_descending_score_order(self) -> None:
        """Higher-scored chunks are admitted first when budget is tight."""
        manager = TokenBudgetManager()
        content = " ".join(["word"] * 30)  # ~30 tokens per chunk

        low_score = _make_chunk("doc-low", content, chunk_index=0, score=0.5)
        high_score = _make_chunk("doc-high", content, chunk_index=0, score=0.9)
        mid_score = _make_chunk("doc-mid", content, chunk_index=0, score=0.7)

        # Budget admits exactly one chunk worth of tokens.
        single_chunk_budget = manager.count_tokens(content)
        result = manager.fit_contexts_to_budget(
            [low_score, high_score, mid_score],
            max_context_tokens=single_chunk_budget,
        )
        assert len(result) == 1
        assert result[0].chunk.document_id == "doc-high"

    def test_empty_input_returns_empty_list(self) -> None:
        """An empty chunk sequence produces an empty result without raising."""
        manager = TokenBudgetManager()
        result = manager.fit_contexts_to_budget([], max_context_tokens=2500)
        assert result == []

    def test_result_preserves_descending_score_ordering(self) -> None:
        """The returned list is sorted by descending score."""
        manager = TokenBudgetManager()
        chunks = [
            _make_chunk("doc-a", "short", chunk_index=0, score=0.6),
            _make_chunk("doc-b", "text", chunk_index=0, score=0.95),
            _make_chunk("doc-c", "hello", chunk_index=0, score=0.75),
        ]
        result = manager.fit_contexts_to_budget(chunks, max_context_tokens=2500)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_zero_budget_drops_all_chunks(self) -> None:
        """A max_context_tokens of zero means no chunk can be included."""
        manager = TokenBudgetManager()
        chunks = [_make_chunk("doc-a", "any content", chunk_index=0, score=0.9)]
        result = manager.fit_contexts_to_budget(chunks, max_context_tokens=0)
        assert result == []
