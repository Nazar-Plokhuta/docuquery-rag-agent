"""Token budget management for RAG context assembly.

Enforces the architectural invariant (§6.3) that no unbounded context is
injected into LLM prompts.  Every context payload is measured with tiktoken
and truncated to fit within a configurable token ceiling before the prompt
is handed to the OpenAI API.

The ``TokenBudgetManager`` is intentionally synchronous.  tiktoken's BPE
encode/decode is CPU-bound but fast (typically < 1 ms for chunks bounded
to 400 tokens), so dispatching it to a thread pool would introduce more
scheduling overhead than it saves.  The event loop is not meaningfully
blocked at normal document-chunk sizes.
"""

from __future__ import annotations

from collections.abc import Sequence

import tiktoken

from src.core.models import RetrievalResult


class TokenBudgetManager:
    """Measure and constrain token counts for RAG context assembly.

    Attributes:
        _encoding: Cached ``tiktoken`` encoding instance, constructed once to
            avoid repeated BPE vocabulary loading on every call.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """Initialise the manager with a cached tiktoken encoding.

        ``cl100k_base`` is the BPE encoding shared by OpenAI's
        ``text-embedding-3-small`` and the GPT-4 family.  Using the same
        tokeniser at ingestion time, at retrieval time, and at prompt
        assembly time guarantees that token counts are consistent across the
        entire pipeline.

        Args:
            encoding_name: tiktoken encoding name.  Defaults to
                ``"cl100k_base"`` to match the ingestion chunker.
        """
        self._encoding: tiktoken.Encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Return the token count of ``text`` under the cached encoding.

        Args:
            text: UTF-8 string to tokenise.

        Returns:
            Number of BPE tokens produced by the encoding.
        """
        return len(self._encoding.encode(text))

    def fit_contexts_to_budget(
        self,
        chunks: Sequence[RetrievalResult],
        max_context_tokens: int = 2500,
    ) -> list[RetrievalResult]:
        """Select the highest-scored chunks that collectively fit within the token budget.

        Chunks are sorted by descending similarity score so the most relevant
        context is prioritised when the budget is tight.  Iteration halts as
        soon as the next candidate would push the cumulative token count over
        ``max_context_tokens``; all lower-ranked candidates are dropped.

        This greedy algorithm is O(n log n) due to the sort and guarantees
        that the selected set is always the highest-scored prefix that fits
        within the budget.

        Args:
            chunks: Candidate retrieval results to filter.  May be unsorted.
            max_context_tokens: Upper bound on the combined token count of all
                selected chunk contents.  Defaults to 2 500 tokens, leaving
                ample room for the system prompt and completion within a
                standard 8 k context window.

        Returns:
            Ordered list of ``RetrievalResult`` instances whose combined
            content token count does not exceed ``max_context_tokens``,
            sorted by descending score.
        """
        sorted_chunks = sorted(chunks, key=lambda r: r.score, reverse=True)
        selected: list[RetrievalResult] = []
        cumulative_tokens: int = 0

        for result in sorted_chunks:
            chunk_tokens = self.count_tokens(result.chunk.content)
            if cumulative_tokens + chunk_tokens > max_context_tokens:
                # Adding this chunk would exceed the ceiling — discard it and
                # all remaining lower-scored chunks.
                break
            cumulative_tokens += chunk_tokens
            selected.append(result)

        return selected
