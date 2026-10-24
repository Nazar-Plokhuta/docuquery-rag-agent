"""RAG prompt construction and anti-hallucination guardrails.

This module owns the boundary between retrieved context and LLM input.
It enforces two invariants at prompt-design time rather than post-processing:

1. **Strict grounding** — the system prompt prohibits the model from using
   any knowledge beyond the explicitly provided context chunks.

2. **Mandatory citations** — every factual claim must be followed by a
   citation token in the exact format ``[Source: <source>, Section: <section>]``.
   This format is enforced by prompt instruction, not regex substitution.

Grounding is deliberately language-agnostic: the retrieval gate is tuned for recall
(multilingual embeddings score cross-language matches low), so the prompt is the
precision gate and must refuse when the context is irrelevant.

No I/O, no driver imports, and no business logic belong here.
"""

from __future__ import annotations

from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Deterministic fallback phrase
# ---------------------------------------------------------------------------

FALLBACK_REFUSAL_MESSAGE: str = (
    "I am sorry, but the provided documentation does not contain sufficient "
    "information to answer your question."
)

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: str = f"""You are a precise document question-answering assistant.

Follow these rules:

1. Base every statement ONLY on the context chunks provided in the user message.
   Never use outside knowledge, and never invent facts, figures, names or sources.

2. The question and the context may be written in different languages. Match them by
   meaning, and write the answer in the language of the question. Translating,
   paraphrasing and combining facts from several chunks is expected; adding facts
   that no chunk states is forbidden.

3. Answer whenever the context contains information that addresses the question,
   even if it is only partial. State clearly which part of the question the context
   does not cover instead of refusing the whole question.

4. Only when the context contains nothing relevant to the question, output EXACTLY the
   following phrase and nothing else:
   {FALLBACK_REFUSAL_MESSAGE}

5. After every factual claim, append the citation of the chunk it came from, copied
   verbatim from that chunk's header line, in this format:
   [Source: <source>, Section: <section>]
   Never cite a chunk that does not support the claim."""


def format_context_block(source: str, section: str, content: str) -> str:
    """Prefix a chunk with a provenance header the model can copy into citations.

    Without the header the model sees only anonymous text and has no truthful
    ``<source>``/``<section>`` to cite, so it would have to fabricate them.

    Args:
        source: Document identifier recorded at ingestion time.
        section: Section label recorded at ingestion time; may be empty.
        content: Raw chunk text.

    Returns:
        The chunk text preceded by its ``[Source: ..., Section: ...]`` header line.
    """
    return f"[Source: {source}, Section: {section}]\n{content}"


def build_rag_prompt(
    query: str,
    retrieved_contexts: Sequence[str],
) -> list[dict[str, str]]:
    """Construct the two-message Chat Completions payload for a RAG query.

    The system message embeds the anti-hallucination ruleset and the mandatory
    citation format.  The user message wraps all retrieved context chunks inside
    ``--- BEGIN CONTEXT ---`` / ``--- END CONTEXT ---`` delimiters and appends
    the user's question.

    This separation keeps the invariant rules in a position the model weights
    most heavily (the system turn) while placing variable content (context +
    query) in the user turn, matching the canonical instruction-following
    prompt pattern for GPT-4-class models.

    Args:
        query: The raw user question to answer from the retrieved context.
        retrieved_contexts: Ordered sequence of chunk content strings to inject
            into the prompt.  Consumers must ensure this sequence has already
            been token-budgeted before calling this function.

    Returns:
        A two-element list of ``{"role": ..., "content": ...}`` dicts ready
        for direct submission to ``openai.chat.completions.create(messages=...)``.
    """
    combined_context = "\n\n".join(retrieved_contexts)
    user_content = (
        f"--- BEGIN CONTEXT ---\n{combined_context}\n--- END CONTEXT ---\n\nQuestion: {query}"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
