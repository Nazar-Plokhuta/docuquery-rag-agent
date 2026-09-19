"""RAG prompt construction and anti-hallucination guardrails.

This module owns the boundary between retrieved context and LLM input.
It enforces two invariants at prompt-design time rather than post-processing:

1. **Strict grounding** — the system prompt prohibits the model from using
   any knowledge beyond the explicitly provided context chunks.

2. **Mandatory citations** — every factual claim must be followed by a
   citation token in the exact format ``[Source: <source>, Section: <section>]``.
   This format is enforced by prompt instruction, not regex substitution.

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

You MUST follow these rules strictly and without exception:

1. Base your answer EXCLUSIVELY on the context chunks provided in the user message.
   Do not use any external knowledge, assumptions, training data, or extrapolations
   beyond what is explicitly stated in the provided context.

2. If the provided context does not contain sufficient information to answer the
   question, output EXACTLY the following phrase and nothing else:
   {FALLBACK_REFUSAL_MESSAGE}

3. Do not fabricate facts, invent sources, hallucinate details, or speculate
   beyond the information present in the context.

4. After every factual claim you make, you MUST immediately append a citation in
   this exact format (no deviations):
   [Source: <source>, Section: <section>]
   where <source> is the document filename and <section> is the Markdown section
   heading from which the information was derived. If the section is unknown,
   use an empty string for <section>."""


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
        f"--- BEGIN CONTEXT ---\n{combined_context}\n--- END CONTEXT ---\n\n"
        f"Question: {query}"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
