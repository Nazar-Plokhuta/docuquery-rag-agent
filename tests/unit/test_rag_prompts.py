"""Unit tests for RAG prompt construction.

Verifies:
- The returned message list has exactly two entries with correct roles.
- The system message embeds the fallback refusal phrase verbatim.
- The system message contains explicit citation-format instructions.
- The system message prohibits outside knowledge and fabrication.
- The user message wraps context inside the correct delimiters.
- The user message appends the original query.
"""

from __future__ import annotations

from src.core.rag.prompts import FALLBACK_REFUSAL_MESSAGE, build_rag_prompt

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_QUERY = "What is the deployment strategy for the RAG agent?"
_CONTEXTS = [
    "The agent is deployed as a Docker container on Kubernetes.",
    "Blue-green deployment is used to achieve zero-downtime upgrades.",
]


# ---------------------------------------------------------------------------
# TestBuildRagPromptStructure
# ---------------------------------------------------------------------------


class TestBuildRagPromptStructure:
    """Verify the top-level structure of the returned message list."""

    def test_returns_exactly_two_messages(self) -> None:
        """build_rag_prompt must return a list with exactly two messages."""
        messages = build_rag_prompt(_QUERY, _CONTEXTS)
        assert len(messages) == 2

    def test_first_message_role_is_system(self) -> None:
        """The first message must have role 'system'."""
        messages = build_rag_prompt(_QUERY, _CONTEXTS)
        assert messages[0]["role"] == "system"

    def test_second_message_role_is_user(self) -> None:
        """The second message must have role 'user'."""
        messages = build_rag_prompt(_QUERY, _CONTEXTS)
        assert messages[1]["role"] == "user"

    def test_all_messages_have_content_key(self) -> None:
        """Both messages must carry a non-empty 'content' key."""
        messages = build_rag_prompt(_QUERY, _CONTEXTS)
        for msg in messages:
            assert "content" in msg
            assert isinstance(msg["content"], str)
            assert len(msg["content"]) > 0

    def test_works_with_single_context(self) -> None:
        """A single-element context sequence must not raise."""
        messages = build_rag_prompt(_QUERY, ["Only context chunk."])
        assert len(messages) == 2

    def test_works_with_empty_context_list(self) -> None:
        """An empty context sequence must not raise (edge case for callers)."""
        messages = build_rag_prompt(_QUERY, [])
        assert len(messages) == 2


# ---------------------------------------------------------------------------
# TestSystemPromptAntiHallucination
# ---------------------------------------------------------------------------


class TestSystemPromptAntiHallucination:
    """Verify anti-hallucination invariants embedded in the system prompt."""

    def _system_content(self) -> str:
        return build_rag_prompt(_QUERY, _CONTEXTS)[0]["content"]

    def test_system_prompt_contains_fallback_refusal_phrase(self) -> None:
        """The exact FALLBACK_REFUSAL_MESSAGE string must appear in the system prompt."""
        assert FALLBACK_REFUSAL_MESSAGE in self._system_content()

    def test_system_prompt_contains_citation_format(self) -> None:
        """The system prompt must specify the [Source: ..., Section: ...] citation format."""
        content = self._system_content()
        assert "[Source:" in content
        assert "Section:" in content

    def test_system_prompt_forbids_external_knowledge(self) -> None:
        """The system prompt must explicitly prohibit using external knowledge."""
        content = self._system_content()
        # Any of these phrases indicate the restriction is present.
        restriction_phrases = [
            "external knowledge",
            "outside knowledge",
            "training data",
            "Do not use any external",
        ]
        assert any(phrase in content for phrase in restriction_phrases), (
            "System prompt must explicitly forbid use of external knowledge."
        )

    def test_system_prompt_forbids_fabrication(self) -> None:
        """The system prompt must explicitly prohibit fabricating facts."""
        content = self._system_content()
        fabrication_phrases = [
            "fabricate",
            "hallucinate",
            "invent",
            "speculate",
        ]
        assert any(phrase in content for phrase in fabrication_phrases), (
            "System prompt must explicitly forbid fabricating facts."
        )

    def test_system_prompt_instructs_exclusive_context_use(self) -> None:
        """The system prompt must instruct the model to answer from context only."""
        content = self._system_content()
        grounding_phrases = [
            "EXCLUSIVELY",
            "exclusively",
            "ONLY",
            "only based on",
            "strictly",
        ]
        assert any(phrase in content for phrase in grounding_phrases), (
            "System prompt must instruct exclusive reliance on provided context."
        )


# ---------------------------------------------------------------------------
# TestUserMessageContextFraming
# ---------------------------------------------------------------------------


class TestUserMessageContextFraming:
    """Verify context delimiters and query placement in the user message."""

    def _user_content(self) -> str:
        return build_rag_prompt(_QUERY, _CONTEXTS)[1]["content"]

    def test_user_message_contains_begin_context_delimiter(self) -> None:
        """User message must open with the --- BEGIN CONTEXT --- delimiter."""
        assert "--- BEGIN CONTEXT ---" in self._user_content()

    def test_user_message_contains_end_context_delimiter(self) -> None:
        """User message must close context with the --- END CONTEXT --- delimiter."""
        assert "--- END CONTEXT ---" in self._user_content()

    def test_begin_delimiter_precedes_end_delimiter(self) -> None:
        """BEGIN CONTEXT delimiter must appear before END CONTEXT delimiter."""
        content = self._user_content()
        begin_pos = content.index("--- BEGIN CONTEXT ---")
        end_pos = content.index("--- END CONTEXT ---")
        assert begin_pos < end_pos

    def test_query_is_present_in_user_message(self) -> None:
        """The original query string must appear in the user message."""
        assert _QUERY in self._user_content()

    def test_query_appears_after_end_context_delimiter(self) -> None:
        """The user question must follow the END CONTEXT delimiter."""
        content = self._user_content()
        end_pos = content.index("--- END CONTEXT ---")
        query_pos = content.index(_QUERY)
        assert query_pos > end_pos

    def test_all_context_strings_present_in_user_message(self) -> None:
        """Every provided context string must appear verbatim in the user message."""
        content = self._user_content()
        for ctx in _CONTEXTS:
            assert ctx in content
