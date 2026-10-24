"""Unit tests for the LiteLLM helper module (credential routing, flags, embeddings)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import litellm
import pytest

from src.config.settings import Settings
from src.core.llm import configure_litellm, embed_texts, provider_credentials


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "openai_api_key": "sk-test",
        "openai_base_url": None,
        "openai_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "database_url": "sqlite+aiosqlite:///:memory:",
    }
    return Settings(**{**base, **overrides})  # type: ignore[arg-type]


class TestProviderCredentials:
    """OpenAI credentials must only reach OpenAI-routed models."""

    def test_openai_model_receives_key(self) -> None:
        assert provider_credentials("gpt-4o-mini", _settings()) == {"api_key": "sk-test"}

    def test_openai_model_receives_custom_base_url(self) -> None:
        settings = _settings(openai_base_url="http://vllm:8000/v1")
        assert provider_credentials("openai/my-model", settings) == {
            "api_key": "sk-test",
            "api_base": "http://vllm:8000/v1",
        }

    @pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4-5", "openrouter/x/y"])
    def test_other_providers_never_receive_openai_credentials(self, model: str) -> None:
        settings = _settings(openai_base_url="http://vllm:8000/v1")
        assert provider_credentials(model, settings) == {}

    def test_unresolvable_model_receives_no_credentials(self) -> None:
        assert provider_credentials("no-such-provider-model-xyz", _settings()) == {}


class TestConfigureLiteLLM:
    def test_applies_global_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "drop_params", False)
        monkeypatch.setattr(litellm, "telemetry", True, raising=False)

        configure_litellm(_settings(litellm_drop_params=True))

        assert litellm.drop_params is True
        assert litellm.telemetry is False

    def test_drop_params_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "drop_params", True)
        configure_litellm(_settings(litellm_drop_params=False))
        assert litellm.drop_params is False

    def test_drop_params_defaults_to_true(self) -> None:
        assert _settings().litellm_drop_params is True


class TestEmbedTexts:
    async def test_returns_vectors_for_dict_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = MagicMock()
        response.data = [{"embedding": [1.0, 2.0]}, {"embedding": [3.0, 4.0]}]
        mock = AsyncMock(return_value=response)
        monkeypatch.setattr("litellm.aembedding", mock)

        vectors = await embed_texts(["a", "b"], _settings())

        assert vectors == [[1.0, 2.0], [3.0, 4.0]]
        kwargs = mock.await_args.kwargs
        assert kwargs["model"] == "text-embedding-3-small"
        assert kwargs["input"] == ["a", "b"]
        assert kwargs["api_key"] == "sk-test"

    async def test_returns_vectors_for_object_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = MagicMock()
        response.data = [SimpleNamespace(embedding=[0.5])]
        monkeypatch.setattr("litellm.aembedding", AsyncMock(return_value=response))

        assert await embed_texts(["a"], _settings()) == [[0.5]]
