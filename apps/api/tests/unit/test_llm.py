"""Unit tests for the LiteLLM helper module (credential routing, flags, embeddings)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import litellm
import pytest

from src.config.settings import OPENAI_PLACEHOLDER_API_KEY, Settings
from src.core.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from src.core.llm import (
    audit_llm_configuration,
    configure_litellm,
    embed_texts,
    provider_credentials,
    resolve_provider,
    translate_litellm_errors,
)


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "openai_api_key": "sk-test",
        "openai_base_url": None,
        "openai_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "database_url": "sqlite+aiosqlite:///:memory:",
    }
    # `_env_file=None` keeps the shared repo-root `.env` from leaking into any field a test
    # does not set, so results are identical on a developer machine and in CI.
    return Settings(**{**base, **overrides}, _env_file=None)  # type: ignore[arg-type]


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


def _provider_fault(error_class: type[Exception], provider: str = "openai") -> Exception:
    """Instantiate a LiteLLM exception the way a failed provider call would."""
    request = httpx.Request("POST", "https://provider.test/v1")
    # Two classes have a mandatory transport argument the others default.
    extra: dict[str, object] = {}
    if error_class is litellm.PermissionDeniedError:
        extra["response"] = httpx.Response(403, request=request)
    elif error_class is litellm.APIConnectionError:
        extra["request"] = request
    return error_class(message="provider said no", model="m", llm_provider=provider, **extra)  # type: ignore[call-arg]


class TestTranslateLiteLLMErrors:
    """Provider faults must leave the LLM layer as domain exceptions."""

    @pytest.mark.parametrize(
        ("error_class", "expected"),
        [
            (litellm.AuthenticationError, LLMAuthenticationError),
            (litellm.PermissionDeniedError, LLMAuthenticationError),
            (litellm.RateLimitError, LLMRateLimitError),
            (litellm.Timeout, LLMTimeoutError),
            (litellm.APIConnectionError, LLMUnavailableError),
            (litellm.ServiceUnavailableError, LLMUnavailableError),
            (litellm.InternalServerError, LLMUnavailableError),
            (litellm.BadGatewayError, LLMUnavailableError),
            (litellm.NotFoundError, LLMProviderError),
            (litellm.BadRequestError, LLMProviderError),
        ],
    )
    def test_maps_exception_class_to_domain_type(
        self, error_class: type[Exception], expected: type[LLMProviderError]
    ) -> None:
        fault = _provider_fault(error_class)

        with (
            pytest.raises(LLMProviderError) as excinfo,
            translate_litellm_errors("gpt-4o-mini", "chat completion"),
        ):
            raise fault

        assert type(excinfo.value) is expected
        assert excinfo.value.__cause__ is fault

    def test_authentication_message_for_bare_embedding_model_suggests_rerouting(self) -> None:
        with (
            pytest.raises(LLMAuthenticationError) as excinfo,
            translate_litellm_errors("text-embedding-3-small", "embedding"),
        ):
            raise _provider_fault(litellm.AuthenticationError)

        message = excinfo.value.message
        assert "The openai provider rejected the credentials" in message
        assert "OPENAI_API_KEY" in message
        assert "EMBEDDING_MODEL" in message
        assert "openrouter/openai/text-embedding-3-small" in message

    def test_authentication_message_for_openrouter_names_its_key(self) -> None:
        with (
            pytest.raises(LLMAuthenticationError) as excinfo,
            translate_litellm_errors("openrouter/google/gemini-2.5-flash", "chat completion"),
        ):
            raise _provider_fault(litellm.AuthenticationError, provider="openrouter")

        assert "OPENROUTER_API_KEY" in excinfo.value.message

    def test_message_never_echoes_raw_provider_text(self) -> None:
        leaky = litellm.AuthenticationError(
            message="Incorrect API key provided: sk-live-ABCDEF***XYZ for org-secret",
            model="m",
            llm_provider="openai",
        )

        with (
            pytest.raises(LLMAuthenticationError) as excinfo,
            translate_litellm_errors("gpt-4o-mini", "chat completion"),
        ):
            raise leaky

        assert "sk-live" not in excinfo.value.message
        assert "org-secret" not in excinfo.value.message

    def test_unrelated_exceptions_pass_through_untranslated(self) -> None:
        with (
            pytest.raises(ValueError, match="boom"),
            translate_litellm_errors("gpt-4o-mini", "chat completion"),
        ):
            raise ValueError("boom")


class TestEmbedTextsErrors:
    async def test_provider_authentication_failure_becomes_domain_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = AsyncMock(side_effect=_provider_fault(litellm.AuthenticationError))
        monkeypatch.setattr("litellm.aembedding", mock)

        with pytest.raises(LLMAuthenticationError, match="EMBEDDING_MODEL"):
            await embed_texts(["a"], _settings())

    async def test_rate_limit_becomes_domain_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = AsyncMock(side_effect=_provider_fault(litellm.RateLimitError))
        monkeypatch.setattr("litellm.aembedding", mock)

        with pytest.raises(LLMRateLimitError):
            await embed_texts(["a"], _settings())


class TestResolveProvider:
    @pytest.mark.parametrize(
        ("model", "provider"),
        [
            ("text-embedding-3-small", "openai"),
            ("openrouter/openai/text-embedding-3-small", "openrouter"),
            ("anthropic/claude-sonnet-4-5", "anthropic"),
            ("no-such-provider-model-xyz", None),
        ],
    )
    def test_resolves_provider_from_prefix(self, model: str, provider: str | None) -> None:
        assert resolve_provider(model) == provider


class TestAuditLLMConfiguration:
    """Startup audit: report combinations that are certain to fail, never raise."""

    def test_flags_bare_embedding_model_with_placeholder_openai_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reported deployment: OpenRouter chat model, bare embedding model, no OpenAI key."""
        # The audit reads provider keys from os.environ (not Settings); without this the
        # chat model adds a second, unrelated problem on machines that have no `.env`.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-dummy")
        settings = _settings(
            openai_api_key=OPENAI_PLACEHOLDER_API_KEY,
            openai_model="openrouter/google/gemini-2.5-flash",
            embedding_model="text-embedding-3-small",
        )

        problems = audit_llm_configuration(settings)

        assert len(problems) == 1
        assert "EMBEDDING_MODEL='text-embedding-3-small'" in problems[0]
        assert "OPENAI_API_KEY" in problems[0]
        assert "openrouter/openai/text-embedding-3-small" in problems[0]

    @pytest.mark.parametrize("key", ["", "   ", OPENAI_PLACEHOLDER_API_KEY])
    def test_flags_unset_or_placeholder_openai_key_for_both_models(self, key: str) -> None:
        problems = audit_llm_configuration(_settings(openai_api_key=key))

        assert len(problems) == 2
        assert {p.split("=")[0] for p in problems} == {"OPENAI_MODEL", "EMBEDDING_MODEL"}

    def test_accepts_real_openai_key(self) -> None:
        assert audit_llm_configuration(_settings(openai_api_key="sk-real")) == []

    def test_flags_missing_key_of_non_openai_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "litellm.validate_environment",
            lambda model: {"keys_in_environment": False, "missing_keys": ["OPENROUTER_API_KEY"]},
        )
        settings = _settings(
            openai_model="openrouter/google/gemini-2.5-flash",
            embedding_model="openrouter/openai/text-embedding-3-small",
        )

        problems = audit_llm_configuration(settings)

        assert len(problems) == 2
        assert all("OPENROUTER_API_KEY is not set" in p for p in problems)

    def test_accepts_non_openai_provider_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "litellm.validate_environment",
            lambda model: {"keys_in_environment": True, "missing_keys": []},
        )
        settings = _settings(
            openai_api_key=OPENAI_PLACEHOLDER_API_KEY,
            openai_model="openrouter/google/gemini-2.5-flash",
            embedding_model="openrouter/openai/text-embedding-3-small",
        )

        assert audit_llm_configuration(settings) == []

    def test_flags_model_without_resolvable_provider(self) -> None:
        problems = audit_llm_configuration(_settings(embedding_model="no-such-provider-model-xyz"))

        assert len(problems) == 1
        assert "does not resolve to a known provider" in problems[0]
