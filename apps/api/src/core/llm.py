"""LiteLLM integration helpers shared by the RAG engine and the ingestion pipeline.

LiteLLM is the single abstraction over chat-completion and embedding providers.
Routing is driven entirely by the model identifier (``gpt-4o-mini``,
``anthropic/claude-sonnet-4-5``, ``openrouter/<vendor>/<model>``,
``openai/<model>`` + ``OPENAI_BASE_URL`` for a custom gateway), so no caller
needs provider-specific code.

Provider faults are translated into the ``LLMProviderError`` family here, at the
only layer that knows LiteLLM's exception vocabulary, so callers and the HTTP
layer deal exclusively in domain exceptions.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Literal

import litellm

from src.config.settings import OPENAI_PLACEHOLDER_API_KEY, Settings
from src.core.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

_OPENAI_ROUTED_PROVIDER: str = "openai"

LLMOperation = Literal["chat completion", "embedding"]

# LiteLLM mirrors the OpenAI SDK hierarchy, so its exceptions do not share one
# public base class; the tuple enumerates every class a call can raise.
_LITELLM_ERRORS: tuple[type[Exception], ...] = (
    litellm.AuthenticationError,
    litellm.PermissionDeniedError,
    litellm.RateLimitError,
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.BadGatewayError,
    litellm.NotFoundError,
    litellm.BadRequestError,
    litellm.UnprocessableEntityError,
    litellm.APIError,
)

_PROVIDER_KEY_ENV: Mapping[str, str] = {
    _OPENAI_ROUTED_PROVIDER: "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Concrete re-routing examples quoted in operator-facing messages, keyed by the
# setting that selects the model. Same underlying models, reached via OpenRouter.
_ROUTING_EXAMPLES: Mapping[str, str] = {
    "OPENAI_MODEL": "openrouter/openai/gpt-4o-mini",
    "EMBEDDING_MODEL": "openrouter/openai/text-embedding-3-small",
}

_MAX_LOGGED_DETAIL_CHARS: int = 500


def configure_litellm(settings: Settings) -> None:
    """Apply process-wide LiteLLM flags once at application startup.

    ``drop_params`` makes LiteLLM silently discard parameters a provider does not
    support instead of failing the request, which keeps one call signature valid
    across heterogeneous backends.  Telemetry is disabled because the service
    handles private enterprise documents.

    Args:
        settings: Application settings supplying the ``litellm_drop_params`` flag.
    """
    litellm.drop_params = settings.litellm_drop_params
    litellm.telemetry = False


def resolve_provider(model: str) -> str | None:
    """Return the provider LiteLLM routes ``model`` to, or ``None`` if it cannot tell.

    A bare name such as ``text-embedding-3-small`` resolves to ``openai`` even
    when the deployment only holds an OpenRouter key; the provider prefix is the
    sole routing signal, which is why mismatches are worth surfacing explicitly.
    """
    try:
        provider: str = litellm.get_llm_provider(model)[1]
    except litellm.BadRequestError:
        logger.debug("LiteLLM could not resolve a provider for model %r.", model)
        return None
    return provider


def provider_credentials(model: str, settings: Settings) -> dict[str, str]:
    """Return ``api_key``/``api_base`` kwargs for models routed to OpenAI.

    The OpenAI credentials must never be forwarded to another vendor: Anthropic,
    OpenRouter and similar providers resolve their own keys from their dedicated
    environment variables (``ANTHROPIC_API_KEY``, ``OPENROUTER_API_KEY``, ...).
    A model LiteLLM cannot attribute to a provider receives no credentials; the
    subsequent completion call then fails with LiteLLM's own routing error.

    Args:
        model: LiteLLM model identifier, optionally carrying a provider prefix.
        settings: Application settings holding the OpenAI key and optional gateway URL.

    Returns:
        Keyword arguments to splat into ``litellm.acompletion``/``aembedding``.
    """
    if resolve_provider(model) != _OPENAI_ROUTED_PROVIDER:
        return {}

    credentials = {"api_key": settings.openai_api_key}
    if settings.openai_base_url is not None:
        credentials["api_base"] = settings.openai_base_url
    return credentials


def audit_llm_configuration(settings: Settings) -> list[str]:
    """Describe model/credential combinations that are certain to fail at request time.

    Runs once at startup and only reports; it never raises, so the service still
    boots and the health, telemetry and ingestion-listing surfaces stay usable
    while the operator fixes the LLM configuration.

    Args:
        settings: Application settings carrying both model identifiers and the
            OpenAI credentials.

    Returns:
        One operator-facing sentence per problem; empty when nothing is wrong.
    """
    problems: list[str] = []
    configured_models = (
        ("OPENAI_MODEL", settings.openai_model),
        ("EMBEDDING_MODEL", settings.embedding_model),
    )
    for setting_name, model in configured_models:
        provider = resolve_provider(model)
        if provider is None:
            problems.append(
                f"{setting_name}={model!r} does not resolve to a known provider; "
                f"add a provider prefix such as {_ROUTING_EXAMPLES[setting_name]!r}."
            )
        elif provider == _OPENAI_ROUTED_PROVIDER:
            api_key = settings.openai_api_key.strip()
            if not api_key or api_key == OPENAI_PLACEHOLDER_API_KEY:
                problems.append(
                    f"{setting_name}={model!r} routes to OpenAI but OPENAI_API_KEY is unset "
                    f"or the template placeholder. Set a real key, or prefix the model with "
                    f"a provider you hold credentials for, e.g. "
                    f"{_ROUTING_EXAMPLES[setting_name]!r}."
                )
        else:
            missing: list[str] = litellm.validate_environment(model=model)["missing_keys"]
            if missing:
                problems.append(
                    f"{setting_name}={model!r} routes to {provider!r} but "
                    f"{', '.join(missing)} is not set."
                )
    return problems


def _credential_hint(provider: str, operation: LLMOperation) -> str:
    """Return the remedy for a credential rejection, specific to ``provider``."""
    env_var = _PROVIDER_KEY_ENV.get(provider)
    hint = f"Check {env_var}." if env_var else "Check the provider API key."
    if provider == _OPENAI_ROUTED_PROVIDER:
        setting_name = "EMBEDDING_MODEL" if operation == "embedding" else "OPENAI_MODEL"
        hint += (
            f" To use another provider, prefix {setting_name}, "
            f"e.g. {_ROUTING_EXAMPLES[setting_name]}."
        )
    return hint


def _to_provider_error(exc: Exception, *, model: str, operation: LLMOperation) -> LLMProviderError:
    """Map one LiteLLM exception onto the domain hierarchy with an actionable message.

    The message is built from the exception *class* and HTTP status only. Raw
    provider text is logged but never returned, because it can echo masked
    credentials or fragments of the prompt.
    """
    provider = resolve_provider(model) or "unknown"
    status: int | None = getattr(exc, "status_code", None)
    subject = f"{operation} with model {model!r}"

    detail = " ".join(str(exc).split())[:_MAX_LOGGED_DETAIL_CHARS]
    logger.warning(
        "LiteLLM %s failed: provider=%s model=%s error=%s status=%s detail=%s",
        operation,
        provider,
        model,
        type(exc).__name__,
        status,
        detail,
    )

    if isinstance(exc, litellm.AuthenticationError | litellm.PermissionDeniedError):
        return LLMAuthenticationError(
            f"The {provider} provider rejected the credentials used for {subject}. "
            f"{_credential_hint(provider, operation)}"
        )
    if isinstance(exc, litellm.RateLimitError):
        return LLMRateLimitError(
            f"The {provider} provider rate-limited {subject} or the quota is exhausted. "
            "Retry shortly or check the account quota."
        )
    if isinstance(exc, litellm.Timeout):
        return LLMTimeoutError(f"The {provider} provider timed out during {subject}.")
    if isinstance(
        exc,
        litellm.APIConnectionError
        | litellm.ServiceUnavailableError
        | litellm.InternalServerError
        | litellm.BadGatewayError,
    ):
        status_note = f" (HTTP {status})" if status is not None else ""
        return LLMUnavailableError(
            f"The {provider} provider is unreachable or reported a server error"
            f"{status_note} during {subject}."
        )
    if isinstance(exc, litellm.NotFoundError):
        return LLMProviderError(
            f"The {provider} provider does not recognise the model used for {subject}. "
            "Check the model identifier and its provider prefix."
        )
    status_note = f", HTTP {status}" if status is not None else ""
    return LLMProviderError(
        f"The {provider} provider rejected {subject} ({type(exc).__name__}{status_note})."
    )


@contextmanager
def translate_litellm_errors(model: str, operation: LLMOperation) -> Iterator[None]:
    """Re-raise any LiteLLM provider exception inside the block as ``LLMProviderError``.

    Wrap both the call that opens a streaming completion and the loop that
    consumes it: providers can fail before the first chunk or between chunks.

    Args:
        model: LiteLLM identifier of the model being called, used in the message.
        operation: Which kind of call the block performs, used in the message.

    Raises:
        LLMProviderError: A subclass matching the provider failure category.
    """
    try:
        yield
    except _LITELLM_ERRORS as exc:
        raise _to_provider_error(exc, model=model, operation=operation) from exc


async def embed_texts(texts: Sequence[str], settings: Settings) -> list[list[float]]:
    """Embed ``texts`` through ``litellm.aembedding`` and return vectors in input order.

    Args:
        texts: Non-empty sequence of strings to embed.
        settings: Application settings providing ``embedding_model`` and credentials.

    Returns:
        One dense vector per input string, in the same order as ``texts``.

    Raises:
        LLMProviderError: If the embedding provider rejects or fails the request.
    """
    with translate_litellm_errors(settings.embedding_model, "embedding"):
        response = await litellm.aembedding(
            model=settings.embedding_model,
            input=list(texts),
            **provider_credentials(settings.embedding_model, settings),
        )
    # LiteLLM yields plain dicts for some providers and objects for others.
    items: list[Any] = list(response.data)
    return [_embedding_of(item) for item in items]


def _embedding_of(item: Mapping[str, Any] | Any) -> list[float]:
    """Read the vector from a LiteLLM embedding item, whether dict or object."""
    if isinstance(item, Mapping):
        return list(item["embedding"])
    return list(item.embedding)
