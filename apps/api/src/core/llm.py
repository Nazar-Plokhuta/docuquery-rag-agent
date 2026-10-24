"""LiteLLM integration helpers shared by the RAG engine and the ingestion pipeline.

LiteLLM is the single abstraction over chat-completion and embedding providers.
Routing is driven entirely by the model identifier (``gpt-4o-mini``,
``anthropic/claude-sonnet-4-5``, ``openrouter/<vendor>/<model>``,
``openai/<model>`` + ``OPENAI_BASE_URL`` for a custom gateway), so no caller
needs provider-specific code.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import litellm

from src.config.settings import Settings

logger = logging.getLogger(__name__)

_OPENAI_ROUTED_PROVIDER: str = "openai"


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
    try:
        provider: str = litellm.get_llm_provider(model)[1]
    except litellm.BadRequestError:
        logger.debug("LiteLLM could not resolve a provider for model %r.", model)
        return {}

    if provider != _OPENAI_ROUTED_PROVIDER:
        return {}

    credentials = {"api_key": settings.openai_api_key}
    if settings.openai_base_url is not None:
        credentials["api_base"] = settings.openai_base_url
    return credentials


async def embed_texts(texts: Sequence[str], settings: Settings) -> list[list[float]]:
    """Embed ``texts`` through ``litellm.aembedding`` and return vectors in input order.

    Args:
        texts: Non-empty sequence of strings to embed.
        settings: Application settings providing ``embedding_model`` and credentials.

    Returns:
        One dense vector per input string, in the same order as ``texts``.
    """
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
