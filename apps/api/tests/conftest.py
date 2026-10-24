"""Shared pytest fixtures for the DocuQuery RAG Agent test suite.

The ``async_client`` fixture provides an ``httpx.AsyncClient`` bound to the
FastAPI application instance.  Using the ASGI transport avoids starting a real
TCP server, which keeps tests hermetic and fast.

Settings are overridden before the client boots so tests never require a real
``.env`` file or valid LLM provider credentials.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Must be set before ``litellm`` is first imported: otherwise it downloads its model
# price map from GitHub at import time, which would break the offline guarantee.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from src.config.settings import Settings, get_settings  # noqa: E402
from src.main import app  # noqa: E402


def _test_settings() -> Settings:
    """Return a minimal ``Settings`` instance safe for unit tests.

    Overrides the LRU-cached singleton so the test process never reads from an
    actual ``.env`` file or touches the filesystem outside ``/tmp``.
    """
    return Settings(
        app_env="development",
        app_port=8000,
        log_level="DEBUG",
        openai_api_key="sk-test-placeholder",
        openai_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        chroma_persist_directory="/tmp/test_chroma_db",
        database_url="sqlite+aiosqlite:///:memory:",
    )


@pytest_asyncio.fixture(scope="session", autouse=True)
async def override_settings() -> AsyncGenerator[None, None]:
    """Override the ``get_settings`` dependency for the full test session."""
    app.dependency_overrides[get_settings] = _test_settings
    get_settings.cache_clear()
    yield
    app.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Yield an ``httpx.AsyncClient`` wired to the FastAPI ASGI app.

    Lifespan events (startup/shutdown) are triggered so directory provisioning
    logic is exercised in integration-style unit tests.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


@pytest.fixture(autouse=True)
def forbid_live_llm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any un-mocked LiteLLM call fail loudly instead of reaching the network.

    Tests that exercise LLM paths override these with their own mocks.
    """
    forbidden = AsyncMock(side_effect=AssertionError("Live LiteLLM call attempted in a test."))
    monkeypatch.setattr("litellm.acompletion", forbidden)
    monkeypatch.setattr("litellm.aembedding", forbidden)
