"""Unit tests for the Settings configuration model.

Covers the ``openai_base_url`` field introduced in Sprint 5:
- Defaults to ``None`` when the ``OPENAI_BASE_URL`` environment variable is
  absent and no ``.env`` file is consulted.
- Accepts a valid URL string and surfaces it on the instance without mutation.
- Confirms the field name is mapped to the ``OPENAI_BASE_URL`` env var
  (Pydantic Settings upper-cases field names to derive env var names).
- Verifies that all previously existing fields still resolve to their compiled-in
  defaults when ``.env`` files and ambient env vars are suppressed.

Isolation strategy
------------------
Every ``Settings(...)`` construction in this module passes ``_env_file=None``
so that the local ``.env`` file on the developer's machine (or CI environment)
is never consulted.  Tests that assert a field is ``None`` additionally use
``monkeypatch.delenv`` to remove any ambient process-level environment variable
that may have been set before pytest launched.  This makes the entire suite
reproducible regardless of the host machine's configuration.
"""

from __future__ import annotations

import pytest

from src.config.settings import Settings

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_REQUIRED_OVERRIDES: dict[str, str] = {
    "openai_api_key": "sk-test",
    "chroma_persist_directory": "/tmp/chroma",
    "sqlite_database_path": "/tmp/telemetry.db",
}


def _make_settings(**extra: str | None) -> Settings:
    """Construct a ``Settings`` instance fully isolated from the host environment.

    ``_env_file=None`` prevents Pydantic Settings from reading the on-disk
    ``.env`` file.  Callers should use ``monkeypatch.delenv`` for any env var
    that must also be absent from the process environment.

    Args:
        **extra: Additional keyword arguments forwarded to ``Settings.__init__``.

    Returns:
        A ``Settings`` instance whose field values come exclusively from
        compiled-in defaults and the explicitly supplied ``extra`` kwargs.
    """
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        **{**_REQUIRED_OVERRIDES, **extra},  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# TestOpenAIBaseUrl
# ---------------------------------------------------------------------------


class TestOpenAIBaseUrl:
    """Validate the ``openai_base_url`` field on the Settings model."""

    def test_defaults_to_none_when_not_supplied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``openai_base_url`` must be ``None`` when neither the env var nor a kwarg is set."""
        # Remove the env var from the process environment so it cannot bleed in
        # even with ``_env_file=None`` (which only suppresses .env file reads).
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        settings = _make_settings()
        assert settings.openai_base_url is None

    def test_accepts_openrouter_url(self) -> None:
        """A valid OpenRouter base URL must be stored without modification."""
        url = "https://openrouter.ai/api/v1"
        settings = _make_settings(openai_base_url=url)
        assert settings.openai_base_url == url

    def test_accepts_arbitrary_openai_compatible_url(self) -> None:
        """Any OpenAI-compatible base URL (vLLM, LocalAI) must be accepted."""
        url = "http://localhost:8080/v1"
        settings = _make_settings(openai_base_url=url)
        assert settings.openai_base_url == url

    def test_env_var_name_is_openai_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The field must be resolved from the ``OPENAI_BASE_URL`` process env var."""
        expected_url = "https://my-gateway.example.com/v1"
        monkeypatch.setenv("OPENAI_BASE_URL", expected_url)

        # Do NOT pass openai_base_url as a kwarg — the env var alone must provide it.
        settings = _make_settings()
        assert settings.openai_base_url == expected_url

    def test_none_is_preserved_when_env_var_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ``OPENAI_BASE_URL`` is removed from the process env, the field is ``None``."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        settings = _make_settings()
        assert settings.openai_base_url is None

    def test_kwarg_takes_precedence_over_none_default(self) -> None:
        """Supplying ``openai_base_url`` as a kwarg must override the ``None`` default."""
        url = "https://custom.example.com/v1"
        settings = _make_settings(openai_base_url=url)
        assert settings.openai_base_url == url
        assert isinstance(settings.openai_base_url, str)


# ---------------------------------------------------------------------------
# TestSettingsFieldRegressions
# ---------------------------------------------------------------------------


class TestSettingsFieldRegressions:
    """Guard against regressions in pre-existing Settings fields and their defaults.

    All instances are constructed via ``_make_settings()`` so the ``.env`` file
    and ambient env vars are suppressed — only compiled-in default values are
    tested here.
    """

    def test_openai_model_default(self) -> None:
        """``openai_model`` must default to ``'gpt-4o-mini'``."""
        assert _make_settings().openai_model == "gpt-4o-mini"

    def test_embedding_model_default(self) -> None:
        """``embedding_model`` must default to ``'text-embedding-3-small'``."""
        assert _make_settings().embedding_model == "text-embedding-3-small"

    def test_app_env_default(self) -> None:
        """``app_env`` must default to ``'development'``."""
        assert _make_settings().app_env == "development"

    def test_app_port_default(self) -> None:
        """``app_port`` must default to ``8000``."""
        assert _make_settings().app_port == 8000

    def test_log_level_default(self) -> None:
        """``log_level`` must default to ``'INFO'``."""
        assert _make_settings().log_level == "INFO"

    def test_openai_base_url_defaults_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``openai_base_url`` must be ``None`` in the default configuration.

        Uses ``monkeypatch`` to guarantee the env var is absent so the test
        does not depend on the developer's shell or CI environment state.
        """
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        s = _make_settings()
        assert hasattr(s, "openai_base_url")
        assert s.openai_base_url is None
