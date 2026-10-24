"""Typed application settings loaded from environment variables via Pydantic Settings.

All runtime configuration is centralised here.  Downstream layers import
`get_settings()` rather than accessing ``os.environ`` directly, which keeps
secrets isolated and enables dependency-injection-based overrides in tests.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# The monorepo keeps one shared `.env` at the repository root, two levels above
# `apps/api`, so every app reads the same file regardless of the process working
# directory.  In the container image the app root is `/app`, where the extra
# `.parent` hops saturate at `/` and the candidate simply does not exist.
_API_ROOT: Path = Path(__file__).resolve().parents[2]
_REPO_ENV_FILE: Path = _API_ROOT.parent.parent / ".env"


class Settings(BaseSettings):
    """Application-wide configuration resolved from the `.env` file and environment.

    Fields are grouped by concern:
    - **Application runtime**: env label, serving port, log verbosity.
    - **LLM (LiteLLM)**: OpenAI credentials, chat/embedding model identifiers, and
      LiteLLM behaviour flags.
    - **Vector store**: ChromaDB local persistence directory or remote server.
    - **Relational store**: SQLAlchemy async URL for the telemetry / audit trail.
    """

    model_config = SettingsConfigDict(
        # Later files win: a `.env` in the working directory overrides the shared one.
        env_file=(_REPO_ENV_FILE, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application runtime
    # ------------------------------------------------------------------
    app_env: Literal["development", "staging", "production"] = "development"
    app_port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ------------------------------------------------------------------
    # LLM routing (LiteLLM)
    # ------------------------------------------------------------------
    # Model names are LiteLLM identifiers and may carry a provider prefix, e.g.
    # `gpt-4o-mini`, `anthropic/claude-sonnet-4-5`, `openrouter/<vendor>/<model>`.
    # `openai_api_key`/`openai_base_url` are forwarded only to OpenAI-routed models;
    # other providers read their own env vars (`ANTHROPIC_API_KEY`, ...).
    openai_api_key: str = "sk-placeholder"
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    # Optional custom OpenAI-compatible gateway (vLLM, LocalAI, ...). Pair it with an
    # `openai/<model>` identifier; when None, requests go to the official OpenAI API.
    openai_base_url: str | None = None
    # Drop request parameters a provider does not support instead of raising, so one
    # call signature stays valid across backends.
    litellm_drop_params: bool = True

    # ------------------------------------------------------------------
    # HTTP / CORS
    # ------------------------------------------------------------------
    # Comma-separated browser origins allowed to call the API cross-origin, e.g.
    # `http://localhost:3000,http://127.0.0.1:3000`. Empty (default) installs no CORS
    # middleware, so the API stays same-origin only.
    cors_allow_origins: str = ""

    # ------------------------------------------------------------------
    # Vector store (ChromaDB)
    # ------------------------------------------------------------------
    # Local file mode (default) writes to `chroma_persist_directory`. Setting
    # `chroma_host` switches to a standalone Chroma server over HTTP and the
    # persistence directory is then unused.
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_host: str | None = None
    chroma_port: int = 8000

    # ------------------------------------------------------------------
    # Relational / audit store (SQLAlchemy async)
    # ------------------------------------------------------------------
    # PostgreSQL in production. Tests and throwaway local runs may point this at
    # `sqlite+aiosqlite:///:memory:` (or a file path) to avoid a database server.
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/docuquery"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton ``Settings`` instance, constructed once and cached.

    Using ``lru_cache`` avoids repeated `.env` file I/O on every request while
    still allowing tests to clear the cache via ``get_settings.cache_clear()``.
    """
    return Settings()
