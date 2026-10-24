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
    - **OpenAI**: API key, chat model, and embedding model identifiers.
    - **Vector store**: ChromaDB persistence directory.
    - **Relational store**: SQLite file path for the telemetry / audit trail.
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
    # OpenAI
    # ------------------------------------------------------------------
    openai_api_key: str = "sk-placeholder"
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    # Optional custom OpenAI-compatible gateway (e.g. OpenRouter, vLLM, LocalAI).
    # When None, the AsyncOpenAI client defaults to the official OpenAI platform.
    openai_base_url: str | None = None

    # ------------------------------------------------------------------
    # Vector store (ChromaDB)
    # ------------------------------------------------------------------
    chroma_persist_directory: str = "./data/chroma_db"

    # ------------------------------------------------------------------
    # Relational / audit store (SQLite via aiosqlite)
    # ------------------------------------------------------------------
    sqlite_database_path: str = "./data/telemetry.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton ``Settings`` instance, constructed once and cached.

    Using ``lru_cache`` avoids repeated `.env` file I/O on every request while
    still allowing tests to clear the cache via ``get_settings.cache_clear()``.
    """
    return Settings()
