"""Unit tests for the opt-in CORS configuration.

Each test builds a throwaway ``FastAPI`` instance: middleware must be attached
before an app starts serving, so the module-level ``app`` (configured from the
ambient environment at import time) cannot be reconfigured per test.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.config.settings import Settings
from src.main import install_cors, parse_cors_origins

_ORIGIN = "http://localhost:3000"


def _settings(cors_allow_origins: str) -> Settings:
    return Settings(_env_file=None, cors_allow_origins=cors_allow_origins)


def _build_app(origins: list[str]) -> FastAPI:
    test_app = FastAPI()

    @test_app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    install_cors(test_app, origins)
    return test_app


def _client(test_app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=test_app), base_url="http://testserver")


def test_cors_origins_default_to_empty() -> None:
    assert Settings(_env_file=None).cors_allow_origins == ""
    assert parse_cors_origins(_settings("")) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://localhost:3000", ["http://localhost:3000"]),
        (
            " http://localhost:3000 , http://127.0.0.1:3000 ,, ",
            ["http://localhost:3000", "http://127.0.0.1:3000"],
        ),
        (" , ,", []),
    ],
)
def test_parse_cors_origins(raw: str, expected: list[str]) -> None:
    assert parse_cors_origins(_settings(raw)) == expected


@pytest.mark.asyncio
async def test_cors_headers_absent_when_unconfigured() -> None:
    async with _client(_build_app([])) as client:
        simple = await client.get("/ping", headers={"Origin": _ORIGIN})
        preflight = await client.options(
            "/ping",
            headers={"Origin": _ORIGIN, "Access-Control-Request-Method": "GET"},
        )

    assert "access-control-allow-origin" not in simple.headers
    assert "access-control-allow-origin" not in preflight.headers


@pytest.mark.asyncio
async def test_cors_headers_attached_for_allowed_origin() -> None:
    async with _client(_build_app([_ORIGIN])) as client:
        simple = await client.get("/ping", headers={"Origin": _ORIGIN})
        preflight = await client.options(
            "/ping",
            headers={
                "Origin": _ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

    assert simple.headers["access-control-allow-origin"] == _ORIGIN
    assert "access-control-allow-credentials" not in simple.headers
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == _ORIGIN
    assert set(preflight.headers["access-control-allow-methods"].split(", ")) == {
        "GET",
        "POST",
        "OPTIONS",
    }
    assert "content-type" in preflight.headers["access-control-allow-headers"].lower()


@pytest.mark.asyncio
async def test_cors_rejects_unlisted_origin() -> None:
    async with _client(_build_app([_ORIGIN])) as client:
        response = await client.get("/ping", headers={"Origin": "http://evil.example"})

    assert "access-control-allow-origin" not in response.headers
