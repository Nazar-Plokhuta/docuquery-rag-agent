"""Unit tests for the offline OpenAPI schema export."""

from __future__ import annotations

import json
from pathlib import Path

from src import __version__
from src.export_openapi import build_schema, export_schema, main


def test_build_schema_describes_health_contract() -> None:
    schema = build_schema()

    assert schema["info"]["version"] == __version__  # type: ignore[index]
    assert "/api/v1/health/" in schema["paths"]  # type: ignore[operator]
    health = schema["components"]["schemas"]["HealthResponse"]  # type: ignore[index]
    assert {"active_model", "embedding_model"} <= set(health["properties"])


def test_export_schema_writes_valid_json(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "openapi.json"

    written = export_schema(target)

    assert written == target
    assert json.loads(target.read_text(encoding="utf-8")) == json.loads(json.dumps(build_schema()))


def test_export_is_deterministic(tmp_path: Path) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"

    export_schema(first)
    export_schema(second)

    assert first.read_bytes() == second.read_bytes()


def test_cli_main_honours_output_flag(tmp_path: Path) -> None:
    target = tmp_path / "openapi.json"

    assert main(["--output", str(target)]) == 0
    assert target.is_file()
