"""Dump the OpenAPI schema to disk without starting a Uvicorn server.

Lets the web console generate TypeScript types from a committed contract
(`docs/openapi.json`) in CI or offline. Building the schema only inspects the
route table, so the lifespan never runs and no database, vector store, or LLM
provider is contacted.

Usage (from the repository root)::

    uv run --project apps/api python -m src.export_openapi [-o PATH]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from src.main import app

# Resolved from this file so the default works from any working directory.
_DEFAULT_OUTPUT: Path = Path(__file__).resolve().parents[3] / "docs" / "openapi.json"


def build_schema() -> dict[str, object]:
    """Return the OpenAPI document generated from the application's routes."""
    return app.openapi()


def export_schema(output: Path) -> Path:
    """Write the schema to ``output`` as stable, diff-friendly JSON.

    Keys are sorted and a trailing newline is added so regenerating an unchanged
    contract yields a byte-identical file.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_schema(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description="Export the OpenAPI schema as JSON.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Destination file (default: docs/openapi.json at the repository root).",
    )
    args = parser.parse_args(argv)
    written = export_schema(args.output)
    print(f"OpenAPI schema written to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
