"""CSV document loader with Markdown table output for chunk section metadata."""

from __future__ import annotations

import asyncio
import csv
import io
from pathlib import Path
from typing import Any

from src.core.exceptions import DocumentParsingError
from src.ingestion.loaders.base import BaseDocumentLoader, LoadedDocument

_WIDE_COLUMN_THRESHOLD: int = 8


def _decode_csv_bytes(file_bytes: bytes) -> str:
    """Decode CSV bytes, preferring UTF-8 with pragmatic fallbacks for legacy encodings."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1251"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _normalize_row(raw_row: list[str | None]) -> list[str]:
    return [_escape_markdown_cell("" if cell is None else str(cell)) for cell in raw_row]


def _rows_to_markdown_table(rows: list[list[str]]) -> str:
    """Render tabular rows as a Markdown table or key-value lines when very wide."""
    if not rows:
        return ""

    column_count = max(len(row) for row in rows)
    if column_count == 0:
        return ""

    normalized: list[list[str]] = []
    for row in rows:
        padded = row + [""] * (column_count - len(row))
        normalized.append(padded[:column_count])

    if column_count > _WIDE_COLUMN_THRESHOLD:
        headers = normalized[0]
        lines: list[str] = []
        for data_row in normalized[1:]:
            pairs = [
                f"{header}: {value}"
                for header, value in zip(headers, data_row, strict=False)
                if header or value
            ]
            if pairs:
                lines.append("; ".join(pairs))
        return "\n".join(lines)

    header = normalized[0]
    separator = "| " + " | ".join("---" for _ in header) + " |"
    header_line = "| " + " | ".join(header) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in normalized[1:]]
    return "\n".join([header_line, separator, *body_lines])


def _parse_csv_text(text: str, filename: str) -> list[list[str]]:
    """Parse CSV text into rows, inferring delimiter when possible."""
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(io.StringIO(text), dialect=dialect)
    rows: list[list[str]] = []
    for raw_row in reader:
        if not raw_row or all(not (cell or "").strip() for cell in raw_row):
            continue
        rows.append(_normalize_row(raw_row))

    if not rows:
        raise DocumentParsingError(
            f"Unable to read CSV {filename!r}: file contains no rows or columns."
        )

    column_count = max(len(row) for row in rows)
    if column_count == 0:
        raise DocumentParsingError(
            f"Unable to read CSV {filename!r}: file contains no rows or columns."
        )

    return rows


def _extract_csv_text(file_bytes: bytes, filename: str) -> LoadedDocument:
    """Synchronous CSV parse — intended to run inside ``asyncio.to_thread``."""
    source = Path(filename).stem
    text = _decode_csv_bytes(file_bytes)
    rows = _parse_csv_text(text, filename)
    table_block = _rows_to_markdown_table(rows)
    if not table_block.strip():
        raise DocumentParsingError(
            f"Unable to read CSV {filename!r}: file contains no rows or columns."
        )

    heading = f"## CSV: {source}"
    content = f"{heading}\n\n{table_block}"

    metadata: dict[str, Any] = {
        "format": "csv",
        "filename": filename,
        "row_count": len(rows),
        "column_count": max(len(row) for row in rows),
    }

    return LoadedDocument(content=content, source=source, metadata=metadata)


class CsvDocumentLoader(BaseDocumentLoader):
    """Extracts CSV rows as Markdown tables for chunk section metadata."""

    async def load(self, file_bytes: bytes, filename: str) -> LoadedDocument:
        """Parse ``file_bytes`` off the event loop and return tabular Markdown text."""
        if not file_bytes.strip():
            raise DocumentParsingError(f"Unable to read CSV {filename!r}: file is empty.")

        return await asyncio.to_thread(_extract_csv_text, file_bytes, filename)
