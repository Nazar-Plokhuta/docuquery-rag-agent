"""Unit tests for filename sanitization and sandboxing in the ingestion route.

The handler is invoked directly with a hand-built ``UploadFile`` so hostile
filenames (including NUL bytes, which multipart encoders mangle) reach the
route logic unchanged.

Coverage:
- Traversal payloads are reduced to a basename that lives directly inside the
  dedicated temporary directory.
- Blank, whitespace-only, NUL-containing, and basename-less filenames yield 422.
- The extension allowlist is enforced on the sanitized name.
- Cleanup removes only the dedicated sandbox, on success and on failure.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, UploadFile

from src.api.routes import ingestion
from src.api.routes.ingestion import ingest_file
from src.ingestion.pipeline import IngestionPipeline


def _upload(filename: str) -> UploadFile:
    return UploadFile(file=io.BytesIO(b"payload"), filename=filename)


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin ``tempfile.mkdtemp`` to a known directory nested under a guarded parent.

    A sentinel file in the parent proves cleanup never reaches ancestors.
    """
    parent = tmp_path / "parent"
    temp_dir = parent / "sandbox"
    temp_dir.mkdir(parents=True)
    (parent / "sentinel.txt").write_text("keep")
    monkeypatch.setattr(ingestion.tempfile, "mkdtemp", lambda: str(temp_dir))
    return temp_dir


@pytest.fixture
def pipeline() -> AsyncMock:
    return AsyncMock(spec=IngestionPipeline)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_filename", "expected"),
    [
        ("../../test.md", "test.md"),
        ("/etc/passwd.txt", "passwd.txt"),
        ("subdir\\windows_style.pdf", "windows_style.pdf"),
        ("..\\..\\win_escape.md", "win_escape.md"),
        ("plain.md", "plain.md"),
    ],
)
async def test_traversal_filenames_are_confined_to_sandbox(
    raw_filename: str,
    expected: str,
    sandbox: Path,
    pipeline: AsyncMock,
) -> None:
    """The pipeline receives a file that is a direct child of ``temp_dir``."""
    seen: list[tuple[Path, bool]] = []

    async def _capture(path: Path) -> int:
        seen.append((path, path.is_file()))
        return 4

    pipeline.ingest_document.side_effect = _capture

    response = await ingest_file(_upload(raw_filename), pipeline)

    assert response.filename == expected
    assert response.chunks_ingested == 4
    [(path, existed)] = seen
    assert existed
    assert path.name == expected
    assert path.parent == sandbox.resolve()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_filename",
    ["", "   ", "\t\n", "test\x00.md", "\x00", "..", "dir/.."],
)
async def test_invalid_filenames_return_422(
    raw_filename: str,
    sandbox: Path,
    pipeline: AsyncMock,
) -> None:
    """Blank, NUL-containing, or basename-less filenames never reach the pipeline."""
    with pytest.raises(HTTPException) as exc_info:
        await ingest_file(_upload(raw_filename), pipeline)

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Uploaded file must include a valid filename."
    pipeline.ingest_document.assert_not_called()


@pytest.mark.asyncio
async def test_extension_allowlist_applies_to_sanitized_name(
    sandbox: Path,
    pipeline: AsyncMock,
) -> None:
    """A directory component cannot smuggle an allowed extension past the check."""
    with pytest.raises(HTTPException) as exc_info:
        await ingest_file(_upload("evil.md/payload.exe"), pipeline)

    assert exc_info.value.status_code == 415
    pipeline.ingest_document.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_removes_only_the_sandbox(
    sandbox: Path,
    pipeline: AsyncMock,
) -> None:
    """Success path deletes ``temp_dir`` while its parent and siblings survive."""
    pipeline.ingest_document.return_value = 1

    await ingest_file(_upload("../../escaped.md"), pipeline)

    assert not sandbox.exists()
    assert sandbox.parent.is_dir()
    assert (sandbox.parent / "sentinel.txt").read_text() == "keep"
    assert not (sandbox.parent.parent.parent / "escaped.md").exists()


@pytest.mark.asyncio
async def test_cleanup_runs_when_pipeline_fails(
    sandbox: Path,
    pipeline: AsyncMock,
) -> None:
    """A pipeline error still removes the sandbox and nothing beyond it."""
    pipeline.ingest_document.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await ingest_file(_upload("../../escaped.md"), pipeline)

    assert not sandbox.exists()
    assert (sandbox.parent / "sentinel.txt").exists()
