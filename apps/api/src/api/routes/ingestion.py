"""Ingestion router: multipart document upload and indexing endpoint.

The route validates the uploaded file extension, persists the content to a
dedicated temporary directory (preserving the sanitized basename so the
ingestion pipeline derives a meaningful ``document_id`` from the stem), delegates the
full ingestion flow to ``IngestionPipeline``, and returns an ``IngestResponse``
confirming the number of indexed chunks.

Responsibilities explicitly excluded from this module:
- Text chunking logic (owned by ``TokenSlidingWindowChunker``).
- Embedding generation (owned by ``IngestionPipeline``).
- Vector store writes (owned by ``ChromaVectorStore``).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from src.api.dependencies import get_ingestion_pipeline
from src.api.schemas.ingestion import IngestResponse
from src.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# Allowlist of accepted file extensions handled by ``DocumentLoaderFactory``.
_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".txt", ".pdf", ".docx", ".csv", ".xlsx", ".html", ".htm"}
)

_INVALID_FILENAME_DETAIL = "Uploaded file must include a valid filename."


@router.post(
    "/file",
    response_model=IngestResponse,
    summary="Ingest a document",
    description=(
        "Upload a ``.md``, ``.txt``, ``.pdf``, ``.docx``, ``.csv``, ``.xlsx``, "
        "``.html``, or ``.htm`` file. "
        "The service extracts text, "
        "chunks, embeds, and indexes the content into the vector store, then returns "
        "the chunk count."
    ),
    status_code=200,
)
async def ingest_file(
    file: UploadFile,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Validate, store, and ingest an uploaded document.

    Processing flow:
    1. Reject empty, whitespace-only, or NUL-containing filenames with HTTP 422.
    2. Reduce the client filename to a pure basename (directories and both
       path-separator styles stripped) and reject unsupported extensions with
       HTTP 415.
    3. Read the uploaded content into memory.
    4. Write to a dedicated temporary directory under the sanitized filename,
       asserting the target is a direct child of that directory, so that
       ``pipeline.ingest_document`` derives the correct ``document_id`` from
       the file stem.
    5. Delegate the full chunk → embed → index pipeline to ``IngestionPipeline``.
    6. Remove exactly that temporary directory, unconditionally.

    Args:
        file: Multipart ``UploadFile`` provided by FastAPI.
        pipeline: Injected ``IngestionPipeline`` resolved from application state.

    Returns:
        ``IngestResponse`` confirming the sanitized filename and number of
        indexed chunks.

    Raises:
        HTTPException 415: When the uploaded file extension is not in
            ``_ALLOWED_EXTENSIONS``.
        HTTPException 422: When the filename is missing, blank, contains a NUL
            byte, or does not reduce to a usable basename.
    """
    raw_filename: str = file.filename or ""
    if not raw_filename.strip() or "\x00" in raw_filename:
        raise HTTPException(status_code=422, detail=_INVALID_FILENAME_DETAIL)

    # Clients may send Windows-style separators, which ``Path`` on POSIX would
    # treat as part of a single name; normalize first so only the basename survives.
    safe_filename: str = Path(raw_filename.replace("\\", "/")).name
    if not safe_filename.strip() or safe_filename in {".", ".."}:
        raise HTTPException(status_code=422, detail=_INVALID_FILENAME_DETAIL)

    suffix: str = Path(safe_filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Accepted extensions: {', '.join(sorted(_ALLOWED_EXTENSIONS))}."
            ),
        )

    content: bytes = await file.read()

    # The sandbox is owned by this request alone: cleanup targets ``temp_dir``
    # directly and never a path derived from client input.
    temp_dir: Path = Path(await asyncio.to_thread(tempfile.mkdtemp))
    try:
        dest_path: Path = (temp_dir / safe_filename).resolve()
        if dest_path.parent != temp_dir.resolve():
            raise HTTPException(status_code=422, detail=_INVALID_FILENAME_DETAIL)

        # ``IngestionPipeline.ingest_document`` uses ``Path.stem`` as the
        # ``document_id``, so maintaining the sanitized name is load-bearing.
        await asyncio.to_thread(dest_path.write_bytes, content)
        chunks_ingested: int = await pipeline.ingest_document(dest_path)
    finally:
        # Unconditional cleanup — errors during ingestion must not leave
        # temporary data on disk.
        await asyncio.to_thread(shutil.rmtree, temp_dir, ignore_errors=True)

    logger.info(
        "Document ingested. filename=%r chunks=%d",
        safe_filename,
        chunks_ingested,
    )
    return IngestResponse(
        status="success",
        filename=safe_filename,
        chunks_ingested=chunks_ingested,
    )
