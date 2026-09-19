"""Ingestion router: multipart document upload and indexing endpoint.

The route validates the uploaded file extension, persists the content to a
temporary filesystem path (preserving the original filename so the ingestion
pipeline derives a meaningful ``document_id`` from the stem), delegates the
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

# Allowlist of accepted file extensions.  Only plain-text formats supported by
# the ``TokenSlidingWindowChunker`` are permitted; binary formats (PDF, DOCX)
# require a dedicated extraction layer that is out of scope for this sprint.
_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt"})


@router.post(
    "/file",
    response_model=IngestResponse,
    summary="Ingest a document",
    description=(
        "Upload a ``.md`` or ``.txt`` file. The service chunks, embeds, and "
        "indexes the content into the vector store, then returns the chunk count."
    ),
    status_code=200,
)
async def ingest_file(
    file: UploadFile,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Validate, store, and ingest an uploaded document.

    Processing flow:
    1. Reject unsupported file extensions with HTTP 415.
    2. Read the uploaded content into memory.
    3. Write to a temporary directory preserving the original filename so that
       ``pipeline.ingest_document`` derives the correct ``document_id`` from
       the file stem.
    4. Delegate the full chunk → embed → index pipeline to ``IngestionPipeline``.
    5. Clean up the temporary directory unconditionally.

    Args:
        file: Multipart ``UploadFile`` provided by FastAPI.
        pipeline: Injected ``IngestionPipeline`` resolved from application state.

    Returns:
        ``IngestResponse`` confirming the filename and number of indexed chunks.

    Raises:
        HTTPException 415: When the uploaded file extension is not in
            ``_ALLOWED_EXTENSIONS``.
        HTTPException 422: When no filename is present in the multipart upload.
    """
    filename: str = file.filename or ""
    if not filename:
        raise HTTPException(status_code=422, detail="Uploaded file must include a filename.")

    suffix: str = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Accepted extensions: {', '.join(sorted(_ALLOWED_EXTENSIONS))}."
            ),
        )

    content: bytes = await file.read()

    # Write to a named temp directory that preserves the original filename.
    # ``IngestionPipeline.ingest_document`` uses ``Path.stem`` as the
    # ``document_id``, so maintaining the original name is load-bearing.
    def _write_temp() -> Path:
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_file = tmp_dir / filename
        tmp_file.write_bytes(content)
        return tmp_file

    tmp_path: Path = await asyncio.to_thread(_write_temp)

    try:
        chunks_ingested: int = await pipeline.ingest_document(tmp_path)
    finally:
        # Unconditional cleanup — errors during ingestion must not leave
        # temporary data on disk.
        await asyncio.to_thread(lambda: shutil.rmtree(tmp_path.parent, ignore_errors=True))

    logger.info(
        "Document ingested. filename=%r chunks=%d",
        filename,
        chunks_ingested,
    )
    return IngestResponse(
        status="success",
        filename=filename,
        chunks_ingested=chunks_ingested,
    )
