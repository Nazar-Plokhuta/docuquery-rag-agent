"""Request and response schemas for the /ingest resource.

The ingestion endpoint accepts multipart file uploads directly (no wrapping
DTO) and returns ``IngestResponse`` to confirm successful indexing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class IngestResponse(BaseModel):
    """Confirmation payload returned after a successful document ingestion.

    Attributes:
        status: Ingestion outcome descriptor — always ``"success"`` on the
            happy path; a domain exception propagates to the error handler
            on failure.
        filename: Original filename of the uploaded document, preserved from
            the multipart upload for client-side correlation.
        chunks_ingested: Total number of ``DocumentChunk`` objects indexed into
            the vector store. Zero indicates the document was empty or contained
            only whitespace.
    """

    model_config = ConfigDict(frozen=True)

    status: str
    filename: str
    chunks_ingested: int
