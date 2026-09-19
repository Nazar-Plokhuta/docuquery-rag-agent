"""Analytics router: telemetry audit log retrieval endpoint.

Exposes the SQLite audit trail to API consumers for observability, cost
attribution, and compliance reporting.  The router depends exclusively on
``AuditRepositoryInterface`` — no direct database or ORM access occurs here.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_audit_repository
from src.api.schemas.analytics import AnalyticsResponse
from src.core.interfaces import AuditRepositoryInterface

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get(
    "/recent",
    response_model=AnalyticsResponse,
    summary="Recent telemetry records",
    description=(
        "Return the most recent RAG query telemetry records from the audit log, "
        "ordered by descending creation time."
    ),
)
async def get_recent_analytics(
    limit: int = Query(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of records to return (1–1000).",
    ),
    audit_repo: AuditRepositoryInterface = Depends(get_audit_repository),
) -> AnalyticsResponse:
    """Retrieve the most recent telemetry records from the audit repository.

    Records are returned in descending ``created_at`` order (newest first).
    The ``limit`` parameter caps the result set size to prevent unbounded
    response payloads under high query volumes.

    Args:
        limit: Maximum number of records to return. Defaults to 50.
        audit_repo: Injected ``AuditRepositoryInterface`` resolved from
            application state.

    Returns:
        ``AnalyticsResponse`` containing the paged record list and total count.
    """
    records = await audit_repo.get_recent_logs(limit=limit)
    record_list = list(records)
    logger.debug("Returning %d telemetry records (limit=%d).", len(record_list), limit)
    return AnalyticsResponse(
        records=record_list,
        total_count=len(record_list),
    )
