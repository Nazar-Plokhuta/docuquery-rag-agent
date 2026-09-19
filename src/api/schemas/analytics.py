"""Request and response schemas for the /analytics resource.

Analytics endpoints expose the SQLite audit trail to API consumers without
leaking internal storage types.  ``AnalyticsResponse`` wraps a page of
``TelemetryRecord`` objects with a total count field for client-side display.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.core.models import TelemetryRecord


class AnalyticsResponse(BaseModel):
    """Page of recent telemetry records returned by the analytics endpoint.

    Attributes:
        records: Ordered list of ``TelemetryRecord`` instances, sorted by
            descending ``created_at`` (most recent first), bounded by the
            ``limit`` query parameter.
        total_count: Number of records in this response page.  Equals
            ``len(records)`` and is included for client-side convenience.
    """

    model_config = ConfigDict(frozen=True)

    records: list[TelemetryRecord]
    total_count: int
