"""SQLAlchemy declarative models for the relational audit store.

Kept separate from ``src.core.models``: those are immutable domain DTOs, while
the classes here are persistence mappings that only the storage layer and
Alembic may import.

Column names intentionally mirror ``TelemetryRecord`` field-for-field so the
repository can translate between the two without a mapping table.
"""

from __future__ import annotations

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base whose ``metadata`` is the Alembic autogenerate target."""


class QueryTelemetryRow(Base):
    """One persisted ``TelemetryRecord`` per completed RAG request.

    ``created_at`` is stored as the ISO-8601 UTC string produced by the RAG
    engine rather than a native timestamp. The fixed ``YYYY-MM-DDTHH:MM:SSZ``
    format sorts lexicographically in chronological order, and the string
    round-trips byte-for-byte on every backend, including SQLite, which has no
    timestamp type.
    """

    __tablename__ = "query_telemetry"

    request_id: Mapped[str] = mapped_column(String, primary_key=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
