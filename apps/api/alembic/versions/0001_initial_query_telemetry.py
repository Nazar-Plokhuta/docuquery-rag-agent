"""Create the query_telemetry audit table.

Revision ID: 0001
Revises:
Create Date: 2026-10-24

The service's ``initialize_db()`` bootstrap can create this table before Alembic
ever runs against a database, for example a PostgreSQL instance first started
by the service. The create is skipped when the table already exists so that
adopting migrations on such a database does not fail with "relation already
exists".
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE_NAME = "query_telemetry"


def upgrade() -> None:
    # Offline (--sql) rendering has no connection to inspect.
    if not context.is_offline_mode() and sa.inspect(op.get_bind()).has_table(_TABLE_NAME):
        return

    op.create_table(
        _TABLE_NAME,
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("response_text", sa.Text(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("request_id"),
    )


def downgrade() -> None:
    op.drop_table(_TABLE_NAME)
