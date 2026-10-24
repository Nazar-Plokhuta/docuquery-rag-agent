"""Alembic migration environment for the async audit database.

The connection URL comes from ``Settings.database_url`` instead of
``alembic.ini`` so migrations always target the same database as the running
service (``DATABASE_URL`` overrides it for both) and no credentials are
duplicated in a tracked file.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from src.config.settings import get_settings
from src.storage.orm_models import Base

config = context.config

if config.config_file_name is not None:
    # Keep application loggers alive when migrations run inside another process.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Render migration SQL to stdout without connecting to the database."""
    context.configure(
        url=get_settings().database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    # NullPool: a one-shot CLI process gains nothing from pooling, and it lets
    # the process exit without lingering connections.
    connectable = create_async_engine(get_settings().database_url, poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Apply migrations over a live async connection."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
