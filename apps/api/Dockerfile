# ============================================================
# DocuQuery RAG Agent — Dockerfile
# Target: Production-grade, non-root, minimal attack surface
# Base:   python:3.11-slim (Debian bookworm)
# Build context: apps/api (see docker/docker-compose.yml)
# Dependencies: resolved from uv.lock via `uv sync --frozen --no-dev`
# ============================================================

# --------------- Stage 1: dependency builder ---------------
FROM python:3.11-slim AS builder

# Install only the build tools needed to compile native wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# UV_PYTHON_DOWNLOADS=never pins the venv to the image interpreter, which is
# the same one the runtime stage ships, so the copied venv's symlinks resolve.
# UV_LINK_MODE=copy keeps the venv self-contained instead of hard-linking
# into a build cache that does not exist in the runtime stage.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# The venv is created at its final runtime path: console-script shebangs
# embed absolute paths and break if the venv is relocated.
WORKDIR /app

# Copy dependency manifests first to maximise Docker layer cache hit rate.
# The project itself is not installed (--no-install-project): the service runs
# from the copied `src/` tree, and only third-party dependencies are needed.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# --------------- Stage 2: lean runtime image ---------------
FROM python:3.11-slim AS runtime

# Security: run as an unprivileged user.
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy the resolved virtual environment from the builder stage.
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Copy application source and the Alembic environment so the image can run
# `alembic upgrade head` against the configured DATABASE_URL.
COPY src/ ./src/
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Persistent data directory — mounted as a Docker volume in production so that
# ChromaDB (and the optional SQLite audit file) survives container replacement.
RUN mkdir -p /app/data && chown -R appuser:appgroup /app

USER appuser

# Expose the application port (overridable via APP_PORT env var in compose).
EXPOSE 8000

# Health-check wired to the liveness probe endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health/')"

# Start Uvicorn in production mode.  The number of workers should be tuned via
# the UVICORN_WORKERS environment variable in the deployment manifest.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
