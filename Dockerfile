# ============================================================
# DocuQuery RAG Agent — Dockerfile
# Target: Production-grade, non-root, minimal attack surface
# Base:   python:3.11-slim (Debian bookworm)
# ============================================================

# --------------- Stage 1: dependency builder ---------------
FROM python:3.11-slim AS builder

# Install only the build tools needed to compile native wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency manifests first to maximise Docker layer cache hit rate.
COPY pyproject.toml ./

# Install project dependencies into an isolated prefix so the runtime stage
# can copy a clean, reproducible set of wheels without build tooling.
RUN pip install --upgrade pip \
    && pip install --prefix=/install . --no-deps \
    && pip install --prefix=/install \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.30.0" \
        "pydantic>=2.8.0" \
        "pydantic-settings>=2.4.0" \
        "chromadb>=0.5.0" \
        "openai>=1.40.0" \
        "tiktoken>=0.7.0" \
        "aiosqlite>=0.20.0" \
        "httpx>=0.27.0"

# --------------- Stage 2: lean runtime image ---------------
FROM python:3.11-slim AS runtime

# Security: run as an unprivileged user.
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Copy installed packages from the builder stage.
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source.
COPY src/ ./src/

# Persistent data directory — mounted as a Docker volume in production so that
# ChromaDB and SQLite state survives container replacement.
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
