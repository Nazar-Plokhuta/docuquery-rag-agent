# DocuQuery RAG Agent — Architecture Reference

**Version:** 2.0.0  
**Status:** Authoritative — reflects implementation state as of Sprint 7 (v2.0.0 Enterprise Core)  
**Audience:** Internal engineering team  

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Clean Architecture Layers](#2-clean-architecture-layers)
3. [API Endpoint Reference](#3-api-endpoint-reference)
4. [Data Contracts](#4-data-contracts)
5. [Telemetry Schema](#5-telemetry-schema)
6. [Core Interfaces](#6-core-interfaces)
7. [Non-Negotiable Invariants](#7-non-negotiable-invariants)
8. [Architectural Decision Log](#8-architectural-decision-log--sprints-37)

---

## 1. System Overview

DocuQuery RAG Agent is an asynchronous Retrieval-Augmented Generation (RAG) microservice built on FastAPI. Its primary purpose is to ingest unstructured documents, store their semantically dense representations in a persistent vector database, and answer user queries with grounded, citation-carrying LLM responses — while maintaining a tamper-resistant audit trail of every query processed.

The service is designed for enterprise deployment. It prioritises:

- **Strict grounding** — the LLM is forbidden from fabricating facts; every factual claim must cite a source chunk by filename and section header.
- **Full async I/O** — every network call, database operation, and filesystem interaction is non-blocking. The sole exception is the synchronous ChromaDB client (local or HTTP), which is explicitly offloaded to a thread pool via `asyncio.to_thread`.
- **Token budget enforcement** — no unbounded context is injected into LLM prompts; `tiktoken` measures and bounds every payload before it leaves the ingestion or query layer.
- **Clean separation of concerns** — presentation, domain, ingestion, storage, and configuration layers have explicit responsibility boundaries; cross-layer imports are directed inward only.

---

## 2. Clean Architecture Layers

The `src/` tree is partitioned into five layers. Dependency arrows point strictly inward: `api` → `core`, `ingestion` → `core`, `storage` → `core`. No inner layer imports from an outer one.

> **Path convention:** since the monorepo restructure, the API service lives in `apps/api/`. Every `src/...` and `tests/...` path in this document is relative to that directory (e.g. `src/main.py` is `apps/api/src/main.py`). Import paths (`from src.core...`) are unchanged.

```
apps/api/src/
├── config/          # Environment-bound configuration
├── core/            # Domain models, interfaces, exceptions
├── ingestion/       # Text parsing, chunking, batch embedding
├── storage/         # ChromaDB and SQLAlchemy (PostgreSQL) concrete adapters
└── api/             # FastAPI routers, request/response DTOs
```

### 2.1 `src/config/`

**Sole responsibility:** Deliver a fully validated, immutable `Settings` object to any layer that needs it.

- `settings.py` — `Settings` extends `pydantic_settings.BaseSettings`. All fields are typed; the model reads from a `.env` file and environment variables. The module exposes `get_settings() -> Settings`, an LRU-cached factory that is the single allowed entry point for runtime configuration across the entire service.
- No layer other than `config` may call `os.environ` directly.
- The `openai_base_url: str | None` field (mapped to `OPENAI_BASE_URL`) enables routing OpenAI-routed LiteLLM calls through any OpenAI-compatible gateway (vLLM, LocalAI; pair with an `openai/<model>` identifier) without code changes. When `None` (the default), requests target the official OpenAI platform. `src/core/llm.py::provider_credentials` forwards it as `api_base` only for OpenAI-routed models. Other providers (Anthropic, OpenRouter) are selected via the model identifier prefix and read their own API-key env vars.

### 2.2 `src/core/`

**Sole responsibility:** Own the domain language — immutable data models, abstract storage contracts, the LiteLLM provider gateway, and the exception hierarchy.

- `models.py` — Frozen Pydantic v2 models (`DocumentChunk`, `RetrievalResult`, `TelemetryRecord`). These are the canonical data contracts shared across all layers.
- `interfaces.py` — Abstract base classes (`VectorStoreInterface`, `AuditRepositoryInterface`). All storage consumers depend on these ABCs, never on concrete implementations.
- `llm.py` — LiteLLM gateway helpers: `configure_litellm` (global flags, applied once in the lifespan), `provider_credentials` (per-model credential routing), and `embed_texts` (`litellm.aembedding`). `RAGEngine` calls `litellm.acompletion` with the same credentials. Every completion and embedding goes through LiteLLM; no provider SDK (`openai`, `anthropic`) is imported anywhere. The model identifier alone selects the provider (OpenAI, Anthropic Claude, OpenRouter, Azure, or an OpenAI-compatible gateway such as vLLM), and `OPENAI_API_KEY` / `OPENAI_BASE_URL` are forwarded to OpenAI-routed models only.
- `exceptions.py` — `AppException` base and its concrete subclasses (`ConfigurationError`, `ResourceNotFoundError`, `StorageError`). All domain faults are expressed as typed subclasses of `AppException`.

No I/O, no driver imports, no business logic belongs here.

### 2.3 `src/ingestion/`

**Sole responsibility:** Transform raw document bytes into indexed, embedded `DocumentChunk` objects.

- `chunker.py` — `TokenSlidingWindowChunker` encodes text with `tiktoken`, slides a bounded window over the token sequence, decodes each window to UTF-8, and extracts the nearest preceding Markdown heading for semantic attribution.
- `pipeline.py` — `IngestionPipeline` orchestrates read → chunk → embed (via `litellm.aembedding`, default `text-embedding-3-small`) → index. It depends on `VectorStoreInterface` and LiteLLM, never on ChromaDB directly.

The ingestion layer never formats prompt templates or writes to audit storage.

### 2.4 `src/storage/`

**Sole responsibility:** Provide concrete I/O adapters that implement the core interfaces.

- `vector_store.py` — `ChromaVectorStore` implements `VectorStoreInterface` and builds its client lazily: `chromadb.PersistentClient` against `chroma_persist_directory` by default, or `chromadb.HttpClient(host, port)` when `chroma_host` is set. All blocking Chroma calls are wrapped in `asyncio.to_thread()`.
- `orm_models.py` — SQLAlchemy declarative `Base` and the `query_telemetry` table mapping; `Base.metadata` is Alembic's autogenerate target.
- `audit_db.py` — `SQLAlchemyAuditRepository` implements `AuditRepositoryInterface` over an `AsyncEngine`. Each operation uses a short-lived `AsyncSession`; the engine is created at application startup and disposed at shutdown.

Storage adapters must never format response payloads for API consumers or parse document text.

### 2.5 `src/api/`

**Sole responsibility:** Expose HTTP endpoints, enforce request validation, and serialise domain results to JSON.

- `routes/health.py` — `GET /api/v1/health/` liveness probe. Returns `HealthResponse` populated from injected `Settings`.
- `routes/query.py` — `POST /api/v1/query/` (synchronous RAG) and `POST /api/v1/query/stream` (SSE streaming RAG). Delegates entirely to `RAGEngine`.
- `routes/ingestion.py` — `POST /api/v1/ingest/file` multipart document upload. Validates extension, writes to a named temp path, delegates to `IngestionPipeline`, and cleans up unconditionally.
- `routes/analytics.py` — `GET /api/v1/analytics/recent` telemetry retrieval. Delegates to `AuditRepositoryInterface`.
- `schemas/` — Input DTOs (`QueryRequest`, `IngestResponse`) enforce `model_config = ConfigDict(extra="forbid")`. Output DTOs (`QueryResponse`, `AnalyticsResponse`) are frozen Pydantic models.
- `dependencies.py` — FastAPI `Depends` factories that resolve long-lived service instances from `app.state` on each request without incurring per-request reconnection overhead.

All endpoints use FastAPI `Depends` for service injection. Zero direct calls to ChromaDB, the audit database, or an LLM provider are permitted inside routers.

### 2.6 Composition root (`src/main.py`)

**Sole responsibility:** Assemble the dependency graph once during the lifespan and expose a single logging surface for the process.

`_configure_logging()` is the first lifespan step. It installs one `rich.logging.RichHandler` on the root logger (`rich_tracebacks=True`, `show_path=False`, `markup=True`, `log_time_format="[%X]"`) through `logging.basicConfig(..., force=True)`. `force=True` replaces any handler already attached to the root logger, including handlers left by a reloader or an earlier import.

Uvicorn installs its own handlers and disables propagation before the lifespan runs. The helper clears those handlers on `uvicorn`, `uvicorn.error`, and `uvicorn.access`, sets `propagate = True`, and aligns each logger's level with `Settings.log_level`. Application domain logs and Uvicorn server/access logs then share one timestamped Rich layout, without the concatenated `INFO:logger:` prefix.

---

## 3. API Endpoint Reference

All endpoints are served under the `/api/v1` versioning prefix. OpenAPI documentation is available at `/api/docs` (Swagger UI) and `/api/redoc`.

### 3.1 `GET /api/v1/health/`

**Purpose:** Liveness probe for load balancers, orchestrators, and monitoring systems.  
**Auth:** None.  
**Response:** `HealthResponse` — `status`, `app_name`, `version` (`2.0.0`), `environment`.  
**Side effects:** None. Does not probe downstream dependencies (DB, vector store, LLM).

### 3.2 `POST /api/v1/query/`

**Purpose:** Execute a synchronous RAG query. Blocks until the full LLM completion is available.  
**Auth:** None.  
**Request body:** `QueryRequest` — `query` (string, `min_length=1`), `top_k` (int, `1–20`, default `4`), `score_threshold` (float, `0.0–1.0`, default `0.3`). Unknown fields are rejected (`extra="forbid"`).  
**Response:** `QueryResponse` — `query`, `answer`, `citations[]`, `latency_ms`, `total_tokens`.  
**Anti-hallucination:** When no retrieved chunk meets `score_threshold`, `FALLBACK_REFUSAL_MESSAGE` is returned as `answer` and `citations` is empty. The LLM is never called on this fast path.  
**Telemetry:** Every call (including refusals) persists a `TelemetryRecord` to the audit database.

### 3.3 `POST /api/v1/query/stream`

**Purpose:** Execute a streaming RAG query, emitting incremental token deltas as Server-Sent Events.  
**Auth:** None.  
**Request body:** Identical to `QueryRequest`.  
**Response:** `StreamingResponse` with `Content-Type: text/event-stream`.

SSE event format:
```
data: {"token": "<delta_text>"}\n\n   — one event per token from the LiteLLM stream
data: [DONE]\n\n                       — explicit stream termination sentinel
```

**Proxy bypass headers:** `Cache-Control: no-cache` and `X-Accel-Buffering: no` are set unconditionally to prevent nginx and CDN reverse proxies from buffering the event stream (see ADR-05).  
**Telemetry:** Not logged for streaming queries; provider streams do not return a usage object in standard responses.

### 3.4 `POST /api/v1/ingest/file`

**Purpose:** Upload, chunk, embed, and index a supported document.  
**Auth:** None.  
**Request:** `multipart/form-data` with a single `file` field. Accepted extensions: `.md`, `.txt`, `.pdf`, `.docx`, `.csv`, `.xlsx`, `.html`, `.htm`. All other extensions return HTTP 415.  
**Response:** `IngestResponse` — `status` (`"success"`), `filename`, `chunks_ingested`.  
**Lifecycle:** Content is written to a temporary directory (preserving the original filename so `document_id` is derived from the stem), delegated to `IngestionPipeline`, and the temporary directory is deleted unconditionally in a `finally` block.  
**Edge case:** An empty file produces zero chunks and returns HTTP 200 (`chunks_ingested: 0`).

### 3.5 `GET /api/v1/analytics/recent`

**Purpose:** Retrieve recent RAG query telemetry for observability and cost attribution.  
**Auth:** None.  
**Query parameter:** `limit` (int, `1–1000`, default `50`).  
**Response:** `AnalyticsResponse` — `records[]` (list of `TelemetryRecord`), `total_count`.  
**Ordering:** Records are returned in descending `created_at` order (most recent first).

---

## 4. Data Contracts

All domain models live in `src/core/models.py`. Additional domain models introduced in Sprint 3 and 4 (`Citation`, `RAGResult`) are listed below. Every model carries `model_config = ConfigDict(frozen=True)`, making instances immutable and hashable across async call boundaries.

### 4.1 `DocumentChunk`

Represents a single bounded text segment extracted from a source document.

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `str` | Deterministic identifier: `{document_id}#c{chunk_index:04d}` |
| `document_id` | `str` | Identifier of the parent document (typically the file stem) |
| `content` | `str` | Decoded UTF-8 text of this chunk window |
| `metadata` | `dict[str, str \| int]` | Scalar key-value provenance: `document_id`, `section`, `chunk_index`, `token_count` |
| `token_count` | `int` | `cl100k_base` token count; stored explicitly to avoid re-tokenisation on every access |

The `metadata` type is intentionally restricted to `dict[str, str | int]` — ChromaDB's HNSW index accepts only scalar metadata values, and this constraint propagates from the storage boundary up to the model definition.

### 4.2 `RetrievalResult`

A scored chunk returned from a vector similarity search.

| Field | Type | Description |
|---|---|---|
| `chunk` | `DocumentChunk` | The retrieved chunk with full provenance |
| `score` | `float` | Cosine similarity in `[0, 1]`; higher is more semantically relevant |

The score is computed as `score = max(0.0, 1.0 - cosine_distance)`, where `cosine_distance ∈ [0, 1]` for unit-normalised OpenAI embeddings. The clamp guards against floating-point arithmetic producing values below zero.

### 4.3 `TelemetryRecord`

An immutable audit log entry capturing the full lifecycle of one RAG query.

| Field | Type | Description |
|---|---|---|
| `request_id` | `str` | UUID-based trace identifier for cross-service correlation |
| `query_text` | `str` | Raw user query text |
| `response_text` | `str` | Grounded LLM response returned to the caller |
| `latency_ms` | `float` | End-to-end wall-clock duration in milliseconds |
| `prompt_tokens` | `int` | Tokens consumed by the prompt (system + user messages) |
| `completion_tokens` | `int` | Tokens generated in the LLM completion |
| `total_tokens` | `int` | `prompt_tokens + completion_tokens` |
| `model_name` | `str` | LLM identifier used for generation (e.g. `gpt-4o-mini`) |
| `created_at` | `str` | ISO-8601 UTC timestamp (e.g. `2026-09-19T09:00:00Z`) |

### 4.4 `Citation`

A single source attribution, extracted from chunk metadata after token budgeting.

| Field | Type | Description |
|---|---|---|
| `source` | `str` | Document identifier, typically the file stem (e.g. `sla_policy`) |
| `section` | `str` | Nearest Markdown heading preceding the chunk, or `""` if none |
| `chunk_id` | `str` | Deterministic chunk identifier in `{document_id}#c{index:04d}` format |

### 4.5 `RAGResult`

The structured result returned by `RAGEngine.query()` to the presentation layer.

| Field | Type | Description |
|---|---|---|
| `query` | `str` | Original user query, echoed for client-side correlation |
| `answer` | `str` | Grounded LLM response, or `FALLBACK_REFUSAL_MESSAGE` |
| `citations` | `list[Citation]` | One entry per chunk injected into the LLM prompt |
| `latency_ms` | `float` | End-to-end wall-clock duration in milliseconds |
| `prompt_tokens` | `int` | Tokens consumed by the prompt |
| `completion_tokens` | `int` | Tokens generated in the completion |
| `total_tokens` | `int` | `prompt_tokens + completion_tokens` |
| `retrieved_chunks_count` | `int` | Number of chunks that passed both threshold filtering and token budgeting |

### 4.6 `HealthResponse`

Typed liveness probe payload, defined in `src/api/routes/health.py` rather than `core/models.py` because it is purely a presentation-layer contract with no domain usage.

| Field | Type | Description |
|---|---|---|
| `status` | `str` | Always `"ok"` while the process is alive |
| `app_name` | `str` | Human-readable service name |
| `version` | `str` | SemVer application version. Current release: `2.0.0` |
| `environment` | `str` | Active `app_env` value from settings (`development`, `staging`, `production`) |

---

## 5. Telemetry Schema

The audit database contains a single table. No foreign keys, no multi-table joins. The schema is declared once as a SQLAlchemy model (`src/storage/orm_models.py`) and versioned by Alembic (`apps/api/alembic/`). PostgreSQL is the production target; in-memory SQLite backs the test suite.

### 5.1 Table DDL

PostgreSQL DDL as emitted by migration `0001_initial_query_telemetry`:

```sql
CREATE TABLE query_telemetry (
    request_id        VARCHAR NOT NULL,
    query_text        TEXT NOT NULL,
    response_text     TEXT NOT NULL,
    latency_ms        FLOAT NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens      INTEGER NOT NULL,
    model_name        VARCHAR NOT NULL,
    created_at        VARCHAR NOT NULL,
    PRIMARY KEY (request_id)
);
```

### 5.2 Column Definitions

| Column | PostgreSQL Type | Constraints | Notes |
|---|---|---|---|
| `request_id` | `VARCHAR` | `PRIMARY KEY` | UUID string; implicit B-tree index for O(log n) point lookups |
| `query_text` | `TEXT` | `NOT NULL` | Raw user query; unbounded length |
| `response_text` | `TEXT` | `NOT NULL` | Full LLM response; may be the anti-hallucination refusal phrase |
| `latency_ms` | `DOUBLE PRECISION` | `NOT NULL` | 64-bit IEEE-754 float (SQLAlchemy `Float`); millisecond wall-clock precision |
| `prompt_tokens` | `INTEGER` | `NOT NULL` | Prompt token count from the LiteLLM usage object |
| `completion_tokens` | `INTEGER` | `NOT NULL` | Completion token count from the LiteLLM usage object |
| `total_tokens` | `INTEGER` | `NOT NULL` | Sum of prompt and completion tokens |
| `model_name` | `VARCHAR` | `NOT NULL` | LiteLLM model identifier; enables per-model cost attribution |
| `created_at` | `VARCHAR` | `NOT NULL` | ISO-8601 UTC string (`YYYY-MM-DDTHH:MM:SSZ`), stored as text so it round-trips byte-for-byte on every backend; sorts lexicographically for `ORDER BY created_at DESC` |

### 5.3 Indexing Rationale

The only explicit index is the `PRIMARY KEY` on `request_id`, which both PostgreSQL and SQLite back with a B-tree index automatically. No additional secondary indexes are provisioned at this stage because:

- The dominant query pattern (`get_recent_logs`) reads the most recent rows ordered by `created_at`. At current expected volumes (≪ 100 k rows), a sequential scan with a top-N sort (`ORDER BY created_at DESC LIMIT n`) is inexpensive.
- A secondary index on `created_at` is a candidate Alembic migration once query volume benchmarks justify it.

### 5.4 Connection Strategy

`SQLAlchemyAuditRepository` is built from one `AsyncEngine`, created in the lifespan with `create_async_engine(settings.database_url, pool_pre_ping=True, hide_parameters=True)` and shared for the application's lifetime. Each operation opens a short-lived `AsyncSession` and returns its pooled connection on exit, so concurrent requests do not queue behind a single connection. The repository is stored on `app.state.audit_repo`; `close()` disposes the engine in the lifespan shutdown hook.

For `sqlite+aiosqlite:///:memory:` SQLAlchemy backs the engine with a `StaticPool` that shares one connection, so the schema created by `initialize_db()` stays visible to later sessions. This is what keeps the test suite offline without a PostgreSQL server.

**Schema bootstrap.** `initialize_db()` runs `metadata.create_all`, which is idempotent and never alters an existing table; it exists for tests and first start. Migration `0001` skips table creation when the table already exists, so a database bootstrapped by the service can still be adopted by `alembic upgrade head`. Evolve production schemas with Alembic.

**Error reporting.** Database failures, including the raw `OSError` that asyncpg raises for an unreachable server, are re-raised as `StorageError`. The message carries exception type names only: `str()` of a SQLAlchemy error embeds bound parameters (user queries and model responses), and `StorageError.message` is returned to API clients.

### 5.5 Runtime Logging

The `query_telemetry` table is the durable audit trail. Process stdout is a separate channel and is never written into that table.

At lifespan startup, `_configure_logging()` in `src/main.py` routes both log streams through one `rich.logging.RichHandler`:

- Application domain loggers (for example `src.main`) inherit the root handler installed by `logging.basicConfig(..., force=True)`.
- Uvicorn server and access loggers — `uvicorn`, `uvicorn.error`, and `uvicorn.access` — have their default handlers cleared and `propagate` set to `True`, so access lines use the same timestamped Rich layout as domain logs.

---

## 6. Core Interfaces

Both abstract base classes live in `src/core/interfaces.py`. They define the contracts that bind the domain layer to the storage layer without introducing any coupling to a specific driver.

### 6.1 `VectorStoreInterface`

```python
class VectorStoreInterface(ABC):
    @abstractmethod
    async def add_documents(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> None: ...

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> Sequence[RetrievalResult]: ...
```

**Design note — decoupled `embeddings` parameter:** Pre-computed embedding vectors are passed as a separate `list[list[float]]` parameter rather than being embedded inside `DocumentChunk`. This preserves the primitive-scalar-only constraint on `DocumentChunk.metadata` (required by ChromaDB's HNSW layer) and keeps embedding generation — an I/O-bound embedding call — in the ingestion layer where it belongs, not inside the storage adapter.

### 6.2 `AuditRepositoryInterface`

```python
class AuditRepositoryInterface(ABC):
    @abstractmethod
    async def log_query(self, record: TelemetryRecord) -> None: ...

    @abstractmethod
    async def get_recent_logs(self, limit: int = 50) -> Sequence[TelemetryRecord]: ...
```

Both methods are `async`. Concrete implementations must not perform blocking I/O on the event loop thread.

---

## 7. Non-Negotiable Invariants

These constraints are enforced across the entire codebase. Violations must be rejected at code review.

### 7.1 Strict Async Rule

Every function that performs I/O — filesystem reads, database operations, network calls — must be declared `async` and awaited at call sites. `time.sleep`, synchronous `requests`, and blocking `open()` in hot paths are prohibited. The sole exception is `tiktoken`'s `encode`/`decode`, which is CPU-bound and runs synchronously within async handlers (it does not block the event loop for meaningful durations at normal document sizes).

### 7.2 Threadpool Offloading for ChromaDB

The ChromaDB Python clients (`chromadb.PersistentClient` and `chromadb.HttpClient`) are synchronous. Every call that touches it — collection creation, `add`, `query` — must be dispatched via `asyncio.to_thread()`. Direct invocation on the event loop thread is a blocking-call violation. The `ChromaVectorStore` adapter encapsulates this pattern; callers never interact with the raw client.

### 7.3 Token Bounding

No raw, unbounded document text may be passed to an LLM API call. Every context payload destined for an LLM prompt must be measured with `tiktoken` (`cl100k_base`) and truncated to fit within the configured budget before the API call is made. The `TokenSlidingWindowChunker` enforces this at ingestion time; the RAG orchestration layer (Sprint 3+) must enforce it again at query time when assembling the final prompt.

### 7.4 Anti-Hallucination Refusals

The system prompt injected into every LLM call must contain an explicit prohibition against fabricating facts. If the retrieved context is insufficient to answer a query, the model must return the deterministic fallback refusal phrase rather than speculating. Every factual statement in an LLM response must carry a citation in the form:

```
[Source: <filename>, Section: <header>]
```

This citation format is enforced by prompt design, not post-processing.

### 7.5 Domain Exception Hierarchy

Application code must never raise bare `Exception` or `RuntimeError` at the domain boundary. All expected failure modes are expressed as typed subclasses of `AppException`:

| Exception | HTTP Mapping | Trigger |
|---|---|---|
| `ResourceNotFoundError` | 404 | Requested document or entity not found |
| `StorageError` | 503 | ChromaDB or audit database operation failure |
| `ConfigurationError` | 500 | Missing or invalid settings at boot time |
| `AppException` (catch-all) | 500 | Any other domain fault |

The centralised `app_exception_handler` in `src/main.py` translates all `AppException` subclasses to structured JSON error envelopes without leaking stack traces or internal paths.

### 7.6 Secrets Isolation

API keys, database paths, and all environment-specific values must be read exclusively from the `Settings` object. Zero hardcoded secrets or literal paths are permitted in source files. Test overrides use `Settings(openai_api_key="sk-test-placeholder", ...)` without touching the filesystem.

---

## 8. Architectural Decision Log — Sprints 3–7

### ADR-07 — Multi-Container Compose Topology with Migrate-on-Start

**Context:** The single-container preview used file-backed SQLite and embedded Chroma, which does not exercise the production PostgreSQL path and cannot scale the vector store independently of the API.

**Decision:** `docker/docker-compose.yml` defines three services:

- `db` — `postgres:16-alpine`, named volume `pg_data`, `pg_isready` healthcheck.
- `api` — `depends_on: db (service_healthy)`; its command is `alembic upgrade head && exec uvicorn ...`, so the schema is current before the first request. Migrations are idempotent, so running them on every start is safe for a single replica. Its `DATABASE_URL` is assembled from `POSTGRES_*` and points at `db` on the internal network; the `DATABASE_URL` from the root `.env` (a `localhost` URL for `uv run`) is intentionally not forwarded.
- `chroma` — optional standalone `chromadb/chroma` server (named volume `chroma_data`) behind the `chroma` Compose profile. `api` depends on it with `required: false`, so it is only awaited when the profile is active.

`ChromaVectorStore` selects its client from settings: `chroma_host` unset → embedded `PersistentClient` (default, no server required); set → `HttpClient(chroma_host, chroma_port)`. The sync HTTP client is used rather than `AsyncHttpClient` so both modes share one `asyncio.to_thread()` boundary and one collection handle type (see §7.2). An empty `CHROMA_HOST` is treated as unset because Compose interpolates unset variables to an empty string.

**Consequences:**

- `uv run` development is unchanged: no new required settings, and `.env.example` keeps a local PostgreSQL or SQLite `DATABASE_URL`.
- Tests patch both Chroma client constructors and use in-memory SQLite; no Chroma or PostgreSQL server is contacted.
- Running several `api` replicas would race on `alembic upgrade head`; move migrations into a one-shot job before scaling out.

### ADR-05 — Server-Sent Events Streaming Protocol with Proxy-Buffering Bypass

**Context:** The `POST /api/v1/query/stream` endpoint must deliver incremental LLM token deltas to clients in real time. In typical enterprise deployments, the FastAPI service runs behind an nginx reverse proxy or a CDN edge layer. Without explicit header instructions, both nginx and most CDNs buffer the response body until either the connection closes or the buffer fills, negating the perceived latency benefit of streaming.

**Decision:** Every `StreamingResponse` from the streaming query endpoint sets two headers unconditionally:

```python
"Cache-Control": "no-cache"
"X-Accel-Buffering": "no"
```

`X-Accel-Buffering: no` is the nginx-specific directive that instructs the reverse proxy to flush each chunk to the client immediately rather than accumulating a buffer. `Cache-Control: no-cache` provides an RFC-standard equivalent signal for other intermediaries (CDNs, shared proxies) and prevents the event stream from being served from cache.

The SSE event format is the [WHATWG EventSource](https://html.spec.whatwg.org/multipage/server-sent-events.html) wire format: each event is a `data:` line terminated by two newlines (`\n\n`). Token deltas are JSON-encoded (`{"token": "<delta>"}`). The stream terminates with the explicit sentinel `data: [DONE]\n\n`, allowing clients to detect stream completion without relying on connection closure or a timeout.

**Consequences:**
- Clients consuming the stream via browser `EventSource` or any SSE-aware HTTP client receive token deltas with sub-second latency on the first token.
- The `[DONE]` sentinel is an explicit application-level terminator, making client-side stream handling deterministic regardless of network conditions.
- nginx deployments require no configuration changes; the `X-Accel-Buffering: no` header overrides any server-level `proxy_buffering on` directive.
- For deployments that do not use nginx, `Cache-Control: no-cache` alone signals intermediaries not to buffer. This dual-header strategy is the most portable proxy-bypass pattern available without modifying infrastructure configuration.

---

### ADR-06 — Single-Instance Lifespan Dependency Graph Shared Between Ingestion and Query Engines

**Context:** Both `IngestionPipeline` and `RAGEngine` depend on the same `ChromaVectorStore` and `Settings` instance. A naïve FastAPI implementation might construct these dependencies fresh for every `Depends` resolution, resulting in multiple `chromadb.PersistentClient` handles open concurrently (which triggers file-locking conflicts).

**Decision:** All long-lived service instances — `SQLAlchemyAuditRepository`, `ChromaVectorStore`, `TokenBudgetManager`, `RAGEngine`, and `IngestionPipeline` — are constructed exactly once inside the `@asynccontextmanager lifespan()` function and attached to `app.state` before the application begins accepting requests:

```python
app.state.audit_repo = audit_repo
app.state.vector_store = vector_store
app.state.rag_engine = rag_engine
app.state.ingestion_pipeline = ingestion_pipeline
```

Thin `Depends` factory functions in `src/api/dependencies.py` resolve these from `request.app.state` per request, returning the shared singleton without constructing a new object. The dependency tree is therefore built once at startup and torn down once at shutdown.

**Consequences:**
- A single `chromadb.PersistentClient` holds the exclusive write lock on the HNSW index throughout the application's lifetime. Concurrent `add_documents` and `similarity_search` calls serialize through `asyncio.to_thread()`, which is safe because ChromaDB manages its own internal locking.
- LiteLLM is process-global; `configure_litellm` applies its flags once at startup and no provider client needs to be shared.
- `SQLAlchemyAuditRepository` shares one pooled `AsyncEngine` (see ADR-02); each operation checks a connection out of the pool in its own short-lived session, so concurrent requests do not contend on a single connection.
- The ingestion and query paths share the same `ChromaVectorStore` instance, so documents indexed via `POST /api/v1/ingest/file` are immediately queryable via `POST /api/v1/query/` without any cache invalidation, replica lag, or consistency delay.
- `TokenBudgetManager` loads the `cl100k_base` tiktoken BPE vocabulary once in its `__init__` and caches the `tiktoken.Encoding` object on the instance. Sharing this instance avoids repeated BPE vocabulary deserialisation across requests.
