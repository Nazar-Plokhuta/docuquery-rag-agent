# DocuQuery RAG Agent — Architecture Reference

**Version:** 0.1.0  
**Status:** Authoritative — reflects implementation state as of Sprint 2 completion  
**Audience:** Internal engineering team  

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Clean Architecture Layers](#2-clean-architecture-layers)
3. [Data Contracts](#3-data-contracts)
4. [SQLite Telemetry Schema](#4-sqlite-telemetry-schema)
5. [Core Interfaces](#5-core-interfaces)
6. [Non-Negotiable Invariants](#6-non-negotiable-invariants)

---

## 1. System Overview

DocuQuery RAG Agent is an asynchronous Retrieval-Augmented Generation (RAG) microservice built on FastAPI. Its primary purpose is to ingest unstructured documents, store their semantically dense representations in a persistent vector database, and answer user queries with grounded, citation-carrying LLM responses — while maintaining a tamper-resistant audit trail of every query processed.

The service is designed for enterprise deployment. It prioritises:

- **Strict grounding** — the LLM is forbidden from fabricating facts; every factual claim must cite a source chunk by filename and section header.
- **Full async I/O** — every network call, database operation, and filesystem interaction is non-blocking. The sole exception is the ChromaDB local driver, which is explicitly offloaded to a thread pool via `asyncio.to_thread`.
- **Token budget enforcement** — no unbounded context is injected into LLM prompts; `tiktoken` measures and bounds every payload before it leaves the ingestion or query layer.
- **Clean separation of concerns** — presentation, domain, ingestion, storage, and configuration layers have explicit responsibility boundaries; cross-layer imports are directed inward only.

---

## 2. Clean Architecture Layers

The `src/` tree is partitioned into five layers. Dependency arrows point strictly inward: `api` → `core`, `ingestion` → `core`, `storage` → `core`. No inner layer imports from an outer one.

```
src/
├── config/          # Environment-bound configuration
├── core/            # Domain models, interfaces, exceptions
├── ingestion/       # Text parsing, chunking, batch embedding
├── storage/         # ChromaDB and SQLite concrete adapters
└── api/             # FastAPI routers, request/response DTOs
```

### 2.1 `src/config/`

**Sole responsibility:** Deliver a fully validated, immutable `Settings` object to any layer that needs it.

- `settings.py` — `Settings` extends `pydantic_settings.BaseSettings`. All fields are typed; the model reads from a `.env` file and environment variables. The module exposes `get_settings() -> Settings`, an LRU-cached factory that is the single allowed entry point for runtime configuration across the entire service.
- No layer other than `config` may call `os.environ` directly.

### 2.2 `src/core/`

**Sole responsibility:** Own the domain language — immutable data models, abstract storage contracts, and the exception hierarchy.

- `models.py` — Frozen Pydantic v2 models (`DocumentChunk`, `RetrievalResult`, `TelemetryRecord`). These are the canonical data contracts shared across all layers.
- `interfaces.py` — Abstract base classes (`VectorStoreInterface`, `AuditRepositoryInterface`). All storage consumers depend on these ABCs, never on concrete implementations.
- `exceptions.py` — `AppException` base and its concrete subclasses (`ConfigurationError`, `ResourceNotFoundError`, `StorageError`). All domain faults are expressed as typed subclasses of `AppException`.

No I/O, no driver imports, no business logic belongs here.

### 2.3 `src/ingestion/`

**Sole responsibility:** Transform raw document bytes into indexed, embedded `DocumentChunk` objects.

- `chunker.py` — `TokenSlidingWindowChunker` encodes text with `tiktoken`, slides a bounded window over the token sequence, decodes each window to UTF-8, and extracts the nearest preceding Markdown heading for semantic attribution.
- `pipeline.py` — `IngestionPipeline` orchestrates read → chunk → embed (OpenAI `text-embedding-3-small`) → index. It depends on `VectorStoreInterface` and `AsyncOpenAI`, never on ChromaDB directly.

The ingestion layer never formats prompt templates or writes to audit storage.

### 2.4 `src/storage/`

**Sole responsibility:** Provide concrete I/O adapters that implement the core interfaces.

- `vector_store.py` — `ChromaVectorStore` implements `VectorStoreInterface` using `chromadb.PersistentClient`. All blocking Chroma calls are wrapped in `asyncio.to_thread()`.
- `audit_db.py` — `SQLiteAuditRepository` implements `AuditRepositoryInterface` using `aiosqlite`. A single persistent connection is opened at application startup and closed at shutdown.

Storage adapters must never format response payloads for API consumers or parse document text.

### 2.5 `src/api/`

**Sole responsibility:** Expose HTTP endpoints, enforce request validation, and serialise domain results to JSON.

- `routes/health.py` — `GET /api/v1/health/` liveness probe. Returns `HealthResponse` populated from injected `Settings`.
- All endpoints use FastAPI `Depends` for service injection. Zero direct calls to ChromaDB, SQLite, or OpenAI are permitted inside routers.

---

## 3. Data Contracts

All domain models live in `src/core/models.py`. Every model carries `model_config = ConfigDict(frozen=True)`, making instances immutable and hashable across async call boundaries.

### 3.1 `DocumentChunk`

Represents a single bounded text segment extracted from a source document.

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `str` | Deterministic identifier: `{document_id}#c{chunk_index:04d}` |
| `document_id` | `str` | Identifier of the parent document (typically the file stem) |
| `content` | `str` | Decoded UTF-8 text of this chunk window |
| `metadata` | `dict[str, str \| int]` | Scalar key-value provenance: `document_id`, `section`, `chunk_index`, `token_count` |
| `token_count` | `int` | `cl100k_base` token count; stored explicitly to avoid re-tokenisation on every access |

The `metadata` type is intentionally restricted to `dict[str, str | int]` — ChromaDB's HNSW index accepts only scalar metadata values, and this constraint propagates from the storage boundary up to the model definition.

### 3.2 `RetrievalResult`

A scored chunk returned from a vector similarity search.

| Field | Type | Description |
|---|---|---|
| `chunk` | `DocumentChunk` | The retrieved chunk with full provenance |
| `score` | `float` | Cosine similarity in `[0, 1]`; higher is more semantically relevant |

The score is computed as `score = max(0.0, 1.0 - cosine_distance)`, where `cosine_distance ∈ [0, 1]` for unit-normalised OpenAI embeddings. The clamp guards against floating-point arithmetic producing values below zero.

### 3.3 `TelemetryRecord`

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

### 3.4 `HealthResponse`

Typed liveness probe payload, defined in `src/api/routes/health.py` rather than `core/models.py` because it is purely a presentation-layer contract with no domain usage.

| Field | Type | Description |
|---|---|---|
| `status` | `str` | Always `"ok"` while the process is alive |
| `app_name` | `str` | Human-readable service name |
| `version` | `str` | SemVer application version |
| `environment` | `str` | Active `app_env` value from settings (`development`, `staging`, `production`) |

---

## 4. SQLite Telemetry Schema

The audit database contains a single table. No foreign keys, no multi-table joins — the design prioritises zero-dependency portability and instant startup migrations.

### 4.1 Table DDL

```sql
CREATE TABLE IF NOT EXISTS query_telemetry (
    request_id        TEXT PRIMARY KEY,
    query_text        TEXT NOT NULL,
    response_text     TEXT NOT NULL,
    latency_ms        REAL NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens      INTEGER NOT NULL,
    model_name        TEXT NOT NULL,
    created_at        TEXT NOT NULL
);
```

### 4.2 Column Definitions

| Column | SQLite Type | Constraints | Notes |
|---|---|---|---|
| `request_id` | `TEXT` | `PRIMARY KEY` | UUID string; implicit B-tree index for O(log n) point lookups |
| `query_text` | `TEXT` | `NOT NULL` | Raw user query; unbounded length |
| `response_text` | `TEXT` | `NOT NULL` | Full LLM response; may be the anti-hallucination refusal phrase |
| `latency_ms` | `REAL` | `NOT NULL` | 64-bit IEEE-754 float; millisecond wall-clock precision |
| `prompt_tokens` | `INTEGER` | `NOT NULL` | Prompt token count from the OpenAI usage object |
| `completion_tokens` | `INTEGER` | `NOT NULL` | Completion token count from the OpenAI usage object |
| `total_tokens` | `INTEGER` | `NOT NULL` | Sum of prompt and completion tokens |
| `model_name` | `TEXT` | `NOT NULL` | OpenAI model identifier; enables per-model cost attribution |
| `created_at` | `TEXT` | `NOT NULL` | ISO-8601 UTC string; sorted lexicographically for `ORDER BY created_at DESC` |

### 4.3 Indexing Rationale

The only explicit index is the `PRIMARY KEY` on `request_id`, which SQLite implements as a B-tree index automatically. No additional secondary indexes are provisioned at this stage because:

- The dominant query pattern (`get_recent_logs`) reads the most recent rows ordered by `created_at`. At current expected volumes (≪ 100 k rows), a full sequential scan with `ORDER BY created_at DESC LIMIT ?` is faster than a secondary index scan due to SQLite's page cache behaviour.
- A secondary index on `created_at` is a planned Sprint 3 migration once query volume benchmarks justify it.

### 4.4 Connection Strategy

`SQLiteAuditRepository` opens a **single persistent `aiosqlite.Connection`** for the application's lifetime. This is the only strategy compatible with in-memory databases (`":memory:"`), where each new `aiosqlite.connect(":memory:")` call produces a completely isolated, empty database. The persistent connection is stored on `app.state.audit_repo` and closed in the lifespan shutdown hook.

---

## 5. Core Interfaces

Both abstract base classes live in `src/core/interfaces.py`. They define the contracts that bind the domain layer to the storage layer without introducing any coupling to a specific driver.

### 5.1 `VectorStoreInterface`

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

**Design note — decoupled `embeddings` parameter:** Pre-computed embedding vectors are passed as a separate `list[list[float]]` parameter rather than being embedded inside `DocumentChunk`. This preserves the primitive-scalar-only constraint on `DocumentChunk.metadata` (required by ChromaDB's HNSW layer) and keeps embedding generation — an I/O-bound OpenAI call — in the ingestion layer where it belongs, not inside the storage adapter.

### 5.2 `AuditRepositoryInterface`

```python
class AuditRepositoryInterface(ABC):
    @abstractmethod
    async def log_query(self, record: TelemetryRecord) -> None: ...

    @abstractmethod
    async def get_recent_logs(self, limit: int = 50) -> Sequence[TelemetryRecord]: ...
```

Both methods are `async`. Concrete implementations must not perform blocking I/O on the event loop thread.

---

## 6. Non-Negotiable Invariants

These constraints are enforced across the entire codebase. Violations must be rejected at code review.

### 6.1 Strict Async Rule

Every function that performs I/O — filesystem reads, database operations, network calls — must be declared `async` and awaited at call sites. `time.sleep`, synchronous `requests`, and blocking `open()` in hot paths are prohibited. The sole exception is `tiktoken`'s `encode`/`decode`, which is CPU-bound and runs synchronously within async handlers (it does not block the event loop for meaningful durations at normal document sizes).

### 6.2 Threadpool Offloading for ChromaDB

The ChromaDB Python client (`chromadb.PersistentClient`) is synchronous. Every call that touches it — collection creation, `add`, `query` — must be dispatched via `asyncio.to_thread()`. Direct invocation on the event loop thread is a blocking-call violation. The `ChromaVectorStore` adapter encapsulates this pattern; callers never interact with the raw client.

### 6.3 Token Bounding

No raw, unbounded document text may be passed to an LLM API call. Every context payload destined for an LLM prompt must be measured with `tiktoken` (`cl100k_base`) and truncated to fit within the configured budget before the API call is made. The `TokenSlidingWindowChunker` enforces this at ingestion time; the RAG orchestration layer (Sprint 3+) must enforce it again at query time when assembling the final prompt.

### 6.4 Anti-Hallucination Refusals

The system prompt injected into every LLM call must contain an explicit prohibition against fabricating facts. If the retrieved context is insufficient to answer a query, the model must return the deterministic fallback refusal phrase rather than speculating. Every factual statement in an LLM response must carry a citation in the form:

```
[Source: <filename>, Section: <header>]
```

This citation format is enforced by prompt design, not post-processing.

### 6.5 Domain Exception Hierarchy

Application code must never raise bare `Exception` or `RuntimeError` at the domain boundary. All expected failure modes are expressed as typed subclasses of `AppException`:

| Exception | HTTP Mapping | Trigger |
|---|---|---|
| `ResourceNotFoundError` | 404 | Requested document or entity not found |
| `StorageError` | 503 | ChromaDB or SQLite operation failure |
| `ConfigurationError` | 500 | Missing or invalid settings at boot time |
| `AppException` (catch-all) | 500 | Any other domain fault |

The centralised `app_exception_handler` in `src/main.py` translates all `AppException` subclasses to structured JSON error envelopes without leaking stack traces or internal paths.

### 6.6 Secrets Isolation

API keys, database paths, and all environment-specific values must be read exclusively from the `Settings` object. Zero hardcoded secrets or literal paths are permitted in source files. Test overrides use `Settings(openai_api_key="sk-test-placeholder", ...)` without touching the filesystem.
