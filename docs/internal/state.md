# DocuQuery RAG Agent — Living Implementation State

**Version:** 0.1.0  
**Sprint:** 5 (complete)  
**Test suite:** 105 tests — all passing  
**Last updated:** 2026-09-19  

> This document is a living record. Update it whenever a sprint closes or an ADR is ratified.

---

## Table of Contents

1. [Sprint Summary](#1-sprint-summary)
2. [Module Breakdown](#2-module-breakdown)
   - [2.1 `src/config/settings.py`](#21-srcconfigsettingspy)
   - [2.2 `src/storage/audit_db.py`](#22-srcstorageaudit_dbpy)
   - [2.3 `src/storage/vector_store.py`](#23-srcstoragevector_storepy)
   - [2.4 `src/ingestion/chunker.py`](#24-srcingestionchunkerpy)
   - [2.5 `src/ingestion/pipeline.py`](#25-srcingestionpipelinepy)
   - [2.6 `src/core/rag/prompts.py`](#26-srccoreragpromptspy)
   - [2.7 `src/core/rag/token_counter.py`](#27-srccoreragtoken_counterpy)
   - [2.8 `src/core/rag/engine.py`](#28-srccoreragenginepy)
   - [2.9 `src/api/routes/query.py`](#29-srcapiroutesquerypy)
   - [2.10 `src/api/routes/ingestion.py`](#210-srcapiroutesingestionpy)
   - [2.11 `src/api/routes/analytics.py`](#211-srcapiroutesanalyticspy)
   - [2.12 `src/main.py`](#212-srcmainpy)
3. [Test Suite Inventory](#3-test-suite-inventory)
4. [Architectural Decision Log](#4-architectural-decision-log)

---

## 1. Sprint Summary

| Sprint | Deliverable | Status |
|---|---|---|
| Sprint 1 | Project scaffold, `Settings`, `AppException` hierarchy, `VectorStoreInterface`, `AuditRepositoryInterface`, `DocumentChunk`, `RetrievalResult`, `TelemetryRecord`, `HealthResponse`, `GET /api/v1/health/` | Complete |
| Sprint 2 | `SQLiteAuditRepository`, `ChromaVectorStore`, `TokenSlidingWindowChunker`, `IngestionPipeline`, lifespan startup/teardown with audit DB lifecycle, full 33-test suite | Complete |
| Sprint 3 | `RAGEngine` (non-streaming query), `TokenBudgetManager`, `build_rag_prompt`, anti-hallucination system prompt, `Citation` and `RAGResult` domain models, `POST /api/v1/query/` endpoint, audit telemetry logging | Complete |
| Sprint 4 | SSE streaming query (`POST /api/v1/query/stream`), document ingestion endpoint (`POST /api/v1/ingest/file`), analytics endpoint (`GET /api/v1/analytics/recent`), full 105-test suite | Complete |
| Sprint 5 | `docs/internal/architecture.md` update (endpoints, ADR-05, ADR-06), enterprise sample datasets (`data/sample_docs/`), E2E smoke test script (`scripts/smoke_test.py`), B2B showcase `README.md` | Complete |

---

## 2. Module Breakdown

### 2.1 `src/config/settings.py`

**Class:** `Settings(BaseSettings)`

#### Pydantic Settings Schema

| Field | Type | Default | Source env var |
|---|---|---|---|
| `app_env` | `Literal["development", "staging", "production"]` | `"development"` | `APP_ENV` |
| `app_port` | `int` | `8000` | `APP_PORT` |
| `log_level` | `Literal["DEBUG","INFO","WARNING","ERROR","CRITICAL"]` | `"INFO"` | `LOG_LEVEL` |
| `openai_api_key` | `str` | `"sk-placeholder"` | `OPENAI_API_KEY` |
| `openai_model` | `str` | `"gpt-4o-mini"` | `OPENAI_MODEL` |
| `embedding_model` | `str` | `"text-embedding-3-small"` | `EMBEDDING_MODEL` |
| `openai_base_url` | `str \| None` | `None` | `OPENAI_BASE_URL` |
| `chroma_persist_directory` | `str` | `"./data/chroma_db"` | `CHROMA_PERSIST_DIRECTORY` |
| `sqlite_database_path` | `str` | `"./data/telemetry.db"` | `SQLITE_DATABASE_PATH` |

`model_config` uses `extra="ignore"` (not `"forbid"`). `Settings` is not an inbound API DTO; permitting unknown environment variables prevents startup failures in environments that inject additional system-level variables.

#### LRU Cache Behaviour

`get_settings()` is decorated with `@lru_cache(maxsize=1)`. Pydantic Settings resolves the `.env` file exactly once per process lifetime, regardless of how many modules call `get_settings()`. In tests, the cache is invalidated with `get_settings.cache_clear()` before injecting a `Settings` override, ensuring test isolation without restarting the process.

#### Environment Override Pattern (Tests)

```python
app.dependency_overrides[get_settings] = _test_settings
get_settings.cache_clear()
```

Both lines are required: the `dependency_overrides` dict intercepts FastAPI `Depends` resolution, while `cache_clear()` resets the singleton so direct `get_settings()` calls outside the DI container also receive the test configuration.

---

### 2.2 `src/storage/audit_db.py`

**Class:** `SQLiteAuditRepository(AuditRepositoryInterface)`

#### Connection Management

The repository holds a single `aiosqlite.Connection` as `self._connection`. The connection lifecycle is:

1. **`initialize_db()`** — opens the connection, sets `row_factory = aiosqlite.Row` for named column access, executes the `CREATE TABLE IF NOT EXISTS` DDL, commits, and stores the connection on the instance.
2. **`_get_connection()`** — internal accessor that auto-initialises if `self._connection is None`. This convenience path supports ad-hoc CLI usage; the canonical production path is explicit `initialize_db()` during lifespan startup.
3. **`close()`** — closes and nullifies the connection. Called from the FastAPI lifespan shutdown hook.

#### Schema Initialisation

The DDL uses `CREATE TABLE IF NOT EXISTS`, making `initialize_db()` idempotent. Repeated calls on a file-based database across process restarts are safe. On `":memory:"` databases, each new `SQLiteAuditRepository` instance opens a fresh, isolated schema — shared-connection semantics are mandatory because `:memory:` databases are scoped to their connection handle.

#### Parameterisation

All SQL statements use positional `?` placeholders. String interpolation in SQL is prohibited throughout this module. The tuple parameter order in `_INSERT_SQL` matches the `CREATE TABLE` column declaration order exactly, preventing silent field misalignment on schema changes.

#### Lifecycle Integration via `app.state`

During lifespan startup in `src/main.py`, the initialised repository is attached to `app.state.audit_repo`:

```python
audit_repo = SQLiteAuditRepository(settings)
await audit_repo.initialize_db()
app.state.audit_repo = audit_repo
```

Dependency injection resolves `SQLiteAuditRepository` from `request.app.state.audit_repo` via a `Depends` factory, avoiding per-request reconnection overhead.

#### Error Handling

Both `log_query` and `get_recent_logs` wrap their database calls in `try/except Exception` blocks that re-raise as `StorageError`. This ensures the presentation layer always receives a typed domain exception regardless of the underlying `aiosqlite` error.

---

### 2.3 `src/storage/vector_store.py`

**Class:** `ChromaVectorStore(VectorStoreInterface)`

#### Lazy Collection Instantiation

The `chromadb.PersistentClient` and the `Collection` handle are **not** created in `__init__`. Instantiation is deferred to `_get_collection()`, which is called on the first `add_documents` or `similarity_search` invocation. This ensures the constructor performs no I/O, keeping FastAPI startup fast and making the adapter trivially constructable in tests.

The lazy init guard:

```python
async def _get_collection(self) -> Collection:
    if self._collection is None:
        self._collection = await asyncio.to_thread(self._init_collection_sync)
    return self._collection
```

The cached `Collection` handle is safe to share across coroutines because all mutations are dispatched through `asyncio.to_thread()`, and Chroma manages its own internal locking.

#### Cosine Similarity Configuration

The collection is created with:

```python
metadata = {"hnsw:space": "cosine"}
```

This instructs Chroma's HNSW index to measure angular distance rather than L2 Euclidean distance. OpenAI `text-embedding-3-small` embeddings are unit-normalised, making cosine distance the correct metric — identical vectors yield distance 0, orthogonal vectors yield distance 1.

#### Score Conversion

Chroma returns cosine **distance** (not similarity). The adapter converts to a similarity score for API consumers:

```
score = max(0.0, 1.0 - distance)
```

The `max(0.0, ...)` clamp guards against floating-point edge cases where arithmetic produces a marginally negative distance value.

#### `asyncio.to_thread` Wrapping

Every call that touches the synchronous ChromaDB driver is wrapped:

```python
await asyncio.to_thread(
    collection.add,
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings,
)
```

```python
raw = await asyncio.to_thread(
    collection.query,
    query_embeddings=[query_embedding],
    n_results=top_k,
    include=["documents", "metadatas", "distances"],
)
```

The `_get_collection()` call that precedes `add_documents`'s `to_thread` block runs on the event loop thread (it is async and only touches Python object state); only the blocking `.add()` call is dispatched to a thread. This two-phase pattern avoids nesting `asyncio.to_thread` inside `asyncio.to_thread`.

#### Default Collection

The default collection name is `enterprise_knowledge_base`. It is configurable at construction time via the `collection_name` parameter to support multi-tenant or test-isolation scenarios.

---

### 2.4 `src/ingestion/chunker.py`

**Class:** `TokenSlidingWindowChunker` (Python `dataclass`)

#### `cl100k_base` Encoding

`tiktoken.get_encoding("cl100k_base")` is called once in `__post_init__` and stored as `self._encoding`. `cl100k_base` is the BPE encoding shared by `text-embedding-3-small` and the GPT-4 family — using the same tokenizer at ingestion time and at LLM prompt assembly time guarantees that token counts are consistent across the pipeline.

#### Window Bounding

The sliding-window algorithm:

1. Encode the full document text to a flat `list[int]` of tokens.
2. Iterate with `step = chunk_size - chunk_overlap` tokens per window advance.
3. Slice `tokens[start:start + chunk_size]` (clamped at EOF).
4. Decode the slice back to UTF-8 text.

Each resulting chunk is strictly bounded to `≤ chunk_size` tokens. The final chunk of a document may be shorter if the document's length is not an exact multiple of `step`. Defaults: `chunk_size=400`, `chunk_overlap=50`.

**Guard:** `__post_init__` raises `ValueError` if `chunk_overlap >= chunk_size`, preventing degenerate configurations where the window never advances.

#### Heading Context Extraction

`_nearest_containing_section(text, chunk_end_offset)` scans the text slice `text[:chunk_end_offset]` for ATX Markdown headings using the compiled pattern `r"^#{1,6}\s+(.+)$"` (MULTILINE). It returns the **last** match in the slice — the heading that most recently precedes or opens this chunk.

The `chunk_end_offset` is computed as:

```python
prefix_text = encoding.decode(tokens[:start_token])  # text before this chunk
chunk_end_offset = len(prefix_text) + len(chunk_text)
```

Scanning up to and **including** the chunk's last character, rather than up to the chunk's first character, is the critical design choice that correctly attributes the very first chunk of a document that opens with a heading. If the scan stopped before the chunk start, an opening heading would be missed.

#### Deterministic Chunk ID Format

```
{document_id}#c{chunk_index:04d}
```

Examples: `annual-report-2026#c0000`, `annual-report-2026#c0001`, `annual-report-2026#c0027`.

The four-digit zero-padded index supports up to 9 999 chunks per document before the format pads further. The `#c` separator prevents ambiguity when `document_id` itself contains digits.

#### Metadata Fields per Chunk

| Key | Type | Value |
|---|---|---|
| `document_id` | `str` | The `document_id` argument passed to `chunk_document` |
| `section` | `str` | Nearest preceding Markdown heading text, or `""` |
| `chunk_index` | `int` | Zero-based sequential chunk index |
| `token_count` | `int` | `cl100k_base` token count of this chunk's `content` |

`token_count` is stored both in `DocumentChunk.token_count` (the typed field) and in `metadata["token_count"]` (the scalar dict). Both values are derived from the same `len(window_tokens)` expression and are guaranteed to agree.

---

### 2.5 `src/ingestion/pipeline.py`

**Class:** `IngestionPipeline`

#### Constructor Dependencies

```python
def __init__(
    self,
    chunker: TokenSlidingWindowChunker,
    vector_store: VectorStoreInterface,
    openai_client: AsyncOpenAI,
    settings: Settings,
) -> None:
```

The pipeline depends on abstractions (`VectorStoreInterface`), not on `ChromaVectorStore` directly. In tests, a stub `VectorStoreInterface` implementation replaces the real adapter.

#### Batch Embedding with OpenAI `text-embedding-3-small`

Embedding generation is batched to respect OpenAI API rate limits:

```
_EMBEDDING_BATCH_SIZE = 100
```

The OpenAI Embeddings API accepts up to 2 048 inputs per call. The service caps batches at 100 to stay well within burst rate-limit budgets and to bound individual request latency. For a document producing `N` chunks, the pipeline issues `ceil(N / 100)` embedding API calls.

The `_embed_batch` helper calls:

```python
await self._openai.embeddings.create(
    model=self._settings.embedding_model,
    input=texts,
)
```

The API guarantees response items are returned in the same order as the input list, so `response.data[i].embedding` corresponds to `texts[i]` without additional sorting.

#### Ingest Flow

```
ingest_document(file_path)
    │
    ├── file_path.read_text(encoding="utf-8")
    ├── chunker.chunk_document(document_id, text)  → []  (early return on empty)
    │
    └── for batch in chunks[::_EMBEDDING_BATCH_SIZE]:
            embeddings = await _embed_batch([c.content for c in batch])
            await vector_store.add_documents(batch, embeddings)
```

Returns the total number of chunks indexed. Returns `0` without raising for documents that produce zero chunks (empty or whitespace-only files).

---

### 2.6 `src/core/rag/prompts.py`

**Constants:** `FALLBACK_REFUSAL_MESSAGE`, `_SYSTEM_PROMPT`  
**Function:** `build_rag_prompt(query, retrieved_contexts) -> list[dict[str, str]]`

#### Anti-Hallucination System Prompt

The system prompt enforces four rules by prompt instruction (not post-processing):

1. Answer **exclusively** from the provided context chunks.
2. If context is insufficient, output `FALLBACK_REFUSAL_MESSAGE` verbatim.
3. Do not fabricate facts, invent sources, hallucinate, or speculate.
4. Every factual claim must carry `[Source: <source>, Section: <section>]`.

#### Message Structure

`build_rag_prompt` returns a two-element list:

```
[
  {"role": "system", "content": _SYSTEM_PROMPT},
  {"role": "user",   "content": "--- BEGIN CONTEXT ---\n{chunks}\n--- END CONTEXT ---\n\nQuestion: {query}"}
]
```

The `--- BEGIN CONTEXT ---` / `--- END CONTEXT ---` delimiters provide unambiguous framing so the model can distinguish retrieved evidence from the user question. The structure matches the canonical instruction-following pattern for GPT-4-class models, where system-turn instructions receive the highest weight.

---

### 2.7 `src/core/rag/token_counter.py`

**Class:** `TokenBudgetManager`

#### `count_tokens(text, encoding_name)`

Encodes `text` with `tiktoken.get_encoding(encoding_name)` and returns `len(tokens)`. The default encoding is `"cl100k_base"`, matching the ingestion chunker and the GPT-4 / `text-embedding-3-small` family.

#### `fit_contexts_to_budget(chunks, max_context_tokens)`

Greedy bin-packing algorithm:

1. Sort `chunks` by descending `score` (highest-confidence evidence first).
2. Walk the sorted list; admit each chunk whose `token_count` fits within the remaining budget.
3. Return the admitted subset in the same descending-score order.

This greedy approach is optimal for the common case where all chunks are similar in size. The `max_context_tokens` default of `2 500` leaves headroom within GPT-4o-mini's 128 k context for the system prompt, user message framing, and completion output.

---

### 2.8 `src/core/rag/engine.py`

**Class:** `RAGEngine`

#### Query Pipeline (non-streaming)

`RAGEngine.query()` orchestrates the following pipeline stages in order:

| Stage | Implementation |
|---|---|
| Query embedding | `AsyncOpenAI.embeddings.create` |
| Top-k retrieval | `VectorStoreInterface.similarity_search` |
| Score threshold filtering | Inline list comprehension (`score >= threshold`) |
| Fast-path refusal | Return `FALLBACK_REFUSAL_MESSAGE` with zero tokens if no chunks survive |
| Token budgeting | `TokenBudgetManager.fit_contexts_to_budget` |
| Prompt construction | `build_rag_prompt(query, contexts)` |
| LLM completion | `AsyncOpenAI.chat.completions.create(temperature=0.0)` |
| Citation extraction | `_extract_citations(budgeted_chunks)` |
| Telemetry logging | `AuditRepositoryInterface.log_query(TelemetryRecord)` |

`temperature=0.0` is enforced unconditionally to maximise determinism and minimise hallucination risk from stochastic sampling.

#### Streaming Pipeline

`RAGEngine.stream_query()` is an `async def` generator. The retrieval, filtering, and budgeting stages are identical to `query()`. After prompt construction, it calls `chat.completions.create(stream=True)` and yields each `delta.content` string from the async stream. `None` deltas (produced by the first streaming chunk which carries only the role announcement) are silently skipped.

Telemetry is not logged for streaming queries because the OpenAI streaming API does not include a usage object in the default `stream=True` response.

---

### 2.9 `src/api/routes/query.py`

**Endpoints:** `POST /api/v1/query/` and `POST /api/v1/query/stream`

The non-streaming endpoint delegates to `RAGEngine.query()` and serialises the result as `QueryResponse`. The streaming endpoint wraps `RAGEngine.stream_query()` in an `_event_generator()` async generator that formats each token delta as a `data: {"token": "..."}\n\n` SSE event and appends `data: [DONE]\n\n` after the generator is exhausted.

Both endpoints share the same `QueryRequest` DTO, which enforces `extra="forbid"` and `min_length=1` on `query`.

---

### 2.10 `src/api/routes/ingestion.py`

**Endpoint:** `POST /api/v1/ingest/file`

Validates the uploaded file extension against `_ALLOWED_EXTENSIONS = {".md", ".txt"}`. Writes content to a named temp directory preserving the original filename (so `IngestionPipeline` derives the correct `document_id` from `Path.stem`). Delegates to `IngestionPipeline.ingest_document()`. Cleans up the temp directory unconditionally in a `finally` block regardless of ingestion success or failure.

---

### 2.11 `src/api/routes/analytics.py`

**Endpoint:** `GET /api/v1/analytics/recent`

Delegates to `AuditRepositoryInterface.get_recent_logs(limit)` and wraps the result in `AnalyticsResponse`. The `limit` query parameter is validated in the range `[1, 1000]` by FastAPI's `Query(ge=1, le=1000)` constraint.

---

### 2.12 `src/main.py`

#### Lifespan Startup/Teardown Sequence

The `@asynccontextmanager` lifespan function controls the full application boot:

| Step | Action |
|---|---|
| 1 | `logging.basicConfig(level=settings.log_level)` — configure root logger from settings |
| 2 | `_provision_directories()` — idempotent `mkdir(parents=True, exist_ok=True)` for `chroma_persist_directory` and `sqlite_database_path` parent |
| 3 | `SQLiteAuditRepository(settings).initialize_db()` — open persistent connection, create schema |
| 4 | `app.state.audit_repo = audit_repo` — attach to application state for downstream DI |
| 5 | `ChromaVectorStore(settings)` — lazy adapter; no I/O yet |
| 6 | `AsyncOpenAI(api_key=settings.openai_api_key)` — shared httpx connection pool |
| 7 | `TokenBudgetManager()` — loads `cl100k_base` BPE vocabulary once |
| 8 | `RAGEngine(...)` — assemble with all dependencies |
| 9 | `IngestionPipeline(...)` — assemble with chunker, vector store, OpenAI client |
| 10 | Attach all services to `app.state`; `yield` — application is live |
| 11 | `audit_repo.close()` — drain in-flight writes, close the SQLite connection |

The teardown hook executes whether the application shuts down gracefully (SIGTERM) or raises during request handling.

#### Exception Dispatching Table

The `_exception_to_status` function uses a declarative `dict[type[AppException], int]` mapping:

| Domain Exception | HTTP Status Code |
|---|---|
| `ResourceNotFoundError` | `404 Not Found` |
| `StorageError` | `503 Service Unavailable` |
| Any other `AppException` | `500 Internal Server Error` |

The `@app.exception_handler(AppException)` decorator intercepts all subclasses. The JSON error envelope is:

```json
{
  "error": "StorageError",
  "message": "ChromaDB add_documents failed: ..."
}
```

Stack traces and internal filesystem paths are suppressed to prevent information leakage in non-development environments.

#### Router Registration

```python
app.include_router(health_router.router, prefix="/api/v1")
app.include_router(query_router.router, prefix="/api/v1")
app.include_router(ingestion_router.router, prefix="/api/v1")
app.include_router(analytics_router.router, prefix="/api/v1")
```

All routers are registered under the same `/api/v1` versioning prefix.

---

## 3. Test Suite Inventory

All 105 tests pass with `pytest --asyncio-mode=auto`. No test requires live OpenAI credentials or a running ChromaDB server.

### `tests/unit/test_health.py` — 4 tests

| Test | What it verifies |
|---|---|
| `test_health_returns_200` | Endpoint responds HTTP 200 |
| `test_health_response_content_type` | `Content-Type` header contains `application/json` |
| `test_health_response_payload` | `status`, `app_name`, `version`, `environment` values are correct |
| `test_health_response_field_types` | All fields are non-empty strings |

### `tests/unit/test_audit_db.py` — 10 tests

**`TestInitialiseDb` (2)**

| Test | What it verifies |
|---|---|
| `test_initialize_db_does_not_raise` | `initialize_db()` completes without error on a fresh in-memory DB |
| `test_initialize_db_is_idempotent` | A second `initialize_db()` call on a new instance does not raise |

**`TestLogQuery` (3)**

| Test | What it verifies |
|---|---|
| `test_log_query_persists_record` | A logged record is retrievable via `get_recent_logs` |
| `test_log_query_all_fields_round_trip` | Every `TelemetryRecord` field survives a write-read cycle without mutation |
| `test_log_multiple_records` | Five distinct records are all persisted and retrievable |

**`TestGetRecentLogs` (5)**

| Test | What it verifies |
|---|---|
| `test_empty_table_returns_empty_list` | Returns `[]` on an empty table |
| `test_limit_is_respected` | `get_recent_logs(limit=3)` returns exactly 3 rows from 10 |
| `test_default_limit_is_fifty` | Default `limit=50` caps at 50 rows when 60 records exist |
| `test_ordering_is_descending_by_created_at` | Rows are returned in descending `created_at` order |
| `test_returns_list_of_telemetry_records` | Every element is a `TelemetryRecord` instance |

### `tests/unit/test_chunker.py` — 19 tests

**`TestChunkIdFormat` (3)**

| Test | What it verifies |
|---|---|
| `test_ids_match_pattern` | All chunk IDs match `^{doc_id}#c\d{4}$` |
| `test_ids_are_zero_based_and_sequential` | Indices start at 0 and increment with no gaps |
| `test_document_id_embedded_in_chunk_id` | `document_id` prefix is exact |

**`TestChunkSizeBoundary` (2)**

| Test | What it verifies |
|---|---|
| `test_no_chunk_exceeds_chunk_size` | `chunk.token_count <= chunk_size` for all chunks |
| `test_token_count_matches_actual_encoding` | `chunk.token_count` equals live `tiktoken` re-count |

**`TestChunkOverlap` (2)**

| Test | What it verifies |
|---|---|
| `test_overlap_tokens_shared_between_adjacent_chunks` | Adjacent chunks share ≥ `chunk_overlap - 2` tokens at the boundary |
| `test_multiple_chunk_overlaps_are_consistent` | Overlap is consistent across all consecutive pairs |

**`TestMarkdownSectionMetadata` (3)**

| Test | What it verifies |
|---|---|
| `test_first_chunk_captures_h2_section` | Opening H2 heading is captured on the first chunk |
| `test_section_transitions_across_chunks` | Section transitions correctly when a new heading appears mid-document |
| `test_no_heading_yields_empty_section` | `metadata["section"] == ""` for documents with no headings |

**`TestEdgeCases` (6)**

| Test | What it verifies |
|---|---|
| `test_empty_string_returns_empty_list` | `""` input returns `[]` without raising |
| `test_whitespace_only_returns_empty_list` | Whitespace-only input returns `[]` without raising |
| `test_short_document_yields_single_chunk` | Sub-threshold document produces exactly one chunk |
| `test_document_exactly_chunk_size_yields_single_chunk` | Exactly `chunk_size` tokens produces one chunk |
| `test_invalid_overlap_raises_value_error` | `chunk_overlap == chunk_size` raises `ValueError` |
| `test_overlap_greater_than_size_raises_value_error` | `chunk_overlap > chunk_size` raises `ValueError` |

**`TestMetadataConsistency` (3)**

| Test | What it verifies |
|---|---|
| `test_token_count_agrees_between_field_and_metadata` | `chunk.token_count == chunk.metadata["token_count"]` for all chunks |
| `test_chunk_index_in_metadata_matches_id` | `metadata["chunk_index"]` equals the numeric suffix in `chunk_id` |
| `test_document_id_in_metadata_matches_argument` | `metadata["document_id"]` equals the `document_id` argument |

### `tests/unit/test_rag_prompts.py` — 17 tests

**`TestBuildRagPromptStructure` (6)**

| Test | What it verifies |
|---|---|
| `test_returns_exactly_two_messages` | `build_rag_prompt` returns a list with exactly two messages |
| `test_first_message_role_is_system` | First message has `role == "system"` |
| `test_second_message_role_is_user` | Second message has `role == "user"` |
| `test_all_messages_have_content_key` | Both messages carry a non-empty `content` key |
| `test_works_with_single_context` | Single-element context does not raise |
| `test_works_with_empty_context_list` | Empty context does not raise |

**`TestSystemPromptAntiHallucination` (5)**

| Test | What it verifies |
|---|---|
| `test_system_prompt_contains_fallback_refusal_phrase` | `FALLBACK_REFUSAL_MESSAGE` appears verbatim in the system prompt |
| `test_system_prompt_contains_citation_format` | `[Source:` and `Section:` markers are present |
| `test_system_prompt_forbids_external_knowledge` | Explicit prohibition against using external knowledge |
| `test_system_prompt_forbids_fabrication` | Explicit prohibition against fabricating facts |
| `test_system_prompt_instructs_exclusive_context_use` | Model is instructed to answer exclusively from context |

**`TestUserMessageContextFraming` (6)**

| Test | What it verifies |
|---|---|
| `test_user_message_contains_begin_context_delimiter` | `--- BEGIN CONTEXT ---` is present |
| `test_user_message_contains_end_context_delimiter` | `--- END CONTEXT ---` is present |
| `test_begin_delimiter_precedes_end_delimiter` | BEGIN appears before END |
| `test_query_is_present_in_user_message` | Original query string appears in user message |
| `test_query_appears_after_end_context_delimiter` | Query follows the END delimiter |
| `test_all_context_strings_present_in_user_message` | Every context string appears verbatim |

### `tests/unit/test_token_counter.py` — 10 tests

**`TestCountTokens` (4)**

| Test | What it verifies |
|---|---|
| `test_count_tokens_matches_tiktoken` | Token count equals the reference `tiktoken cl100k_base` encode length |
| `test_count_tokens_empty_string_returns_zero` | Empty string encodes to zero tokens |
| `test_count_tokens_single_word` | Single common word produces exactly one token |
| `test_count_tokens_custom_encoding` | Alternative encoding name loads correctly |

**`TestFitContextsToBudget` (6)**

| Test | What it verifies |
|---|---|
| `test_all_chunks_admitted_when_budget_is_ample` | All chunks admitted when combined count fits the budget |
| `test_overflow_chunk_is_dropped` | A chunk exceeding the remaining budget is dropped |
| `test_chunks_selected_in_descending_score_order` | Higher-scored chunks are admitted first |
| `test_empty_input_returns_empty_list` | Empty input returns `[]` without raising |
| `test_result_preserves_descending_score_ordering` | Returned list is sorted by descending score |
| `test_zero_budget_drops_all_chunks` | `max_context_tokens=0` drops all chunks |

### `tests/unit/test_rag_engine.py` — 12 tests

**`TestRAGEngineQuery` (8)**

| Test | What it verifies |
|---|---|
| `test_below_threshold_returns_fallback_without_completion_call` | No LLM call when no chunk meets the score threshold |
| `test_below_threshold_logs_telemetry_with_zero_tokens` | Fallback path persists a `TelemetryRecord` with zero token counts |
| `test_successful_retrieval_invokes_chat_completion` | Chat completions are called exactly once when chunks pass the threshold |
| `test_successful_retrieval_returns_correct_citations` | Citations are extracted from the metadata of budgeted chunks |
| `test_successful_retrieval_logs_audit_record` | A `TelemetryRecord` is persisted after a successful completion |
| `test_successful_retrieval_populates_rag_result_fields` | `RAGResult` fields reflect the completion response and budgeted chunks |
| `test_empty_similarity_search_returns_fallback` | Empty similarity search result triggers refusal |
| `test_token_budget_is_applied_before_completion` | `fit_contexts_to_budget` is called with the filtered chunk list |

**`TestRAGEngineStream` (4)**

| Test | What it verifies |
|---|---|
| `test_stream_query_yields_fallback_when_no_chunks_pass_threshold` | Exactly the refusal phrase is yielded; LLM not called |
| `test_stream_query_yields_tokens_sequentially` | Streaming tokens are yielded in the order they arrive |
| `test_stream_query_skips_none_delta_content` | `None` delta content (role announcement chunk) is skipped |
| `test_stream_query_empty_similarity_result_yields_fallback` | Empty similarity result triggers the stream fallback |

### `tests/integration/test_api_query.py` — 9 tests

| Test | What it verifies |
|---|---|
| `test_query_returns_200_with_valid_response` | HTTP 200 with well-formed `QueryResponse` |
| `test_query_engine_called_with_correct_parameters` | Route forwards all parameters to `RAGEngine.query` |
| `test_query_response_includes_citations` | Citations pass through verbatim from engine result |
| `test_query_rejects_empty_query_string` | Empty `query` returns HTTP 422 |
| `test_query_rejects_missing_query_field` | Missing `query` field returns HTTP 422 |
| `test_query_rejects_top_k_below_minimum` | `top_k=0` returns HTTP 422 |
| `test_query_rejects_top_k_above_maximum` | `top_k=21` returns HTTP 422 |
| `test_query_rejects_extra_fields` | Unknown fields return HTTP 422 |
| `test_query_token_usage_is_preserved` | `total_tokens` passes through without modification |

### `tests/integration/test_api_analytics.py` — 9 tests

| Test | What it verifies |
|---|---|
| `test_analytics_returns_200` | HTTP 200 on a valid request |
| `test_analytics_returns_recent_records` | Records from repository appear verbatim in response |
| `test_analytics_total_count_equals_records_length` | `total_count == len(records)` invariant |
| `test_analytics_empty_audit_log_returns_empty_list` | Empty log returns `records=[]` and `total_count=0` |
| `test_analytics_limit_parameter_is_forwarded` | `limit` query parameter is forwarded to `get_recent_logs` |
| `test_analytics_default_limit_is_fifty` | Default `limit=50` is applied when omitted |
| `test_analytics_rejects_limit_zero` | `limit=0` returns HTTP 422 |
| `test_analytics_rejects_limit_above_maximum` | `limit=1001` returns HTTP 422 |
| `test_analytics_record_fields_are_complete` | All `TelemetryRecord` fields are present in the serialised response |

### `tests/integration/test_api_ingestion.py` — 8 tests

| Test | What it verifies |
|---|---|
| `test_ingest_rejects_pdf_extension` | `.pdf` upload returns HTTP 415 |
| `test_ingest_rejects_exe_extension` | `.exe` upload returns HTTP 415 |
| `test_ingest_rejects_docx_extension` | `.docx` upload returns HTTP 415 |
| `test_ingest_processes_md_file` | `.md` file returns HTTP 200 with `IngestResponse` |
| `test_ingest_processes_txt_file` | `.txt` file returns HTTP 200 with `IngestResponse` |
| `test_ingest_response_status_is_success` | `status == "success"` on the happy path |
| `test_ingest_zero_chunks_for_empty_document` | Empty file returns HTTP 200 with `chunks_ingested=0` |
| `test_ingest_pipeline_called_with_path_argument` | Pipeline receives a `Path` with the correct stem |

### `tests/integration/test_api_stream.py` — 7 tests

| Test | What it verifies |
|---|---|
| `test_stream_returns_text_event_stream_content_type` | Response has `Content-Type: text/event-stream` |
| `test_stream_contains_done_sentinel` | Stream body includes the `[DONE]` termination sentinel |
| `test_stream_sse_token_format` | Each token delta is wrapped in `data: {"token": "..."}\n\n` |
| `test_stream_sse_events_are_valid_json` | All `data:` lines except `[DONE]` contain parseable JSON with a `token` key |
| `test_stream_fallback_refusal_is_yielded` | Fallback refusal phrase appears in the stream body |
| `test_stream_rejects_empty_query` | Empty `query` returns HTTP 422 before streaming begins |
| `test_stream_engine_called_with_correct_parameters` | Route forwards all parameters to `stream_query` |

---

## 4. Architectural Decision Log

### ADR-01 — Threadpool Isolation for ChromaDB Local Operations

**Context:** The FastAPI event loop is single-threaded. ChromaDB's `PersistentClient` — including `get_or_create_collection`, `collection.add`, and `collection.query` — is a synchronous, blocking implementation backed by SQLite and in-process HNSW indexing (hnswlib). Calling these methods directly on the event loop thread would stall all concurrent requests for the duration of each operation.

**Decision:** All ChromaDB driver calls are dispatched to Python's default `ThreadPoolExecutor` via `asyncio.to_thread()`. The `ChromaVectorStore` adapter encapsulates this pattern completely. No caller outside the adapter touches the raw `chromadb` client.

**Consequences:**
- The event loop remains unblocked during vector index operations.
- Each `add_documents` and `similarity_search` call acquires a thread from the pool and releases it on completion. At very high concurrency, thread pool exhaustion is a potential bottleneck; this will be revisited with a configurable executor.
- The `_get_collection()` lazy-init coroutine first checks `self._collection is None` on the event loop thread (a pure Python check, zero I/O), then dispatches `_init_collection_sync` to a thread only when initialisation is actually needed. Subsequent calls skip the thread dispatch entirely.

---

### ADR-02 — Shared Persistent `aiosqlite` Connection Across Lifespan

**Context:** `aiosqlite` wraps SQLite with an async interface by running each database operation in a background thread. A naïve implementation might open a fresh connection for every `log_query` call. This is problematic for two reasons: (1) it imposes a per-call file open/close overhead, and (2) it is incompatible with in-memory SQLite databases, where `aiosqlite.connect(":memory:")` creates an entirely new, isolated database on each call.

**Decision:** `SQLiteAuditRepository` opens a single `aiosqlite.Connection` in `initialize_db()` and reuses it for all subsequent operations. The connection is attached to `app.state` at startup and closed explicitly in the lifespan teardown hook.

**Consequences:**
- In-memory databases work correctly in tests: the single open connection holds the schema alive for the repository's entire lifetime.
- File-based databases benefit from connection reuse: the OS file descriptor is held open, avoiding repeated open/stat/close syscalls under moderate request rates.
- The `_get_connection()` auto-init guard provides a safety net for ad-hoc non-lifespan usage (e.g., one-off CLI scripts) without requiring callers to manage lifecycle explicitly.
- Thread safety: `aiosqlite` serialises all database operations through an internal `asyncio.Queue`, so the shared connection is safe to use from multiple concurrent coroutines.

---

### ADR-03 — Decoupled `embeddings` List Parameter in `VectorStoreInterface.add_documents`

**Context:** An earlier design considered embedding the dense vector inside `DocumentChunk` as an additional field. This was rejected because: (1) `DocumentChunk.metadata` is constrained to `dict[str, str | int]` for ChromaDB compatibility — a `list[float]` embedding would violate this constraint and require a parallel `metadata` structure; (2) `DocumentChunk` is a domain model shared between ingestion and retrieval contexts; forcing it to carry a mutable embedding couples the ingestion concern to all consumers of the type.

**Decision:** Pre-computed embedding vectors are passed as a separate `embeddings: list[list[float]] | None` parameter on `VectorStoreInterface.add_documents`. The `IngestionPipeline` calls the OpenAI Embeddings API, collects the resulting vectors, and passes them alongside the `DocumentChunk` sequence in a single `add_documents` call.

**Consequences:**
- `DocumentChunk` remains a pure, primitive-typed domain model. It can be stored, logged, and passed across layers without carrying embedding infrastructure.
- The storage adapter receives embeddings as a native Python list, which it passes directly to `collection.add(embeddings=...)` without type conversion.
- The interface allows `embeddings=None` to support hypothetical future adapters that generate embeddings internally (e.g., a local embedding server co-located with ChromaDB). The current `ChromaVectorStore` implementation passes `None` through to `collection.add`, deferring to any Chroma-side embedding function if configured.

---

### ADR-04 — Header-Inclusive Scanning for Markdown Chunks

**Context:** A heading that opens a chunk (i.e., the chunk begins with a Markdown ATX heading on its first line) should be attributed to that chunk's `section` metadata. A naive implementation that scans `text[:start_offset]` — only the text *before* the chunk starts — would miss this case, leaving the first chunk of a heading-delimited section with `section=""`.

**Decision:** `_nearest_containing_section(text, chunk_end_offset)` scans `text[:chunk_end_offset]` — the prefix extending up to and **including the chunk's last character**. The `chunk_end_offset` is computed as `len(prefix_text) + len(chunk_text)`, where `prefix_text` is the decoded text of all tokens before the current window.

**Consequences:**
- A heading that starts at the exact first character of a chunk is correctly captured, not skipped.
- A heading that falls mid-chunk is also captured — the last heading seen before the chunk's end is the nearest containing section, which is the correct attribution for the chunk's content.
- If a document begins with a heading and has no preceding text, the first chunk's `section` metadata will correctly reflect that heading.
- The trade-off is that the scan window is slightly larger than the strict "prefix only" approach, but the cost is a constant-time regex scan over a string slice — negligible relative to the `tiktoken` encode/decode operations.

---

### ADR-05 — Server-Sent Events Streaming Protocol with Proxy-Buffering Bypass

**Context:** The `POST /api/v1/query/stream` endpoint must deliver incremental LLM token deltas to clients in real time. In typical enterprise deployments, the FastAPI service runs behind an nginx reverse proxy or a CDN edge layer. Without explicit header instructions, both nginx and most CDNs buffer the response body until either the connection closes or the buffer fills, negating the perceived latency benefit of streaming.

**Decision:** Every `StreamingResponse` from the streaming query endpoint sets two headers unconditionally:

```python
"Cache-Control": "no-cache"
"X-Accel-Buffering": "no"
```

`X-Accel-Buffering: no` is the nginx-specific directive that instructs the reverse proxy to flush each chunk to the client immediately rather than accumulating a buffer. `Cache-Control: no-cache` provides an RFC-standard equivalent signal for other intermediaries (CDNs, shared proxies) and prevents the event stream from being served from cache.

The SSE event format follows the [WHATWG EventSource](https://html.spec.whatwg.org/multipage/server-sent-events.html) wire format: each event is a `data:` line terminated by two newlines (`\n\n`). Token deltas are JSON-encoded (`{"token": "<delta>"}`). The stream terminates with the explicit sentinel `data: [DONE]\n\n`, allowing clients to detect stream completion without relying on connection closure or a timeout.

**Consequences:**
- Clients consuming the stream via browser `EventSource` or any SSE-aware HTTP client receive token deltas with sub-second latency on the first token.
- The `[DONE]` sentinel is an explicit application-level terminator, making client-side stream handling deterministic regardless of network conditions.
- nginx deployments require no configuration changes; the `X-Accel-Buffering: no` header overrides any server-level `proxy_buffering on` directive.
- For deployments that do not use nginx, `Cache-Control: no-cache` alone signals intermediaries not to buffer.

---

### ADR-06 — Single-Instance Lifespan Dependency Graph Shared Between Ingestion and Query Engines

**Context:** Both `IngestionPipeline` and `RAGEngine` depend on the same `ChromaVectorStore`, `AsyncOpenAI` client, and `Settings` instance. A naïve FastAPI implementation might construct these dependencies fresh for every `Depends` resolution, resulting in multiple `chromadb.PersistentClient` handles open concurrently (which triggers file-locking conflicts) and multiple `AsyncOpenAI` instances that each maintain their own `httpx` connection pool (inflating file descriptor usage and TLS handshake overhead).

**Decision:** All long-lived service instances — `SQLiteAuditRepository`, `ChromaVectorStore`, `AsyncOpenAI`, `TokenBudgetManager`, `RAGEngine`, and `IngestionPipeline` — are constructed exactly once inside the `@asynccontextmanager lifespan()` function and attached to `app.state` before the application begins accepting requests. Thin `Depends` factory functions in `src/api/dependencies.py` resolve these from `request.app.state` per request.

**Consequences:**
- A single `chromadb.PersistentClient` holds the exclusive write lock on the HNSW index throughout the application's lifetime.
- A single `AsyncOpenAI` instance shares one `httpx.AsyncClient` and its underlying connection pool across all concurrent requests, reducing TLS handshake overhead and file descriptor usage under load.
- Documents indexed via `POST /api/v1/ingest/file` are immediately queryable via `POST /api/v1/query/` without any cache invalidation, replica lag, or consistency delay — both paths share the same `ChromaVectorStore` instance.
- `TokenBudgetManager` loads the `cl100k_base` tiktoken BPE vocabulary once and caches the `tiktoken.Encoding` object on the instance, avoiding repeated BPE vocabulary deserialisation across requests.
