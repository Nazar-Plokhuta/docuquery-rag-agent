# DocuQuery RAG Agent

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-105%20Passing-brightgreen?style=flat-square&logo=pytest&logoColor=white)
![Ruff](https://img.shields.io/badge/Linting-Ruff-D7FF64?style=flat-square)
![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> **Enterprise-grade Retrieval-Augmented Generation microservice.** Strict grounding. Zero hallucinations. Full async. Production-ready from day one.

---

## The Problem With Naive RAG in Enterprise B2B

Standard off-the-shelf RAG pipelines fail in production for three predictable reasons:

| Failure Mode | Root Cause | Business Impact |
|---|---|---|
| **Hallucinated answers** | LLM generates facts not present in retrieved context | Legal liability, lost client trust, compliance failures |
| **Unbounded token costs** | No context window budgeting; full documents injected into every prompt | Unpredictable OpenAI API bills at scale |
| **Ungrounded responses** | No citation enforcement; impossible to audit where an answer came from | Cannot satisfy SOC 2, HIPAA, or internal governance requirements |

**DocuQuery solves all three problems by design, not by convention.**

---

## Architecture

DocuQuery is a Clean Architecture FastAPI microservice with five strictly bounded layers. Dependency arrows point strictly inward: the presentation layer (`api`) never touches the database; the storage layer never formats response payloads.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client / Browser                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS / SSE
┌──────────────────────────▼──────────────────────────────────────┐
│                  Presentation Layer  (src/api)                   │
│   POST /query   POST /query/stream   POST /ingest/file           │
│   GET  /health  GET  /analytics/recent                           │
│   FastAPI routers · Pydantic v2 DTOs · SSE StreamingResponse    │
└──────────────┬───────────────────────────────┬──────────────────┘
               │ RAGEngine                     │ IngestionPipeline
┌──────────────▼───────────────────────────────▼──────────────────┐
│                   Domain & Core Layer  (src/core)                │
│   RAGEngine · TokenBudgetManager · build_rag_prompt              │
│   Anti-hallucination system prompt · FALLBACK_REFUSAL_MESSAGE    │
│   VectorStoreInterface · AuditRepositoryInterface (ABCs)         │
│   DocumentChunk · RetrievalResult · Citation · RAGResult         │
└──────────────┬───────────────────────────────┬──────────────────┘
               │ similarity_search             │ add_documents
┌──────────────▼──────────────┐ ┌─────────────▼──────────────────┐
│  Ingestion Layer (src/       │ │  Storage Layer  (src/storage)   │
│  ingestion)                 │ │                                  │
│  TokenSlidingWindowChunker  │ │  ChromaVectorStore               │
│  IngestionPipeline          │ │    └─ asyncio.to_thread wrapping │
│  Batch embedding (OpenAI)   │ │  SQLiteAuditRepository           │
└─────────────────────────────┘ │    └─ aiosqlite persistent conn  │
                                └──────────────────────────────────┘
```

**Query flow (non-streaming):**

```
POST /api/v1/query/
        │
        ├─ 1. Embed query          → OpenAI text-embedding-3-small
        ├─ 2. Retrieve top-k       → ChromaDB HNSW cosine similarity
        ├─ 3. Score filter         → discard chunks below score_threshold
        ├─ 4. Fast-path refusal    → if no chunks pass → return FALLBACK_REFUSAL_MESSAGE
        ├─ 5. Token budgeting      → TokenBudgetManager (cl100k_base, greedy descending score)
        ├─ 6. Prompt assembly      → build_rag_prompt() with anti-hallucination system prompt
        ├─ 7. LLM completion       → OpenAI GPT-4o-mini, temperature=0.0
        ├─ 8. Citation extraction  → one Citation per budgeted chunk
        └─ 9. Telemetry logging    → TelemetryRecord → SQLite audit trail
```

---

## Production Features

### Anti-Hallucination Guardrails

Every LLM call is governed by a system prompt that **explicitly prohibits**:
- Using knowledge beyond the provided context chunks.
- Fabricating facts, inventing sources, or speculating.
- Returning anything other than `FALLBACK_REFUSAL_MESSAGE` when context is insufficient.

When no retrieved chunk meets the similarity score threshold, the LLM is **never called** — the deterministic refusal phrase is returned immediately with zero token cost.

### Mandatory Source Citations

Every factual statement in every LLM response must carry:

```
[Source: <filename>, Section: <markdown-heading>]
```

This citation format is enforced by prompt design. Every `Citation` object in the API response maps back to an exact `DocumentChunk` in the vector index, with full provenance (`source`, `section`, `chunk_id`).

### Token Budget Enforcement

`TokenBudgetManager` applies a greedy descending-score bin-packing algorithm before every LLM call. Chunks are selected in order of semantic relevance until the configured token ceiling is reached. No unbounded context is ever injected into a prompt.

### SSE Streaming

`POST /api/v1/query/stream` yields incremental token deltas via Server-Sent Events with proxy-buffering bypass headers (`X-Accel-Buffering: no`, `Cache-Control: no-cache`). The stream terminates with the explicit `[DONE]` sentinel, compatible with browser `EventSource` and all SSE-aware HTTP clients.

### Full Async I/O

Every network call, database operation, and filesystem interaction is `async`/`await`. The sole synchronous exception — ChromaDB's local driver — is explicitly offloaded via `asyncio.to_thread()`. The FastAPI event loop is never blocked.

### Clean Architecture with Dependency Inversion

All service instances (vector store, OpenAI client, audit repository, RAG engine) are constructed **once** at application startup and shared across requests via `app.state`. Routers depend on abstract interfaces, not on concrete adapters. Swapping ChromaDB for Weaviate requires changing one file.

---

## Quickstart

### Option A: Docker Compose (Recommended for Production Preview)

```bash
# 1. Clone and configure
git clone https://github.com/your-org/docuquery-rag-agent.git
cd docuquery-rag-agent
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 2. Build and start
docker compose up --build

# 3. Verify health
curl http://localhost:8000/api/v1/health/
```

The `./data` directory on the host is bind-mounted into the container, so ChromaDB vector indices and SQLite telemetry persist across container restarts.

### Option B: Local Development

**Prerequisites:** Python 3.11+

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\Activate.ps1       # Windows PowerShell

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 4. Start the server with hot-reload
uvicorn src.main:app --reload --port 8000

# 5. Open the interactive API docs
# http://localhost:8000/api/docs
```

---

## API Reference

All endpoints are served under `/api/v1`. Interactive documentation is available at `/api/docs` (Swagger UI) and `/api/redoc`.

### Liveness Probe

```bash
curl -s http://localhost:8000/api/v1/health/ | jq
```

```json
{
  "status": "ok",
  "app_name": "DocuQuery RAG Agent",
  "version": "0.1.0",
  "environment": "development"
}
```

---

### Ingest a Document

Upload a `.md` or `.txt` file. The service chunks, embeds, and indexes the content.

```bash
curl -s -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@data/sample_docs/sla_policy.md" | jq
```

```json
{
  "status": "success",
  "filename": "sla_policy.md",
  "chunks_ingested": 42
}
```

---

### Standard Query with Citations

```bash
curl -s -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the initial response times for a P0 outage under Tier 1 Enterprise Platinum?",
    "top_k": 5,
    "score_threshold": 0.3
  }' | jq
```

```json
{
  "query": "What are the initial response times for a P0 outage under Tier 1 Enterprise Platinum?",
  "answer": "Under the Tier 1 Enterprise Platinum SLA, a P0 (Service Outage) incident — defined as complete API unavailability or data loss — requires an initial response within 15 minutes. [Source: sla_policy, Section: 2.1 Tier 1 — Enterprise Platinum]",
  "citations": [
    {
      "source": "sla_policy",
      "section": "2.1 Tier 1 — Enterprise Platinum",
      "chunk_id": "sla_policy#c0003"
    }
  ],
  "latency_ms": 1842.7,
  "total_tokens": 312
}
```

---

### Streaming Query (Token-by-Token SSE)

```bash
curl -s -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What encryption standards are used for data at rest?"}' \
  --no-buffer
```

```
data: {"token": "Data"}
data: {"token": " at"}
data: {"token": " rest"}
data: {"token": " is"}
data: {"token": " encrypted"}
data: {"token": " using"}
data: {"token": " AES"}
data: {"token": "-256"}
data: {"token": "-GCM"}
...
data: [DONE]
```

---

### Telemetry & Latency Audit

```bash
curl -s "http://localhost:8000/api/v1/analytics/recent?limit=5" | jq
```

```json
{
  "records": [
    {
      "request_id": "a3f8b21c-4d92-4e1a-b7f3-9c2d1e6a8b05",
      "query_text": "What are the P0 response times?",
      "response_text": "Under Tier 1 Enterprise Platinum...",
      "latency_ms": 1842.7,
      "prompt_tokens": 289,
      "completion_tokens": 23,
      "total_tokens": 312,
      "model_name": "gpt-4o-mini",
      "created_at": "2026-09-19T10:42:31Z"
    }
  ],
  "total_count": 1
}
```

---

### Out-of-Scope Query (Anti-Hallucination Verification)

Queries with no grounding evidence in the indexed corpus return the deterministic refusal phrase — never a hallucinated answer.

```bash
curl -s -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the meaning of life according to the documentation?"}' | jq '.answer'
```

```
"I am sorry, but the provided documentation does not contain sufficient information to answer your question."
```

---

## Quality Assurance

### Running the Test Suite

```bash
# Run all 105 tests
pytest

# Run with verbose output and coverage
pytest -v

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

All tests are fully isolated: no live OpenAI credentials, no running ChromaDB server, and no network access are required. Mocks cover all external boundaries.

### Linting and Formatting

```bash
# Check for lint violations
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/

# Check formatting without modifying files
ruff format --check src/ tests/
```

### End-to-End Smoke Test

After starting the server, run the automated smoke test to verify the full pipeline end-to-end:

```bash
# Against the local dev server
python scripts/smoke_test.py

# Against a custom URL (e.g. Docker Compose)
python scripts/smoke_test.py --base-url http://localhost:8000
```

The smoke test probes all six critical paths: health, document ingestion, grounded query with citations, out-of-scope refusal, SSE streaming, and analytics telemetry.

---

## Configuration

All configuration is read from environment variables (or `.env` file) via Pydantic Settings. See `.env.example` for the full reference.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key (or your gateway's key) |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model for completions |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `OPENAI_BASE_URL` | *(unset — uses OpenAI platform)* | Custom OpenAI-compatible gateway URL |
| `APP_ENV` | `development` | Runtime environment (`development`, `staging`, `production`) |
| `APP_PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Root log level |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma_db` | ChromaDB persistence path |
| `SQLITE_DATABASE_PATH` | `./data/telemetry.db` | SQLite audit database path |

> **OpenRouter & Custom Gateways:** Set `OPENAI_BASE_URL=https://openrouter.ai/api/v1` and `OPENAI_API_KEY=<your-openrouter-key>` to route all LLM and embedding calls through OpenRouter or any other OpenAI-compatible provider (vLLM, LocalAI, Azure OpenAI). When unset, the service targets the official OpenAI platform. No code changes required.

---

## Project Structure

```
docuquery-rag-agent/
├── src/
│   ├── config/          # Pydantic Settings — typed environment configuration
│   ├── core/
│   │   ├── models.py    # Domain models: DocumentChunk, RetrievalResult, Citation, RAGResult
│   │   ├── interfaces.py# ABCs: VectorStoreInterface, AuditRepositoryInterface
│   │   ├── exceptions.py# AppException hierarchy (ConfigurationError, StorageError, …)
│   │   └── rag/
│   │       ├── engine.py       # RAGEngine — full query and streaming pipeline
│   │       ├── token_counter.py# TokenBudgetManager — greedy bin-packing
│   │       └── prompts.py      # Anti-hallucination system prompt + citation format
│   ├── ingestion/
│   │   ├── chunker.py   # TokenSlidingWindowChunker — cl100k_base BPE sliding window
│   │   └── pipeline.py  # IngestionPipeline — chunk → embed → index
│   ├── storage/
│   │   ├── vector_store.py# ChromaVectorStore — asyncio.to_thread HNSW adapter
│   │   └── audit_db.py  # SQLiteAuditRepository — aiosqlite persistent connection
│   ├── api/
│   │   ├── routes/      # health · query · ingestion · analytics routers
│   │   ├── schemas/     # QueryRequest · QueryResponse · IngestResponse · AnalyticsResponse
│   │   └── dependencies.py# Depends factories resolving services from app.state
│   └── main.py          # FastAPI app · lifespan DI graph · exception handlers
├── tests/
│   ├── unit/            # 75 unit tests (zero external dependencies)
│   └── integration/     # 33 integration tests (mocked engine/pipeline/repo)
├── data/
│   └── sample_docs/     # Enterprise sample documents for testing and demos
├── scripts/
│   └── smoke_test.py    # Async E2E smoke test script (httpx)
├── docs/
│   └── internal/
│       ├── architecture.md  # Authoritative architecture reference + ADR log
│       └── state.md         # Living implementation state + test inventory
├── Dockerfile           # Multi-stage production image (non-root, slim)
├── docker-compose.yml   # Local development compose with bind-mounted data volume
├── pyproject.toml       # Project metadata, dependencies, Ruff, and Pytest config
└── .env.example         # Environment variable reference template
```

---

## Architectural Decision Records

The full ADR log is maintained in [`docs/internal/architecture.md`](docs/internal/architecture.md). Key decisions:

| ADR | Decision |
|---|---|
| ADR-01 | ChromaDB blocking calls dispatched via `asyncio.to_thread()` |
| ADR-02 | Single persistent `aiosqlite` connection shared across lifespan |
| ADR-03 | Embedding vectors passed as a separate parameter, not embedded in `DocumentChunk` |
| ADR-04 | Heading-inclusive scan for Markdown section attribution |
| ADR-05 | SSE streaming with `X-Accel-Buffering: no` proxy-bypass headers |
| ADR-06 | Single-instance lifespan DI graph shared between ingestion and query engines |

---

## License

MIT © 2026 DocuQuery Contributors
