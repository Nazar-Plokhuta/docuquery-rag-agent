# DocuQuery RAG Agent — Project Memory & Architecture Guide

## Role & Philosophy
Staff Python Systems Architect & AI Infrastructure Engineer designing an enterprise-grade RAG microservice.
Deliver production-ready, maintainable, strictly typed code adhering to high-reliability software engineering principles.
- **Codebase Communication Standard:** All source code, architectural documentation, inline docstrings, logging schemas, and git commits must strictly follow international engineering English standards.
- **Documentation Intent:** No AI fluff or obvious comments (e.g. `# loop over items`). Document architectural intent ("why", not "what") using Google Python Style Guide.

---

## Clean Architecture Boundaries
Dependencies point strictly inward: `api` → `core`, `ingestion` → `core`, `storage` → `core`.
- **Presentation Layer (`src/api`):** Controllers, routers, request/response DTOs, SSE streaming. Zero direct DB/Chroma/OpenAI calls here.
- **Domain & Core Layer (`src/core`):** Domain entities, interfaces/ABCs, business logic, RAG orchestration, and token bounding.
- **Ingestion Layer (`src/ingestion`):** Text parsing, loaders, deterministic chunking, metadata enrichment.
- **Storage Layer (`src/storage`):** Concrete adapters for ChromaDB and SQLite audit trails.
- **Configuration (`src/config`):** Typed environment settings via Pydantic Settings (`.env`).
- **Single Responsibility:** An API endpoint must never parse chunks; a vector store adapter must never format prompt templates.

---

## Tech Stack Standards & Invariants
- **Python:** 3.11+ / 3.12+.
- **AsyncIO First:** All endpoints, DB operations, and network calls must be asynchronous (`async`/`await`). No blocking calls (`time.sleep`, synchronous `requests`) in hot paths.
- **ChromaDB Threadpool Isolation:** All synchronous/blocking ChromaDB driver interactions must be offloaded via `asyncio.to_thread()` behind repository interfaces.
- **Strict Typing:** Zero untyped function signatures. Enforce 100% type coverage. Use PEP 604 pipe syntax (`T | None`, `A | B`) instead of `Optional` or `Union`.
- **Data Validation & DTOs:** Pydantic v2 and `pydantic-settings`. Enforce `model_config = ConfigDict(extra="forbid")` on input DTOs. Domain/output models should be immutable (`frozen=True`).
- **Error Handling:** Map domain faults to typed subclasses of `AppException`. Centralized FastAPI exception handlers translate them to structured JSON envelopes. Never catch bare `Exception` without context; never write `pass` in exception blocks.
- **Secrets Isolation:** Zero hardcoded API keys, paths, or credentials. Read strictly through `Settings` (`.env`).

---

## RAG & Anti-Hallucination Guardrails
- **Strict Grounding:** System prompts must strictly forbid fabricating facts. If retrieved context is insufficient or similarity falls below threshold, return the deterministic fallback refusal phrase without calling LLM completion.
- **Mandatory Citations:** Every factual claim must cite source metadata: ``.
- **Token Budgeting:** Never inject unbounded context. Always measure, truncate, and bound context payloads using `tiktoken` (`cl100k_base`).
- **Streaming Ready:** Server-Sent Events (SSE) via `StreamingResponse` using standard format (`data: <json>\n\n`) and explicit termination sentinel (`data: [DONE]\n\n`) with proxy-bypass headers (`X-Accel-Buffering: no`, `Cache-Control: no-cache`).

---

## CLI & Verification Commands
Always use these commands to verify changes before marking tasks complete:
- **Run test suite:** `pytest -v`
- **Run unit tests:** `pytest tests/unit/`
- **Run integration tests:** `pytest tests/integration/`
- **Linting check:** `ruff check src/ tests/`
- **Auto-fix linting:** `ruff check --fix src/ tests/`
- **Formatting check:** `ruff format --check src/ tests/`
- **Apply formatting:** `ruff format src/ tests/`
- **Start server locally:** `uvicorn src.main:app --reload --port 8000`

---

## Operational Workflow for Claude Code
1. **Zero Regressions:** After any file modification, run `pytest` to guarantee all tests continue to pass (100% green suite).
2. **Offline Tests:** Do not introduce unmocked live network calls to OpenAI or external endpoints in tests.
3. **Concise Execution:** Lead directly with concrete code changes, diffs, and command execution.

---

## Git & Commit Policy
1. **No Autonomous Commits:** Never run `git commit`, `git push`, or `git tag`. Committing and pushing are strictly reserved for the human operator (Nazar).
2. **No AI Attribution / Co-Authorship:** Never append `Co-authored-by: Claude ...`, `Signed-off-by`, or any AI attribution tags to commit messages or repository metadata.
3. **Commit Messages on Request:** When explicitly asked to prepare a commit message, output only the clean Conventional Commit text (`feat:`, `fix:`, `refactor:`, `chore:`, `docs:`) without executing any Git commands.
4. **Git Inspection Only:** You are only permitted to run read-only Git inspection commands (`git status`, `git diff`, `git log`).