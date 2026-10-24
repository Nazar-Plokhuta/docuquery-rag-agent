# DocuQuery RAG Agent — Project Memory & Architecture Guide

## Role & Philosophy
Staff Python Systems Architect & AI Infrastructure Engineer designing an enterprise-grade RAG microservice.
Deliver production-ready, maintainable, strictly typed code adhering to high-reliability software engineering principles.
- **Codebase Communication Standard:** All source code, architectural documentation, inline docstrings, logging schemas, and git commits must strictly follow international engineering English standards.
- **Documentation Intent:** No AI fluff or obvious comments (e.g. `# loop over items`). Document architectural intent ("why", not "what") using Google Python Style Guide.

---

## Monorepo Layout
- `apps/api/` — the FastAPI service: `src/`, `tests/`, `pyproject.toml` (Ruff + Pytest config), `Dockerfile`. Also the Docker build context.
- `apps/web/` — DocuQuery Studio, the Next.js (App Router, Tailwind v4, shadcn/ui) frontend. Standalone pnpm project (no root workspace). `next.config.ts` rewrites `/api/v1/*` to `API_INTERNAL_URL` (default `http://localhost:8000`; see `apps/web/.env.example`).
- `docker/` — `docker-compose.yml`: `db` (PostgreSQL 16), `api` (build context `../apps/api`, runs `alembic upgrade head` before Uvicorn, bind-mounts the repo-level `data/`), and an optional standalone `chroma` service behind the `chroma` profile. Compose builds the api `DATABASE_URL` from `POSTGRES_*` and ignores the one in `.env`.
- `data/sample_docs/`, `scripts/`, `docs/` — shared, repository-level assets. Scripts resolve `apps/api` and `data/` from their own location, so they run from any directory.
- `.env` lives at the repository root and is shared across apps. `Settings` resolves it by absolute path; a `.env` in the working directory overrides it.
- Every `src/...` and `tests/...` path in this guide is relative to `apps/api/` (the `apps/web` section below is relative to `apps/web/`).

---

## Clean Architecture Boundaries
Dependencies point strictly inward: `api` → `core`, `ingestion` → `core`, `storage` → `core`.
- **Presentation Layer (`src/api`):** Controllers, routers, request/response DTOs, SSE streaming. Zero direct DB/Chroma/LLM-provider calls here.
- **Domain & Core Layer (`src/core`):** Domain entities, interfaces/ABCs, business logic, RAG orchestration, and token bounding.
- **Ingestion Layer (`src/ingestion`):** Text parsing, loaders, deterministic chunking, metadata enrichment.
- **Storage Layer (`src/storage`):** Concrete adapters for ChromaDB and SQLite audit trails.
- **Configuration (`src/config`):** Typed environment settings via Pydantic Settings (`.env`).
- **Single Responsibility:** An API endpoint must never parse chunks; a vector store adapter must never format prompt templates.

---

## Tech Stack Standards & Invariants
- **Python:** 3.11+ / 3.12+.
- **AsyncIO First:** All endpoints, DB operations, and network calls must be asynchronous (`async`/`await`). No blocking calls (`time.sleep`, synchronous `requests`) in hot paths.
- **ChromaDB Threadpool Isolation:** All synchronous/blocking ChromaDB driver interactions must be offloaded via `asyncio.to_thread()` behind repository interfaces. This covers both modes: local `PersistentClient` and the remote sync `HttpClient` (selected when `CHROMA_HOST` is set).
- **Strict Typing:** Zero untyped function signatures. Enforce 100% type coverage. Use PEP 604 pipe syntax (`T | None`, `A | B`) instead of `Optional` or `Union`.
- **Data Validation & DTOs:** Pydantic v2 and `pydantic-settings`. Enforce `model_config = ConfigDict(extra="forbid")` on input DTOs. Domain/output models should be immutable (`frozen=True`).
- **Error Handling:** Map domain faults to typed subclasses of `AppException`. Centralized FastAPI exception handlers translate them to structured JSON envelopes. Never catch bare `Exception` without context; never write `pass` in exception blocks.
- **Secrets Isolation:** Zero hardcoded API keys, paths, or credentials. Read strictly through `Settings` (`.env`).

---

## RAG & Anti-Hallucination Guardrails
- **Provider Abstraction (LiteLLM):** All chat completions and embeddings go through `litellm.acompletion` / `litellm.aembedding` (helpers in `src/core/llm.py`). Never import a provider SDK (`openai`, `anthropic`) directly. Model names are LiteLLM identifiers (`gpt-4o-mini`, `anthropic/claude-sonnet-4-5`, `openrouter/<vendor>/<model>`); `OPENAI_API_KEY`/`OPENAI_BASE_URL` are forwarded to OpenAI-routed models only, other providers read their own env vars. Global flags (`drop_params`, telemetry off) are set once in the lifespan via `configure_litellm`.
- **Strict Grounding:** System prompts must strictly forbid fabricating facts. If retrieved context is insufficient or similarity falls below threshold, return the deterministic fallback refusal phrase without calling LLM completion.
- **Mandatory Citations:** Every factual claim must cite source metadata: ``.
- **Token Budgeting:** Never inject unbounded context. Always measure, truncate, and bound context payloads using `tiktoken` (`cl100k_base`).
- **Streaming Ready:** Server-Sent Events (SSE) via `StreamingResponse` using standard format (`data: <json>\n\n`) and explicit termination sentinel (`data: [DONE]\n\n`) with proxy-bypass headers (`X-Accel-Buffering: no`, `Cache-Control: no-cache`).

---

## CLI & Verification Commands
Always use these commands to verify changes before marking tasks complete.
The toolchain is `uv`; `apps/api/uv.lock` is the single source of truth for dependencies. Run commands from the **repository root** with `--directory apps/api` (paths after it resolve against `apps/api/`), or drop that flag when already inside `apps/api/`. Pytest and Ruff pick up `apps/api/pyproject.toml` automatically.
- **One-time setup / sync:** `uv sync --directory apps/api` (creates `apps/api/.venv`, installs runtime + `dev` group from the lockfile)
- **Refresh lockfile after editing dependencies:** `uv lock --directory apps/api`
- **Run test suite:** `uv run --directory apps/api pytest`
- **Run unit tests:** `uv run --directory apps/api pytest tests/unit/`
- **Run integration tests:** `uv run --directory apps/api pytest tests/integration/`
- **Linting check:** `uv run --directory apps/api ruff check .`
- **Auto-fix linting:** `uv run --directory apps/api ruff check --fix .`
- **Formatting check:** `uv run --directory apps/api ruff format --check .`
- **Apply formatting:** `uv run --directory apps/api ruff format .`
- **Start server locally:** `uv run --directory apps/api uvicorn src.main:app --reload --port 8000`
- **Export OpenAPI schema (no server needed):** `uv run --project apps/api python -m src.export_openapi` (writes `docs/openapi.json`; `-o PATH` overrides)
- **Alembic migrations:** `uv run --directory apps/api alembic upgrade head` (drift check: `uv run --directory apps/api alembic check`)
- **Docker Compose:** `docker compose -f docker/docker-compose.yml --env-file .env up --build` (`--env-file` is required; Compose does not read the root `.env` for interpolation otherwise). Add `--profile chroma` and set `CHROMA_HOST=chroma` to use the standalone Chroma server. Validate with `docker compose -f docker/docker-compose.yml --env-file .env config`.

---

## Web Frontend (`apps/web`) Standards & Verification
Run from the repository root with `--dir apps/web` (or drop it inside `apps/web/`). Next.js here is 16.x: `apps/web/AGENTS.md` mandates reading `node_modules/next/dist/docs/` before relying on remembered APIs.
- **Install:** `pnpm --dir apps/web install`
- **Dev server:** `pnpm --dir apps/web dev` (port 3000; start the API on 8000 first)
- **Lint:** `pnpm --dir apps/web lint`
- **Type check:** `pnpm --dir apps/web exec tsc --noEmit`
- **Production build:** `pnpm --dir apps/web build` (must stay green; `cacheComponents` is enabled)
- **Proxy invariants (do not regress):** call FastAPI with its trailing slashes (`/api/v1/health/`); keep `skipTrailingSlashRedirect: true`, the `:path(.*)` rewrite (plain `:path*` drops the trailing slash and FastAPI then 307s to `:8000`, bypassing the proxy), and `compress: false` (Next's gzip buffers proxied SSE into one burst).
- **shadcn/ui:** components live in `src/components/ui/` and import `cn` from `@/lib/utils`. After `shadcn add`, verify no component imports `cn` from the npm package of that name.
- **Streaming check:** run the stream through `http://localhost:3000` with `curl -N` and a browser-style `-H "Accept-Encoding: gzip"`; events must arrive incrementally, not at the end.

---

## Operational Workflow for Claude Code
1. **Zero Regressions:** After any file modification, run `uv run --directory apps/api pytest` to guarantee all tests continue to pass (100% green suite). After changes under `apps/web/`, also run its lint and build.
2. **Offline Tests:** Do not introduce unmocked live network calls to LLM providers or external endpoints in tests. Patch `litellm.acompletion` / `litellm.aembedding`; `tests/conftest.py` installs an autouse guard that fails any un-mocked call.
3. **Concise Execution:** Lead directly with concrete code changes, diffs, and command execution.

---

## Git & Commit Policy
1. **No Autonomous Commits:** Never run `git commit`, `git push`, or `git tag`. Committing and pushing are strictly reserved for the human operator (Nazar).
2. **No AI Attribution / Co-Authorship:** Never append `Co-authored-by: Claude ...`, `Signed-off-by`, or any AI attribution tags to commit messages or repository metadata.
3. **Commit Messages on Request:** When explicitly asked to prepare a commit message, output only the clean Conventional Commit text (`feat:`, `fix:`, `refactor:`, `chore:`, `docs:`) without executing any Git commands.
4. **Git Inspection Only:** You are only permitted to run read-only Git inspection commands (`git status`, `git diff`, `git log`).