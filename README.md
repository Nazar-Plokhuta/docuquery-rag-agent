# DocuQuery RAG Agent

Production-grade, asynchronous Enterprise RAG (Retrieval-Augmented Generation) microservice designed for knowledge base automation and customer support intelligence.

## Tech Stack
- **Runtime:** Python 3.11+
- **API Engine:** FastAPI (Async lifespan, Dependency Injection)
- **Vector Engine:** ChromaDB + OpenAI `text-embedding-3-small`
- **Orchestration & LLM:** OpenAI `gpt-4o-mini` with strict citation bounding
- **Data Integrity & Audit:** Pydantic v2, SQLite (`aiosqlite`)
- **Quality Gates:** Ruff, Pytest

---
*Work in progress. Architectural foundations and sprints are actively being implemented.*