"""Health-check router.

Provides a lightweight liveness probe that returns application metadata
without touching any downstream dependency (DB, vector store, or LLM).
Downstream infrastructure (Kubernetes, Docker Compose health-checks, load
balancers) can poll this endpoint to determine service readiness.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from src.config.settings import Settings, get_settings

router = APIRouter(prefix="/health", tags=["Health"])


class HealthResponse(BaseModel):
    """Typed response payload returned by the health-check endpoint."""

    model_config = ConfigDict(frozen=True)

    status: str
    app_name: str
    version: str
    environment: str


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns service metadata. No downstream dependencies are checked.",
)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Return a lightweight liveness payload sourced entirely from configuration.

    Args:
        settings: Injected application settings via FastAPI ``Depends``.

    Returns:
        A ``HealthResponse`` confirming the service is alive.
    """
    return HealthResponse(
        status="ok",
        app_name="DocuQuery RAG Agent",
        version="0.1.0",
        environment=settings.app_env,
    )
