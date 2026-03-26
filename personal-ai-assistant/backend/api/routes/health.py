"""Health check route."""
from fastapi import APIRouter, Depends

from backend.api.schemas.health import HealthResponse
from backend.core.config import Settings, getSettings
from backend.database.vector_store import loadExisting

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def getHealth(settings: Settings = Depends(getSettings)) -> HealthResponse:
    """Check app and vector store / Postgres reachability."""
    vectorStoreLoaded = await loadExisting(settings)
    return HealthResponse(
        status="ok",
        app="personal-ai-assistant",
        vector_store_loaded=vectorStoreLoaded,
        postgres_ok=vectorStoreLoaded,
    )
