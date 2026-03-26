"""API routes: aggregate and mount under /api/v1."""
from fastapi import APIRouter

from backend.api.routes import documents, health, query, tools as tools_route

apiRouter = APIRouter(prefix="/api/v1")
apiRouter.include_router(health.router)  # GET /api/v1/health
apiRouter.include_router(tools_route.router)  # GET /api/v1/tools
apiRouter.include_router(documents.router)  # POST /api/v1/documents/upload, /api/v1/documents/ingest
apiRouter.include_router(query.router)  # POST /api/v1/query

__all__ = ["apiRouter"]
