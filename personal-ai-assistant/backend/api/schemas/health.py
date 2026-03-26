"""Health check schema."""
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(..., description="ok or error")
    app: str = Field(..., description="Application name")
    vector_store_loaded: bool = Field(..., description="Whether pgvector/table is reachable")
    postgres_ok: Optional[bool] = Field(default=None, description="Whether Postgres connection is ok")
