"""Document upload and ingestion schemas."""
import re
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

_USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


class DocumentUploadRequest(BaseModel):
    """Metadata for document upload (file sent as multipart)."""

    filename: Optional[str] = Field(default=None, description="Original filename")


class DocumentChunk(BaseModel):
    """A single chunk of a document."""

    content: str = Field(..., description="Chunk text")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    chunk_index: int = Field(..., description="Index of chunk in document")


class DocumentMetadata(BaseModel):
    """High-level document metadata for API."""

    source: str = Field(..., description="Document source path or ID")
    filename: str = Field(..., description="Filename")
    chunk_count: int = Field(..., description="Number of chunks")


def validate_user_id_field(v: str) -> str:
    """Shared user_id rules: non-empty, no path chars."""
    if not v or not v.strip():
        raise ValueError("user_id is required")
    s = v.strip()
    if ".." in s or "/" in s or "\\" in s:
        raise ValueError("invalid user_id")
    if not _USER_ID_PATTERN.match(s):
        raise ValueError("user_id must match [a-zA-Z0-9_-]{1,128}")
    return s


class UploadResponse(BaseModel):
    """Response for POST /documents/upload."""

    path: str = Field(..., description="Path relative to uploads_dir (content-addressed under user_id/)")
    filename: str = Field(..., description="Basename on disk")
    content_hash: str = Field(..., description="SHA-256 hex of file bytes")
    deduplicated: bool = Field(..., description="True if file already existed (same user, same content)")


class IngestRequest(BaseModel):
    """Request body for POST /documents/ingest."""

    user_id: str = Field(..., description="User scope for dedupe and retrieval metadata")
    paths: Optional[list[str]] = Field(default=None, description="Paths relative to uploads_dir (under user_id/...)")
    ingest_all: Optional[bool] = Field(default=False, description="If true, ingest all supported files under this user's uploads tree")

    @field_validator("user_id")
    @classmethod
    def _user_id_ok(cls, v: str) -> str:
        return validate_user_id_field(v)


class SkippedDocument(BaseModel):
    """A path not ingested and why."""

    path: str = Field(..., description="Relative path that was skipped")
    reason: str = Field(..., description="e.g. already_ingested")


class IngestResponse(BaseModel):
    """Response for POST /documents/ingest."""

    ingested_count: int = Field(..., description="Number of source documents newly embedded")
    chunk_counts: dict[str, int] = Field(default_factory=dict, description="Source -> chunk count")
    document_metadata: list[DocumentMetadata] = Field(default_factory=list, description="Per-document metadata")
    skipped_documents: list[SkippedDocument] = Field(
        default_factory=list,
        description="Paths skipped (e.g. already ingested for this user)",
    )
