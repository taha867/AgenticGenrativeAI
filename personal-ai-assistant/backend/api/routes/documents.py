"""Document upload and ingest routes."""
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from backend.api.schemas.documents import (
    DocumentMetadata,
    IngestRequest,
    IngestResponse,
    SkippedDocument,
    UploadResponse,
    validate_user_id_field,
)
from backend.api.utils.uploads import content_hash_from_path, relative_path_from_uploads, save_upload_if_new
from backend.core.config import Settings, getSettings
from backend.database.ingest_registry import is_ingested, mark_ingested
from backend.database.vector_store import addDocuments
from backend.rag import loadAndChunk
from backend.rag.ingestion import _safePath

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def uploadDocument(
    file: UploadFile = File(...),
    user_id: Annotated[Optional[str], Query(description="User scope; use if not sending form field user_id")] = None,
    user_id_form: Annotated[Optional[str], Form(alias="user_id")] = None,
    settings: Settings = Depends(getSettings),
) -> UploadResponse:
    """Save uploaded file under content-addressed path per user; dedupe by hash.

    Provide **user_id** either as query ``?user_id=...`` or as multipart form field ``user_id``
    (Swagger/curl often omit the form field and only send ``file``, which caused 422 before).
    """
    uid = (user_id_form or user_id or "").strip()
    if not uid:
        raise HTTPException(
            400,
            "user_id is required: add query ?user_id=your_id or form field user_id (alphanumeric, _, -, max 128).",
        )
    try:
        validate_user_id_field(uid)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    content = await file.read()
    path, content_hash, dedup = save_upload_if_new(
        settings.uploads_dir,
        uid,
        file.filename or "file",
        content,
    )
    rel = relative_path_from_uploads(settings.uploads_dir, path)
    return UploadResponse(
        path=rel,
        filename=path.name,
        content_hash=content_hash,
        deduplicated=dedup,
    )


def _collect_paths_to_ingest(
    settings: Settings,
    user_id: str,
    postgres_uri: str,
    paths: list[str] | None,
    ingest_all: bool,
) -> tuple[list[str], list[SkippedDocument]]:
    """Return relative paths (from uploads_dir) to ingest and skipped entries."""
    root = Path(settings.uploads_dir).resolve()
    skipped: list[SkippedDocument] = []
    pending: list[str] = []

    if ingest_all:
        user_root = (root / user_id).resolve()
        if not str(user_root).startswith(str(root)):
            raise HTTPException(400, "Invalid user_id")
        if not user_root.is_dir():
            return [], []
        for ext in ("*.pdf", "*.txt", "*.md", "*.markdown"):
            for p in user_root.rglob(ext):
                if not p.is_file():
                    continue
                rel = str(p.relative_to(root)).replace("\\", "/")
                ch = content_hash_from_path(p)
                if is_ingested(postgres_uri, user_id, ch):
                    skipped.append(SkippedDocument(path=rel, reason="already_ingested"))
                else:
                    pending.append(rel)
        return pending, skipped

    if not paths:
        raise HTTPException(400, "Provide paths or set ingest_all to true")

    for rel in paths:
        try:
            full = _safePath(str(root), rel)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if not full.is_file():
            skipped.append(SkippedDocument(path=rel, reason="not_found"))
            continue
        ch = content_hash_from_path(full)
        if is_ingested(postgres_uri, user_id, ch):
            skipped.append(SkippedDocument(path=rel, reason="already_ingested"))
        else:
            pending.append(rel)
    return pending, skipped


@router.post("/ingest", response_model=IngestResponse)
async def ingestDocuments(
    body: IngestRequest,
    settings: Settings = Depends(getSettings),
) -> IngestResponse:
    """Load and chunk documents from paths (or all under user) and add to vector store; skip already-ingested hashes."""
    uid = body.user_id
    uri = settings.postgres_uri

    pending, skipped = _collect_paths_to_ingest(
        settings,
        uid,
        uri,
        body.paths,
        body.ingest_all or False,
    )

    if not pending:
        return IngestResponse(
            ingested_count=0,
            chunk_counts={},
            document_metadata=[],
            skipped_documents=skipped,
        )

    chunks, metaList = loadAndChunk(
        paths=pending,
        ingestAll=False,
        settings=settings,
        user_id=uid,
    )
    if not chunks:
        return IngestResponse(
            ingested_count=0,
            chunk_counts={},
            document_metadata=[],
            skipped_documents=skipped,
        )

    await addDocuments(chunks, settings=settings)
    hashes_written = {c.metadata.get("content_hash") for c in chunks if c.metadata.get("content_hash")}
    for h in hashes_written:
        if h:
            mark_ingested(uri, uid, h)

    chunk_counts = {}
    for m in metaList:
        chunk_counts[m.source] = m.chunk_count
    return IngestResponse(
        ingested_count=len(metaList),
        chunk_counts=chunk_counts,
        document_metadata=[
            DocumentMetadata(source=m.source, filename=m.filename, chunk_count=m.chunk_count) for m in metaList
        ],
        skipped_documents=skipped,
    )
