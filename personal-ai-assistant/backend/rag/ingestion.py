"""
Load PDF/TXT/MD from paths under uploads/; chunk with RecursiveCharacterTextSplitter.
Returns list of LangChain Documents and optional DocumentMetadata for API.
"""
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.api.schemas.documents import DocumentMetadata
from backend.api.utils.uploads import content_hash_from_path
from backend.core.config import Settings, getSettings


def _safePath(baseDir: str, path: str) -> Path:
    """Resolve path under baseDir; avoid path traversal."""
    base = Path(baseDir).resolve()
    full = (base / path).resolve()
    if not str(full).startswith(str(base)):
        raise ValueError(f"Path outside base: {path}")
    return full


def loadDocuments(
    paths: list[str],
    uploadsDir: Optional[str] = None,
    settings: Optional[Settings] = None,
    user_id: Optional[str] = None,
) -> list[Document]:
    """Load documents from given paths under uploadsDir; attach user_id and content_hash when provided."""
    s = settings or getSettings()
    baseDir = uploadsDir or s.uploads_dir
    docs: list[Document] = []
    for path in paths:
        p = _safePath(baseDir, path)
        if not p.exists():
            continue
        suffix = p.suffix.lower()
        batch: list[Document] = []
        if suffix == ".pdf":
            batch.extend(PyPDFLoader(str(p)).load())
        elif suffix == ".txt":
            batch.extend(TextLoader(str(p), encoding="utf-8", autodetect_encoding=True).load())
        elif suffix in (".md", ".markdown"):
            try:
                batch.extend(UnstructuredMarkdownLoader(str(p)).load())
            except Exception:
                batch.extend(TextLoader(str(p), encoding="utf-8").load())
        ch = content_hash_from_path(p)
        for d in batch:
            if "source" not in d.metadata:
                d.metadata["source"] = str(p)
            if "filename" not in d.metadata:
                d.metadata["filename"] = p.name
            if user_id:
                d.metadata["user_id"] = user_id
            d.metadata["content_hash"] = ch
        docs.extend(batch)
    return docs


def loadAndChunk(
    paths: Optional[list[str]] = None,
    ingestAll: bool = False,
    settings: Optional[Settings] = None,
    uploadsDir: Optional[str] = None,
    user_id: Optional[str] = None,
) -> tuple[list[Document], list[DocumentMetadata]]:
    """
    Load and chunk documents. If ingestAll is True, use all PDF/TXT/MD under baseDir
    (when user_id is set, baseDir is uploads_dir/user_id so only that user's tree is scanned).
    Returns (chunks, document_metadata_list).
    """
    s = settings or getSettings()
    root = Path(uploadsDir or s.uploads_dir).resolve()
    # ingest_all + user_id: only scan that user's directory; otherwise paths are relative to root (e.g. user_id/objects/...)
    if ingestAll and user_id:
        baseDir = (root / user_id).resolve()
        if not str(baseDir).startswith(str(root)):
            return [], []
    else:
        baseDir = root
    if not baseDir.exists():
        return [], []

    if ingestAll:
        paths = []
        for ext in ("*.pdf", "*.txt", "*.md", "*.markdown"):
            paths.extend(str(p.relative_to(baseDir)) for p in baseDir.rglob(ext) if p.is_file())
    if not paths:
        return [], []

    rawDocs = loadDocuments(paths, str(baseDir), s, user_id=user_id)
    if not rawDocs:
        return [], []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(rawDocs)

    sourceToCount: dict[str, int] = {}
    sourceToFilename: dict[str, str] = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        sourceToCount[src] = sourceToCount.get(src, 0) + 1
        sourceToFilename[src] = c.metadata.get("filename", Path(src).name)

    docMetadataList = [
        DocumentMetadata(source=src, filename=sourceToFilename[src], chunk_count=count)
        for src, count in sourceToCount.items()
    ]
    return chunks, docMetadataList
