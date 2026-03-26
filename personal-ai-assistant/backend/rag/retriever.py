"""Thin retriever wrapper and formatDocs for prompt formatting."""
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from backend.database.vector_store import getRetriever as _getDbRetriever


def getRetriever(
    topK: Optional[int] = None,
    settings=None,
    collectionName: str = "documents",
    metadataFilter: Optional[dict[str, Any]] = None,
) -> BaseRetriever:
    """Return a LangChain retriever over pgvector; async-friendly (ainvoke in callers)."""
    return _getDbRetriever(
        topK=topK,
        settings=settings,
        collectionName=collectionName,
        metadataFilter=metadataFilter,
    )


def formatDocs(docs: list[Document], separator: str = "\n\n---\n\n") -> str:
    """Format documents for prompt context."""
    return separator.join(d.page_content for d in docs).strip()
