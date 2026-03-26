"""
pgvector-backed vector store for document embeddings.
Uses LangChain PGVector and OpenAIEmbeddings; same Postgres can be used for checkpointer and store.
Caches store per (uri, collection) to avoid repeated creation and PGVector __del__ teardown quirks.
"""
import asyncio
import warnings
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings

from backend.core.config import Settings, getSettings

# Cache: (postgres_uri, collection_name) -> VectorStore instance (avoids per-request PGVector creation and __del__ issues)
_vectorStoreCache: dict[tuple[str, str], VectorStore] = {}


def getEmbeddings(settings: Optional[Settings] = None) -> Embeddings:
    """Build OpenAI embeddings from settings."""
    s = settings or getSettings()
    return OpenAIEmbeddings(
        model=s.embedding_model,
        openai_api_key=s.openai_api_key,
    )


def getVectorStore(
    settings: Optional[Settings] = None,
    collectionName: str = "documents",
) -> VectorStore:
    """
    Return a PGVector store connected to POSTGRES_URI.
    Creates the collection/table if needed. Cached per (uri, collection) to avoid teardown quirks.
    """
    s = settings or getSettings()
    cacheKey = (s.postgres_uri, collectionName) # ("postgresql://chatbot:123@localhost/db", "documents")
    if cacheKey in _vectorStoreCache:
        return _vectorStoreCache[cacheKey]
    embeddings = getEmbeddings(s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langchain_community.vectorstores import PGVector

        store = PGVector(
            connection_string=s.postgres_uri,
            embedding_function=embeddings,
            collection_name=collectionName,
            use_jsonb=True, # stores metadata as JSONB
            engine_args={
                "connect_args": {"options": "-c search_path=public"},
            },
        )
    _vectorStoreCache[cacheKey] = store
    return store


async def addDocuments(
    docs: list[Document],
    settings: Optional[Settings] = None,
    collectionName: str = "documents",
) -> None:
    """Add documents to the vector store (async where supported)."""
    store = getVectorStore(settings, collectionName)
    if hasattr(store, "aadd_documents"):
        await store.aadd_documents(docs)
    else:
        store.add_documents(docs)


def getRetriever(
    topK: Optional[int] = None,
    settings: Optional[Settings] = None,
    collectionName: str = "documents",
    metadataFilter: Optional[dict[str, Any]] = None,
) -> BaseRetriever:
    """Return a retriever over the pgvector store; optional JSONB metadata filter (e.g. user_id)."""
    s = settings or getSettings()
    k = topK if topK is not None else s.retrieval_top_k
    store = getVectorStore(s, collectionName)
    search_kwargs: dict[str, Any] = {"k": k}
    if metadataFilter:
        search_kwargs["filter"] = metadataFilter
    return store.as_retriever(search_kwargs=search_kwargs)


async def loadExisting(
    settings: Optional[Settings] = None,
    collectionName: str = "documents",
) -> bool:
    """
    Verify vector store end-to-end: embeddings + DB query (not just store construction).

    Note: calls the embedding API and runs a similarity search; use sparingly on /health if cost matters.
    """
    try:
        store = getVectorStore(settings, collectionName)
        await asyncio.to_thread(store.similarity_search, "health", k=1)
        return True
    except Exception:
        return False
