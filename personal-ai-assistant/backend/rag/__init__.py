"""RAG domain: ingestion, retriever, Self-RAG pipeline."""
from backend.rag.ingestion import loadAndChunk
from backend.rag.retriever import getRetriever, formatDocs
from backend.rag.self_rag_pipeline import runSelfRag

__all__ = ["loadAndChunk", "getRetriever", "formatDocs", "runSelfRag"]
