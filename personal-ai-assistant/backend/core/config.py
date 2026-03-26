"""
Central configuration via Pydantic Settings.
All env and tunables in one place; no magic strings.
"""
from functools import lru_cache # we created the settings object once ans reuse it (cached)
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.core.constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_RETRIES,
    RETRIEVAL_TOP_K,
    SELF_RAG_MAX_REFINE_ROUNDS,
    STM_KEEP_LAST_N,
    STM_MAX_TOKENS_DEFAULT,
    STM_SUMMARIZE_AFTER_MESSAGES,
)


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI / LLM
    openai_api_key: str = Field(..., description="OpenAI API key")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model name")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")

    # LangSmith
    langchain_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    langchain_tracing_v2: Optional[str] = Field(default=None, description="Enable LangSmith tracing")
    langchain_project: Optional[str] = Field(default="personal-ai-assistant", description="LangSmith project name")

    # PostgreSQL (pgvector, checkpointer, store)
    postgres_uri: str = Field(
        ...,
        description="PostgreSQL connection URI for pgvector, checkpoint, and store",
    )

    # RAG (defaults from constants)
    chunk_size: int = Field(default=CHUNK_SIZE, description="Chunk size for document splitting")
    chunk_overlap: int = Field(default=CHUNK_OVERLAP, description="Chunk overlap")
    retrieval_top_k: int = Field(default=RETRIEVAL_TOP_K, description="Number of docs to retrieve")
    self_rag_max_refine_rounds: int = Field(default=SELF_RAG_MAX_REFINE_ROUNDS, description="MAX_REWRITE_TRIES for Self-RAG")

    # Memory (defaults from constants)
    stm_max_tokens: Optional[int] = Field(default=STM_MAX_TOKENS_DEFAULT, description="Max tokens for STM trimming")
    stm_summarize_after_messages: Optional[int] = Field(
        default=STM_SUMMARIZE_AFTER_MESSAGES,
        description="Run STM summarization when len(messages) > this; 0 or None = disabled",
    )
    stm_keep_last_n: int = Field(default=STM_KEEP_LAST_N, description="After summarization, keep this many most recent messages verbatim")

    @field_validator("stm_max_tokens", mode="before")
    @classmethod
    def emptyStrToNoneStmMaxTokens(cls, v: object) -> Optional[int]:
        if v is None or v == "":
            return None
        return v

    @field_validator("stm_summarize_after_messages", mode="before")
    @classmethod
    def emptyStrToNoneStmSummarize(cls, v: object) -> Optional[int]:
        if v is None or v == "":
            return None
        return int(v) if v is not None else None

    # MCP
    mcp_servers: str = Field(default="{}", description="JSON string of MCP server configs")

    # Paths
    uploads_dir: str = Field(default="uploads", description="Directory for uploaded files")


@lru_cache
def getSettings() -> Settings:
    """Cached settings dependency."""
    return Settings()


__all__ = ["Settings", "getSettings", "MAX_RETRIES"]
