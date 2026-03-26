"""Query request and response schemas."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str = Field(..., description="User query")
    use_tools: Optional[bool] = Field(default=True, description="Whether to use MCP tools")
    sourced_pipeline: Optional[bool] = Field(
        default=True,
        description="When true: documents first, then MCP tools, then web search; no Self-RAG direct generation; sourced final answer.",
    )
    max_retrieval_docs: Optional[int] = Field(
        default=None,
        description="Override retriever top_k (number of chunks). Omit or use 0 or negative for server default.",
    )
    thread_id: Optional[str] = Field(default=None, description="Conversation thread ID for STM")
    user_id: Optional[str] = Field(default=None, description="User ID for LTM and personalization")


AnswerSource = Literal["document", "web_search", "tool", "unsourced"]


class QueryResponse(BaseModel):
    """Response for POST /query."""

    answer: str = Field(..., description="Final answer")
    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Uploaded document chunks only when answer_source is document; empty for tool/web_search/unsourced",
    )
    web_sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured web results when web_search was used (title, href, snippet)",
    )
    tool_calls_used: list[dict[str, Any]] = Field(default_factory=list, description="Tools invoked")
    refinement_rounds: Optional[int] = Field(default=None, description="Self-RAG rewrite/retry count")
    answer_source: AnswerSource = Field(
        ...,
        description="document=uploaded docs; web_search=public web; tool=MCP tools; unsourced=no attributed source",
    )
