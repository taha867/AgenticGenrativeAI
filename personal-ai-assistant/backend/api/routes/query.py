"""Query route: POST /query runs the agent and returns answer + sources."""
import uuid

from fastapi import APIRouter, Depends, Request
from langchain_core.messages import HumanMessage

from backend.agents import buildAgentGraph
from backend.api.answer_source import compute_answer_source
from backend.api.schemas.query import QueryRequest, QueryResponse
from backend.core.config import Settings, getSettings

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def postQuery(request: Request, body: QueryRequest, settings: Settings = Depends(getSettings)) -> QueryResponse:
    """Run the agent with optional thread_id and user_id; return answer, sources, tool_calls_used."""
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        # Fallback: build graph without checkpointer/store if lifespan did not set it
        from backend.tools.mcp_client import getMcpTools
        from backend.tools.web_search import getWebSearchTool

        mcp_tools = await getMcpTools(settings)
        graph = buildAgentGraph(
            settings=settings,
            mcp_tools=mcp_tools,
            web_tools=[getWebSearchTool()],
        )
    threadId = body.thread_id or str(uuid.uuid4())
    userId = body.user_id or ""
    sourced_pipeline = body.sourced_pipeline if body.sourced_pipeline is not None else True
    config = {
        "configurable": {
            "thread_id": threadId,
            "user_id": userId,
            "max_retrieval_docs": body.max_retrieval_docs,
            "use_tools": body.use_tools if body.use_tools is not None else True,
            "sourced_pipeline": sourced_pipeline,
        }
    }
    # Per-request reset: tool_calls_used / tool_results / sources must not carry over from prior
    # POSTs on the same thread_id (checkpointer merges state; omitting keys keeps stale lists).
    # Keep summary, pending_ai_message, etc. out of initialState so STM fields persist.
    initialState = {
        "query": body.query,
        "messages": [HumanMessage(content=body.query)],
        "tool_results": [],
        "tool_calls_used": [],
        "web_sources": [],
        "sources": [],
        "retrieved_docs": [],
        "sourced_pipeline": sourced_pipeline,
    }
    finalState = await graph.ainvoke(initialState, config=config)
    answer = finalState.get("final_answer") or "No answer generated."
    sources = finalState.get("sources") or []
    toolCallsUsed = finalState.get("tool_calls_used") or []
    refinementRounds = finalState.get("refinement_rounds")
    answer_source = compute_answer_source(finalState)
    web_sources = finalState.get("web_sources") or []
    # Document chunks only when the answer is attributed to uploaded docs; otherwise PDF chunks
    # are misleading (e.g. GitHub MCP tool answers still showed curriculum snippets).
    sources_for_response = sources if answer_source == "document" else []
    return QueryResponse(
        answer=answer,
        sources=[
            {"content": s.get("content", ""), "metadata": s.get("metadata", {})}
            if isinstance(s, dict)
            else {"content": str(s), "metadata": {}}
            for s in sources_for_response
        ],
        web_sources=web_sources,
        tool_calls_used=toolCallsUsed,
        refinement_rounds=refinementRounds,
        answer_source=answer_source,
    )
