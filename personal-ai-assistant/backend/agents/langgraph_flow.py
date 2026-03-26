"""
LangGraph agent: build and compile with checkpointer (STM) and store (LTM).
Sourced pipeline: START → recover_legacy → tool | remember → … → retriever →
  (docs sufficient → answer) | (sourced: reasoning_mcp → tool | reasoning_web → tool | answer) |
  (legacy: reasoning_legacy → tool | answer).
"""
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from backend.agents import nodes as agent_nodes
from backend.graph_routing import (
    route_after_recover,
    route_after_retriever,
    route_after_reasoning_legacy,
    route_after_reasoning_mcp,
    route_after_reasoning_web,
    route_after_tool,
)
from backend.core.config import Settings, getSettings


class AgentState(TypedDict, total=False):
    """State for the agent graph; messages use add_messages for checkpointer."""

    messages: Annotated[list[BaseMessage], add_messages]
    pending_ai_message: Any  # AIMessage | None; checkpoint-serializable
    summary: str
    query: str
    planned_steps: str
    retrieved_docs: list
    tool_results: list
    tool_calls_used: list
    sources: list
    retrieval_from_documents: bool
    final_answer: str
    refinement_rounds: int
    last_tool_phase: str  # "mcp" | "web"
    sourced_pipeline: bool
    web_sources: list


def buildAgentGraph(
    *,
    checkpointer: Any = None,
    store: Any = None,
    mcp_tools: Optional[list] = None,
    web_tools: Optional[list] = None,
    settings: Optional[Settings] = None,
):
    """Build and compile the agent StateGraph with optional checkpointer and store."""
    settings = settings or getSettings()
    mcp_tools = mcp_tools or []
    web_tools = web_tools or []

    builder = StateGraph(AgentState)

    async def remember(state: dict, config: RunnableConfig):
        return await agent_nodes.rememberNode(state, config, store=store, settings=settings)

    async def planner(state: dict, config: RunnableConfig):
        return await agent_nodes.plannerNode(state, config, settings=settings)

    async def retriever(state: dict, config: RunnableConfig):
        return await agent_nodes.retrieverNode(state, config, settings=settings)

    async def reasoning_mcp(state: dict, config: RunnableConfig):
        cfg = config.get("configurable") or {}
        use_tools = cfg.get("use_tools", True)
        active = mcp_tools if use_tools else []
        return await agent_nodes.reasoningMcpNode(state, config, tools=active, settings=settings, store=store)

    async def reasoning_web(state: dict, config: RunnableConfig):
        return await agent_nodes.reasoningWebNode(state, config, tools=web_tools, settings=settings, store=store)

    async def reasoning_legacy(state: dict, config: RunnableConfig):
        cfg = config.get("configurable") or {}
        use_tools = cfg.get("use_tools", True)
        merged = [*mcp_tools, *web_tools] if use_tools else []
        return await agent_nodes.reasoningLegacyNode(state, config, tools=merged, settings=settings, store=store)

    async def tool(state: dict, config: RunnableConfig):
        return await agent_nodes.toolNode(state, config, settings=settings)

    async def answer(state: dict, config: RunnableConfig):
        return await agent_nodes.answerNode(state, config, store=store, settings=settings)

    async def summarize(state: dict, config: RunnableConfig):
        return await agent_nodes.summarizeNode(state, config, settings=settings)

    async def recover_legacy(state: dict, config: RunnableConfig):
        return await agent_nodes.recoverLegacyToolCallsNode(state, config)

    def should_summarize(state: dict) -> str:
        if not settings.stm_summarize_after_messages:
            return "no"
        messages = state.get("messages") or []
        return "yes" if len(messages) > settings.stm_summarize_after_messages else "no"

    builder.add_node("recover_legacy", recover_legacy)
    builder.add_node("remember", remember)
    builder.add_node("planner", planner)
    builder.add_node("retriever", retriever)
    builder.add_node("reasoning_mcp", reasoning_mcp)
    builder.add_node("reasoning_web", reasoning_web)
    builder.add_node("reasoning_legacy", reasoning_legacy)
    builder.add_node("tool", tool)
    builder.add_node("answer", answer)
    builder.add_node("summarize", summarize)

    builder.add_edge(START, "recover_legacy")
    builder.add_conditional_edges(
        "recover_legacy",
        route_after_recover,
        {"tool": "tool", "remember": "remember"},
    )
    builder.add_edge("remember", "planner")
    builder.add_edge("planner", "retriever")
    builder.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {"answer": "answer", "reasoning_mcp": "reasoning_mcp", "reasoning_legacy": "reasoning_legacy"},
    )
    builder.add_conditional_edges(
        "reasoning_mcp",
        route_after_reasoning_mcp,
        {"tool": "tool", "reasoning_web": "reasoning_web"},
    )
    builder.add_conditional_edges(
        "reasoning_web",
        route_after_reasoning_web,
        {"tool": "tool", "answer": "answer"},
    )
    builder.add_conditional_edges(
        "reasoning_legacy",
        route_after_reasoning_legacy,
        {"tool": "tool", "answer": "answer"},
    )
    builder.add_conditional_edges(
        "tool",
        route_after_tool,
        {"reasoning_mcp": "reasoning_mcp", "reasoning_web": "reasoning_web", "reasoning_legacy": "reasoning_legacy"},
    )
    builder.add_conditional_edges("answer", should_summarize, {"yes": "summarize", "no": END})
    builder.add_edge("summarize", END)

    return builder.compile(checkpointer=checkpointer, store=store)
