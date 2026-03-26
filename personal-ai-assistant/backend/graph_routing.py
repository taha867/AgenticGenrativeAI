"""
Pure routing helpers for the phased agent graph (no LangChain imports).
"""
from typing import Literal

_NO_DOC_ANSWER_PHRASE = "No answer found."


def doc_answer_sufficient(state: dict) -> bool:
    if not state.get("retrieval_from_documents"):
        return False
    fa = (state.get("final_answer") or "").strip()
    if not fa or fa == _NO_DOC_ANSWER_PHRASE:
        return False
    return True


def route_after_recover(state: dict) -> Literal["tool", "remember"]:
    if state.get("pending_ai_message") is not None:
        return "tool"
    return "remember"


def route_after_retriever(state: dict) -> Literal["answer", "reasoning_mcp", "reasoning_legacy"]:
    """
    After Self-RAG, either go straight to final answer (legacy only) or continue to MCP.

    In sourced mode we never skip MCP/web here: loosely related PDF chunks can still yield a
    plausible but wrong draft (e.g. listing repos from memory while answer_source would say
    \"document\"). Tools such as GitHub search must always get a chance after retrieval.
    """
    if not state.get("sourced_pipeline", True):
        if doc_answer_sufficient(state):
            return "answer"
        return "reasoning_legacy"
    return "reasoning_mcp"


def route_after_reasoning_mcp(state: dict) -> Literal["tool", "reasoning_web"]:
    pending = state.get("pending_ai_message")
    if pending is not None:
        tcs = getattr(pending, "tool_calls", None) or []
        if tcs:
            return "tool"
    return "reasoning_web"


def route_after_reasoning_web(state: dict) -> Literal["tool", "answer"]:
    pending = state.get("pending_ai_message")
    if pending is not None:
        tcs = getattr(pending, "tool_calls", None) or []
        if tcs:
            return "tool"
    return "answer"


def route_after_reasoning_legacy(state: dict) -> Literal["tool", "answer"]:
    pending = state.get("pending_ai_message")
    if pending is not None:
        tcs = getattr(pending, "tool_calls", None) or []
        if tcs:
            return "tool"
    return "answer"


def route_after_tool(state: dict) -> Literal["reasoning_mcp", "reasoning_web", "reasoning_legacy"]:
    if not state.get("sourced_pipeline", True):
        return "reasoning_legacy"
    phase = state.get("last_tool_phase") or "mcp"
    if phase == "web":
        return "reasoning_web"
    return "reasoning_mcp"
