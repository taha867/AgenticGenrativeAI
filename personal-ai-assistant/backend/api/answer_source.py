"""Compute API answer_source label from final agent state (no FastAPI dependency)."""


def compute_answer_source(final_state: dict) -> str:
    """
    Prefer tools over passive retrieval when both apply.
    Returns: document | web_search | tool | unsourced
    """
    tools_used = final_state.get("tool_calls_used") or []
    names = [t.get("name") for t in tools_used if isinstance(t, dict) and t.get("name")]
    if "web_search" in names:
        return "web_search"
    non_web = [n for n in names if n != "web_search"]
    if non_web:
        return "tool"
    from_docs = bool(final_state.get("retrieval_from_documents"))
    src = final_state.get("sources") or []
    if from_docs and len(src) > 0:
        return "document"
    return "unsourced"
