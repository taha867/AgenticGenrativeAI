"""
Optional web search for facts not in user documents (DuckDuckGo text API, no API key).
"""
from typing import Any

from langchain_core.tools import tool


def ddg_search_payload(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Run DuckDuckGo text search. Returns content (for ToolMessage) and web_sources for API responses.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return {
            "content": "Web search is unavailable (install duckduckgo-search).",
            "web_sources": [],
        }
    if not (query or "").strip():
        return {"content": "Empty search query.", "web_sources": []}
    try:
        with DDGS() as ddgs:
            rows = list(ddgs.text(query.strip(), max_results=max_results))
        if not rows:
            return {"content": "No web results found.", "web_sources": []}
        web_sources: list[dict[str, Any]] = []
        lines: list[str] = []
        for i, r in enumerate(rows, 1):
            title = r.get("title") or ""
            body = (r.get("body") or "")[:400]
            href = r.get("href") or ""
            web_sources.append({"title": title, "href": href, "snippet": body})
            lines.append(f"{i}. {title}\n   {body}\n   Source: {href}")
        return {"content": "\n\n".join(lines), "web_sources": web_sources}
    except Exception as e:
        return {"content": f"Web search failed: {e}", "web_sources": []}


def _run_ddg(query: str, max_results: int = 5) -> str:
    return str(ddg_search_payload(query, max_results=max_results).get("content") or "")


@tool
def web_search(query: str) -> str:
    """Search the public web for current events, people, or facts not in the user's uploaded documents. Use when retrieval context does not answer the question."""
    return _run_ddg(query)


def getWebSearchTool() -> Any:
    """Return LangChain tool for binding to the agent."""
    return web_search
