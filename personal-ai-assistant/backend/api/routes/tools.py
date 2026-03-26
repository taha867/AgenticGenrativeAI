"""MCP tools listing: for checking that MCP servers are connected."""
from fastapi import APIRouter, Depends

from backend.core.config import Settings, getSettings
from backend.tools.mcp_client import getMcpTools

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("")
async def listMcpTools(settings: Settings = Depends(getSettings)):
    """
    List MCP tools currently available (from configured servers, e.g. GitHub).
    Use this to verify MCP is working: if you see tool names, the client connected successfully.
    """
    tools = await getMcpTools(settings)
    return {
        "count": len(tools),
        "tools": [{"name": t.name, "description": (t.description or "")[:200]} for t in tools],
    }
