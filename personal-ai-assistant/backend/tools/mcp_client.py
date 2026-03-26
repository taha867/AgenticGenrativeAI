"""
MCP client for discovering and invoking tools from configured MCP servers.
Uses langchain-mcp-adapters MultiServerMCPClient; returns LangChain-style tools.
Supports ${VAR_NAME} in config (substituted from os.environ) and auto-adds
GitHub MCP server when GITHUB_TOKEN or GITHUB_PERSONAL_ACCESS_TOKEN is set.
"""
import json
import logging
import os
import re
from typing import Any, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

from backend.core.config import Settings, getSettings

logger = logging.getLogger(__name__)

# Cache: resolved server config (JSON string) -> (client, tools list)
_mcpCache: dict[str, tuple[Any, list[BaseTool]]] = {}

_ENV_PLACEHOLDER = re.compile(r"\$\{([^}]+)\}")


def _substituteEnv(val: Any) -> Any:
    """Replace ${VAR_NAME} in strings with os.environ.get(VAR_NAME, ''). Recursively process dicts/lists."""
    if isinstance(val, dict):
        return {k: _substituteEnv(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_substituteEnv(v) for v in val]
    if isinstance(val, str):
        return _ENV_PLACEHOLDER.sub(lambda m: os.environ.get(m.group(1), ""), val)
    return val


def _parseMcpServers(mcpServersStr: str) -> dict[str, dict[str, Any]]:
    """Parse MCP_SERVERS JSON; substitute ${VAR} from os.environ; return empty dict if invalid or empty."""
    if not (mcpServersStr or mcpServersStr.strip()):
        return {}
    try:
        parsed = json.loads(mcpServersStr)
        if not isinstance(parsed, dict):
            return {}
        return _substituteEnv(parsed)
    except json.JSONDecodeError as e:
        logger.warning("Invalid MCP_SERVERS JSON: %s", e)
        return {}


def _getResolvedServers(settings: Settings) -> dict[str, dict[str, Any]]:
    """
    Return resolved server config: parsed MCP_SERVERS with env substitution,
    plus auto-added GitHub server when GITHUB_TOKEN or GITHUB_PERSONAL_ACCESS_TOKEN is set and "github" not already in config.
    """
    servers = _parseMcpServers(settings.mcp_servers) # parses the JSON config and replaces ${VAR_NAME} placeholders with environment variables.
    github_token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if "github" not in servers and github_token:
        servers = dict(servers)
        servers["github"] = {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
        }
    return servers


async def getMcpTools(settings: Optional[Settings] = None) -> list[BaseTool]:
    """
    Discover and return LangChain tools from all configured MCP servers.
    Resolves env placeholders and auto-adds GitHub server when token is set.
    If no servers (empty or invalid), returns empty list so the app still runs.
    """
    settings = settings or getSettings()
    servers = _getResolvedServers(settings)
    if not servers:
        return []

    cacheKey = json.dumps(servers, sort_keys=True)
    if cacheKey in _mcpCache:
        _, tools = _mcpCache[cacheKey]
        return tools

    try:
        

        client = MultiServerMCPClient(servers)
        tools = await client.get_tools()
        _mcpCache[cacheKey] = (client, tools)
        return tools
    except Exception as e:
        logger.warning("Failed to load MCP tools: %s", e)
        return []


async def invokeTool(
    name: str,
    args: dict[str, Any],
    settings: Optional[Settings] = None,
) -> Any:
    """
    Invoke an MCP tool by name with the given arguments.
    Built-in `web_search` is handled without MCP servers.
    Returns the tool result (string or dict). Raises if tool not found or invocation fails.
    """
    if name == "web_search":
        from backend.tools.web_search import ddg_search_payload

        q = (args or {}).get("query", "") if isinstance(args, dict) else ""
        import asyncio

        return await asyncio.to_thread(ddg_search_payload, str(q))

    tools = await getMcpTools(settings)
    toolByName = {t.name: t for t in tools}
    if name not in toolByName:
        raise ValueError(f"Unknown MCP tool: {name}")
    tool = toolByName[name]
    return await tool.ainvoke(args)