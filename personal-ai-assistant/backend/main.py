"""
FastAPI app: lifespan for checkpointer/store and graph, router include only.
Run: uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()  # so GITHUB_TOKEN etc. are in os.environ for MCP client

from fastapi import FastAPI

from backend.api.routes import apiRouter
from backend.core.config import getSettings
from backend.database.checkpoint_store import getCheckpointerAndStore
from backend.database.ingest_registry import ensure_ingest_table
from backend.agents import buildAgentGraph
from backend.tools.mcp_client import getMcpTools
from backend.tools.web_search import getWebSearchTool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: open Postgres checkpointer and store, build agent graph with MCP tools. On shutdown: close."""
    settings = getSettings()
    ensure_ingest_table(settings.postgres_uri)
    async with getCheckpointerAndStore(settings) as (checkpointer, store):
        mcp_tools = await getMcpTools(settings)
        web_tools = [getWebSearchTool()]
        app.state.graph = buildAgentGraph(
            checkpointer=checkpointer,
            store=store,
            mcp_tools=mcp_tools,
            web_tools=web_tools,
            settings=settings,
        )
        yield


def createApp() -> FastAPI: 
    app = FastAPI(
        title="Personal AI Research & Knowledge Assistant",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(apiRouter)
    return app


app = createApp()
